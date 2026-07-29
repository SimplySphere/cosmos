from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.model.architecture import GlobalMarketFactorTransformer, ModelShape
from src.utils.config import load_project
from src.utils.files import ensure_parent, read_frame
from src.utils.runtime import seed_everything, select_device

LOGGER = logging.getLogger(__name__)


class FactorWindowDataset(Dataset):
    def __init__(self, factors: np.ndarray, target_indices: np.ndarray, context_length: int) -> None:
        writable_factors = np.array(factors, dtype=np.float32, copy=True, order="C")
        self.factors = torch.from_numpy(writable_factors)
        self.target_indices = np.array(target_indices, dtype=np.int64, copy=True)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_index = int(self.target_indices[index])
        context = self.factors[target_index - self.context_length : target_index]
        target = self.factors[target_index]
        return context, target


def _shape(config: dict, factor_count: int) -> ModelShape:
    model = config["model"]
    return ModelShape(
        factor_count=factor_count,
        context_length=int(model["context_length"]),
        model_dimension=int(model["model_dimension"]),
        layers=int(model["layers"]),
        heads=int(model["heads"]),
        feedforward_dimension=int(model["feedforward_dimension"]),
        dropout=float(model["dropout"]),
        mixture_components=int(model["mixture_components"]),
        covariance_type=str(model.get("covariance_type", "full")),
        minimum_scale=float(model["minimum_scale"]),
        maximum_scale=float(model["maximum_scale"]),
        maximum_off_diagonal=float(model.get("maximum_off_diagonal", 0.75)),
        minimum_degrees_of_freedom=float(model["minimum_degrees_of_freedom"]),
        maximum_degrees_of_freedom=float(model["maximum_degrees_of_freedom"]),
    )


def _indices_for_period(
    dates: pd.DatetimeIndex,
    factor_values: np.ndarray,
    context_length: int,
    start_exclusive: str | None,
    end_inclusive: str,
) -> np.ndarray:
    valid = np.arange(context_length, len(dates))
    target_dates = dates[valid]
    mask = target_dates <= pd.Timestamp(end_inclusive)
    if start_exclusive is not None:
        mask &= target_dates > pd.Timestamp(start_exclusive)
    candidate = valid[mask]
    complete: list[int] = []
    for target_index in candidate:
        window = factor_values[target_index - context_length : target_index + 1]
        if np.isfinite(window).all():
            complete.append(int(target_index))
    return np.asarray(complete, dtype=np.int64)


def _learning_rate_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _factor_normalizer(
    factor_values: np.ndarray,
    dates: pd.DatetimeIndex,
    fit_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    fit = factor_values[dates <= pd.Timestamp(fit_end)]
    fit = fit[np.isfinite(fit).all(axis=1)]
    if len(fit) < 2:
        raise ValueError("Too few finite factor rows to fit the model normalizer")
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _save_checkpoint(
    path: Path,
    model: GlobalMarketFactorTransformer,
    shape: ModelShape,
    epoch: int,
    validation_loss: float | None,
    training_config: dict,
    factor_mean: np.ndarray,
    factor_scale: np.ndarray,
    device: torch.device,
    metadata: dict | None = None,
) -> None:
    ensure_parent(path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "shape": asdict(shape),
            "epoch": int(epoch),
            "validation_loss": validation_loss,
            "training_config": training_config,
            "factor_mean": np.asarray(factor_mean, dtype=np.float32),
            "factor_scale": np.asarray(factor_scale, dtype=np.float32),
            "calibration_scale": float(model.calibration_scale.detach().cpu()),
            "device_used": str(device),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            **(metadata or {}),
        },
        path,
    )


def _evaluate_loader(
    model: GlobalMarketFactorTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for context, target in loader:
            loss = model.negative_log_likelihood(context.to(device), target.to(device))
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def _calibrate_scale(
    model: GlobalMarketFactorTransformer,
    validation_loader: DataLoader,
    device: torch.device,
    grid: list[float],
) -> tuple[float, float]:
    original = float(model.calibration_scale.detach().cpu())
    best_scale = original
    best_loss = float("inf")
    for value in grid:
        model.set_calibration_scale(float(value))
        loss = _evaluate_loader(model, validation_loader, device)
        if loss < best_loss:
            best_loss = loss
            best_scale = float(value)
    model.set_calibration_scale(best_scale)
    return best_scale, best_loss


def train_model(config: dict, paths: dict, phase: str, force: bool = False) -> None:
    if phase not in {"development", "final"}:
        raise ValueError("phase must be development or final")
    checkpoint_path = Path(
        paths["artifacts"][
            "development_checkpoint" if phase == "development" else "final_checkpoint"
        ]
    )
    if checkpoint_path.exists() and not force:
        LOGGER.info("Using existing %s checkpoint", phase)
        return

    factor_path = Path(
        paths["artifacts"]["development_factors" if phase == "development" else "final_factors"]
    )
    factor_frame = read_frame(factor_path).sort_index()
    raw_factor_values = np.array(
        factor_frame.to_numpy(dtype=np.float32), dtype=np.float32, copy=True, order="C"
    )
    dates = pd.DatetimeIndex(factor_frame.index)
    shape = _shape(config, raw_factor_values.shape[1])
    if len(factor_frame) <= shape.context_length:
        raise ValueError("Not enough factor weeks for the configured context length")

    fit_end = config["data"]["train_end"] if phase == "development" else config["data"]["validation_end"]
    factor_mean, factor_scale = _factor_normalizer(raw_factor_values, dates, fit_end)
    factor_values = (raw_factor_values - factor_mean) / factor_scale

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = select_device(str(config["project"]["device"]))
    model = GlobalMarketFactorTransformer(shape).to(device)
    training = config["training"]
    batch_size = int(training["batch_size"])

    if phase == "development":
        train_indices = _indices_for_period(
            dates,
            factor_values,
            shape.context_length,
            None,
            config["data"]["train_end"],
        )
        # Split 2025 into two strictly ordered roles: the earlier block chooses the
        # checkpoint/epoch, while the later block calibrates predictive dispersion.
        # Calibration therefore never reuses the observations that selected the model.
        validation_indices = _indices_for_period(
            dates,
            factor_values,
            shape.context_length,
            config["data"]["train_end"],
            config["data"]["model_selection_end"],
        )
        calibration_indices = _indices_for_period(
            dates,
            factor_values,
            shape.context_length,
            config["data"]["model_selection_end"],
            config["data"]["validation_end"],
        )
        maximum_epochs = int(training["maximum_epochs"])
        selected_calibration = 1.0
    else:
        development_checkpoint = torch.load(
            paths["artifacts"]["development_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        maximum_epochs = int(development_checkpoint["epoch"]) + 1
        selected_calibration = float(development_checkpoint.get("calibration_scale", 1.0))
        train_indices = _indices_for_period(
            dates,
            factor_values,
            shape.context_length,
            None,
            config["data"]["validation_end"],
        )
        validation_indices = np.array([], dtype=np.int64)
        calibration_indices = np.array([], dtype=np.int64)

    if len(train_indices) == 0:
        raise ValueError(f"No finite training windows are available for the {phase} phase")
    if phase == "development" and len(validation_indices) == 0:
        raise ValueError("No finite model-selection windows fall inside the configured 2025 selection period")
    if phase == "development" and len(calibration_indices) == 0:
        raise ValueError("No finite calibration windows fall after model_selection_end")

    checkpoint_metadata = {
        "phase": phase,
        "fit_end": str(pd.Timestamp(fit_end).date()),
        "factor_count": int(shape.factor_count),
        "train_target_start": str(dates[train_indices].min().date()),
        "train_target_end": str(dates[train_indices].max().date()),
        "validation_target_start": (
            None if len(validation_indices) == 0 else str(dates[validation_indices].min().date())
        ),
        "validation_target_end": (
            None if len(validation_indices) == 0 else str(dates[validation_indices].max().date())
        ),
        "calibration_target_start": (
            None if len(calibration_indices) == 0 else str(dates[calibration_indices].min().date())
        ),
        "calibration_target_end": (
            None if len(calibration_indices) == 0 else str(dates[calibration_indices].max().date())
        ),
        "model_selection_end": str(pd.Timestamp(config["data"]["model_selection_end"]).date()),
        "post_validation_targets_used": bool(
            dates[train_indices].max() > pd.Timestamp(config["data"]["validation_end"])
        ),
    }

    train_dataset = FactorWindowDataset(factor_values, train_indices, shape.context_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training["num_workers"]),
        generator=generator,
    )
    validation_loader = None
    calibration_loader = None
    if len(validation_indices):
        validation_loader = DataLoader(
            FactorWindowDataset(factor_values, validation_indices, shape.context_length),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training["num_workers"]),
        )
    if len(calibration_indices):
        calibration_loader = DataLoader(
            FactorWindowDataset(factor_values, calibration_indices, shape.context_length),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training["num_workers"]),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = max(len(train_loader) * maximum_epochs, 1)
    warmup_steps = int(total_steps * float(training["warmup_fraction"]))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_multiplier(step, total_steps, warmup_steps),
    )

    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(maximum_epochs):
        model.train()
        training_losses: list[float] = []
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(context, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            training_losses.append(float(loss.detach().cpu()))

        validation_loss: float | None = None
        if validation_loader is not None:
            validation_loss = _evaluate_loader(model, validation_loader, device)
        train_loss = float(np.mean(training_losses))
        history.append(
            {
                "phase": phase,
                "epoch": epoch + 1,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected_checkpoint": False,
                "calibration_scale": 1.0,
            }
        )
        LOGGER.info(
            "%s epoch %d/%d | train %.5f | validation %s",
            phase,
            epoch + 1,
            maximum_epochs,
            train_loss,
            "n/a" if validation_loss is None else f"{validation_loss:.5f}",
        )

        if phase == "final":
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path,
                model,
                shape,
                epoch,
                None,
                training,
                factor_mean,
                factor_scale,
                device,
                checkpoint_metadata,
            )
        else:
            monitored = float(validation_loss)
            if monitored < best_loss - 1e-6:
                best_loss = monitored
                best_epoch = epoch
                stale_epochs = 0
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    shape,
                    epoch,
                    validation_loss,
                    training,
                    factor_mean,
                    factor_scale,
                    device,
                    checkpoint_metadata,
                )
            else:
                stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                LOGGER.info("Early stopping after epoch %d", epoch + 1)
                break

    if phase == "development":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        grid = [float(value) for value in training.get("calibration_scale_grid", [1.0])]
        if calibration_loader is None:
            raise RuntimeError("Development calibration loader was not created")
        selected_calibration, calibrated_loss = _calibrate_scale(
            model, calibration_loader, device, grid
        )
        _save_checkpoint(
            checkpoint_path,
            model,
            shape,
            int(checkpoint["epoch"]),
            calibrated_loss,
            training,
            factor_mean,
            factor_scale,
            device,
            checkpoint_metadata,
        )
        LOGGER.info(
            "Selected holdout covariance calibration scale %.3f (NLL %.5f)",
            selected_calibration,
            calibrated_loss,
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.set_calibration_scale(selected_calibration)
        _save_checkpoint(
            checkpoint_path,
            model,
            shape,
            best_epoch,
            None,
            training,
            factor_mean,
            factor_scale,
            device,
            checkpoint_metadata,
        )

    for row in history:
        if int(row["epoch"]) == best_epoch + 1:
            row["selected_checkpoint"] = True
            row["calibration_scale"] = selected_calibration
    history_path = Path(paths["artifacts"]["training_history"])
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if "phase" in existing.columns:
        existing = existing[existing["phase"] != phase]
    else:
        existing = pd.DataFrame()
    combined = pd.DataFrame(existing.to_dict("records") + history)
    combined.to_csv(ensure_parent(history_path), index=False)
    LOGGER.info(
        "Saved %s checkpoint from epoch %d with %d parameters",
        phase,
        best_epoch + 1,
        sum(parameter.numel() for parameter in model.parameters()),
    )


def load_model_bundle(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GlobalMarketFactorTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    shape = ModelShape(**checkpoint["shape"])
    model = GlobalMarketFactorTransformer(shape)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_model(checkpoint_path: str | Path, device: torch.device) -> GlobalMarketFactorTransformer:
    return load_model_bundle(checkpoint_path, device)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train development or final transformer model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--phase", choices=["development", "final", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    phases = ["development", "final"] if args.phase == "both" else [args.phase]
    for phase in phases:
        train_model(config, paths, phase, force=args.force)


if __name__ == "__main__":
    main()
