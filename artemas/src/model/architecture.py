from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelShape:
    factor_count: int
    context_length: int
    model_dimension: int
    layers: int
    heads: int
    feedforward_dimension: int
    dropout: float
    mixture_components: int
    covariance_type: str
    minimum_scale: float
    maximum_scale: float
    maximum_off_diagonal: float
    minimum_degrees_of_freedom: float
    maximum_degrees_of_freedom: float


class GlobalMarketFactorTransformer(nn.Module):
    """Causal transformer with a bounded multivariate Student-t mixture head."""

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        if shape.factor_count < 1:
            raise ValueError("factor_count must be positive")
        if shape.context_length < 1:
            raise ValueError("context_length must be positive")
        if shape.model_dimension % shape.heads != 0:
            raise ValueError("model_dimension must be divisible by heads")
        if shape.covariance_type not in {"diagonal", "full"}:
            raise ValueError("covariance_type must be 'diagonal' or 'full'")
        if not 0 < shape.minimum_scale < shape.maximum_scale:
            raise ValueError("minimum_scale must be positive and below maximum_scale")
        if not 2 < shape.minimum_degrees_of_freedom < shape.maximum_degrees_of_freedom:
            raise ValueError("degrees-of-freedom bounds must satisfy 2 < minimum < maximum")

        self.shape = shape
        self.input_projection = nn.Linear(shape.factor_count, shape.model_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, shape.context_length, shape.model_dimension)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shape.model_dimension,
            nhead=shape.heads,
            dim_feedforward=shape.feedforward_dimension,
            dropout=shape.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=shape.layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(shape.model_dimension)

        factors = shape.factor_count
        mixtures = shape.mixture_components
        if shape.covariance_type == "full":
            covariance_parameters = factors * (factors + 1) // 2
            output_size = mixtures * (2 + factors + covariance_parameters)
        else:
            output_size = mixtures * (2 + 2 * factors)
        self.output_projection = nn.Linear(shape.model_dimension, output_size)

        causal_mask = torch.full(
            (shape.context_length, shape.context_length), float("-inf")
        )
        self.register_buffer(
            "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
        )
        self.register_buffer(
            "calibration_scale", torch.tensor(1.0, dtype=torch.float32), persistent=True
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def set_calibration_scale(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("calibration scale must be positive and finite")
        self.calibration_scale.fill_(float(value))

    def _bounded_scale(self, raw: torch.Tensor) -> torch.Tensor:
        span = self.shape.maximum_scale - self.shape.minimum_scale
        return self.shape.minimum_scale + span * torch.sigmoid(raw)

    def _bounded_df(self, raw: torch.Tensor) -> torch.Tensor:
        span = (
            self.shape.maximum_degrees_of_freedom
            - self.shape.minimum_degrees_of_freedom
        )
        return self.shape.minimum_degrees_of_freedom + span * torch.sigmoid(raw)

    def _full_cholesky(self, raw: torch.Tensor) -> torch.Tensor:
        batch, mixtures, _ = raw.shape
        factors = self.shape.factor_count
        rows, columns = torch.tril_indices(factors, factors, device=raw.device)
        transformed = torch.tanh(raw) * self.shape.maximum_off_diagonal
        diagonal_positions = rows == columns
        transformed[..., diagonal_positions] = self._bounded_scale(
            raw[..., diagonal_positions]
        )
        cholesky = raw.new_zeros((batch, mixtures, factors, factors))
        cholesky[..., rows, columns] = transformed
        return cholesky * self.calibration_scale.to(raw.dtype)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, time, factors]")
        if context.shape[1] != self.shape.context_length:
            raise ValueError(
                f"Expected context length {self.shape.context_length}, got {context.shape[1]}"
            )
        if context.shape[2] != self.shape.factor_count:
            raise ValueError(
                f"Expected {self.shape.factor_count} factors, got {context.shape[2]}"
            )
        hidden = self.input_projection(context) + self.position_embedding
        hidden = self.encoder(hidden, mask=self.causal_mask)
        final = self.final_norm(hidden[:, -1])
        raw = self.output_projection(final)

        batch = context.shape[0]
        mixtures = self.shape.mixture_components
        factors = self.shape.factor_count
        cursor = 0
        logits = raw[:, cursor : cursor + mixtures]
        cursor += mixtures
        loc = raw[:, cursor : cursor + mixtures * factors].reshape(
            batch, mixtures, factors
        )
        cursor += mixtures * factors

        if self.shape.covariance_type == "full":
            covariance_count = factors * (factors + 1) // 2
            raw_covariance = raw[
                :, cursor : cursor + mixtures * covariance_count
            ].reshape(batch, mixtures, covariance_count)
            cursor += mixtures * covariance_count
            covariance = self._full_cholesky(raw_covariance)
        else:
            raw_scale = raw[:, cursor : cursor + mixtures * factors].reshape(
                batch, mixtures, factors
            )
            cursor += mixtures * factors
            covariance = self._bounded_scale(raw_scale) * self.calibration_scale.to(
                raw_scale.dtype
            )

        raw_df = raw[:, cursor : cursor + mixtures]
        degrees_of_freedom = self._bounded_df(raw_df)
        return logits, loc, covariance, degrees_of_freedom

    def component_log_prob(
        self,
        target: torch.Tensor,
        loc: torch.Tensor,
        covariance: torch.Tensor,
        degrees_of_freedom: torch.Tensor,
    ) -> torch.Tensor:
        factors = self.shape.factor_count
        difference = target[:, None, :] - loc
        df = degrees_of_freedom
        if self.shape.covariance_type == "full":
            solved = torch.linalg.solve_triangular(
                covariance,
                difference.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            mahalanobis = torch.sum(solved * solved, dim=-1)
            log_scale = torch.log(
                torch.diagonal(covariance, dim1=-2, dim2=-1)
            ).sum(dim=-1)
        else:
            z = difference / covariance
            mahalanobis = torch.sum(z * z, dim=-1)
            log_scale = torch.log(covariance).sum(dim=-1)

        log_normalizer = (
            torch.lgamma((df + factors) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * factors * torch.log(df * math.pi)
            - log_scale
        )
        log_kernel = -0.5 * (df + factors) * torch.log1p(mahalanobis / df)
        return log_normalizer + log_kernel

    def negative_log_likelihood(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logits, loc, covariance, degrees_of_freedom = self(context)
        component_log_prob = self.component_log_prob(
            target, loc, covariance, degrees_of_freedom
        )
        mixture_log_prob = torch.log_softmax(logits, dim=-1) + component_log_prob
        return -torch.logsumexp(mixture_log_prob, dim=-1).mean()

    @torch.no_grad()
    def distribution_parameters(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, loc, covariance, degrees_of_freedom = self(context)
        result = {
            "probabilities": torch.softmax(logits, dim=-1),
            "loc": loc,
            "degrees_of_freedom": degrees_of_freedom,
        }
        if self.shape.covariance_type == "full":
            result["cholesky"] = covariance
        else:
            result["scale"] = covariance
        return result
