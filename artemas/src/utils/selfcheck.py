from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _check_adjacent_week_returns() -> None:
    from src.data.preprocess import _adjacent_log_return

    dates = pd.date_range("2026-01-02", periods=4, freq="W-FRI")
    prices = pd.Series([100.0, np.nan, 121.0, 133.1], index=dates)
    returns = _adjacent_log_return(prices)
    if not np.isnan(returns.iloc[2]):
        raise RuntimeError("Missing weeks were bridged into a multi-week return")
    if not np.isclose(returns.iloc[3], np.log(133.1 / 121.0), atol=1e-12):
        raise RuntimeError("Adjacent weekly return calculation is incorrect")


def _check_currency_inference() -> None:
    from src.data.universe import _infer_listing_currency

    cases = {
        ("2330.TW", "TW"): "TWD",
        ("005930.KS", "KR"): "KRW",
        ("0700.HK", "HK"): "HKD",
        ("HSBA.L", "GB"): "GBp",
        ("AAPL", "US"): "USD",
    }
    for inputs, expected in cases.items():
        actual, _ = _infer_listing_currency(*inputs)
        if actual != expected:
            raise RuntimeError(f"Currency inference failed for {inputs}: {actual} != {expected}")


def _check_full_covariance_model(config: dict) -> None:
    from src.model.architecture import GlobalMarketFactorTransformer, ModelShape

    rank = 4
    model_cfg = config["model"]
    shape = ModelShape(
        factor_count=rank,
        context_length=8,
        model_dimension=32,
        layers=1,
        heads=4,
        feedforward_dimension=64,
        dropout=0.0,
        mixture_components=3,
        covariance_type="full",
        minimum_scale=float(model_cfg["minimum_scale"]),
        maximum_scale=float(model_cfg["maximum_scale"]),
        maximum_off_diagonal=float(model_cfg["maximum_off_diagonal"]),
        minimum_degrees_of_freedom=float(model_cfg["minimum_degrees_of_freedom"]),
        maximum_degrees_of_freedom=float(model_cfg["maximum_degrees_of_freedom"]),
    )
    model = GlobalMarketFactorTransformer(shape)
    x = torch.randn(5, shape.context_length, rank)
    y = torch.randn(5, rank)
    loss = model.negative_log_likelihood(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Full-covariance Student-t loss was non-finite")
    loss.backward()
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("Full-covariance model did not produce finite gradients")
    params = model.distribution_parameters(x)
    diagonal = torch.diagonal(params["cholesky"], dim1=-2, dim2=-1)
    if not torch.all(diagonal > 0):
        raise RuntimeError("Student-t Cholesky factor has a non-positive diagonal")


def _check_expanding_splits() -> None:
    from src.factors.selection import _expanding_time_splits

    splits = _expanding_time_splits(420, 3, 180)
    previous_train = 0
    for train, validation in splits:
        if len(train) <= previous_train:
            raise RuntimeError("Expanding fold training window did not grow")
        if train.max() >= validation.min():
            raise RuntimeError("Expanding fold leaks future weeks into training")
        previous_train = len(train)


def run_self_checks(config: dict) -> None:
    _check_adjacent_week_returns()
    _check_currency_inference()
    _check_full_covariance_model(config)
    _check_expanding_splits()
