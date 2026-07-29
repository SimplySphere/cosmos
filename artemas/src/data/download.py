from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.universe import prepare_universe
from src.utils.config import load_project
from src.utils.files import ensure_parent, read_frame, write_frame
from src.utils.runtime import batches

LOGGER = logging.getLogger(__name__)

# Yahoo sometimes reports securities in currency subunits. The multiplier converts
# one quoted price unit into one unit of the canonical ISO currency.
_CURRENCY_ALIASES: dict[str, tuple[str, float]] = {
    "GBX": ("GBP", 0.01),
    "GBP": ("GBP", 1.0),
    "GBPENCE": ("GBP", 0.01),
    "GBPENNY": ("GBP", 0.01),
    "GBPEN": ("GBP", 0.01),
    "GBp": ("GBP", 0.01),
    "ZAC": ("ZAR", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
    # Yahoo labels Boursa Kuwait quotes as KWF (Kuwaiti fils).
    # One Kuwaiti dinar is 1,000 fils, so convert quoted units to KWD.
    "KWF": ("KWD", 0.001),
    "USC": ("USD", 0.01),
    "USc": ("USD", 0.01),
    "CAD": ("CAD", 1.0),
    "USD": ("USD", 1.0),
}


def normalize_currency(value: object) -> tuple[str, float]:
    raw = str(value or "").strip()
    if not raw:
        return "", 1.0
    if raw in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[raw]
    upper = raw.upper().replace(" ", "")
    if upper in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[upper]
    if len(upper) == 3 and upper.isalpha():
        return upper, 1.0
    return upper, 1.0


def _safe_name(identifier: str) -> str:
    return identifier.replace("/", "_").replace("^", "INDEX_").replace("=", "_")


def _requested_end(config: dict) -> pd.Timestamp:
    configured = config["data"].get("download_end")
    if configured:
        end = pd.Timestamp(configured)
    else:
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if end <= pd.Timestamp(config["data"]["validation_end"]):
        raise ValueError("data.download_end must be after data.validation_end")
    return end


def _extract_ticker_frame(payload: pd.DataFrame, ticker: str, ticker_count: int) -> pd.DataFrame:
    if payload.empty:
        raise ValueError("No price rows returned")

    frame: pd.DataFrame
    if isinstance(payload.columns, pd.MultiIndex):
        first_level = set(map(str, payload.columns.get_level_values(0)))
        second_level = set(map(str, payload.columns.get_level_values(1)))
        if ticker in first_level:
            frame = payload.xs(ticker, axis=1, level=0, drop_level=True).copy()
        elif ticker in second_level:
            frame = payload.xs(ticker, axis=1, level=1, drop_level=True).copy()
        elif ticker_count == 1:
            level = 0 if len(first_level) == 1 else 1
            frame = payload.xs(
                payload.columns.get_level_values(level)[0], axis=1, level=level
            ).copy()
        else:
            raise KeyError(f"Ticker {ticker} was not present in the returned columns")
    else:
        if ticker_count != 1:
            raise KeyError(f"Expected multi-ticker columns for {ticker_count} requested tickers")
        frame = payload.copy()

    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError("No non-empty price rows returned")
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None)
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "date"
    return frame


def _completed_security_ids(
    manifest: pd.DataFrame,
    requested_end: pd.Timestamp,
    raw_dir: Path,
) -> set[str]:
    if manifest.empty or "status" not in manifest.columns:
        return set()
    completed: set[str] = set()
    for row in manifest[manifest["status"] == "ok"].itertuples(index=False):
        security_id = str(row.security_id)
        destination = raw_dir / f"{_safe_name(security_id)}.parquet"
        last_date = pd.to_datetime(getattr(row, "last_date", None), errors="coerce")
        sufficiently_current = pd.notna(last_date) and last_date >= requested_end - pd.Timedelta(days=10)
        if destination.exists() and sufficiently_current:
            completed.add(security_id)
    return completed


def _configure_yfinance_cache(yf: object, paths: dict) -> Path:
    cache_dir = Path(paths["data"]["vanguard_holdings_dir"]).parent / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    setter = getattr(yf, "set_tz_cache_location", None)
    if callable(setter):
        setter(str(cache_dir))
    return cache_dir


def _download_with_retries(
    yf: object,
    tickers: list[str] | str,
    *,
    start: str,
    end: str,
    retries: int,
    backoff: float,
    threads: bool,
    group_by: str | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            kwargs = {
                "tickers": tickers,
                "start": start,
                "end": end,
                "auto_adjust": False,
                "actions": False,
                "threads": threads,
                "progress": False,
            }
            if group_by is not None:
                kwargs["group_by"] = group_by
            return yf.download(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                delay = backoff * (2**attempt)
                LOGGER.warning(
                    "Yahoo download attempt %d/%d failed for %d ticker(s): %s; retrying in %.1fs",
                    attempt + 1,
                    retries,
                    len(tickers) if isinstance(tickers, list) else 1,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
    assert last_error is not None
    raise last_error


def _write_price_checkpoint(
    existing_manifest: pd.DataFrame,
    records: list[dict],
    failures: list[dict],
    universe: pd.DataFrame,
    manifest_path: Path,
    failures_path: Path,
) -> pd.DataFrame:
    pieces = [frame for frame in (existing_manifest, pd.DataFrame(records)) if not frame.empty]
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not combined.empty:
        if "downloaded_at" in combined.columns:
            combined = combined.sort_values("downloaded_at")
        combined = combined.drop_duplicates("security_id", keep="last")
        write_frame(combined, ensure_parent(manifest_path), index=False)

    failure_columns = list(universe.columns) + ["error"]
    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=failure_columns)
    else:
        for column in failure_columns:
            if column not in failure_frame.columns:
                failure_frame[column] = ""
        failure_frame = failure_frame.reindex(columns=failure_columns)
        failure_frame = failure_frame.drop_duplicates("security_id", keep="last")
    failure_frame.to_csv(ensure_parent(failures_path), index=False)
    return combined


def _single_price_retry(
    yf: object,
    row: dict,
    *,
    start: str,
    end: str,
    retries: int,
    backoff: float,
) -> tuple[pd.DataFrame | None, str | None]:
    ticker = str(row["yahoo_ticker"])
    try:
        payload = _download_with_retries(
            yf,
            ticker,
            start=start,
            end=end,
            retries=retries,
            backoff=backoff,
            threads=False,
        )
        return _extract_ticker_frame(payload, ticker, 1), None
    except Exception as exc:
        return None, repr(exc)


def download_prices(config: dict, paths: dict, force: bool = False) -> pd.DataFrame:
    import yfinance as yf

    universe = prepare_universe(config, paths, force=False)
    raw_dir = Path(paths["data"]["raw_prices_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(paths["data"]["download_manifest"])
    failures_path = Path(paths["data"]["download_failures"])
    cache_dir = _configure_yfinance_cache(yf, paths)
    LOGGER.info("Using project-local yfinance cache for prices: %s", cache_dir)

    existing_manifest = (
        read_frame(manifest_path) if manifest_path.exists() and not force else pd.DataFrame()
    )
    requested_end = _requested_end(config)
    completed = _completed_security_ids(existing_manifest, requested_end, raw_dir)

    start = str(config["data"]["start_date"])
    exclusive_end = requested_end + pd.Timedelta(days=1)
    end_string = exclusive_end.strftime("%Y-%m-%d")
    batch_size = int(config["data"]["download_batch_size"])
    pause = float(config["data"]["download_pause_seconds"])
    retries = int(config["data"].get("download_retries", 3))
    backoff = float(config["data"].get("download_retry_backoff_seconds", 5.0))
    failure_limit = int(config["data"].get("download_consecutive_failure_limit", 3))
    checkpoint_batches = max(1, int(config["data"].get("download_checkpoint_batches", 1)))
    individual_retry = bool(config["data"].get("download_individual_retry", True))

    records: list[dict] = []
    failures: list[dict] = []
    rows = universe.to_dict("records")
    pending = [row for row in rows if force or str(row["security_id"]) not in completed]
    total_batches = (len(pending) + batch_size - 1) // batch_size
    LOGGER.info(
        "Price acquisition: %d already current, %d pending across %d batches",
        len(completed),
        len(pending),
        total_batches,
    )

    consecutive_failed_batches = 0
    processed_batches = 0
    try:
        for group in tqdm(batches(pending, batch_size), total=total_batches, desc="Downloading price batches"):
            processed_batches += 1
            tickers = [str(row["yahoo_ticker"]) for row in group]
            try:
                payload = _download_with_retries(
                    yf,
                    tickers,
                    start=start,
                    end=end_string,
                    retries=retries,
                    backoff=backoff,
                    threads=True,
                    group_by="ticker",
                )
            except Exception as exc:
                consecutive_failed_batches += 1
                for row in group:
                    failures.append({**row, "error": repr(exc)})
                _write_price_checkpoint(
                    existing_manifest, records, failures, universe, manifest_path, failures_path
                )
                if consecutive_failed_batches >= failure_limit:
                    raise RuntimeError(
                        "Yahoo price acquisition failed for "
                        f"{consecutive_failed_batches} consecutive batches. Successful files and "
                        "the download manifest were checkpointed. Re-run the same pipeline command; "
                        "completed securities will be skipped automatically. Last error: "
                        f"{exc!r}"
                    ) from exc
                if pause > 0:
                    time.sleep(pause)
                continue

            successful_in_batch = 0
            for row in group:
                ticker = str(row["yahoo_ticker"])
                frame: pd.DataFrame | None = None
                error: str | None = None
                try:
                    frame = _extract_ticker_frame(payload, ticker, len(tickers))
                except Exception as exc:
                    error = repr(exc)
                    if individual_retry:
                        frame, retry_error = _single_price_retry(
                            yf,
                            row,
                            start=start,
                            end=end_string,
                            retries=retries,
                            backoff=backoff,
                        )
                        if retry_error is not None:
                            error = f"batch={error}; individual={retry_error}"

                if frame is None:
                    failures.append({**row, "error": error or "No usable Yahoo price history"})
                    continue

                destination = raw_dir / f"{_safe_name(str(row['security_id']))}.parquet"
                write_frame(frame, destination, index=True)
                records.append(
                    {
                        "security_id": row["security_id"],
                        "yahoo_ticker": ticker,
                        "status": "ok",
                        "rows": len(frame),
                        "first_date": frame.index.min(),
                        "last_date": frame.index.max(),
                        "requested_end": requested_end,
                        "file": str(destination),
                        "downloaded_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
                successful_in_batch += 1

            # yfinance can fail by returning an empty/malformed frame instead of raising.
            # Treat a batch with zero usable histories as an outage-like batch so a global
            # empty-response incident stops after a few checkpointed batches rather than
            # crawling through thousands of securities as individual failures.
            if successful_in_batch == 0 and group:
                consecutive_failed_batches += 1
            else:
                consecutive_failed_batches = 0

            if processed_batches % checkpoint_batches == 0:
                _write_price_checkpoint(
                    existing_manifest, records, failures, universe, manifest_path, failures_path
                )

            if successful_in_batch == 0 and consecutive_failed_batches >= failure_limit:
                raise RuntimeError(
                    "Yahoo returned no usable price histories for "
                    f"{consecutive_failed_batches} consecutive batches. Successful files and "
                    "the download manifest were checkpointed. Re-run the same pipeline command; "
                    "completed securities will be skipped automatically."
                )
            if pause > 0:
                time.sleep(pause)
    except KeyboardInterrupt:
        _write_price_checkpoint(
            existing_manifest, records, failures, universe, manifest_path, failures_path
        )
        raise

    combined = _write_price_checkpoint(
        existing_manifest, records, failures, universe, manifest_path, failures_path
    )
    LOGGER.info(
        "Downloaded or refreshed %d securities; %d failures; %d already current",
        len(records),
        len(failures),
        len(completed),
    )
    return combined


def _fx_candidates(canonical_currency: str) -> list[tuple[str, str]]:
    if canonical_currency == "USD":
        return []
    return [
        (f"{canonical_currency}USD=X", "direct"),
        (f"USD{canonical_currency}=X", "inverse"),
    ]


def _close_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("adj_close", "close"):
        if column in frame.columns:
            result = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not result.empty:
                return result
    raise ValueError("FX response has no usable close column")


def _write_fx_checkpoint(existing: pd.DataFrame, records: list[dict], manifest_path: Path) -> pd.DataFrame:
    pieces = [frame for frame in (existing, pd.DataFrame(records)) if not frame.empty]
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates("source_currency", keep="last")
        combined.to_csv(ensure_parent(manifest_path), index=False)
    return combined


def download_fx_rates(config: dict, paths: dict, force: bool = False) -> pd.DataFrame:
    """Download one USD conversion series for every listing currency in the universe.

    Successful currencies are checkpointed one at a time.  A later rerun skips those
    currencies and retries only failed/missing conversions.
    """
    import yfinance as yf

    universe = prepare_universe(config, paths, force=False)
    _configure_yfinance_cache(yf, paths)
    fx_dir = Path(paths["data"]["fx_rates_dir"])
    fx_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(paths["data"]["fx_manifest"])
    existing = pd.read_csv(manifest_path) if manifest_path.exists() and not force else pd.DataFrame()
    existing_ok = (
        set(existing.loc[existing["status"] == "ok", "source_currency"].astype(str))
        if not existing.empty and "status" in existing.columns
        else set()
    )

    requested_end = _requested_end(config)
    exclusive_end = requested_end + pd.Timedelta(days=1)
    start = str(config["data"]["start_date"])
    end_string = exclusive_end.strftime("%Y-%m-%d")
    specs: dict[str, tuple[str, float]] = {}
    for value in universe["currency"]:
        canonical, unit_scale = normalize_currency(value)
        if canonical:
            specs[str(value)] = (canonical, unit_scale)

    records: list[dict] = []
    pause = float(config["data"].get("fx_download_pause_seconds", 0.5))
    retries = int(config["data"].get("fx_download_retries", 3))
    backoff = float(config["data"].get("fx_download_retry_backoff_seconds", 5.0))

    try:
        for source_currency, (canonical, unit_scale) in sorted(specs.items()):
            destination = fx_dir / f"{_safe_name(source_currency)}_to_USD.parquet"
            if canonical == "USD":
                records.append(
                    {
                        "source_currency": source_currency,
                        "canonical_currency": canonical,
                        "unit_scale": unit_scale,
                        "yahoo_ticker": "",
                        "direction": "identity",
                        "status": "ok",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "file": str(destination),
                    }
                )
                _write_fx_checkpoint(existing, records, manifest_path)
                continue
            if not force and source_currency in existing_ok and destination.exists():
                continue

            error_messages: list[str] = []
            succeeded = False
            for ticker, direction in _fx_candidates(canonical):
                try:
                    payload = _download_with_retries(
                        yf,
                        ticker,
                        start=start,
                        end=end_string,
                        retries=retries,
                        backoff=backoff,
                        threads=False,
                    )
                    frame = _extract_ticker_frame(payload, ticker, 1)
                    close = _close_series(frame)
                    usd_per_canonical = close if direction == "direct" else 1.0 / close
                    usd_per_source_unit = usd_per_canonical * float(unit_scale)
                    output = pd.DataFrame(
                        {"usd_per_currency_unit": usd_per_source_unit.replace([np.inf, -np.inf], np.nan)}
                    ).dropna()
                    if len(output) < 20:
                        raise ValueError("FX series contained fewer than 20 valid observations")
                    write_frame(output, destination, index=True)
                    records.append(
                        {
                            "source_currency": source_currency,
                            "canonical_currency": canonical,
                            "unit_scale": unit_scale,
                            "yahoo_ticker": ticker,
                            "direction": direction,
                            "status": "ok",
                            "rows": len(output),
                            "first_date": output.index.min(),
                            "last_date": output.index.max(),
                            "file": str(destination),
                            "error": "",
                        }
                    )
                    succeeded = True
                    break
                except Exception as exc:
                    error_messages.append(f"{ticker}: {exc!r}")
            if not succeeded:
                records.append(
                    {
                        "source_currency": source_currency,
                        "canonical_currency": canonical,
                        "unit_scale": unit_scale,
                        "yahoo_ticker": "",
                        "direction": "",
                        "status": "failed",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "file": str(destination),
                        "error": " | ".join(error_messages),
                    }
                )
            _write_fx_checkpoint(existing, records, manifest_path)
            if pause > 0:
                time.sleep(pause)
    except KeyboardInterrupt:
        _write_fx_checkpoint(existing, records, manifest_path)
        raise

    combined = _write_fx_checkpoint(existing, records, manifest_path)
    failed = combined[combined["status"] != "ok"] if not combined.empty and "status" in combined else pd.DataFrame()
    if not failed.empty:
        failed_currencies = set(failed["source_currency"].astype(str))
        failed_securities = universe["currency"].astype(str).isin(failed_currencies)
        failed_fraction = float(failed_securities.mean()) if len(universe) else 0.0
        LOGGER.warning(
            "FX download failed for %d currencies used by %.2f%% of prepared Yahoo equities",
            len(failed),
            100 * failed_fraction,
        )
    LOGGER.info("Prepared FX conversion metadata for %d listing-currency codes", len(combined))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache historical market prices and FX rates.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fx-only", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    if args.fx_only:
        download_fx_rates(config, paths, force=args.force)
    else:
        download_prices(config, paths, force=args.force)
        if bool(config["data"].get("require_common_currency", True)):
            download_fx_rates(config, paths, force=args.force)


if __name__ == "__main__":
    main()
