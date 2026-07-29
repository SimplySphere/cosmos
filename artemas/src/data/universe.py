from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_universe import build_reference_files
from src.utils.config import load_project
from src.utils.files import ensure_parent, read_frame, write_frame

LOGGER = logging.getLogger(__name__)

PREPARED_UNIVERSE_SCHEMA_VERSION = 4

REQUIRED_UNIVERSE_COLUMNS = {
    "security_id",
    "company_name",
    "index_ticker",
    "exchange",
    "country",
    "currency",
    "sector",
    "industry",
    "market_cap",
    "snapshot_date",
}



_SUFFIX_CURRENCY = {
    ".TW": "TWD", ".TWO": "TWD", ".KS": "KRW", ".KQ": "KRW",
    ".HK": "HKD", ".T": "JPY", ".L": "GBp", ".SW": "CHF",
    ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".BR": "EUR",
    ".MI": "EUR", ".MC": "EUR", ".LS": "EUR", ".HE": "EUR",
    ".VI": "EUR", ".IR": "EUR", ".PR": "EUR", ".AT": "EUR",
    ".ST": "SEK", ".CO": "DKK", ".OL": "NOK", ".AX": "AUD",
    ".TO": "CAD", ".V": "CAD", ".NS": "INR", ".BO": "INR",
    ".SA": "BRL", ".MX": "MXN", ".SR": "SAR", ".JO": "ZAc",
    ".TA": "ILA", ".SI": "SGD", ".KL": "MYR", ".JK": "IDR",
    ".BK": "THB", ".IS": "TRY", ".NZ": "NZD", ".WA": "PLN",
    ".BD": "HUF", ".SS": "CNY", ".SZ": "CNY", ".QA": "QAR",
    ".KW": "KWD", ".AD": "AED", ".CA": "EGP", ".SN": "CLP",
    ".AE": "AED", ".CL": "COP", ".NE": "CAD", ".IC": "ISK",
    ".MU": "EUR", ".XD": "EUR", ".RO": "RON", ".SG": "EUR",
    ".IL": "USD",
}

_COUNTRY_CURRENCY = {
    "US": "USD", "CA": "CAD", "GB": "GBp", "JP": "JPY", "CN": "CNY",
    "HK": "HKD", "TW": "TWD", "KR": "KRW", "AU": "AUD", "NZ": "NZD",
    "IN": "INR", "BR": "BRL", "MX": "MXN", "CH": "CHF", "SE": "SEK",
    "DK": "DKK", "NO": "NOK", "PL": "PLN", "HU": "HUF", "CZ": "CZK",
    "TR": "TRY", "ZA": "ZAc", "IL": "ILA", "SA": "SAR", "AE": "AED",
    "QA": "QAR", "KW": "KWD", "SG": "SGD", "MY": "MYR", "ID": "IDR",
    "TH": "THB", "PH": "PHP", "VN": "VND", "CL": "CLP", "CO": "COP",
    "PE": "PEN", "AR": "ARS", "EG": "EGP", "PK": "PKR", "BD": "BDT",
    "FR": "EUR", "DE": "EUR", "NL": "EUR", "BE": "EUR", "IT": "EUR",
    "ES": "EUR", "PT": "EUR", "IE": "EUR", "FI": "EUR", "AT": "EUR",
    "GR": "EUR", "LU": "EUR", "EE": "EUR", "LV": "EUR", "LT": "EUR",
    "SK": "EUR", "SI": "EUR", "IS": "ISK", "RO": "RON",
}

def _infer_listing_currency(yahoo_ticker: object, country: object) -> tuple[str, str]:
    ticker = str(yahoo_ticker or "").strip().upper()
    for suffix in sorted(_SUFFIX_CURRENCY, key=len, reverse=True):
        if ticker.endswith(suffix):
            return _SUFFIX_CURRENCY[suffix], f"yahoo_suffix:{suffix}"
    country_code = str(country or "").strip().upper()
    if country_code in _COUNTRY_CURRENCY:
        return _COUNTRY_CURRENCY[country_code], f"country:{country_code}"
    return "", "missing"


_EXCHANGE_CURRENCY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("NASDAQ", "USD"),
    ("NMS", "USD"),
    ("NGM", "USD"),
    ("NCM", "USD"),
    ("NYQ", "USD"),
    ("PCX", "USD"),
    ("ASE", "USD"),
    ("NYSE", "USD"),
    ("NEW YORK STOCK EXCHANGE", "USD"),
    ("NYSE ARCA", "USD"),
    ("NYSE AMERICAN", "USD"),
    ("OTC", "USD"),
    ("TORONTO", "CAD"),
    ("TSX", "CAD"),
    ("NEO", "CAD"),
    ("STUTTGART", "EUR"),
    ("MUNICH", "EUR"),
    ("DUSSELDORF", "EUR"),
    ("DÜSSELDORF", "EUR"),
    ("INTERNATIONAL ORDERBOOK - LONDON", "USD"),
    ("DFM", "AED"),
    ("DUBAI FINANCIAL MARKET", "AED"),
    ("ABU DHABI", "AED"),
    ("NASDAQ ICELAND", "ISK"),
    ("ICELAND", "ISK"),
    ("BUCHAREST", "RON"),
    ("BVB", "RON"),
    ("BOLSA DE VALORES DE COLOMBIA", "COP"),
    ("BVC", "COP"),
)

_CASH_OR_CURRENCY_NAME = re.compile(
    r"(?:^|\b)(?:cash|currency|dollar|yen|sheqel|shekel|rupee|krone|riyal|lira|rupiah)(?:\b|$)",
    flags=re.IGNORECASE,
)


def _infer_exchange_currency(exchange: object) -> tuple[str, str]:
    text = str(exchange or "").strip().upper()
    if not text:
        return "", "missing"
    for keyword, currency in _EXCHANGE_CURRENCY_KEYWORDS:
        if keyword.upper() in text:
            return currency, f"exchange:{keyword}"
    return "", "missing"

REQUIRED_MAP_COLUMNS = {
    "security_id",
    "index_ticker",
    "yahoo_ticker",
    "mapping_status",
    "mapping_method",
    "verified",
    "notes",
}


def _truthy(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "verified"})


def _verified_status(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"verified", "resolved", "ok"})


def _needs_build(universe_path: Path, map_path: Path, config: dict) -> bool:
    """Return True when reference files are missing, malformed, or incomplete.

    A temporary Yahoo failure can leave both CSVs present while the ticker map is only
    partially resolved.  Existence alone is therefore not evidence that bootstrap is
    complete; reruns must re-enter the resumable builder until configured coverage is met.
    """
    if not universe_path.exists() or universe_path.stat().st_size == 0:
        return True
    if not map_path.exists() or map_path.stat().st_size == 0:
        return True
    try:
        universe = pd.read_csv(universe_path, usecols=lambda c: c in {"security_id"}, dtype=str)
        mapping = pd.read_csv(map_path, dtype=str).fillna("")
    except Exception:
        return True
    if universe.empty or "security_id" not in universe.columns:
        return True
    if not REQUIRED_MAP_COLUMNS.issubset(mapping.columns):
        return True
    verified = (
        mapping["verified"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        & mapping["mapping_status"].astype(str).str.strip().str.lower().isin({"verified", "resolved", "ok"})
        & mapping["yahoo_ticker"].astype(str).str.strip().ne("")
    )
    # Count only mappings that belong to the current source universe.
    valid_ids = set(universe["security_id"].fillna("").astype(str))
    covered = mapping["security_id"].fillna("").astype(str).isin(valid_ids) & verified
    fraction = float(covered.sum()) / max(len(valid_ids), 1)
    target = float(config.get("universe", {}).get("minimum_verified_count_fraction", 0.0))
    if bool(config.get("universe", {}).get("require_verified_yahoo_tickers", True)) and fraction < target:
        LOGGER.info(
            "Reference ticker map is incomplete (%.1f%% verified < %.1f%% required); resuming builder",
            100 * fraction,
            100 * target,
        )
        return True
    return False


def _metadata_from_notes(value: object) -> dict[str, str]:
    try:
        payload = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    quote = payload.get("quote", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(quote, dict):
        quote = {}

    def pick(*keys: str) -> str:
        for source in (metadata, quote):
            for key in keys:
                value = source.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    return {
        "resolved_currency": pick("currency"),
        "resolved_exchange": pick("exchangeName", "fullExchangeName", "exchangeDisplay", "exchange"),
        "instrument_type": pick("instrumentType", "quoteType", "typeDisp").upper(),
        "resolved_name": pick("longName", "longname", "shortName", "shortname", "name"),
        "resolved_country": pick("country", "region"),
        "resolved_sector": pick("sector", "sectorDisp"),
        "resolved_industry": pick("industry", "industryDisp"),
    }


def prepare_universe(config: dict, paths: dict, force: bool = False) -> pd.DataFrame:
    """Prepare the Yahoo Finance equity list used by the model.

    The configured universe source determines security membership and may supply
    descriptive metadata only when Yahoo metadata is unavailable. Selector portfolio
    quantities are discarded before the prepared model universe is written.
    """
    output = Path(paths["data"]["prepared_universe"])
    membership_mode = str(
        config.get("universe", {}).get("membership_mode", "point_in_time")
    )
    if output.exists() and not force:
        existing = read_frame(output)
        if (
            "selection_source_weight" not in existing.columns
            and "index_weight" not in existing.columns
            and "market_cap" not in existing.columns
            and "prepared_universe_schema_version" in existing.columns
            and existing["prepared_universe_schema_version"]
            .eq(PREPARED_UNIVERSE_SCHEMA_VERSION)
            .all()
            and "model_aggregation_method" in existing.columns
            and existing["model_aggregation_method"]
            .astype(str)
            .eq("equal_weight")
            .all()
            and "membership_mode" in existing.columns
            and existing["membership_mode"].astype(str).eq(membership_mode).all()
        ):
            LOGGER.info("Using existing weight-free prepared universe: %s", output)
            return existing
        LOGGER.warning(
            "Existing prepared universe predates the current metadata-resolution schema and will be rebuilt"
        )

    universe_path = Path(paths["data"]["universe_source"])
    map_path = Path(paths["data"]["ticker_map"])
    auto_build = bool(config.get("universe", {}).get("auto_build", True))
    if _needs_build(universe_path, map_path, config):
        if not auto_build:
            raise FileNotFoundError(
                "The stock-selection source or Yahoo ticker map is missing and universe.auto_build is false."
            )
        LOGGER.info(
            "Stock-selection reference is missing; acquiring the configured holdings list and resolving Yahoo symbols"
        )
        # A derived-output rebuild must not discard network acquisition checkpoints.
        # Missing references are acquired resumably; intentional source refresh remains
        # available through ``python -m src.data.build_universe --force --no-resume``.
        build_reference_files(config, paths, force=False, resolve_yahoo=True, resume=True)

    universe = pd.read_csv(universe_path, dtype={"security_id": str, "index_ticker": str})
    missing = REQUIRED_UNIVERSE_COLUMNS.difference(universe.columns)
    if missing:
        raise ValueError(f"Universe file is missing columns: {sorted(missing)}")
    if universe.empty:
        raise ValueError("Stock-selection universe contains only headers")

    minimum_source = int(config.get("universe", {}).get("minimum_source_constituents", 0))
    if len(universe) < minimum_source:
        raise ValueError(
            f"Universe contains only {len(universe):,} securities, below the configured minimum "
            f"of {minimum_source:,}; the selector export is probably incomplete."
        )

    for column in (
        "security_id",
        "index_ticker",
        "company_name",
        "exchange",
        "country",
        "currency",
        "sector",
        "industry",
    ):
        universe[column] = universe[column].fillna("").astype(str).str.strip()
    if (universe["security_id"] == "").any():
        raise ValueError("Universe contains blank security_id values")
    if universe["security_id"].duplicated().any():
        raise ValueError("Universe contains duplicate security_id values")

    universe["snapshot_date"] = pd.to_datetime(universe["snapshot_date"], errors="coerce")
    if universe["snapshot_date"].isna().any():
        raise ValueError("Universe contains invalid snapshot_date values")
    snapshot_max = universe["snapshot_date"].max()
    cutoff_key = str(config.get("universe", {}).get("membership_cutoff", "train_end"))
    membership_cutoff = pd.Timestamp(config["data"][cutoff_key])
    snapshot_is_point_in_time = bool(snapshot_max <= membership_cutoff)
    if not snapshot_is_point_in_time:
        message = (
            f"Stock-list snapshot {snapshot_max.date()} is after the configured membership "
            f"cutoff {membership_cutoff.date()} ({cutoff_key})."
        )
        if membership_mode == "point_in_time":
            raise ValueError(
                message
                + " Supply a selector snapshot known by the cutoff, or explicitly use "
                "universe.membership_mode=retrospective_disclosed for exploratory runs. "
                "Do not relabel or backdate the source snapshot."
            )
        if membership_mode != "retrospective_disclosed":
            raise ValueError(f"Unsupported universe.membership_mode: {membership_mode!r}")
        LOGGER.warning(
            "%s Proceeding in retrospective_disclosed exploration mode. Results may be "
            "used for architecture, simulation, and exploratory analysis, but not as a "
            "prospective or survivorship-bias-free test.",
            message,
        )

    universe["market_cap"] = pd.to_numeric(universe["market_cap"], errors="coerce")

    ticker_map = pd.read_csv(map_path, dtype=str).fillna("")
    missing_map = REQUIRED_MAP_COLUMNS.difference(ticker_map.columns)
    if missing_map:
        raise ValueError(f"Ticker map is missing columns: {sorted(missing_map)}")
    for column in ("security_id", "index_ticker", "yahoo_ticker", "mapping_status"):
        ticker_map[column] = ticker_map[column].astype(str).str.strip()
    if ticker_map["security_id"].duplicated().any():
        raise ValueError("Ticker map contains duplicate security_id values")

    metadata = pd.DataFrame(
        [_metadata_from_notes(value) for value in ticker_map["notes"]],
        index=ticker_map.index,
    )
    ticker_map = pd.concat([ticker_map, metadata], axis=1)
    keep_columns = [
        "security_id",
        "yahoo_ticker",
        "mapping_status",
        "mapping_method",
        "verified",
        "notes",
        "resolved_currency",
        "resolved_exchange",
        "instrument_type",
        "resolved_name",
        "resolved_country",
        "resolved_sector",
        "resolved_industry",
    ]
    universe = universe.merge(ticker_map[keep_columns], on="security_id", how="left")
    verified = (
        _truthy(universe["verified"])
        & _verified_status(universe["mapping_status"])
        & universe["yahoo_ticker"].fillna("").astype(str).str.strip().ne("")
    )
    verified_count_fraction = float(verified.mean()) if len(verified) else 0.0
    LOGGER.info(
        "Yahoo symbol coverage for the selector list: %d/%d securities (%.1f%%)",
        int(verified.sum()),
        len(universe),
        100 * verified_count_fraction,
    )

    require_verified = bool(config.get("universe", {}).get("require_verified_yahoo_tickers", True))
    if require_verified:
        minimum_count = float(config.get("universe", {}).get("minimum_verified_count_fraction", 0.0))
        if verified_count_fraction < minimum_count:
            status_counts = universe["mapping_status"].fillna("").astype(str).value_counts()
            raise ValueError(
                f"Only {verified_count_fraction:.1%} of selector securities have verified Yahoo mappings; "
                f"configured minimum is {minimum_count:.1%}. Status counts: {status_counts.to_dict()}"
            )
        universe = universe.loc[verified].copy()
    else:
        universe["yahoo_ticker"] = universe["yahoo_ticker"].fillna("").astype(str).str.strip()
        universe.loc[universe["yahoo_ticker"] == "", "yahoo_ticker"] = universe["index_ticker"]

    # Yahoo metadata is preferred. The holdings selector may provide descriptive
    # fallback metadata when Yahoo Search resolved the symbol but returned sparse quote
    # metadata. Portfolio quantities and source ordering remain excluded from modeling.
    for field in ("company_name", "exchange", "country", "currency", "sector", "industry"):
        universe[f"selector_{field}"] = universe[field].fillna("").astype(str).str.strip()

    field_pairs = {
        "company_name": "resolved_name",
        "exchange": "resolved_exchange",
        "country": "resolved_country",
        "sector": "resolved_sector",
        "industry": "resolved_industry",
    }
    for field, resolved_field in field_pairs.items():
        resolved = universe[resolved_field].fillna("").astype(str).str.strip()
        fallback = universe[f"selector_{field}"].fillna("").astype(str).str.strip()
        universe[field] = resolved.where(resolved.ne(""), fallback)
        universe[f"{field}_source"] = np.select(
            [resolved.ne(""), fallback.ne("")],
            ["yahoo_metadata", "selector_metadata_fallback"],
            default="missing",
        )

    # Currency resolution is intentionally layered by listing evidence. Issuer-country
    # currency is only the final fallback because cross-listed securities can trade in a
    # different currency from their home country.
    universe["currency"] = universe["resolved_currency"].fillna("").astype(str).str.strip()
    universe["currency_source"] = np.where(
        universe["currency"].ne(""), "yahoo_metadata", "missing"
    )

    suffix_inferred = universe.apply(
        lambda row: _infer_listing_currency(row.get("yahoo_ticker"), ""),
        axis=1,
        result_type="expand",
    )
    suffix_inferred.columns = ["suffix_currency", "suffix_currency_source"]
    missing_currency = universe["currency"].eq("")
    usable_suffix = missing_currency & suffix_inferred["suffix_currency"].ne("")
    universe.loc[usable_suffix, "currency"] = suffix_inferred.loc[usable_suffix, "suffix_currency"]
    universe.loc[usable_suffix, "currency_source"] = suffix_inferred.loc[
        usable_suffix, "suffix_currency_source"
    ]

    selector_currency = universe["selector_currency"].fillna("").astype(str).str.strip()
    missing_currency = universe["currency"].eq("")
    usable_selector_currency = missing_currency & selector_currency.ne("")
    universe.loc[usable_selector_currency, "currency"] = selector_currency.loc[
        usable_selector_currency
    ]
    universe.loc[usable_selector_currency, "currency_source"] = "selector_currency_fallback"

    exchange_inferred = universe["exchange"].map(_infer_exchange_currency)
    exchange_currency = exchange_inferred.map(lambda item: item[0])
    exchange_source = exchange_inferred.map(lambda item: item[1])
    missing_currency = universe["currency"].eq("")
    usable_exchange = missing_currency & exchange_currency.ne("")
    universe.loc[usable_exchange, "currency"] = exchange_currency.loc[usable_exchange]
    universe.loc[usable_exchange, "currency_source"] = exchange_source.loc[usable_exchange]

    country_inferred = universe["country"].map(lambda value: _infer_listing_currency("", value))
    country_currency = country_inferred.map(lambda item: item[0])
    country_source = country_inferred.map(lambda item: item[1])
    missing_currency = universe["currency"].eq("")
    usable_country = missing_currency & country_currency.ne("")
    universe.loc[usable_country, "currency"] = country_currency.loc[usable_country]
    universe.loc[usable_country, "currency_source"] = country_source.loc[usable_country]

    exclusion_rows: list[pd.DataFrame] = []

    # Broad holdings exports can include cash-currency rows. When Yahoo mapping metadata
    # is sparse, remove only deterministic cash/currency positions rather than treating
    # their three-letter codes as equities.
    instrument = universe["instrument_type"].fillna("").astype(str).str.upper()
    ticker_code = universe["yahoo_ticker"].fillna("").astype(str).str.strip().str.upper()
    cash_or_currency = (
        instrument.eq("")
        & universe["country"].fillna("").astype(str).str.strip().eq("")
        & ticker_code.str.fullmatch(r"[A-Z]{3}", na=False)
        & universe["company_name"].fillna("").astype(str).str.contains(_CASH_OR_CURRENCY_NAME, na=False)
    )
    if cash_or_currency.any():
        excluded = universe.loc[cash_or_currency].copy()
        excluded["exclusion_reason"] = "cash_or_currency_position"
        exclusion_rows.append(excluded)
        universe = universe.loc[~cash_or_currency].copy()

    instrument = universe["instrument_type"].fillna("").astype(str).str.upper()
    non_equity = instrument.ne("") & ~instrument.isin({"EQUITY", "STOCK"})
    if non_equity.any():
        excluded = universe.loc[non_equity].copy()
        excluded["exclusion_reason"] = "non_equity_instrument"
        exclusion_rows.append(excluded)
        universe = universe.loc[~non_equity].copy()

    universe["yahoo_ticker"] = universe["yahoo_ticker"].fillna("").astype(str).str.strip()
    blank_ticker = universe["yahoo_ticker"].eq("")
    if blank_ticker.any():
        excluded = universe.loc[blank_ticker].copy()
        excluded["exclusion_reason"] = "blank_yahoo_ticker"
        exclusion_rows.append(excluded)
        universe = universe.loc[~blank_ticker].copy()

    if universe.empty:
        raise ValueError("No Yahoo-resolved equities remain after selector and instrument filters")

    known_currency = universe["currency"].fillna("").astype(str).str.strip().ne("")
    known_currency_fraction = float(known_currency.mean())
    minimum_currency_fraction = float(config["data"].get("minimum_currency_count_fraction", 0.0))
    if (
        bool(config["data"].get("require_common_currency", True))
        and known_currency_fraction < minimum_currency_fraction
    ):
        raise ValueError(
            f"Only {known_currency_fraction:.1%} of prepared Yahoo equities have a known listing currency; "
            f"configured minimum is {minimum_currency_fraction:.1%}."
        )

    currency_resolution_counts = {
        str(key): int(value)
        for key, value in universe["currency_source"].value_counts(dropna=False).items()
    }

    # Drop selector portfolio quantities before any model-facing artifact is written.
    universe = universe.drop(
        columns=[
            "index_weight",
            "selection_source_weight",
            "source_weight",
            "market_cap",
            "position_market_value",
            "shares",
        ],
        errors="ignore",
    )
    universe["selection_source"] = "configured holdings selector (membership plus disclosed descriptive-metadata fallback; portfolio quantities discarded)"
    universe["membership_mode"] = membership_mode
    universe["membership_snapshot_date"] = str(snapshot_max.date())
    universe["membership_cutoff_date"] = str(membership_cutoff.date())
    universe["membership_is_point_in_time"] = snapshot_is_point_in_time
    universe["claim_scope"] = np.where(
        snapshot_is_point_in_time,
        "point_in_time_validation",
        "retrospective_exploration",
    )
    universe["price_provider"] = str(config["market"]["price_provider"])
    universe["model_aggregation_method"] = str(config["market"]["aggregation_method"])
    universe["prepared_universe_schema_version"] = PREPARED_UNIVERSE_SCHEMA_VERSION
    universe = universe.sort_values(["security_id", "yahoo_ticker"], kind="stable").reset_index(drop=True)

    exclusions = (
        pd.concat(exclusion_rows, ignore_index=True)
        if exclusion_rows
        else pd.DataFrame(columns=list(universe.columns) + ["exclusion_reason"])
    )
    exclusions.to_csv(ensure_parent(paths["data"]["universe_exclusions"]), index=False)
    write_frame(universe, output, index=False)
    LOGGER.info(
        "Prepared %d Yahoo Finance equities for modeling; %.1f%% have resolved listing "
        "currency. Currency sources: %s. Selector weights were discarded.",
        len(universe),
        100 * known_currency_fraction,
        currency_resolution_counts,
    )
    return universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the stock-selection list and prepare Yahoo Finance equities."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    prepare_universe(config, paths, force=args.force)


if __name__ == "__main__":
    main()
