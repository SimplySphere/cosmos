from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.vanguard import acquire_vanguard_holdings
from src.utils.config import load_project

LOGGER = logging.getLogger(__name__)

UNIVERSE_COLUMNS = [
    "security_id",
    "company_name",
    "index_ticker",
    "exchange",
    "country",
    "currency",
    "sector",
    "industry",
    "market_cap",
    "index_weight",
    "snapshot_date",
    "cusip",
    "sedol",
    "isin",
    "position_market_value",
    "shares",
    "source_name",
    "source_url",
]

MAP_COLUMNS = [
    "security_id",
    "index_ticker",
    "yahoo_ticker",
    "mapping_status",
    "mapping_method",
    "verified",
    "notes",
]

COLUMN_ALIASES: dict[str, set[str]] = {
    "security_id": {"security_id", "security id", "stable security id"},
    "company_name": {
        "holdings",
        "holding",
        "holding name",
        "security",
        "security name",
        "company",
        "company name",
        "name",
        "issuer",
    },
    "index_ticker": {"ticker", "symbol", "local ticker", "local symbol", "security ticker"},
    "index_weight": {
        "% of fund",
        "% of funds",
        "percent of fund",
        "percent of funds",
        "weight",
        "weight %",
        "weight (%)",
        "portfolio weight",
    },
    "position_market_value": {"market value", "market value usd", "position market value"},
    "shares": {"shares", "share quantity", "quantity", "face amount"},
    "cusip": {"cusip", "cusip code"},
    "sedol": {"sedol", "sedol code"},
    "isin": {"isin", "isin code"},
    "country": {"country", "country/region", "country region", "domicile"},
    "currency": {"currency", "trading currency", "local currency"},
    "sector": {"sector", "gics sector", "sector name"},
    "industry": {"industry", "gics industry", "industry name", "sub industry"},
    "exchange": {"exchange", "exchange name", "primary exchange", "listing exchange"},
    "mic": {"mic", "mic code", "market identifier code"},
    "market_cap": {"market cap", "market capitalization"},
    "snapshot_date": {"as of date", "as-of date", "snapshot date", "date"},
    "yahoo_ticker": {"yahoo ticker", "yahoo symbol", "yahoo finance ticker"},
}

EXCHANGE_TO_MIC = {
    "new york stock exchange": "XNYS",
    "nyse": "XNYS",
    "nasdaq": "XNAS",
    "nasdaq global select market": "XNGS",
    "nasdaq global market": "XNAS",
    "nasdaq capital market": "XNCM",
    "nyse american": "XASE",
    "toronto stock exchange": "XTSE",
    "tsx": "XTSE",
    "london stock exchange": "XLON",
    "lse": "XLON",
    "euronext paris": "XPAR",
    "euronext amsterdam": "XAMS",
    "xetra": "XETR",
    "frankfurt": "XFRA",
    "six swiss exchange": "XSWX",
    "milan": "XMIL",
    "borsa italiana": "XMIL",
    "madrid": "XMAD",
    "tokyo stock exchange": "XTKS",
    "tokyo": "XTKS",
    "hong kong stock exchange": "XHKG",
    "hong kong": "XHKG",
    "shanghai stock exchange": "XSHG",
    "shenzhen stock exchange": "XSHE",
    "taiwan stock exchange": "XTAI",
    "taipei exchange": "ROCO",
    "korea exchange": "XKRX",
    "kospi": "XKRX",
    "kosdaq": "XKOS",
    "australian securities exchange": "XASX",
    "asx": "XASX",
    "new zealand exchange": "XNZE",
    "singapore exchange": "XSES",
    "sgx": "XSES",
    "bursa malaysia": "XKLS",
    "stock exchange of thailand": "XBKK",
    "indonesia stock exchange": "XIDX",
    "philippine stock exchange": "XPHS",
    "national stock exchange of india": "XNSE",
    "nse": "XNSE",
    "bombay stock exchange": "XBOM",
    "bse": "XBOM",
    "b3": "BVMF",
    "mexican stock exchange": "XMEX",
    "johannesburg stock exchange": "XJSE",
    "tel aviv stock exchange": "XTAE",
    "saudi exchange": "XSAU",
}


def _normalise_label(value: Any) -> str:
    text = str(value).strip().lower().replace("&", "and")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_LOOKUP = {
    _normalise_label(alias): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def _canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[Any, str] = {}
    used: set[str] = set()
    for column in frame.columns:
        label = _normalise_label(column)
        canonical = _ALIAS_LOOKUP.get(label)
        if canonical is None:
            if "market value" in label:
                canonical = "position_market_value"
            elif "% of fund" in label or "portfolio weight" in label:
                canonical = "index_weight"
            elif "holding" in label and "number" not in label:
                canonical = "company_name"
        if canonical and canonical not in used:
            renamed[column] = canonical
            used.add(canonical)
    output = frame.rename(columns=renamed).copy()
    # The holdings source establishes membership and descriptive fallback metadata.
    # Source-fund weights are optional provenance and never model inputs.
    required = {"company_name"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(
            "Holdings selector export is missing required columns after normalization: "
            f"{sorted(missing)}. Found: {list(map(str, frame.columns))[:30]}"
        )
    if "index_weight" not in output.columns:
        output["index_weight"] = np.nan
    return output


def _split_holding_name_and_ticker(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    # Vanguard's combined format ends with a compact exchange symbol in
    # parentheses, for example ``Berkshire Hathaway Inc (BRK/B)`` or
    # ``Taiwan Semiconductor Manufacturing Co Ltd (2330)``. Requiring a
    # compact ticker-like token avoids treating ordinary parenthetical company
    # descriptors as symbols.
    match = re.match(r"^(.*?)\s*\(([A-Za-z0-9][A-Za-z0-9./_-]{0,24})\)\s*$", text)
    if not match:
        return text, ""
    name = match.group(1).strip()
    ticker = match.group(2).strip()
    return (name or text), ticker


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})


def _numeric(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    negative = raw.str.match(r"^\(.*\)$")
    cleaned = raw.str.replace(r"[$,%(),]", "", regex=True).str.replace("—", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    values.loc[negative & values.notna()] *= -1
    return values


def _normalise_weights(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str)
    values = _numeric(series).clip(lower=0)
    if values.notna().sum() == 0:
        return values
    total = float(values.sum(skipna=True))
    if (
        raw.str.contains("%", regex=False).any()
        or float(values.max(skipna=True)) > 1.0
        or total > 2.0
    ):
        values = values / 100.0
    return values


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert yfinance/Pandas objects into compact JSON-compatible values.

    ``Ticker.history_metadata`` can contain DataFrames (notably trading-period
    tables), NumPy scalars, timestamps, and other objects that ``json.dumps``
    cannot serialize. Mapping notes only need compact diagnostic metadata, so
    large tabular objects are summarized rather than embedded in full.
    """

    if depth > 8:
        return "<maximum depth reached>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item(), depth=depth + 1)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns[:50]],
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "Series",
            "length": int(len(value)),
            "name": None if value.name is None else str(value.name),
        }
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return _json_safe(value.tolist(), depth=depth + 1)
        return {"__type__": "ndarray", "shape": [int(item) for item in value.shape]}
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        converted = [_json_safe(item, depth=depth + 1) for item in items[:100]]
        if len(items) > 100:
            converted.append(f"<{len(items) - 100} additional items omitted>")
        return converted
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _stable_id(row: pd.Series) -> str:
    explicit = str(row.get("security_id", "")).strip().upper()
    if explicit and explicit not in {"NAN", "NONE", "NULL"}:
        return explicit
    for prefix, field in (("CUSIP", "cusip"), ("SEDOL", "sedol"), ("ISIN", "isin")):
        value = str(row.get(field, "")).strip().upper()
        if value and value.lower() != "nan":
            return f"{prefix}-{value}"
    payload = "|".join(
        str(row.get(field, "")).strip().upper()
        for field in ("company_name", "index_ticker", "country", "exchange")
    )
    return "VT-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:18].upper()


def _snapshot_date(frame: pd.DataFrame, metadata: dict[str, Any]) -> str:
    configured = metadata.get("snapshot_date")
    if configured:
        parsed = pd.to_datetime(configured, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    if "snapshot_date" in frame.columns:
        dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna()
        if not dates.empty:
            return dates.max().strftime("%Y-%m-%d")
    # A missing date is uncommon in Vanguard exports. Use acquisition date but label it
    # transparently rather than claiming a historical snapshot.
    fetched = pd.to_datetime(metadata.get("fetched_at"), errors="coerce", utc=True)
    if pd.notna(fetched):
        return fetched.tz_convert(None).strftime("%Y-%m-%d")
    return pd.Timestamp.now(tz="UTC").tz_localize(None).strftime("%Y-%m-%d")


def _normalise_vanguard_holdings(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _canonicalise_columns(raw)
    for column in (
        "security_id",
        "company_name",
        "index_ticker",
        "exchange",
        "country",
        "currency",
        "sector",
        "industry",
        "cusip",
        "sedol",
        "isin",
        "mic",
        "yahoo_ticker",
    ):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = _clean_text(frame[column])

    frame["company_name"] = _clean_text(frame["company_name"])
    frame["index_ticker"] = _clean_text(frame["index_ticker"])

    combined = frame["company_name"].map(_split_holding_name_and_ticker)
    parsed_names = combined.map(lambda item: item[0])
    parsed_tickers = combined.map(lambda item: item[1])
    combined_mask = parsed_tickers.ne("")
    frame.loc[combined_mask, "company_name"] = parsed_names.loc[combined_mask]
    missing_ticker = frame["index_ticker"].eq("") & combined_mask
    frame.loc[missing_ticker, "index_ticker"] = parsed_tickers.loc[missing_ticker]

    # Preserve the raw percentage until after the actual security rows are known.
    # Vanguard rounds many tiny VT positions to 0.00%, so percentage positivity
    # must never be used as a constituent filter.
    frame["position_market_value"] = (
        _numeric(frame["position_market_value"])
        if "position_market_value" in frame.columns
        else np.nan
    )
    frame["shares"] = _numeric(frame["shares"]) if "shares" in frame.columns else np.nan
    frame["market_cap"] = _numeric(frame["market_cap"]) if "market_cap" in frame.columns else np.nan

    invalid_names = {
        "",
        "total",
        "grand total",
        "holdings",
        "holding",
        "portfolio composition file",
        "short term reserve",
        "short term reserves",
        "short-term reserve",
        "short-term reserves",
        "cash",
        "equity",
        "fixed income",
    }
    has_resolvable_identity = (
        frame["index_ticker"].ne("")
        | frame["cusip"].ne("")
        | frame["sedol"].ne("")
        | frame["isin"].ne("")
    )
    frame = frame[
        ~frame["company_name"].str.lower().isin(invalid_names)
        & has_resolvable_identity
    ].copy()
    if frame.empty:
        raise ValueError("No identifiable securities remained after cleaning Vanguard holdings")

    # Preserve any reported source weight only for selector provenance. Do not
    # reconstruct, normalize, gate, prioritize, train, aggregate, or evaluate with it.
    frame["index_weight"] = _normalise_weights(frame["index_weight"])

    frame["security_id"] = frame.apply(_stable_id, axis=1)
    if frame["security_id"].duplicated().any():
        for index in frame.index[frame["security_id"].duplicated(keep=False)]:
            payload = "|".join(
                str(frame.at[index, field])
                for field in ("security_id", "company_name", "index_ticker", "position_market_value")
            )
            frame.at[index, "security_id"] = "VT-" + hashlib.sha1(payload.encode()).hexdigest()[:18].upper()
    if frame["security_id"].duplicated().any():
        raise ValueError("Unable to construct unique identifiers for Vanguard holdings")

    snapshot = _snapshot_date(frame, metadata)
    frame["snapshot_date"] = snapshot
    frame["source_name"] = "Vanguard Total World Stock ETF (VT) holdings"
    frame["source_url"] = str(metadata.get("source_url", ""))

    universe = frame.reindex(columns=UNIVERSE_COLUMNS).copy()
    mapping = pd.DataFrame(
        {
            "security_id": frame["security_id"],
            "index_ticker": frame["index_ticker"],
            "yahoo_ticker": frame["yahoo_ticker"],
            "mapping_status": np.where(frame["yahoo_ticker"].ne(""), "pending_verification", "pending"),
            "mapping_method": np.where(frame["yahoo_ticker"].ne(""), "source_yahoo_ticker", ""),
            "verified": False,
            "notes": "",
        }
    ).reindex(columns=MAP_COLUMNS)
    return universe, mapping


RESOLVER_VERSION = "2026-07-29.1"

PLACEHOLDER_SYMBOLS = {
    "",
    "-",
    "--",
    "---",
    "N/A",
    "NA",
    "NONE",
    "NULL",
    "1",
    "2",
}

# Yahoo suffixes for the primary exchanges represented in Vanguard VT. The
# order matters: the first suffix is the most likely primary listing.
COUNTRY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "US": ("",),
    "CA": (".TO", ".V", ".NE"),
    "GB": (".L", ".IL"),
    "JP": (".T",),
    "HK": (".HK",),
    "TW": (".TW", ".TWO"),
    "KR": (".KS", ".KQ"),
    "CN": (".SS", ".SZ", ".BJ"),
    "AU": (".AX",),
    "NZ": (".NZ",),
    "CH": (".SW",),
    "DE": (".DE",),
    "FR": (".PA",),
    "NL": (".AS",),
    "BE": (".BR",),
    "IT": (".MI",),
    "ES": (".MC",),
    "PT": (".LS",),
    "AT": (".VI",),
    "IE": (".IR", ".L"),
    "SE": (".ST",),
    "DK": (".CO",),
    "NO": (".OL",),
    "FI": (".HE",),
    "PL": (".WA",),
    "GR": (".AT",),
    "HU": (".BD",),
    "CZ": (".PR",),
    "BR": (".SA",),
    "MX": (".MX",),
    "ZA": (".JO",),
    "IN": (".NS", ".BO"),
    "ID": (".JK",),
    "MY": (".KL",),
    "SG": (".SI",),
    "TH": (".BK",),
    "PH": (".PS",),
    "IL": (".TA",),
    "SA": (".SR",),
    "AE": (".AE", ".AD", ".DU"),
    "QA": (".QA",),
    "KW": (".KW",),
    "TR": (".IS",),
    "EG": (".CA",),
    "CL": (".SN",),
}

COUNTRY_MICS: dict[str, tuple[str, ...]] = {
    "US": ("XNYS", "XNAS", "XASE"),
    "CA": ("XTSE", "XTSX"),
    "GB": ("XLON",),
    "JP": ("XTKS",),
    "HK": ("XHKG",),
    "TW": ("XTAI", "ROCO"),
    "KR": ("XKRX", "XKOS"),
    "CN": ("XSHG", "XSHE"),
    "AU": ("XASX",),
    "NZ": ("XNZE",),
    "CH": ("XSWX",),
    "DE": ("XETR", "XFRA"),
    "FR": ("XPAR",),
    "NL": ("XAMS",),
    "BE": ("XBRU",),
    "IT": ("XMIL",),
    "ES": ("XMAD",),
    "PT": ("XLIS",),
    "AT": ("XWBO",),
    "IE": ("XDUB", "XLON"),
    "SE": ("XSTO",),
    "DK": ("XCSE",),
    "NO": ("XOSL",),
    "FI": ("XHEL",),
    "PL": ("XWAR",),
    "BR": ("BVMF",),
    "MX": ("XMEX",),
    "ZA": ("XJSE",),
    "IN": ("XNSE", "XBOM"),
    "ID": ("XIDX",),
    "MY": ("XKLS",),
    "SG": ("XSES",),
    "TH": ("XBKK",),
    "PH": ("XPHS",),
    "IL": ("XTAE",),
    "SA": ("XSAU",),
}


class YahooTemporaryFailure(RuntimeError):
    """Raised when Yahoo is unavailable or throttling the resolver."""


def _name_similarity(left: str, right: str) -> float:
    remove = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "plc",
        "sa",
        "ag",
        "nv",
        "class",
        "ordinary",
        "shares",
        "the",
    }

    def simplify(value: str) -> str:
        return " ".join(token for token in _normalise_label(value).split() if token not in remove)

    return SequenceMatcher(None, simplify(left), simplify(right)).ratio()


def _quote_value(quote: dict[str, Any], *names: str) -> str:
    for name in names:
        value = quote.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _country_code(row: pd.Series | dict[str, Any]) -> str:
    value = row.get("country", "") if hasattr(row, "get") else ""
    return str(value).strip().upper()


def _clean_source_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper().replace("*", "")
    symbol = re.sub(r"\s+", "", symbol)
    return symbol


def _symbol_suffix(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    return "." + text.rsplit(".", 1)[1] if "." in text else ""


def _country_symbol_compatible(country: str, symbol: str) -> bool:
    country = str(country or "").strip().upper()
    symbol = str(symbol or "").strip().upper()
    if not symbol or symbol.startswith("^") or "/" in symbol or "*" in symbol:
        return False
    allowed = COUNTRY_SUFFIXES.get(country)
    if not allowed:
        return True
    if country == "US":
        return "." not in symbol
    suffix = _symbol_suffix(symbol)
    return suffix in allowed


def _resolve_mic(row: pd.Series) -> str:
    mic = str(row.get("mic", "")).strip().upper()
    if re.fullmatch(r"[A-Z0-9]{4}", mic):
        return mic
    return EXCHANGE_TO_MIC.get(_normalise_label(row.get("exchange", "")), "")


def _source_symbol_variants(row: pd.Series) -> list[str]:
    source = _clean_source_symbol(row.get("index_ticker", ""))
    country = _country_code(row)
    if source in PLACEHOLDER_SYMBOLS:
        return []

    variants: list[str] = []

    def add(value: str) -> None:
        value = value.strip().upper()
        if value and value not in PLACEHOLDER_SYMBOLS and value not in variants:
            variants.append(value)

    if country in {"US", "CA"} and "/" in source:
        add(source.replace("/", "-"))
    add(source)
    add(source.replace(" ", "-"))
    add(source.replace(".", "-"))
    add(source.replace("/", "-"))
    if source.endswith("/"):
        add(source[:-1])
    if source.endswith("/F"):
        add(source[:-2])

    # Yahoo pads Hong Kong numeric listings to four digits.
    if country == "HK" and re.fullmatch(r"\d{1,4}", source):
        variants.insert(0, source.zfill(4))

    # Nordic class shares are commonly written without the hyphen in
    # Vanguard's export: VOLVB -> VOLV-B.ST, NOVOB -> NOVO-B.CO.
    if country in {"SE", "DK", "NO", "FI"} and re.fullmatch(r"[A-Z0-9]{4,}[ABC]", source):
        add(f"{source[:-1]}-{source[-1]}")

    # U.S./Canadian class shares use a hyphen on Yahoo.
    if country in {"US", "CA"} and "/" in source:
        add(source.replace("/", "-"))

    return variants


def _suffixes_for_row(row: pd.Series) -> tuple[str, ...]:
    country = _country_code(row)
    source = _clean_source_symbol(row.get("index_ticker", ""))
    if country == "CN" and re.fullmatch(r"\d{6}", source):
        if source.startswith(("5", "6", "9")):
            return (".SS",)
        return (".SZ", ".BJ")
    return COUNTRY_SUFFIXES.get(country, tuple())


def _add_candidate(
    output: list[tuple[str, str]],
    ticker: str,
    method: str,
    country: str,
) -> None:
    ticker = str(ticker or "").strip().upper()
    if not ticker or ticker in {item[0] for item in output}:
        return
    if country and not _country_symbol_compatible(country, ticker):
        return
    output.append((ticker, method))


def _deterministic_candidates(yf: Any, row: pd.Series) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    country = _country_code(row)
    supplied = str(row.get("yahoo_ticker", "")).strip().upper()
    if supplied:
        _add_candidate(output, supplied, "source_yahoo_ticker", country)

    bases = _source_symbol_variants(row)
    suffixes = _suffixes_for_row(row)

    # Country-qualified candidates are cheap and avoid the main failure in the
    # previous resolver: submitting local symbols such as 2330 or HSBA directly
    # to Yahoo without .TW or .L.
    for base in bases:
        for suffix in suffixes:
            candidate_base = base
            if suffix == ".HK" and re.fullmatch(r"\d{1,4}", candidate_base):
                candidate_base = candidate_base.zfill(4)
            _add_candidate(output, candidate_base + suffix, f"country_suffix:{country}", country)

    # yfinance officially accepts (symbol, MIC) tuples. This can recover cases
    # where Yahoo's exact suffix convention differs from the common rule.
    mics: list[str] = []
    resolved_mic = _resolve_mic(row)
    if resolved_mic:
        mics.append(resolved_mic)
    for mic in COUNTRY_MICS.get(country, tuple()):
        if mic not in mics:
            mics.append(mic)
    for base in bases[:4]:
        for mic in mics[:3]:
            try:
                converted = str(yf.Ticker((base, mic)).ticker).strip().upper()
            except Exception:
                converted = ""
            _add_candidate(output, converted, f"mic:{mic}", country)

    # A bare direct symbol is safe as the first-line candidate only for the U.S.
    # For other countries it frequently resolves to an unrelated U.S. company.
    if country == "US" or not country:
        for base in bases:
            _add_candidate(output, base, "direct_symbol", country)
    return output


def _parse_note_payload(raw: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(raw)) if str(raw).strip() else {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolved_name_from_notes(raw: Any) -> str:
    payload = _parse_note_payload(raw)
    quote = payload.get("quote", {}) if isinstance(payload.get("quote", {}), dict) else {}
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    return (
        _quote_value(quote, "longname", "shortname", "name")
        or str(metadata.get("longName") or metadata.get("shortName") or "")
    )



def _merge_previous_mapping(current_mapping: pd.DataFrame, previous_mapping: pd.DataFrame) -> pd.DataFrame:
    """Merge a saved ticker-map checkpoint without dtype-dependent assignment.

    Freshly normalized Vanguard mappings use a real Boolean dtype for ``verified``.
    CSV checkpoints are intentionally read as strings so values such as ``True`` and
    ``False`` round-trip exactly.  Recent Pandas versions reject assigning that
    Arrow-backed string column into a Boolean block.  Converting both frames to
    object before indexed replacement makes the resume path stable across Pandas
    versions while preserving all checkpoint values for the later audit step.
    """

    if not set(MAP_COLUMNS).issubset(current_mapping.columns):
        missing = sorted(set(MAP_COLUMNS).difference(current_mapping.columns))
        raise ValueError(f"Current ticker map is missing columns required for resume: {missing}")
    if not set(MAP_COLUMNS).issubset(previous_mapping.columns):
        return current_mapping.reindex(columns=MAP_COLUMNS).copy()

    current = current_mapping.reindex(columns=MAP_COLUMNS).copy().astype(object)
    previous = previous_mapping.reindex(columns=MAP_COLUMNS).copy().astype(object)
    current["security_id"] = current["security_id"].fillna("").astype(str)
    previous["security_id"] = previous["security_id"].fillna("").astype(str)

    current = current.set_index("security_id", drop=True)
    previous = previous.set_index("security_id", drop=True)
    shared = current.index.intersection(previous.index)
    if len(shared):
        columns = MAP_COLUMNS[2:]
        current_shared = current.loc[shared]
        previous_shared = previous.loc[shared]
        current_verified = (
            current_shared["verified"].astype("string").fillna("").str.lower().isin({"true", "1", "yes"})
            & current_shared["mapping_status"].astype("string").fillna("").str.lower().eq("verified")
            & current_shared["yahoo_ticker"].astype("string").fillna("").str.strip().ne("")
        )
        previous_verified = (
            previous_shared["verified"].astype("string").fillna("").str.lower().isin({"true", "1", "yes"})
            & previous_shared["mapping_status"].astype("string").fillna("").str.lower().eq("verified")
            & previous_shared["yahoo_ticker"].astype("string").fillna("").str.strip().ne("")
        )
        # A failed/partial checkpoint must never overwrite a verified bundled seed.
        # A verified local checkpoint may replace the seed, and any checkpoint may
        # fill a row for which the current mapping has no verified symbol.
        replace = (~current_verified) | previous_verified
        replace_ids = shared[replace.to_numpy()]
        if len(replace_ids):
            current.loc[replace_ids, columns] = previous.loc[replace_ids, columns].to_numpy(dtype=object)

    merged = current.reset_index().reindex(columns=MAP_COLUMNS)
    # Keep string/object storage during resolution.  All downstream truth checks
    # already normalize booleans through ``astype(str)``, and resolver updates may
    # write either Python bools or strings safely into object columns.
    return merged.astype(object)

def _load_selector_seed(
    config: dict[str, Any],
    paths: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load the pinned selector-only membership snapshot for reproducible fresh runs."""

    universe_config = config.get("universe", {})
    if not bool(universe_config.get("prefer_selector_seed", True)):
        return None
    configured = str(universe_config.get("selector_seed_path", "")).strip()
    if not configured:
        return None
    seed_path = Path(configured)
    if not seed_path.is_absolute():
        seed_path = Path(paths["project_root"]) / seed_path
    seed_path = seed_path.resolve()
    manifest_path = seed_path.with_name("manifest.json")
    if not seed_path.exists():
        LOGGER.warning("Configured selector seed is missing: %s", seed_path)
        return None

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = str(manifest.get("seed_sha256", "")).strip().lower()
    actual_sha = hashlib.sha256(seed_path.read_bytes()).hexdigest().lower()
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(
            f"Selector seed checksum mismatch for {seed_path}; expected {expected_sha}, got {actual_sha}"
        )
    frame = pd.read_csv(seed_path, dtype=str).fillna("")
    metadata = {
        "source_type": "bundled_vanguard_membership_seed",
        "source_path": str(seed_path),
        "source_url": manifest.get("source_url", ""),
        "snapshot_date": manifest.get("snapshot_date"),
        "fetched_at": None,
        "row_count": int(len(frame)),
        "seed_sha256": actual_sha,
        "original_official_export_sha256": manifest.get("original_official_export_sha256"),
        "selector_quantities_present": False,
    }
    LOGGER.info(
        "Using pinned selector-only VT membership resource: %s (%d rows; snapshot %s)",
        seed_path,
        len(frame),
        metadata.get("snapshot_date"),
    )
    return frame, metadata


def _load_resolver_seed(
    config: dict[str, Any],
    paths: dict[str, Any],
    universe: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge source-controlled resolver knowledge before any Yahoo requests.

    A clean checkout must not rediscover thousands of stable Yahoo symbols through
    the rate-limited Yahoo Search endpoint.  The seed contains only security IDs and
    verified Yahoo ticker mappings; it contains no prices, returns, selector weights,
    or model artifacts.  A seed verified after the membership cutoff is deliberately
    ignored in point-in-time mode, but is valid for the explicitly retrospective
    exploration mode.
    """

    configured = str(config.get("universe", {}).get("resolver_seed_path", "")).strip()
    if not configured:
        return mapping, {"used": False, "reason": "not_configured"}

    seed_path = Path(configured)
    if not seed_path.is_absolute():
        seed_path = Path(paths["project_root"]) / seed_path
    seed_path = seed_path.resolve()
    manifest_path = seed_path.with_name("manifest.json")
    if not seed_path.exists():
        LOGGER.warning("Configured resolver seed is missing: %s", seed_path)
        return mapping, {"used": False, "reason": "missing", "path": str(seed_path)}

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Ignoring invalid resolver-seed manifest %s: %s", manifest_path, exc)
            manifest = {}

    expected_sha = str(manifest.get("sha256", "")).strip().lower()
    actual_sha = hashlib.sha256(seed_path.read_bytes()).hexdigest().lower()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Resolver seed checksum mismatch for {seed_path}; expected {expected_sha}, got {actual_sha}"
        )

    membership_mode = str(config.get("universe", {}).get("membership_mode", "point_in_time"))
    verified_date = pd.to_datetime(manifest.get("resolver_verified_date"), errors="coerce")
    cutoff_key = str(config.get("universe", {}).get("membership_cutoff", "train_end"))
    cutoff = pd.Timestamp(config["data"][cutoff_key]).normalize()
    if membership_mode == "point_in_time" and pd.notna(verified_date) and verified_date > cutoff:
        LOGGER.info(
            "Skipping resolver seed verified on %s because point_in_time membership cutoff is %s",
            verified_date.date(),
            cutoff.date(),
        )
        return mapping, {
            "used": False,
            "reason": "verified_after_point_in_time_cutoff",
            "path": str(seed_path),
        }

    seed = pd.read_csv(seed_path, dtype=str).fillna("")
    missing = set(MAP_COLUMNS).difference(seed.columns)
    if missing:
        raise ValueError(f"Resolver seed is missing columns: {sorted(missing)}")
    seed = seed.reindex(columns=MAP_COLUMNS)
    verified = (
        seed["verified"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        & seed["mapping_status"].astype(str).str.strip().str.lower().eq("verified")
        & seed["yahoo_ticker"].astype(str).str.strip().ne("")
    )
    seed = seed.loc[verified].copy()
    merged = _merge_previous_mapping(mapping, seed)
    report = _coverage_report(universe, merged)
    LOGGER.info(
        "Loaded bundled Yahoo resolver seed: %d verified mappings; current selector coverage %.1f%%",
        len(seed),
        100 * report["verified_count_fraction"],
    )
    return merged, {
        "used": True,
        "path": str(seed_path),
        "sha256": actual_sha,
        "rows": int(len(seed)),
        "membership_snapshot_date": manifest.get("membership_snapshot_date"),
        "resolver_verified_date": manifest.get("resolver_verified_date"),
        "claim_scope": manifest.get("claim_scope"),
    }


def _audit_cached_mappings(
    universe: pd.DataFrame,
    mapping: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Invalidate cached mappings that the prior resolver accepted incorrectly."""

    minimum_name = float(config["universe"].get("yahoo_cached_mapping_min_name_similarity", 0.55))
    merged = universe[["security_id", "company_name", "country"]].merge(
        mapping, on="security_id", how="left", suffixes=("", "_map")
    )
    invalid: dict[str, str] = {}
    for _, row in merged.iterrows():
        verified = str(row.get("verified", "")).strip().lower() in {"true", "1", "yes"}
        if not verified or str(row.get("mapping_status", "")).strip().lower() != "verified":
            continue
        security_id = str(row["security_id"])
        ticker = str(row.get("yahoo_ticker", "")).strip().upper()
        country = str(row.get("country", "")).strip().upper()
        method = str(row.get("mapping_method", "")).strip()
        reason = ""
        if not ticker:
            reason = "blank Yahoo ticker"
        elif method == "direct_symbol" and country and country != "US":
            reason = "bare direct symbol used for a non-U.S. holding"
        elif country and not _country_symbol_compatible(country, ticker):
            reason = f"Yahoo suffix is incompatible with source country {country}"
        else:
            resolved_name = _resolved_name_from_notes(row.get("notes", ""))
            if resolved_name and method != "direct_symbol":
                similarity = _name_similarity(str(row.get("company_name", "")), resolved_name)
                if similarity < minimum_name:
                    reason = f"resolved company-name similarity {similarity:.3f} is below {minimum_name:.3f}"
        if reason:
            invalid[security_id] = reason

    if not invalid:
        return mapping
    updated = mapping.copy()
    for security_id, reason in invalid.items():
        mask = updated["security_id"].astype(str).eq(security_id)
        updated.loc[mask, "yahoo_ticker"] = ""
        updated.loc[mask, "mapping_status"] = "pending"
        updated.loc[mask, "mapping_method"] = ""
        updated.loc[mask, "verified"] = False
        updated.loc[mask, "notes"] = _json_dumps(
            {"resolver_version": RESOLVER_VERSION, "audit_invalidated": reason}
        )
    LOGGER.warning(
        "Invalidated %d cached Yahoo mappings that failed country/name checks",
        len(invalid),
    )
    return updated


def _valid_symbols_from_download(frame: Any, requested: Iterable[str]) -> set[str]:
    """Return requested symbols that contain at least one real price value.

    yfinance can return either ticker-first or price-field-first MultiIndex
    columns, and single-symbol downloads can be flat.  We deliberately test
    price fields rather than any arbitrary column so a metadata-only frame is
    not accepted as market history.
    """

    requested_set = {str(item).strip().upper() for item in requested if str(item).strip()}
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return set()

    price_names = {"OPEN", "HIGH", "LOW", "CLOSE", "ADJ CLOSE", "ADJ_CLOSE", "VOLUME"}

    def has_prices(subset: pd.DataFrame) -> bool:
        if subset is None or subset.empty:
            return False
        if isinstance(subset.columns, pd.MultiIndex):
            field_mask = np.zeros(len(subset.columns), dtype=bool)
            for level in range(subset.columns.nlevels):
                values = np.array([str(item).strip().upper() for item in subset.columns.get_level_values(level)])
                field_mask |= np.isin(values, list(price_names))
            if field_mask.any():
                subset = subset.loc[:, field_mask]
        else:
            fields = [str(item).strip().upper() for item in subset.columns]
            mask = np.isin(fields, list(price_names))
            if mask.any():
                subset = subset.loc[:, mask]
        numeric = subset.apply(pd.to_numeric, errors="coerce")
        return bool(numeric.notna().any(axis=None))

    columns = frame.columns
    valid: set[str] = set()
    if isinstance(columns, pd.MultiIndex):
        for level in range(columns.nlevels):
            values = np.array([str(item).strip().upper() for item in columns.get_level_values(level)])
            overlap = requested_set.intersection(values.tolist())
            if not overlap:
                continue
            for symbol in overlap:
                subset = frame.loc[:, values == symbol]
                if has_prices(subset):
                    valid.add(symbol)
            if valid:
                return valid
    elif len(requested_set) == 1 and has_prices(frame):
        valid.update(requested_set)
    return valid


def _clear_yfinance_errors(yf: Any) -> None:
    shared = getattr(yf, "shared", None)
    errors = getattr(shared, "_ERRORS", None)
    if isinstance(errors, dict):
        errors.clear()


def _yfinance_error_summary(yf: Any) -> str:
    shared = getattr(yf, "shared", None)
    errors = getattr(shared, "_ERRORS", None)
    if not isinstance(errors, dict) or not errors:
        return ""
    parts: list[str] = []
    for symbol, error in list(errors.items())[:8]:
        parts.append(f"{symbol}: {error}")
    return "; ".join(parts)


def _download_prices(
    yf: Any,
    symbols: list[str],
    *,
    timeout: float,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "tickers": symbols,
        "interval": "1d",
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": False,
        "group_by": "ticker",
        "timeout": timeout,
        "multi_level_index": True,
    }
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    try:
        return yf.download(**kwargs)
    except TypeError:
        # Compatibility with yfinance versions that predate this option.
        kwargs.pop("multi_level_index", None)
        return yf.download(**kwargs)


def _single_canary_history(yf: Any, symbol: str, timeout: float, period: str) -> bool:
    """Use the independent Ticker.history path to disambiguate batch/parser failures."""

    try:
        obj = yf.Ticker(symbol)
        kwargs: dict[str, Any] = {
            "period": period,
            "interval": "1d",
            "auto_adjust": False,
            "actions": False,
            "timeout": timeout,
            "raise_errors": True,
        }
        try:
            frame = obj.history(**kwargs)
        except TypeError:
            kwargs.pop("raise_errors", None)
            try:
                frame = obj.history(**kwargs)
            except TypeError:
                kwargs.pop("timeout", None)
                frame = obj.history(**kwargs)
        return _valid_symbols_from_download(frame, [symbol]) == {symbol}
    except Exception:
        return False


def _probe_yahoo_health(yf: Any, config: dict[str, Any]) -> tuple[bool, str]:
    """Test Yahoo independently of the Vanguard mapping candidates.

    A short period request avoids date-window edge cases.  A second, single-
    ticker path distinguishes a malformed multi-ticker response from a genuine
    Yahoo/cookie/network outage.
    """

    canaries = [
        str(item).strip().upper()
        for item in config["universe"].get("yahoo_canary_symbols", ["AAPL", "MSFT"])
        if str(item).strip()
    ]
    if not canaries:
        return True, "health probe disabled"
    timeout = float(config["universe"].get("yahoo_request_timeout_seconds", 30.0))
    period = str(config["universe"].get("yahoo_health_period", "5d"))
    _clear_yfinance_errors(yf)
    batch_error = ""
    previous_level = logging.getLogger("yfinance").level
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        try:
            frame = _download_prices(yf, canaries, timeout=timeout, period=period)
            valid = _valid_symbols_from_download(frame, canaries)
            if valid:
                return True, f"batch canary returned {sorted(valid)}"
            batch_error = _yfinance_error_summary(yf) or "batch canary returned an empty frame"
        except Exception as exc:
            batch_error = f"{type(exc).__name__}: {exc}"

        individually_valid = [
            symbol for symbol in canaries if _single_canary_history(yf, symbol, timeout, period)
        ]
        if individually_valid:
            return True, (
                "single-ticker canary succeeded after the multi-ticker probe failed; "
                f"valid={individually_valid}; batch={batch_error}"
            )
        individual_error = _yfinance_error_summary(yf)
        details = batch_error
        if individual_error and individual_error not in details:
            details += f"; individual={individual_error}"
        return False, details
    finally:
        logging.getLogger("yfinance").setLevel(previous_level)


def _wait_for_yahoo_health(yf: Any, config: dict[str, Any]) -> str:
    retries = int(config["universe"].get("yahoo_health_retries", 3))
    backoff = float(config["universe"].get("yahoo_health_backoff_seconds", 15.0))
    last_detail = ""
    for attempt in range(max(retries, 1)):
        healthy, detail = _probe_yahoo_health(yf, config)
        last_detail = detail
        if healthy:
            LOGGER.info("Yahoo health check passed: %s", detail)
            return detail
        if attempt + 1 < max(retries, 1):
            delay = backoff * (2**attempt)
            LOGGER.warning(
                "Yahoo health check failed (%s). Retrying in %.0f seconds (%d/%d)",
                detail,
                delay,
                attempt + 1,
                retries,
            )
            time.sleep(delay)
    version = str(getattr(yf, "__version__", "unknown"))
    raise YahooTemporaryFailure(
        "Yahoo returned no recent AAPL/MSFT data through both batch and single-ticker "
        f"yfinance paths (yfinance {version}). This is a Yahoo/session/network health "
        "failure, not a Vanguard ticker-mapping failure. The checkpoint was preserved. "
        f"Details: {last_detail}"
    )


def _batch_download_valid(
    yf: Any,
    symbols: Iterable[str],
    start: str,
    end: str,
    config: dict[str, Any],
) -> set[str]:
    requested = list(dict.fromkeys(str(item).strip().upper() for item in symbols if str(item).strip()))
    if not requested:
        return set()
    retries = int(config["universe"].get("yahoo_batch_retries", 3))
    backoff = float(config["universe"].get("yahoo_retry_backoff_seconds", 5.0))
    timeout = float(config["universe"].get("yahoo_request_timeout_seconds", 30.0))
    last_error = ""

    for attempt in range(max(retries, 1)):
        previous_level = logging.getLogger("yfinance").level
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        _clear_yfinance_errors(yf)
        try:
            frame = _download_prices(
                yf,
                requested,
                timeout=timeout,
                start=start,
                end=end,
            )
            valid = _valid_symbols_from_download(frame, requested)
            if valid:
                return valid

            # An empty candidate batch is ambiguous: all symbols may simply be
            # invalid for this round.  Probe known symbols separately before
            # declaring a transport/rate-limit failure.
            healthy, health_detail = _probe_yahoo_health(yf, config)
            if healthy:
                return set()
            last_error = health_detail or _yfinance_error_summary(yf) or "empty batch response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            healthy, health_detail = _probe_yahoo_health(yf, config)
            if healthy:
                # Yahoo is reachable; this particular candidate batch failed.
                return set()
            if health_detail:
                last_error += f"; health={health_detail}"
        finally:
            logging.getLogger("yfinance").setLevel(previous_level)
        if attempt + 1 < max(retries, 1):
            time.sleep(backoff * (2**attempt))
    raise YahooTemporaryFailure(
        "Yahoo batch validation is unavailable after an independent AAPL/MSFT "
        "health probe; " + (last_error or "health validation failed")
    )

def _update_mapping(mapping: pd.DataFrame, security_id: str, result: dict[str, Any]) -> None:
    location = mapping.index[mapping["security_id"].astype(str).eq(str(security_id))]
    if not len(location):
        return
    for key, value in result.items():
        mapping.loc[location, key] = value


def _mark_retry_later(mapping: pd.DataFrame, security_ids: Iterable[str], reason: str) -> None:
    ids = {str(item) for item in security_ids}
    if not ids:
        return
    mask = mapping["security_id"].astype(str).isin(ids)
    verified = mapping["verified"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    mask &= ~verified
    mapping.loc[mask, "mapping_status"] = "retry_later"
    mapping.loc[mask, "mapping_method"] = "temporary_failure"
    mapping.loc[mask, "notes"] = _json_dumps(
        {"resolver_version": RESOLVER_VERSION, "temporary_failure": reason}
    )


def _resolve_deterministic_batches(
    yf: Any,
    merged: pd.DataFrame,
    mapping: pd.DataFrame,
    pending_indices: list[int],
    config: dict[str, Any],
    map_path: Path,
    verification_start: str,
    verification_end: str,
) -> set[int]:
    candidate_lists = {
        index: _deterministic_candidates(yf, merged.loc[index]) for index in pending_indices
    }
    unresolved = set(pending_indices)
    batch_size = int(config["universe"].get("yahoo_batch_size", 100))
    max_rounds = max((len(items) for items in candidate_lists.values()), default=0)
    resolved_count = 0

    for round_index in range(max_rounds):
        symbol_rows: dict[str, list[tuple[int, str]]] = {}
        for index in sorted(unresolved):
            items = candidate_lists[index]
            if round_index >= len(items):
                continue
            symbol, method = items[round_index]
            symbol_rows.setdefault(symbol, []).append((index, method))
        symbols = list(symbol_rows)
        if not symbols:
            continue
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset : offset + batch_size]
            try:
                valid = _batch_download_valid(
                    yf,
                    batch,
                    verification_start,
                    verification_end,
                    config,
                )
            except YahooTemporaryFailure as exc:
                remaining_ids = [str(merged.loc[index, "security_id"]) for index in unresolved]
                _mark_retry_later(mapping, remaining_ids, str(exc))
                mapping.to_csv(map_path, index=False)
                raise
            for symbol in valid:
                rows = [item for item in symbol_rows.get(symbol, []) if item[0] in unresolved]
                # One Yahoo symbol must not silently represent multiple source
                # securities. Leave collisions for the name-aware search stage.
                if len(rows) != 1:
                    continue
                index, method = rows[0]
                row = merged.loc[index]
                result = {
                    "yahoo_ticker": symbol,
                    "mapping_status": "verified",
                    "mapping_method": method,
                    "verified": True,
                    "notes": _json_dumps(
                        {
                            "resolver_version": RESOLVER_VERSION,
                            "country": _country_code(row),
                            "validation": "batch_price_history",
                        }
                    ),
                }
                _update_mapping(mapping, str(row["security_id"]), result)
                unresolved.discard(index)
                resolved_count += 1
            mapping.to_csv(map_path, index=False)
            batch_pause = float(config["universe"].get("yahoo_batch_pause_seconds", 1.0))
            if batch_pause > 0:
                time.sleep(batch_pause)
        LOGGER.info(
            "Deterministic Yahoo pass %d/%d: %d resolved; %d remain",
            round_index + 1,
            max_rounds,
            resolved_count,
            len(unresolved),
        )
    return unresolved


def _search_quotes(yf: Any, query: str, max_results: int) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_research=False,
            enable_fuzzy_query=True,
            raise_errors=True,
        )
    except TypeError:
        search = yf.Search(query, max_results=max_results, news_count=0)
    return [item for item in (search.quotes or []) if isinstance(item, dict)]


def _search_health_ok(yf: Any, config: dict[str, Any]) -> bool:
    retries = int(config["universe"].get("yahoo_search_health_retries", 3))
    backoff = float(config["universe"].get("yahoo_retry_backoff_seconds", 5.0))
    for attempt in range(retries):
        try:
            quotes = _search_quotes(yf, "AAPL", 3)
            if any(_quote_value(quote, "symbol").upper() == "AAPL" for quote in quotes):
                return True
        except Exception:
            pass
        if attempt + 1 < retries:
            time.sleep(backoff * (2**attempt))
    return False


def _candidate_score(row: pd.Series, quote: dict[str, Any]) -> tuple[float, float]:
    quote_type = _quote_value(quote, "quoteType", "typeDisp").upper()
    if quote_type and "EQUITY" not in quote_type and "STOCK" not in quote_type:
        return -100.0, 0.0
    symbol = _quote_value(quote, "symbol").upper()
    country = _country_code(row)
    if not symbol or (country and not _country_symbol_compatible(country, symbol)):
        return -100.0, 0.0
    resolved_name = _quote_value(quote, "longname", "shortname", "name")
    similarity = _name_similarity(str(row.get("company_name", "")), resolved_name)
    score = 8.0 * similarity + 2.0
    source_symbol = re.sub(r"[^A-Z0-9]", "", _clean_source_symbol(row.get("index_ticker", "")))
    candidate_base = re.sub(r"[^A-Z0-9]", "", re.split(r"[.]", symbol)[0])
    if source_symbol and source_symbol == candidate_base:
        score += 3.0
    elif source_symbol and source_symbol in candidate_base:
        score += 1.0
    return score, similarity


def _select_search_candidate(
    yf: Any,
    row: pd.Series,
    max_results: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    quotes: dict[str, dict[str, Any]] = {}
    queries = [str(row.get("company_name", "")).strip()]
    source_symbol = _clean_source_symbol(row.get("index_ticker", ""))
    if source_symbol not in PLACEHOLDER_SYMBOLS:
        queries.append(source_symbol)
    last_error = ""
    for query in queries:
        try:
            for quote in _search_quotes(yf, query, max_results):
                symbol = _quote_value(quote, "symbol").upper()
                if symbol:
                    quotes[symbol] = quote
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if quotes:
            break
    if not quotes:
        return None, last_error or None

    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for quote in quotes.values():
        score, similarity = _candidate_score(row, quote)
        if score > -50:
            ranked.append((score, similarity, quote))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return None, None
    score, similarity, quote = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -math.inf
    review_min = float(config["universe"].get("yahoo_search_review_name_similarity", 0.55))
    verify_min = float(config["universe"].get("yahoo_search_verified_name_similarity", 0.72))
    margin_min = float(config["universe"].get("yahoo_search_score_margin", 0.75))
    if similarity < review_min:
        return None, None
    status = "verified" if similarity >= verify_min and score - second_score >= margin_min else "review"
    return {
        "symbol": _quote_value(quote, "symbol").upper(),
        "score": score,
        "similarity": similarity,
        "quote": quote,
        "status": status,
    }, None


def _coverage_targets_met(
    universe: pd.DataFrame,
    mapping: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    report = _coverage_report(universe, mapping)
    count_target = float(config["universe"].get("minimum_verified_count_fraction", 0.0))
    return report["verified_count_fraction"] >= count_target, report


def _resolve_search_rows(
    yf: Any,
    merged: pd.DataFrame,
    mapping: pd.DataFrame,
    pending_indices: list[int],
    config: dict[str, Any],
    map_path: Path,
    verification_start: str,
    verification_end: str,
) -> None:
    """Resolve the remaining rows with Yahoo Search, transparently and resumably.

    Yahoo Search is necessarily a one-company-at-a-time fallback.  The previous
    implementation gave no visible output when this module was launched directly,
    slept after every request, and attempted every unresolved holding even after
    the configured coverage targets had already been achieved.  That made a
    healthy run look frozen for tens of minutes or hours.
    """

    max_results = int(config["universe"].get("yahoo_search_results", 8))
    pause = float(config["universe"].get("yahoo_search_pause_seconds", 0.25))
    chunk_size = max(1, int(config["universe"].get("yahoo_search_chunk_size", 25)))
    empty_limit = max(1, int(config["universe"].get("yahoo_search_empty_healthcheck", 5)))
    stop_when_covered = bool(config["universe"].get("yahoo_stop_when_coverage_met", True))
    maximum_rows = int(config["universe"].get("yahoo_search_max_rows_per_run", 0))

    # Resolve in deterministic source order. Holdings weights must not influence
    # model coverage or resolution priority.  Rows already exhausted by Yahoo Search
    # in an earlier checkpoint are not retried forever; temporary-failure rows remain
    # eligible so rerunning the same command can resume after a transient outage.
    mapping_state = mapping.set_index(mapping["security_id"].astype(str), drop=False)
    retryable: list[int] = []
    fresh: list[int] = []
    exhausted = 0
    for index in pending_indices:
        security_id = str(merged.loc[index, "security_id"])
        if security_id not in mapping_state.index:
            fresh.append(index)
            continue
        state = mapping_state.loc[security_id]
        if isinstance(state, pd.DataFrame):
            state = state.iloc[-1]
        status = str(state.get("mapping_status", "")).strip().lower()
        method = str(state.get("mapping_method", "")).strip().lower()
        if status == "retry_later" or method == "temporary_failure":
            retryable.append(index)
        elif method == "yfinance_search" and status in {"unresolved", "review"}:
            exhausted += 1
        else:
            fresh.append(index)
    ordered = retryable + fresh
    if maximum_rows > 0:
        ordered = ordered[:maximum_rows]
    if exhausted:
        LOGGER.info(
            "Skipping %d Yahoo Search rows already exhausted in a prior checkpoint",
            exhausted,
        )

    source_universe = merged.reindex(columns=UNIVERSE_COLUMNS).copy()
    covered, initial_report = _coverage_targets_met(source_universe, mapping, config)
    if stop_when_covered and covered:
        LOGGER.info(
            "Skipping Yahoo name search because the count-coverage target is already met: %.1f%%",
            100 * initial_report["verified_count_fraction"],
        )
        return

    if not ordered:
        LOGGER.info("No retryable or unattempted Yahoo Search rows remain")
        return

    minimum_sleep = len(ordered) * max(pause, 0.0)
    LOGGER.info(
        "Starting Yahoo name-search fallback for %d holdings. The configured %.2fs "
        "request pause alone implies at least %.1f minutes; progress and ETA will "
        "be printed every %d rows.",
        len(ordered),
        pause,
        minimum_sleep / 60.0,
        chunk_size,
    )

    consecutive_empty = 0
    started = time.monotonic()
    processed = 0
    verified_added = 0
    review_added = 0

    try:
        for start_offset in range(0, len(ordered), chunk_size):
            chunk = ordered[start_offset : start_offset + chunk_size]
            selections: dict[int, dict[str, Any]] = {}
            for index in chunk:
                row = merged.loc[index]
                selection, error = _select_search_candidate(yf, row, max_results, config)
                if selection is None:
                    consecutive_empty += 1
                else:
                    selections[index] = selection
                    consecutive_empty = 0
                if consecutive_empty >= empty_limit:
                    if not _search_health_ok(yf, config):
                        remaining = ordered[start_offset:]
                        remaining_ids = [str(merged.loc[item, "security_id"]) for item in remaining]
                        reason = "Yahoo Search health check failed after consecutive empty results"
                        _mark_retry_later(mapping, remaining_ids, reason)
                        mapping.to_csv(map_path, index=False)
                        raise YahooTemporaryFailure(reason)
                    consecutive_empty = 0
                if error:
                    LOGGER.debug("Yahoo Search error for %s: %s", row.get("company_name", ""), error)
                if pause > 0:
                    time.sleep(pause)

            symbols = [selection["symbol"] for selection in selections.values()]
            try:
                valid = _batch_download_valid(
                    yf,
                    symbols,
                    verification_start,
                    verification_end,
                    config,
                )
            except YahooTemporaryFailure as exc:
                remaining = ordered[start_offset:]
                remaining_ids = [str(merged.loc[item, "security_id"]) for item in remaining]
                _mark_retry_later(mapping, remaining_ids, str(exc))
                mapping.to_csv(map_path, index=False)
                raise

            for index in chunk:
                row = merged.loc[index]
                security_id = str(row["security_id"])
                selection = selections.get(index)
                if selection and selection["symbol"] in valid:
                    status = selection["status"]
                    result = {
                        "yahoo_ticker": selection["symbol"],
                        "mapping_status": status,
                        "mapping_method": "yfinance_search",
                        "verified": status == "verified",
                        "notes": _json_dumps(
                            {
                                "resolver_version": RESOLVER_VERSION,
                                "score": round(float(selection["score"]), 4),
                                "name_similarity": round(float(selection["similarity"]), 4),
                                "quote": selection["quote"],
                                "validation": "batch_price_history",
                            }
                        ),
                    }
                    if status == "verified":
                        verified_added += 1
                    else:
                        review_added += 1
                else:
                    result = {
                        "yahoo_ticker": "",
                        "mapping_status": "unresolved",
                        "mapping_method": "yfinance_search",
                        "verified": False,
                        "notes": _json_dumps(
                            {
                                "resolver_version": RESOLVER_VERSION,
                                "reason": "No country-compatible, name-matched Yahoo equity with recent prices",
                            }
                        ),
                    }
                _update_mapping(mapping, security_id, result)

            processed += len(chunk)
            mapping.to_csv(map_path, index=False)
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = processed / elapsed
            remaining_count = len(ordered) - processed
            eta_minutes = remaining_count / rate / 60.0 if rate > 0 else math.inf
            covered, report = _coverage_targets_met(source_universe, mapping, config)
            LOGGER.info(
                "Yahoo name-search progress: %d/%d (%.1f%%), %.2f rows/s, ETA %.1f min; "
                "+%d verified, +%d review; total count coverage %.1f%%",
                processed,
                len(ordered),
                100 * processed / max(len(ordered), 1),
                rate,
                eta_minutes,
                verified_added,
                review_added,
                100 * report["verified_count_fraction"],
            )
            if stop_when_covered and covered:
                LOGGER.info(
                    "Stopping Yahoo name search because configured coverage targets are met "
                    "after %d fallback rows.",
                    processed,
                )
                return
    except KeyboardInterrupt:
        mapping.to_csv(map_path, index=False)
        LOGGER.warning(
            "Yahoo name search interrupted after %d/%d rows. The checkpoint was saved; "
            "rerun the same command to resume.",
            processed,
            len(ordered),
        )
        raise

def _demote_duplicate_verified_tickers(mapping: pd.DataFrame) -> pd.DataFrame:
    updated = mapping.copy()
    verified = (
        updated["verified"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        & updated["mapping_status"].fillna("").astype(str).str.lower().eq("verified")
        & updated["yahoo_ticker"].fillna("").astype(str).str.strip().ne("")
    )
    duplicate = verified & updated.loc[verified, "yahoo_ticker"].duplicated(keep=False).reindex(updated.index, fill_value=False)
    if not duplicate.any():
        return updated
    for ticker in sorted(updated.loc[duplicate, "yahoo_ticker"].astype(str).unique()):
        mask = duplicate & updated["yahoo_ticker"].astype(str).eq(ticker)
        updated.loc[mask, "mapping_status"] = "review"
        updated.loc[mask, "verified"] = False
        updated.loc[mask, "notes"] = _json_dumps(
            {
                "resolver_version": RESOLVER_VERSION,
                "reason": f"Yahoo ticker {ticker} matched multiple Vanguard securities",
            }
        )
    LOGGER.warning("Demoted %d duplicate Yahoo mappings for manual review", int(duplicate.sum()))
    return updated


def _enrich_universe(universe: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    enriched = universe.copy()
    notes = mapping.set_index("security_id")["notes"].to_dict()
    for index, row in enriched.iterrows():
        raw = notes.get(row["security_id"], "")
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}
        quote = payload.get("quote", {}) if isinstance(payload, dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if not enriched.at[index, "exchange"]:
            enriched.at[index, "exchange"] = (
                _quote_value(quote, "exchDisp", "exchangeDisplay", "exchange")
                or str(metadata.get("exchangeName") or metadata.get("exchange") or "")
            )
        if not enriched.at[index, "currency"]:
            enriched.at[index, "currency"] = (
                _quote_value(quote, "currency") or str(metadata.get("currency") or "")
            )
        if not enriched.at[index, "sector"]:
            enriched.at[index, "sector"] = _quote_value(quote, "sector", "sectorDisp")
        if not enriched.at[index, "industry"]:
            enriched.at[index, "industry"] = _quote_value(quote, "industry", "industryDisp")
        if not enriched.at[index, "country"]:
            enriched.at[index, "country"] = _quote_value(quote, "country", "region")
    return enriched


def _coverage_report(universe: pd.DataFrame, mapping: pd.DataFrame) -> dict[str, Any]:
    merged = universe[["security_id"]].merge(mapping, on="security_id", how="left")
    verified = (
        merged["verified"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        & merged["mapping_status"].fillna("").eq("verified")
        & merged["yahoo_ticker"].fillna("").ne("")
    )
    return {
        "source_constituents": int(len(merged)),
        "verified_constituents": int(verified.sum()),
        "verified_count_fraction": float(verified.mean()) if len(verified) else 0.0,
        "selector_weights_used_for_modeling": False,
    }


def build_reference_files(
    config: dict,
    paths: dict,
    force: bool = False,
    source_file: str | Path | None = None,
    resolve_yahoo: bool = True,
    resume: bool = True,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector_seed = None if source_file is not None or force else _load_selector_seed(config, paths)
    if selector_seed is not None:
        raw, source_metadata = selector_seed
    else:
        raw, source_metadata = acquire_vanguard_holdings(
            config, paths, force=force, source_file=source_file
        )
    universe, mapping = _normalise_vanguard_holdings(raw, source_metadata)

    minimum_source = int(config["universe"]["minimum_source_constituents"])
    if len(universe) < minimum_source:
        raise ValueError(
            f"Vanguard source contains only {len(universe):,} identifiable securities; "
            f"configured minimum is {minimum_source:,}. The file is likely not the full holdings export."
        )

    universe_path = Path(paths["data"]["universe_source"])
    map_path = Path(paths["data"]["ticker_map"])
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge source-controlled resolver knowledge first.  It survives deletion of the
    # generated data/ tree and prevents a clean checkout from needing thousands of
    # fragile Yahoo Search calls.
    mapping, seed_metadata = _load_resolver_seed(config, paths, universe, mapping)
    mapping = _audit_cached_mappings(universe, mapping, config)

    # Always preserve a generated resolver checkpoint when resume=True, even if the
    # caller is rebuilding derived outputs with --force.  --force must never throw
    # away hours of successful network acquisition.  The dedicated --no-resume path
    # remains available for an intentional resolver reset.
    if resume and map_path.exists():
        previous = pd.read_csv(map_path, dtype=str).fillna("")
        if set(MAP_COLUMNS).issubset(previous.columns):
            mapping = _merge_previous_mapping(mapping, previous)
            mapping = _audit_cached_mappings(universe, mapping, config)

    # Save the normalized source and audited map before any network work so a
    # temporary Yahoo failure never destroys the recoverable checkpoint.
    universe.to_csv(universe_path, index=False)
    mapping.to_csv(map_path, index=False)

    source_universe = universe.reindex(columns=UNIVERSE_COLUMNS).copy()
    coverage_met_before_network, seeded_report = _coverage_targets_met(
        source_universe, mapping, config
    )
    stop_when_covered = bool(config["universe"].get("yahoo_stop_when_coverage_met", True))

    if resolve_yahoo and coverage_met_before_network and stop_when_covered:
        LOGGER.info(
            "Resolver coverage target already met at %.1f%%; skipping live Yahoo symbol resolution",
            100 * seeded_report["verified_count_fraction"],
        )
    elif resolve_yahoo:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install project dependencies before resolving Yahoo tickers") from exc

        yahoo_cache_dir = Path(paths["data"]["vanguard_holdings_dir"]).parent / "yfinance_cache"
        yahoo_cache_dir.mkdir(parents=True, exist_ok=True)
        set_cache = getattr(yf, "set_tz_cache_location", None)
        if callable(set_cache):
            set_cache(str(yahoo_cache_dir))
        LOGGER.info(
            "Using yfinance %s with project-local cache %s",
            getattr(yf, "__version__", "unknown"),
            yahoo_cache_dir,
        )
        _wait_for_yahoo_health(yf, config)

        lookback = int(config["universe"].get("yahoo_verification_lookback_days", 60))
        cutoff_key = str(config["universe"].get("membership_cutoff", "train_end"))
        membership_mode = str(config["universe"].get("membership_mode", "point_in_time"))
        if membership_mode == "point_in_time":
            verification_anchor = pd.Timestamp(config["data"][cutoff_key]).normalize()
        elif membership_mode == "retrospective_disclosed":
            source_dates = pd.to_datetime(universe.get("snapshot_date"), errors="coerce").dropna()
            if source_dates.empty:
                raise ValueError(
                    "retrospective_disclosed mode requires a valid source snapshot_date"
                )
            # This mode already discloses that membership is retrospective.  Use the
            # current resolver date rather than artificially querying Yahoo as of the
            # holdings snapshot; current symbol availability is materially more robust
            # and matches the provenance of the bundled resolver seed.
            verification_anchor = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
            LOGGER.info(
                "Retrospective resolver verification anchor: %s (source snapshot %s)",
                verification_anchor.date(),
                pd.Timestamp(source_dates.max()).date(),
            )
        else:
            raise ValueError(f"Unsupported universe.membership_mode: {membership_mode!r}")
        verification_end = verification_anchor + pd.Timedelta(days=1)
        verification_start = verification_end - pd.Timedelta(days=lookback)
        start_string = verification_start.strftime("%Y-%m-%d")
        end_string = verification_end.strftime("%Y-%m-%d")

        merged = universe.merge(mapping, on=["security_id", "index_ticker"], how="left")
        verified = merged["verified"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        pending = merged.index[~verified].tolist()
        if limit is not None:
            pending = pending[:limit]
        LOGGER.info(
            "Resolving %d selector securities through Yahoo Finance using batch-first resolver %s",
            len(pending),
            RESOLVER_VERSION,
        )

        try:
            unresolved = _resolve_deterministic_batches(
                yf,
                merged,
                mapping,
                pending,
                config,
                map_path,
                start_string,
                end_string,
            )
            if unresolved:
                covered, interim_report = _coverage_targets_met(source_universe, mapping, config)
                if bool(config["universe"].get("yahoo_stop_when_coverage_met", True)) and covered:
                    LOGGER.info(
                        "Deterministic resolver already met the count-coverage target: %.1f%%; "
                        "skipping the slow one-by-one Yahoo Search fallback.",
                        100 * interim_report["verified_count_fraction"],
                    )
                elif bool(config["universe"].get("yahoo_search_enabled", True)):
                    _resolve_search_rows(
                        yf,
                        merged,
                        mapping,
                        sorted(unresolved),
                        config,
                        map_path,
                        start_string,
                        end_string,
                    )
                else:
                    LOGGER.warning(
                        "Live Yahoo name-search fallback is disabled; leaving %d mappings unresolved",
                        len(unresolved),
                    )
        except YahooTemporaryFailure as exc:
            mapping = _demote_duplicate_verified_tickers(mapping)
            mapping.to_csv(map_path, index=False)
            universe = _enrich_universe(universe, mapping)
            universe.to_csv(universe_path, index=False)
            report = _coverage_report(universe, mapping)
            report.update(source_metadata)
            report["resolver_version"] = RESOLVER_VERSION
            report["resolver_seed"] = seed_metadata
            report["temporary_failure"] = True
            report_path = Path(paths["data"]["universe_report"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            target = float(config["universe"].get("minimum_verified_count_fraction", 0.0))
            if stop_when_covered and report["verified_count_fraction"] >= target:
                LOGGER.warning(
                    "Yahoo resolver became temporarily unavailable after coverage reached %.1f%%; "
                    "continuing because the configured %.1f%% requirement is already satisfied. %s",
                    100 * report["verified_count_fraction"],
                    100 * target,
                    exc,
                )
            else:
                raise

    mapping = _demote_duplicate_verified_tickers(mapping)
    universe = _enrich_universe(universe, mapping)
    universe.to_csv(universe_path, index=False)
    mapping.to_csv(map_path, index=False)
    report = _coverage_report(universe, mapping)
    report.update(source_metadata)
    report["resolver_version"] = RESOLVER_VERSION
    report["resolver_seed"] = seed_metadata
    report["mapping_status_counts"] = {
        str(key): int(value) for key, value in mapping["mapping_status"].value_counts(dropna=False).items()
    }
    report_path = Path(paths["data"]["universe_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    LOGGER.info(
        "Built weight-free stock universe: %d source securities; %d verified Yahoo symbols (%.1f%% of count); source weights are not used by the model",
        report["source_constituents"],
        report["verified_constituents"],
        100 * report["verified_count_fraction"],
    )
    return universe, mapping

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically acquire Vanguard VT holdings and resolve them through Yahoo Finance."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument(
        "--source",
        help="Optional manually downloaded official Vanguard full-holdings CSV/XLSX fallback.",
    )
    parser.add_argument("--no-yahoo", action="store_true", help="Build the source universe without resolving Yahoo symbols.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Resolve only the first N pending Yahoo mappings.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--yahoo-healthcheck",
        action="store_true",
        help="Test AAPL/MSFT through yfinance without modifying the universe or ticker map.",
    )
    args = parser.parse_args()

    config, paths = load_project(args.config, args.paths)
    log_level = str(config.get("project", {}).get("log_level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.yahoo_healthcheck:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise SystemExit("Install project dependencies before running the Yahoo health check") from exc
        cache_dir = Path(paths["data"]["vanguard_holdings_dir"]).parent / "yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        setter = getattr(yf, "set_tz_cache_location", None)
        if callable(setter):
            setter(str(cache_dir))
        try:
            detail = _wait_for_yahoo_health(yf, config)
        except YahooTemporaryFailure as exc:
            raise SystemExit(str(exc)) from None
        print(
            json.dumps(
                {
                    "healthy": True,
                    "yfinance_version": getattr(yf, "__version__", "unknown"),
                    "cache_dir": str(cache_dir),
                    "detail": detail,
                },
                indent=2,
            )
        )
        return
    try:
        build_reference_files(
        config,
        paths,
        force=args.force,
        source_file=args.source,
        resolve_yahoo=not args.no_yahoo,
        resume=not args.no_resume,
        limit=args.limit,
        )
    except YahooTemporaryFailure as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user. The ticker-map checkpoint was preserved; rerun to resume.") from None


if __name__ == "__main__":
    main()
