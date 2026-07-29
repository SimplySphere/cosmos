from __future__ import annotations

import csv
import io
import json
import logging
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.files import ensure_parent

LOGGER = logging.getLogger(__name__)

DEFAULT_PROFILE_URLS = [
    "https://advisors.vanguard.com/investments/products/vt/vanguard-total-world-stock-etf",
    "https://advisors.vanguard.com/investments/products/vt/vanguard-total-world-stock-etf.html",
    "https://investor.vanguard.com/investment-products/etfs/profile/vt",
]

HOLDING_COLUMN_HINTS = {
    "ticker",
    "holdings",
    "holding",
    "security",
    "security name",
    "name",
    "cusip",
    "sedol",
    "% of fund",
    "weight",
    "market value",
    "shares",
}


def _normalise_label(value: Any) -> str:
    text = str(value).strip().lower().replace("&", "and")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _session(timeout: int) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def _request(session: requests.Session, url: str, timeout: int) -> requests.Response:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response


def _candidate_url(value: str, base_url: str) -> str | None:
    raw = str(value or "").strip().replace("\\/", "/")
    if not raw or raw.startswith(("javascript:", "mailto:", "#")):
        return None
    absolute = urljoin(base_url, raw)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    lowered = absolute.lower()
    relevant = any(token in lowered for token in ("holding", "portfolio", "composition"))
    downloadable = any(token in lowered for token in ("export", "download", ".csv", ".xlsx", ".xls", ".zip"))
    return absolute if relevant and downloadable else None


def _walk_json(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _discover_urls(html: str, base_url: str) -> list[str]:
    discovered: list[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        for attribute in (
            "href",
            "src",
            "data-url",
            "data-href",
            "data-download-url",
            "data-export-url",
            "download-url",
        ):
            candidate = _candidate_url(tag.get(attribute, ""), base_url)
            if candidate and candidate not in discovered:
                discovered.append(candidate)

    # Vanguard applications often place API/download URLs inside JSON script tags.
    for script in soup.find_all("script"):
        payload = script.string or script.get_text("", strip=True)
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = None
        if parsed is not None:
            for item in _walk_json(parsed):
                if isinstance(item, str):
                    candidate = _candidate_url(item, base_url)
                    if candidate and candidate not in discovered:
                        discovered.append(candidate)
        for match in re.findall(r"https?:\\?/\\?/[^\"'<>\\s]+|/[A-Za-z0-9_?&=./%+-]+", payload):
            candidate = _candidate_url(match, base_url)
            if candidate and candidate not in discovered:
                discovered.append(candidate)
    return discovered


def _score_table(frame: pd.DataFrame) -> int:
    columns = {_normalise_label(column) for column in frame.columns}
    matches = sum(
        1
        for hint in HOLDING_COLUMN_HINTS
        if hint in columns or any(hint in column for column in columns)
    )
    if any("ticker" == column or "symbol" == column for column in columns):
        matches += 3
    if any("holding" in column or "security" in column for column in columns):
        matches += 3
    if any("% of fund" in column or "weight" in column for column in columns):
        matches += 2
    return matches


def _best_table(tables: Iterable[pd.DataFrame], minimum_rows: int) -> pd.DataFrame | None:
    ranked: list[tuple[int, int, pd.DataFrame]] = []
    for table in tables:
        if table is None or table.empty:
            continue
        ranked.append((_score_table(table), len(table), table))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, rows, frame = ranked[0]
    if score < 5 or rows < minimum_rows:
        return None
    return frame.copy()


def _tables_from_html(html: str, minimum_rows: int) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return None
    return _best_table(tables, minimum_rows)


def _records_from_json(html: str, minimum_rows: int) -> pd.DataFrame | None:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[pd.DataFrame] = []
    for script in soup.find_all("script"):
        payload = script.string or script.get_text("", strip=True)
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        for item in _walk_json(parsed):
            if not isinstance(item, list) or len(item) < minimum_rows:
                continue
            if not item or not all(isinstance(row, dict) for row in item[: min(20, len(item))]):
                continue
            frame = pd.DataFrame(item)
            if _score_table(frame) >= 5:
                candidates.append(frame)
    return _best_table(candidates, minimum_rows)


def _extension_from_response(response: requests.Response, url: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, re.I)
    filename = match.group(1).strip() if match else Path(urlparse(url).path).name
    suffix = Path(filename).suffix.lower()
    if suffix in {".csv", ".xlsx", ".xls", ".zip"}:
        return suffix
    content_type = response.headers.get("content-type", "").lower()
    if "spreadsheetml" in content_type:
        return ".xlsx"
    if "excel" in content_type or "ms-excel" in content_type:
        return ".xls"
    if "zip" in content_type or response.content[:4] == b"PK\x03\x04":
        return ".zip"
    return ".csv"


def _decode_delimited_text(path: Path) -> tuple[str, str]:
    """Decode a Vanguard text export without assuming UTF-8.

    Vanguard exports have appeared as UTF-8, UTF-16, and Windows-1252. A
    UTF-16 file often contains NUL bytes, so attempting UTF-8 first can produce
    misleading parser failures rather than a useful encoding error.
    """

    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"Downloaded Vanguard file is empty: {path}")

    encodings: list[str] = []
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in payload[:4096]:
        encodings.extend(["utf-16", "utf-16-le", "utf-16-be"])
    encodings.extend(["utf-8-sig", "utf-8", "cp1252", "latin-1"])

    errors: list[str] = []
    for encoding in dict.fromkeys(encodings):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        f"Unable to decode Vanguard export {path}. Tried: " + "; ".join(errors)
    )


def _unique_headers(values: list[str]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for position, value in enumerate(values, start=1):
        base = str(value).strip() or f"unnamed_{position}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    return headers


def _header_score(values: list[str]) -> int:
    labels = [_normalise_label(value) for value in values if str(value).strip()]
    if len(labels) < 3:
        return -1

    score = 0
    for label in labels:
        if label in {"ticker", "symbol", "local ticker"}:
            score += 5
        if label in {"holdings", "holding", "security", "security name", "name"}:
            score += 5
        if "% of fund" in label or "% of funds" in label or "weight" in label:
            score += 5
        if "market value" in label:
            score += 2
        if label in {"cusip", "sedol", "isin", "shares", "sub industry"}:
            score += 1

    has_name = any(
        label in {"holdings", "holding", "security", "security name", "name"}
        for label in labels
    )
    has_ticker = any(label in {"ticker", "symbol", "local ticker"} for label in labels)
    has_weight = any(
        "% of fund" in label or "% of funds" in label or "weight" in label
        for label in labels
    )
    if has_weight and (has_name or has_ticker):
        score += 8
    return score


def _frame_from_rows(header: list[str], rows: list[list[str]]) -> pd.DataFrame:
    width = len(header)
    normalized_rows: list[list[str]] = []
    normalized_header = [_normalise_label(value) for value in header]

    for row in rows:
        values = [str(value).strip() for value in row]
        if not any(values):
            continue
        if [_normalise_label(value) for value in values[:width]] == normalized_header:
            # Some Vanguard exports repeat the header between asset sections.
            continue
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        elif len(values) > width:
            # Extra delimiters usually occur in a footer or disclaimer. Preserve
            # the defined table width instead of making the frame ragged.
            values = values[:width]
        normalized_rows.append(values)

    return pd.DataFrame(normalized_rows, columns=_unique_headers(header))


def _delimited_candidates(path: Path) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    text, encoding = _decode_delimited_text(path)
    lines = text.splitlines()
    delimiters = [",", "\t", ";", "|"]
    candidates: list[tuple[int, int, str, list[str]]] = []

    # Search beyond the introductory fund metadata. Vanguard's actual header is
    # usually near the top, but a larger bound makes the parser resilient to
    # added disclosures without scanning a very large file repeatedly.
    for line_number, line in enumerate(lines[:200]):
        if not line.strip():
            continue
        for delimiter in delimiters:
            try:
                row = next(csv.reader([line], delimiter=delimiter))
            except csv.Error:
                continue
            score = _header_score(row)
            if score >= 10:
                candidates.append((score, line_number, delimiter, row))

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "encoding": encoding,
        "line_count": len(lines),
        "candidate_headers": [],
    }

    for score, line_number, delimiter, header in candidates[:12]:
        try:
            reader = csv.reader(lines[line_number + 1 :], delimiter=delimiter)
            rows = list(reader)
        except csv.Error as exc:
            diagnostics.setdefault("errors", []).append(
                f"line {line_number + 1}, delimiter {delimiter!r}: {exc}"
            )
            continue
        frame = _frame_from_rows(header, rows)
        diagnostics["candidate_headers"].append(
            {
                "line": line_number + 1,
                "delimiter": "TAB" if delimiter == "\t" else delimiter,
                "score": score,
                "columns": header,
                "rows": len(frame),
            }
        )
        frames.append(frame)

    return frames, diagnostics


def _read_tabular(
    path: Path,
    minimum_rows: int,
    *,
    log_diagnostics: bool = True,
) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                members = [
                    member
                    for member in archive.namelist()
                    if Path(member).suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
                ]
                for number, member in enumerate(members, start=1):
                    extracted = path.parent / f"extracted_{number}_{Path(member).name}"
                    with archive.open(member) as source, extracted.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    parsed = _read_tabular(extracted, minimum_rows, log_diagnostics=log_diagnostics)
                    if parsed is not None:
                        return parsed
        except (OSError, zipfile.BadZipFile) as exc:
            LOGGER.warning("Could not inspect Vanguard ZIP %s: %s", path, exc)
        return None

    tables: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    if suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(
                path,
                sheet_name=None,
                header=None,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            LOGGER.warning("Could not read Vanguard workbook %s: %s", path, exc)
            sheets = {}
        for sheet_name, raw in sheets.items():
            candidates = _candidate_headers(raw)
            LOGGER.debug(
                "Vanguard workbook sheet %s produced %d header candidates",
                sheet_name,
                len(candidates),
            )
            tables.extend(candidates)
    else:
        try:
            tables, diagnostics = _delimited_candidates(path)
        except Exception as exc:
            LOGGER.warning("Could not decode/scan Vanguard export %s: %s", path, exc)
            return None

    parsed = _best_table(tables, minimum_rows)
    if parsed is None and log_diagnostics:
        if diagnostics:
            LOGGER.warning(
                "Vanguard export parse diagnostics for %s: %s",
                path,
                json.dumps(diagnostics, ensure_ascii=False, default=str),
            )
        else:
            LOGGER.warning(
                "No holdings table with at least %d rows was found in %s",
                minimum_rows,
                path,
            )
    return parsed


def _candidate_headers(raw: pd.DataFrame, max_header_rows: int = 200) -> list[pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    for header in range(min(max_header_rows, len(raw))):
        columns = raw.iloc[header].fillna("").astype(str).str.strip().tolist()
        if _header_score(columns) < 10:
            continue
        frame = raw.iloc[header + 1 :].copy()
        frame.columns = _unique_headers(columns)
        frame = frame.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
        candidates.append(frame)
    return candidates

def _detect_snapshot_date(text: str) -> str | None:
    patterns = [
        r"(?:as\s+of|portfolio\s+composition\s+as\s+of)\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"(?:as\s+of|portfolio\s+composition\s+as\s+of)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        parsed = pd.to_datetime(match.group(1), errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return None


def _download_directories(primary: Path) -> list[Path]:
    directories = [primary.resolve()]
    default_downloads = (Path.home() / "Downloads").resolve()
    if default_downloads not in directories and default_downloads.exists():
        directories.append(default_downloads)
    return directories


def _snapshot_files(directories: list[Path]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.resolve()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _looks_like_download(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    if path.suffix.lower() in {".crdownload", ".part", ".tmp", ".download"}:
        return False
    if path.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".zip"}:
        return True
    try:
        prefix = path.read_bytes()[:8]
    except OSError:
        return False
    return prefix.startswith((b"PK\x03\x04", b"Ticker", b"Holdings", b"Security"))


def _wait_for_download(
    directories: list[Path],
    before: dict[Path, tuple[int, int]],
    timeout: int,
) -> Path | None:
    deadline = time.time() + timeout
    stable: dict[Path, tuple[int, int]] = {}
    while time.time() < deadline:
        candidates: list[Path] = []
        active = False
        for directory in directories:
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                resolved = path.resolve()
                suffix = path.suffix.lower()
                if suffix in {".crdownload", ".part", ".tmp", ".download"}:
                    active = True
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                current = (stat.st_size, stat.st_mtime_ns)
                old = before.get(resolved)
                changed = old is None or current != old
                if changed and _looks_like_download(path):
                    candidates.append(path)
                    if stable.get(resolved) == current and stat.st_size > 0 and not active:
                        return max(candidates, key=lambda item: item.stat().st_mtime_ns)
                    stable[resolved] = current
        time.sleep(0.75)
    return None


def _copy_to_cache(downloaded: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve().parent == directory.resolve():
        return downloaded
    suffix = downloaded.suffix.lower() or ".csv"
    target = directory / f"vt_holdings_browser_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    shutil.copy2(downloaded, target)
    return target


def _switch_frame_path(driver: Any, path: tuple[int, ...]) -> bool:
    from selenium.webdriver.common.by import By

    driver.switch_to.default_content()
    for index in path:
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        if index >= len(frames):
            return False
        driver.switch_to.frame(frames[index])
    return True


def _frame_paths(driver: Any, maximum_depth: int = 3) -> list[tuple[int, ...]]:
    from selenium.webdriver.common.by import By

    paths: list[tuple[int, ...]] = [()]

    def visit(prefix: tuple[int, ...], depth: int) -> None:
        if depth >= maximum_depth or not _switch_frame_path(driver, prefix):
            return
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        for index in range(len(frames)):
            child = prefix + (index,)
            paths.append(child)
            visit(child, depth + 1)

    visit((), 0)
    driver.switch_to.default_content()
    return paths


def _deep_click_current_context(driver: Any, texts: list[str]) -> str | None:
    script = r"""
    const wanted = arguments[0].map(x => x.toLowerCase());
    const normalize = value => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const roots = [document];
    const elements = [];
    for (let i = 0; i < roots.length; i++) {
      const root = roots[i];
      for (const node of root.querySelectorAll('*')) {
        elements.push(node);
        if (node.shadowRoot) roots.push(node.shadowRoot);
      }
    }
    const candidates = [];
    for (const el of elements) {
      const label = normalize([
        el.innerText,
        el.textContent,
        el.getAttribute && el.getAttribute('aria-label'),
        el.getAttribute && el.getAttribute('title'),
        el.getAttribute && el.getAttribute('download')
      ].filter(Boolean).join(' '));
      if (!label) continue;
      const match = wanted.find(text => label === text || label.includes(text));
      if (!match) continue;
      const tag = (el.tagName || '').toLowerCase();
      const role = normalize(el.getAttribute && el.getAttribute('role'));
      const clickable = ['a','button','input'].includes(tag) || role === 'button' ||
        typeof el.onclick === 'function' || getComputedStyle(el).cursor === 'pointer';
      if (!clickable) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) continue;
      candidates.push({el, match, area: rect.width * rect.height, exact: label === match ? 0 : 1});
    }
    candidates.sort((a,b) => a.exact - b.exact || a.area - b.area);
    if (!candidates.length) return null;
    const selected = candidates[0];
    selected.el.scrollIntoView({block:'center', inline:'center'});
    selected.el.click();
    return selected.match;
    """
    try:
        return driver.execute_script(script, texts)
    except Exception:
        return None


def _click_text_deep(driver: Any, texts: list[str], scroll: bool = True) -> bool:
    for path in _frame_paths(driver):
        if not _switch_frame_path(driver, path):
            continue
        if _deep_click_current_context(driver, texts):
            driver.switch_to.default_content()
            return True
        if not scroll:
            continue
        try:
            height = int(driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"))
            viewport = int(driver.execute_script("return window.innerHeight || 800;"))
        except Exception:
            continue
        for offset in range(0, max(height, viewport), max(viewport // 2, 400)):
            try:
                driver.execute_script("window.scrollTo(0, arguments[0]);", offset)
                time.sleep(0.15)
            except Exception:
                break
            if _deep_click_current_context(driver, texts):
                driver.switch_to.default_content()
                return True
    driver.switch_to.default_content()
    return False


def _configure_downloads(driver: Any, directory: Path) -> None:
    parameters = {"behavior": "allow", "downloadPath": str(directory.resolve())}
    for command in ("Browser.setDownloadBehavior", "Page.setDownloadBehavior"):
        try:
            driver.execute_cdp_cmd(command, parameters)
            return
        except Exception:
            continue


def _new_chrome(directory: Path, headless: bool) -> Any:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(directory.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_setting_values.automatic_downloads": 1,
        },
    )
    driver = webdriver.Chrome(options=options)
    _configure_downloads(driver, directory)
    return driver


def _try_export_on_page(driver: Any) -> bool:
    # Investor and advisor pages place holdings under different tabs/components.
    _click_text_deep(driver, ["accept all", "accept cookies", "agree"], scroll=False)
    _click_text_deep(driver, ["portfolio", "portfolio composition"], scroll=True)
    time.sleep(1.0)
    return _click_text_deep(
        driver,
        ["export full holdings", "download full holdings", "export holdings"],
        scroll=True,
    )


def _selenium_download(
    urls: list[str],
    directory: Path,
    browser_timeout: int,
    *,
    headless: bool,
    interactive: bool = False,
) -> tuple[Path | None, str | None, str | None]:
    try:
        import selenium  # noqa: F401
    except Exception as exc:
        LOGGER.info("Selenium browser fallback is unavailable: %s", exc)
        return None, None, None

    directories = _download_directories(directory)
    driver = None
    try:
        driver = _new_chrome(directory, headless=headless)
        driver.set_page_load_timeout(browser_timeout)
        for url in urls:
            before = _snapshot_files(directories)
            try:
                driver.get(url)
            except Exception as exc:
                LOGGER.warning("Chrome could not load %s: %s", url, exc)
                continue
            time.sleep(5)
            clicked = _try_export_on_page(driver)
            if clicked:
                LOGGER.info("Triggered Vanguard full-holdings export at %s", url)
                downloaded = _wait_for_download(directories, before, browser_timeout)
                if downloaded:
                    return _copy_to_cache(downloaded, directory), url, _detect_snapshot_date(driver.page_source)

            if interactive:
                LOGGER.warning(
                    "Vanguard did not allow a fully automatic export. A visible Chrome window is open. "
                    "Go to Portfolio or Portfolio composition, click 'Export full holdings', and leave "
                    "the terminal running. The pipeline will detect the file and continue automatically."
                )
                print(
                    "\nACTION NEEDED: In the Chrome window, click Portfolio/Portfolio composition, "
                    "then Export full holdings. This command is waiting for the download.\n",
                    file=sys.stderr,
                    flush=True,
                )
                downloaded = _wait_for_download(directories, before, browser_timeout)
                if downloaded:
                    return _copy_to_cache(downloaded, directory), url, _detect_snapshot_date(driver.page_source)
        return None, None, None
    except Exception as exc:
        LOGGER.warning("Chrome holdings export failed: %s", exc)
        return None, None, None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

def _cache_path(directory: Path, prefix: str, suffix: str) -> Path:
    """Create a collision-resistant cache path for rapid URL discovery attempts."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return directory / f"{prefix}_{stamp}{normalized_suffix}"


def acquire_vanguard_holdings(
    config: dict,
    paths: dict,
    force: bool = False,
    source_file: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire and parse the full VT holdings file from Vanguard.

    The function first reuses a cached successful export, then tries public HTTP
    discovery, deep shadow-DOM/iframe browser automation, and finally a visible
    Chrome window that waits for the user to click Vanguard's export control. The
    visible fallback remains inside the pipeline: the downloaded file is detected,
    cached, parsed, and processing resumes automatically.
    """

    universe_config = config.get("universe", {})
    minimum_rows = int(universe_config.get("minimum_source_constituents", 5000))
    timeout = int(universe_config.get("http_timeout_seconds", 45))
    browser_timeout = int(universe_config.get("browser_timeout_seconds", 90))
    profile_urls = list(universe_config.get("vanguard_profile_urls") or DEFAULT_PROFILE_URLS)

    raw_dir = Path(paths["data"]["vanguard_holdings_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(paths["data"]["vanguard_holdings_manifest"])

    if source_file:
        source = Path(source_file).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        parsed = _read_tabular(source, minimum_rows)
        if parsed is None:
            raise ValueError(f"{source} does not contain a full Vanguard VT holdings table")
        metadata = {
            "source_type": "manual_vanguard_export",
            "source_path": str(source),
            "source_url": "",
            "snapshot_date": None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(parsed),
        }
        return parsed, metadata

    if manifest_path.exists() and not force:
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached = Path(metadata["source_path"])
            if cached.exists():
                parsed = _read_tabular(cached, minimum_rows)
                if parsed is not None:
                    LOGGER.info("Using cached Vanguard VT holdings export: %s", cached)
                    return parsed, metadata
        except Exception as exc:
            LOGGER.warning("Ignoring invalid Vanguard holdings cache: %s", exc)

    session = _session(timeout)
    for profile_url in profile_urls:
        try:
            page = _request(session, profile_url, timeout)
        except Exception as exc:
            LOGGER.warning("Could not retrieve Vanguard profile %s: %s", profile_url, exc)
            continue

        snapshot = _detect_snapshot_date(page.text)
        embedded = _records_from_json(page.text, minimum_rows)
        if embedded is None:
            embedded = _tables_from_html(page.text, minimum_rows)
        if embedded is not None:
            cache = _cache_path(raw_dir, "vt_holdings_embedded", ".csv")
            embedded.to_csv(cache, index=False)
            metadata = {
                "source_type": "vanguard_profile_embedded_table",
                "source_path": str(cache),
                "source_url": page.url,
                "snapshot_date": snapshot,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(embedded),
            }
            ensure_parent(manifest_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return embedded, metadata

        for candidate_url in _discover_urls(page.text, page.url):
            try:
                response = _request(session, candidate_url, timeout)
            except Exception:
                continue
            if "text/html" in response.headers.get("content-type", "").lower():
                table = _records_from_json(response.text, minimum_rows)
                if table is None:
                    table = _tables_from_html(response.text, minimum_rows)
                if table is None:
                    continue
                cache = _cache_path(raw_dir, "vt_holdings_download", ".csv")
                table.to_csv(cache, index=False)
                parsed = table
            else:
                suffix = _extension_from_response(response, candidate_url)
                cache = _cache_path(raw_dir, "vt_holdings_download", suffix)
                cache.write_bytes(response.content)
                parsed = _read_tabular(cache, minimum_rows, log_diagnostics=False)
            if parsed is None:
                # Page scripts contain many URLs with words such as "portfolio" or
                # "download" that are telemetry/configuration payloads rather than
                # holdings files. Do not preserve or warn about those false candidates.
                try:
                    cache.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            metadata = {
                "source_type": "vanguard_public_download",
                "source_path": str(cache),
                "source_url": candidate_url,
                "snapshot_date": snapshot,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(parsed),
            }
            ensure_parent(manifest_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return parsed, metadata

    if bool(universe_config.get("browser_fallback", True)):
        downloaded, source_url, snapshot = _selenium_download(
            profile_urls,
            raw_dir,
            browser_timeout,
            headless=True,
            interactive=False,
        )
        if downloaded is None and bool(universe_config.get("browser_interactive_fallback", True)):
            interactive_timeout = int(
                universe_config.get("browser_interactive_timeout_seconds", 600)
            )
            downloaded, source_url, snapshot = _selenium_download(
                profile_urls,
                raw_dir,
                interactive_timeout,
                headless=False,
                interactive=True,
            )
        if downloaded is not None:
            parsed = _read_tabular(downloaded, minimum_rows)
            if parsed is not None:
                metadata = {
                    "source_type": "vanguard_browser_export",
                    "source_path": str(downloaded),
                    "source_url": source_url or "",
                    "snapshot_date": snapshot,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "row_count": len(parsed),
                }
                ensure_parent(manifest_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                return parsed, metadata
            raise RuntimeError(
                "Vanguard downloaded the VT holdings file successfully, but the file could not be "
                f"parsed as a full holdings table with at least {minimum_rows:,} rows: {downloaded}. "
                "The parser logged its detected encoding, delimiter, candidate header rows, and row "
                "counts immediately above this error. Do not download the file again; inspect those "
                "diagnostics or pass the same file with `python -m src.data.build_universe --source "
                f"{downloaded} --force`."
            )

    raise RuntimeError(
        "The Vanguard VT full-holdings export was not downloaded. The pipeline attempted HTTP "
        "discovery, headless Chrome automation, and the configured visible-Chrome fallback. Re-run "
        "from an interactive desktop terminal and click 'Export full holdings' when Chrome opens. "
        "You may also pass an already downloaded official Vanguard CSV/XLSX with "
        "`python -m src.data.build_universe --source /path/to/file.csv --force`."
    )
