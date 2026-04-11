"""
Local chat web app. Serves the UI and proxies messages to an LLM.
Set OPENAI_API_KEY to use OpenAI; otherwise uses Ollama at http://127.0.0.1:11434.
Optional online mode uses DuckDuckGo (network), extra queries for social-style questions,
and safe HTTP fetch + text extraction for URLs in the user message.
"""
import asyncio
import contextlib
import ipaddress
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# Long web gather + large context + slow local LLM can exceed default httpx read timeouts.
OLLAMA_HTTP_TIMEOUT = max(60.0, float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600")))
OPENAI_HTTP_TIMEOUT = max(60.0, float(os.getenv("OPENAI_HTTP_TIMEOUT", "300")))
# SSE comment lines every N seconds so browsers and proxies do not close "idle" streams.
SSE_KEEPALIVE_INTERVAL_S = max(3.0, float(os.getenv("SSE_KEEPALIVE_INTERVAL_S", "12")))

STATIC = Path(__file__).parent / "static"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Set to 1 to ask Ollama for a short search query from the user message (extra local round-trip).
REFINE_SEARCH_QUERY = os.getenv("OLLAMA_REFINE_SEARCH_QUERY", "").lower() in ("1", "true", "yes")
# When 1 (default), also run online search for "latest / news / Truth Social / Twitter"-style questions
# even if the user left "Search online" unchecked.
AUTO_SEARCH_RECENCY = os.getenv("AUTO_SEARCH_RECENCY", "1").lower() not in (
    "0",
    "false",
    "no",
)
# DuckDuckGo: parallel queries (faster than sequential sleeps). Set DDG_PARALLEL=0 to disable.
DDG_PARALLEL = os.getenv("DDG_PARALLEL", "1").lower() not in ("0", "false", "no")
# Fewer parallel DDGS clients reduces empty results from rate limits (try 2 before 4).
DDG_MAX_WORKERS = max(1, min(8, int(os.getenv("DDG_MAX_WORKERS", "2"))))
DDG_TIMEOUT_S = max(5, int(os.getenv("DDG_TIMEOUT", "12")))
# Cap total search phrases (each can take several seconds). DDG_CHUNK_SIZE batches + yields SSE between.
DDG_MAX_QUERIES = max(4, min(28, int(os.getenv("DDG_MAX_QUERIES", "8"))))
DDG_CHUNK_SIZE = max(1, min(8, int(os.getenv("DDG_CHUNK_SIZE", "4"))))
DDG_EARLY_EXIT_SNIPPETS = max(6, min(28, int(os.getenv("DDG_EARLY_EXIT_SNIPPETS", "10"))))
# After enough text from direct ABS/DFAT fetches, run at most this many extra DDG query slots (speed).
DDG_EXTRA_QUERY_SLOTS_AFTER_DIRECT = max(
    0, min(DDG_MAX_QUERIES, int(os.getenv("DDG_EXTRA_QUERY_SLOTS_AFTER_DIRECT", "4")))
)
# Fewer DDG backend retries = faster (2 recommended).
DDG_TEXT_ATTEMPTS = max(1, min(6, int(os.getenv("DDG_TEXT_ATTEMPTS", "2"))))
# Hard cap per asyncio.to_thread(DDG) call — prevents 5–10+ minute hangs when DDG stalls.
DDG_BATCH_TIMEOUT_S = max(15.0, float(os.getenv("DDG_BATCH_TIMEOUT_S", "45")))
# Pause between sequential DDG queries inside a worker (lower = faster, higher empty-rate risk).
DDG_PAUSE_S = max(0.0, float(os.getenv("DDG_PAUSE_S", "0.22")))
# Total wall time for all DuckDuckGo batches in one request (0 = no extra cap beyond per-batch).
# Stops multi-batch + fallback stacks from reaching many minutes (e.g. several × per-batch cap).
DDG_TOTAL_BUDGET_S = max(0.0, float(os.getenv("DDG_TOTAL_BUDGET_S", "120")))
# When parallel DDG returns nothing, sequentially retry each query (very slow). Default off.
DDG_SEQUENTIAL_FALLBACK = os.getenv("DDG_SEQUENTIAL_FALLBACK", "0").lower() in (
    "1",
    "true",
    "yes",
)
# If direct ABS/industry pages yield this many characters, skip the first DuckDuckGo batch (speed).
SKIP_FIRST_DDG_IF_DIRECT_CHARS = max(
    0, int(os.getenv("SKIP_FIRST_DDG_IF_DIRECT_CHARS", "500"))
)
# Target DuckDuckGo snippet rows before early exit (lower = faster).
WEB_SNIPPET_GOAL = max(8, min(28, int(os.getenv("WEB_SNIPPET_GOAL", "16"))))
# Per-URL timeout for ABS/DFAT/industry direct fetches (seconds).
DIRECT_FETCH_TIMEOUT_S = max(5.0, float(os.getenv("DIRECT_FETCH_TIMEOUT_S", "12")))
# Max official pages to fetch (coal queries default to the most relevant hosts first).
AU_DIRECT_FETCH_MAX = max(1, min(8, int(os.getenv("AU_DIRECT_FETCH_MAX", "3"))))

SYSTEM_BASE = """You are a helpful assistant. Be direct and readable.

Do not refuse informational questions with boilerplate about being an AI, lacking real-time access,
being unable to browse, or being unable to provide up-to-date information as your whole answer.

Never make rate limits, CAPTCHAs, robots, or search-index limits the center of your reply. Do not say
you "could not find anything because of rate limiting" or similar. If material below is thin, still
synthesize the best supported story from titles, snippets, news mirrors, Reddit, or forum lines.

Do not tell the user to "check the official website" or "check reputable news sources" as the main
answer. If you mention a place to verify, it must come after you have already summarized concrete
claims from the excerpts below (who said what, what outlets reported, approximate timing).

**Banned patterns (do not use as your whole reply):** opening with "I couldn't find" or
"Unfortunately, I couldn't find" **when any usable excerpt appears below**; bullet lists of "sources
to check", "possible sources", "you can try", or "check the Truth Social app/website"; telling the user
to look at Twitter/X instead of answering; generic homework about "reputable news" with no summary
from the excerpts; attributing a statistic to "ABS via news.com.au" unless that chain appears in the
excerpts—prefer naming the **underlying publisher** (ABS, Department of Industry, DFAT) from the text.

Also banned: "Unfortunately, I don't have direct access to [Truth Social / current content]";
"based on my previous knowledge" or "my knowledge up to [month year]"; "up to December 2023" or any
**pretraining cutoff year** as your timeline frame; recommending CNN/BBC/Fox as a substitute for
summarizing the excerpts. If web excerpts exist, they override your training cutoff.

For **statistics, exports, trade volumes, or census-style figures**: prioritize lines from **official
Australian Government** sources when they appear in the excerpts: **abs.gov.au**, **dfat.gov.au**,
**industry.gov.au** (Department of Industry, Science and Resources — resources/energy publications),
**treasury.gov.au**, and state/territory statistical releases. Treat tabloid or SEO news pages as
**secondary** unless they are clearly quoting the same primary figure with a date.

**Australian commodity exports (coal, LNG, iron ore, etc.):** figures are often reported for the
**Australian financial year (FY, July–June)** or for **ABS reference periods** (e.g. “year ending …”),
not always the calendar year in the question. If the user says “2025”, explain whether your number is
**calendar 2025**, **FY 2024–25**, **FY 2025–26**, or **year ending** a month, matching the excerpt.
Do **not** merge tonnage from one period with dollar value from another, or a news headline figure
with a different year from a department table.

**Numbers:** Every **tonnage, value, or percentage** you state must be **explicitly supported by a
retrieved line below** (same sentence, table caption, or adjacent paragraph). If excerpts disagree,
report **both** with their dates/sources—do not pick one arbitrarily. Never “fill in” a missing year
with training-data memory. If excerpts lack the exact year asked, say what period **is** covered and
quote the best official estimate from the excerpts (e.g. latest FY or latest quarterly).

For current events / "latest" / social posts:
- Use search snippets AND any extracted page text below. Third-party reporting counts—summarize it.
- Prefer specifics (dates, paraphrased quotes, outlet names) over vague disclaimers.
- **Recency:** When excerpts include dates, privilege the **most recent** credible items. If everything
  indexed looks years old, say that plainly (e.g. search results skewed to older stories) and still
  report the **newest** dated line you see—do **not** describe 2–3 year old items as if they were
  "today" without that caveat.
- If nothing in the excerpts names a post verbatim, say what *is* described without refusing the question.

Avoid answering with only "no", "I couldn't", or empty denial. Stay concise."""

SYSTEM_WEB_GROUNDING = """
The following material was retrieved with searches biased toward the current year and toward outlets
such as CNN, BBC, Guardian, NBC, Reuters, AP, Australian sources (e.g. news.com.au, 9News, 7 News),
**Australian Government pages (ABS, DFAT, industry.gov.au)**, and Truth Social–related news lines.
You must ground your answer in it. Prefer **official statistics and departmental publications** over
aggregators when they conflict on numbers. Prefer newer-looking items when dates conflict. If one URL
failed, other snippets and pages below remain valid."""

# Prepended only when web_blob is non-empty (before the retrieved text).
SYSTEM_WEB_OUTPUT_RULES = """
OUTPUT RULES FOR THIS TURN (web material is present):
Your FIRST sentences must summarize what the retrieved lines below actually say—topics, rough dates,
financial year or “year ending” labels when visible, and outlet or domain names (abs.gov.au,
industry.gov.au, dfat.gov.au, CNN, BBC, 9News, news.com.au, Truth Social mirrors, etc.) when visible.
Do not substitute a list of places for the user to look. If snippets are vague, still paraphrase the
strongest concrete detail present; only then add a brief caveat—not the reverse order.

Do **not** illustrate "recent" with December 2023 (or other old years) unless that date explicitly
appears in the retrieved lines below. Prefer any line that looks like **this year**, last year, or the
latest **FY** in the excerpts.

For **coal / resource export volumes**: if the excerpts give a **Resources and Energy Quarterly** or
departmental estimate (e.g. metallurgical + thermal, Mt), prefer that narrative over a mismatched ABS
merchandise line from a different period—**do not combine incompatible figures**.

**Citations (required when web material is present):** Use Markdown inline links for the main factual
claims, e.g. `[ABS International trade, Jun 2025](https://www.abs.gov.au/...)` or
`[Resources and Energy Quarterly — September 2025](https://www.industry.gov.au/...)`. Use **only**
`https://` URLs that appear in the retrieved excerpts below or in the URL list the UI will show—do not
invent links. Prefer **.gov.au** official pages for statistics. Do **not** repeat the same sentence or
paragraph; say it once clearly."""


def _calendar_context_block() -> str:
    n = datetime.now(timezone.utc)
    y = n.year
    y1 = y - 1
    return f"""
--- Calendar (authoritative) ---
UTC date: **{n.date().isoformat()}**. For "latest", "recent", and "this year", prioritize material
referencing **{y}** or **{y1}**. Treat older years (e.g. 2023) as **stale** unless the retrieved excerpts
below only contain those—then say the index skewed old. Never use an internal "knowledge cutoff" story
(especially 2023) when the excerpts include newer years.
"""


def _utc_year_month() -> tuple[int, str]:
    n = datetime.now(timezone.utc)
    return n.year, n.strftime("%B")


def _dedupe_queries_in_order(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if len(q) < 2 or q in seen:
            continue
        seen.add(q)
        out.append(q[:400])
    return out


def _years_from_user_message(text: str) -> list[str]:
    found = re.findall(r"\b(20[0-3][0-9])\b", text or "")
    return list(dict.fromkeys(found))[:4]


def _au_trade_or_stats_intent(user_message: str) -> bool:
    """Australia + trade/commodity or stats-style question — skip slow LLM query-refine; prioritize ABS/DFAT."""
    lm = (user_message or "").strip().lower()
    au = bool(
        re.search(
            r"\b(australia|australian|aus\b|australian bureau of statistics|abs\b)",
            lm,
        )
    )
    trade_or_commodity = bool(
        re.search(
            r"\b(export|import|trade|balance|coal|iron ore|lng|gas|commodit|shipping|resource)\b",
            lm,
        )
    )
    stats_intent = bool(
        re.search(
            r"\b(statistic|statistics|census|official data|how much|how many|tonne|tonnes|kilogram|gdp|total export)\b",
            lm,
        )
    )
    return au and (trade_or_commodity or stats_intent)


def _broad_au_trade_leading_queries(user_message: str, seed: str) -> list[str]:
    """Short, broad phrases first — DuckDuckGo often fails on site:abs.gov.au-only queries."""
    year, _ = _utc_year_month()
    um = (user_message or "").strip()
    lm = um.lower()
    years_mentioned = _years_from_user_message(um)
    y_focus = years_mentioned[0] if years_mentioned else str(year)
    out: list[str] = [
        f"Australia international trade exports statistics {y_focus}",
        f"Australian Bureau of Statistics trade in goods {y_focus}",
    ]
    if "coal" in lm:
        out.insert(
            0,
            f"Australia coal exports million tonnes financial year {y_focus}",
        )
        out.append(f"Resources Energy Quarterly Australia coal exports {y_focus}")
        out.append(f"Australia coal export volumes Department Industry Science Resources {y_focus}")
    else:
        out.insert(0, f"Australia merchandise exports trade statistics {y_focus}")
    if "iron" in lm or "ore" in lm:
        out.append(f"Australia iron ore exports {y_focus}")
    if "gas" in lm or "lng" in lm:
        out.append(f"Australia LNG exports {y_focus}")
    return out[:6]


def _au_trade_seed_urls(user_message: str) -> list[str]:
    """Stable landing pages — fetched directly when DDG is unreliable. Order: most relevant first."""
    lm = (user_message or "").lower()
    urls: list[str] = [
        "https://www.abs.gov.au/statistics/economy/international-trade/international-trade-goods-services-australia/latest-release",
        "https://www.dfat.gov.au/trade/trade-and-investment-data-information/trade-statistics",
    ]
    if "coal" in lm or "metallurgical" in lm or "thermal" in lm:
        # Industry RE Quarterly first (coal/energy narrative); skip extra topic pages to save latency.
        urls.insert(
            0,
            "https://www.industry.gov.au/publications/resources-and-energy-quarterly",
        )
    return urls[: AU_DIRECT_FETCH_MAX]


def _contextual_site_queries(user_message: str, seed: str) -> list[str]:
    """Route searches toward the right official sources from question wording (e.g. ABS for AU trade)."""
    year, _ = _utc_year_month()
    um = (user_message or "").strip()
    lm = um.lower()
    core = (seed or um).strip()[:240]
    years_mentioned = _years_from_user_message(um)
    y_focus = years_mentioned[0] if years_mentioned else str(year)

    out: list[str] = []

    au = bool(
        re.search(
            r"\b(australia|australian|aus\b|australian bureau of statistics|abs\b)",
            lm,
        )
    )
    trade_or_commodity = bool(
        re.search(
            r"\b(export|import|trade|balance|coal|iron ore|lng|gas|commodit|shipping|resource)\b",
            lm,
        )
    )
    stats_intent = bool(
        re.search(
            r"\b(statistic|statistics|census|official data|how much|how many|tonne|tonnes|kilogram|gdp)\b",
            lm,
        )
    )

    if au and (trade_or_commodity or stats_intent):
        # Broad lines first; narrow site: filters after (see _broad_au_trade_leading_queries prepended in gather).
        if "coal" in lm:
            out.extend(
                [
                    f"Australia coal export value OR volume {y_focus}",
                    f"Australia coal export million tonnes financial year {y_focus} site:industry.gov.au",
                    f"Australia coal export {y_focus} site:abs.gov.au",
                    f"Australia coal export {y_focus} site:dfat.gov.au",
                    f"Australian Bureau of Statistics international trade goods {y_focus} site:abs.gov.au",
                    f"Australia merchandise trade exports {y_focus} site:abs.gov.au",
                    f"Australia merchandise trade exports {y_focus} site:dfat.gov.au",
                    f"{core} site:abs.gov.au",
                    f"{core} site:dfat.gov.au",
                    f"ABS international trade in goods and services Australia {y_focus}",
                ]
            )
        else:
            out.extend(
                [
                    f"Australia merchandise trade exports {y_focus} site:abs.gov.au",
                    f"Australia merchandise trade exports {y_focus} site:dfat.gov.au",
                    f"{core} site:abs.gov.au",
                    f"{core} site:dfat.gov.au",
                    f"{core} site:industry.gov.au",
                    f"ABS international trade in goods and services Australia {y_focus}",
                ]
            )

    if au and "population" in lm:
        out.append(f"Australia population {y_focus} site:abs.gov.au")

    if re.search(r"\b(unemployment|labour|labor statistics|cpi|inflation)\b", lm) and au:
        out.append(f"Australia {y_focus} labour force site:abs.gov.au")

    return out[:10]


def _priority_this_year_queries(seed: str, user_message: str) -> list[str]:
    """Front-load searches aimed at the current calendar year."""
    year, month = _utc_year_month()
    lm = (user_message or "").lower()
    core = (seed or user_message).strip()[:220]
    pq: list[str] = []
    if (
        re.search(r"\b(australia|australian)\b", lm)
        and "coal" in lm
        and "trump" not in lm
        and "truth" not in lm
    ):
        pq.append(
            f"Australia coal exports million tonnes FY {year - 1} {year} site:industry.gov.au"
        )
        pq.append(f"Resources Energy Quarterly coal exports Australia {year}")
    if "truth" in lm or "trump" in lm:
        pq.extend(
            [
                f"Donald Trump statement {year} site:cnn.com OR site:nbcnews.com OR site:bbc.com",
                f"Trump {year} site:cnn.com OR site:bbc.com OR site:nbcnews.com OR site:news.com.au",
                f"Trump Truth Social {year} {month} news site:cnn.com OR site:9news.com.au OR site:bbc.com",
            ]
        )
    pq.append(
        f"{core} {year} breaking site:reuters.com OR site:apnews.com OR site:theguardian.com"
    )
    return pq[:5]


def _authority_recency_queries(user_message: str, seed: str) -> list[str]:
    year, month = _utc_year_month()
    um = (user_message or "").strip()
    lm = um.lower()
    core = (seed or um).strip()[:260]
    qs: list[str] = []

    qs.append(
        f"{core} {year} {month} breaking news site:nbcnews.com OR site:theguardian.com OR "
        f"site:reuters.com OR site:apnews.com"
    )
    qs.append(
        f"{core} latest news {year} site:theguardian.com OR site:nbcnews.com OR site:reuters.com"
    )
    qs.append(
        f"{core} {year} site:7news.com.au OR site:nbcnews.com OR site:bbc.com"
    )
    qs.append(
        f"{core} OR policy statement {year} site:whitehouse.gov OR site:state.gov OR "
        f"site:justice.gov OR site:congress.gov OR site:govinfo.gov"
    )
    if "trump" in lm or "truth" in lm:
        qs.append(
            f"Donald Trump Truth Social {year} news site:nbcnews.com OR site:theguardian.com OR "
            f"site:reuters.com OR site:apnews.com"
        )
    if "trump" in lm:
        qs.append(
            f"Trump latest {year} site:nbcnews.com OR site:theguardian.com OR site:bbc.com"
        )

    qs.append(
        f"{core} {year} site:cnn.com OR site:bbc.com OR site:news.com.au OR site:9news.com.au"
    )
    if "trump" in lm or "truth" in lm:
        qs.append(
            f"Trump Truth Social {year} site:news.com.au OR site:9news.com.au OR site:cnn.com OR "
            f"site:bbc.com OR truthsocial"
        )

    return qs


def _sort_rows_by_recency(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    year = datetime.now(timezone.utc).year
    y, y1 = str(year), str(year - 1)

    def score(r: dict[str, str]) -> float:
        blob = f"{r.get('title', '')} {r.get('body', '')} {r.get('href', '')}"
        b = blob.lower()
        s = 0.0
        if y in blob:
            s += 26.0
        if y1 in blob:
            s += 14.0
        if any(
            t in b
            for t in (
                "hours ago",
                "hour ago",
                "day ago",
                "days ago",
                "minutes ago",
                "breaking",
                "just in",
            )
        ):
            s += 6.0
        if any(
            d in b
            for d in (
                "theguardian.com",
                "nbcnews.com",
                "reuters.com",
                "apnews.com",
                "bbc.com",
                "bbc.co.uk",
                "7news.com.au",
                "9news.com.au",
                "9news.com",
                "news.com.au",
                "cnn.com",
                "truthsocial.com",
            )
        ):
            s += 3.0
        if any(
            d in b
            for d in (
                "abs.gov.au",
                "dfat.gov.au",
                "industry.gov.au",
                "treasury.gov.au",
                "data.gov.au",
            )
        ):
            s += 12.0
        if ".gov" in b or ".gov.au" in b:
            s += 2.0
        for old in ("2019", "2020", "2021", "2022"):
            if old in blob:
                s -= 12.0
        if "2023" in blob:
            s -= 22.0
            if y in blob or y1 in blob:
                s += 14.0
        return s

    return sorted(rows, key=score, reverse=True)


app = FastAPI(title="Local Chat")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    use_web: bool = False


class ChatResponse(BaseModel):
    reply: str
    source: str  # "openai" | "ollama"
    web_used: bool = False
    source_urls: list[str] = Field(default_factory=list)
    elapsed_ms: int | None = None


class ReplyPartialRequest(BaseModel):
    """Finish with whatever web context was gathered (client may send partial blob)."""

    messages: list[ChatMessage] = Field(min_length=1)
    web_blob: str = ""


URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


def _dedupe_http_urls(urls: list[str], *, limit: int = 48) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        u = (raw or "").strip().rstrip(").,;\"'")
        if not u or u in seen:
            continue
        if not u.startswith(("http://", "https://")):
            continue
        try:
            if not _url_allowed(u) or _discard_fetch_target(u):
                continue
        except Exception:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= limit:
            break
    return out


def _collect_source_urls(
    direct_sections: list[str],
    rows: list[dict[str, str]],
    scraped: list[tuple[str, str]],
) -> list[str]:
    urls: list[str] = []
    for s in direct_sections:
        m = re.search(r"## Official page \((https?://[^)]+)\)", s)
        if m:
            urls.append(m.group(1).strip())
    for r in rows:
        h = (r.get("href") or "").strip()
        if h:
            urls.append(h)
    for url, _ in scraped:
        if url:
            urls.append(url)
    return _dedupe_http_urls(urls)


def _source_urls_from_context_blob(blob: str) -> list[str]:
    """Recover source links from assembled context text (e.g. reply-partial)."""
    urls: list[str] = []
    for m in re.finditer(r"## Official page \((https?://[^)]+)\)", blob or ""):
        urls.append(m.group(1).strip())
    urls.extend(URL_RE.findall(blob or ""))
    return _dedupe_http_urls(urls)

_STOP_GREP = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "that",
        "this",
        "from",
        "with",
        "have",
        "does",
        "your",
        "latest",
        "post",
        "made",
        "about",
        "tell",
        "please",
        "could",
        "would",
        "there",
        "their",
    }
)


def _extract_urls(text: str) -> list[str]:
    raw = URL_RE.findall(text or "")
    out: list[str] = []
    seen: set[str] = set()
    for u in raw:
        u = u.rstrip(".,;:)\"'")
        if len(u) < 8 or u in seen or len(u) > 2000:
            continue
        seen.add(u)
        out.append(u)
    return out


def _url_allowed(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = (p.hostname or "").lower()
        if host in ("localhost", "0.0.0.0") or host.endswith(".local"):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


def _format_ddg_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "(No indexed snippets returned this round; rely on any extracted pages below if present.)"
    lines = []
    for r in rows:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        href = (r.get("href") or "").strip()
        lines.append(f"• {title}\n  {body}\n  {href}")
    return "\n\n".join(lines)


def _merge_ddg_rows(
    primary: list[dict[str, str]], extra: list[dict[str, str]]
) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in primary + extra:
        key = (r.get("href") or "") + "\x00" + (r.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _infer_ddg_region_timelimit(user_message: str) -> tuple[str, str | None]:
    """Region + timelimit for DDG text search (see DuckDuckGo /params for kl= and df=)."""
    lm = (user_message or "").strip().lower()
    # Default wt-wt: automated searches often return *no* snippets with au-en; queries already say "Australia".
    override = (os.getenv("DDG_REGION") or "").strip().lower()
    if override in ("wt-wt", "us-en", "uk-en", "au-en", "ru-ru", "de-de", "fr-fr"):
        region = override
    else:
        region = "wt-wt"
    stats_or_trade = bool(
        re.search(
            r"\b(export|import|trade|statistic|statistics|tonne|tonnes|census|abs|gdp)\b|"
            r"\bhow much\b|\bhow many\b",
            lm,
        )
    )
    news_recency = bool(
        re.search(
            r"\b(breaking|latest|just in|today|this week|right now|this morning|hours ago)\b",
            lm,
        )
    )
    # Past-year filter drops many official statistical releases; keep it off unless user wants "news".
    if stats_or_trade and not news_recency:
        timelimit = None
    elif news_recency:
        timelimit = "y"
    else:
        timelimit = None
    return region, timelimit


def _ddg_text_raw(
    keywords: str,
    *,
    region: str,
    timelimit: str | None,
    per_query: int,
    backend: str,
) -> list[dict[str, str]]:
    """One DDG backend attempt (separate DDGS instance for thread safety)."""
    from duckduckgo_search import DDGS

    out: list[dict[str, str]] = []
    keywords = (keywords or "").strip()[:400]
    if len(keywords) < 2:
        return out
    try:
        with DDGS(timeout=DDG_TIMEOUT_S) as ddgs:
            for r in ddgs.text(
                keywords,
                region=region,
                timelimit=timelimit,
                backend=backend,
                max_results=per_query,
            ):
                href = (r.get("href") or "").strip()
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                if href or title or body:
                    out.append({"title": title, "body": body, "href": href})
    except Exception:
        pass
    return out


def _ddg_text_once(
    keywords: str,
    *,
    region: str,
    timelimit: str | None,
    per_query: int,
) -> list[dict[str, str]]:
    """DDG text search with fallbacks when HTML index returns nothing (common for automated clients)."""
    attempts: list[tuple[str, str | None, str]] = [
        (region, timelimit, "auto"),
    ]
    if region != "wt-wt":
        attempts.append(("wt-wt", timelimit, "auto"))
    if timelimit is not None:
        attempts.append(("wt-wt", None, "auto"))
    attempts.extend(
        [
            ("wt-wt", None, "lite"),
            ("wt-wt", None, "html"),
        ]
    )
    seen: set[tuple[str, str | None, str]] = set()
    ordered: list[tuple[str, str | None, str]] = []
    for reg, tl, bk in attempts:
        key = (reg, tl, bk)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((reg, tl, bk))
    for reg, tl, bk in ordered[:DDG_TEXT_ATTEMPTS]:
        rows = _ddg_text_raw(
            keywords,
            region=reg,
            timelimit=tl,
            per_query=per_query,
            backend=bk,
        )
        if rows:
            return rows
    return []


def _run_ddg_queries(
    queries: list[str],
    *,
    max_total: int = 24,
    per_query: int = 5,
    pause_s: float = 0.5,
    region: str = "wt-wt",
    timelimit: str | None = None,
) -> list[dict[str, str]]:
    """Run DuckDuckGo text searches; parallel by default to avoid long sequential sleeps."""
    if not queries:
        return []

    cleaned: list[tuple[int, str]] = []
    for i, q in enumerate(queries):
        q = (q or "").strip()[:400]
        if len(q) >= 2:
            cleaned.append((i, q))

    if not cleaned:
        return []

    def merge_in_order(
        indexed_results: list[tuple[int, list[dict[str, str]]]]
    ) -> list[dict[str, str]]:
        indexed_results.sort(key=lambda x: x[0])
        seen: set[str] = set()
        rows: list[dict[str, str]] = []
        for _, batch in indexed_results:
            for r in batch:
                href = (r.get("href") or "").strip()
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                key = href or (title + body)
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append({"title": title, "body": body, "href": href})
                if len(rows) >= max_total:
                    return rows
        return rows

    if DDG_PARALLEL and len(cleaned) > 1:
        workers = min(DDG_MAX_WORKERS, len(cleaned))
        indexed_batches: list[tuple[int, list[dict[str, str]]]] = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {}
                for idx, q in cleaned:
                    fut = ex.submit(
                        _ddg_text_once,
                        q,
                        region=region,
                        timelimit=timelimit,
                        per_query=per_query,
                    )
                    futs[fut] = idx
                    time.sleep(random.uniform(0.02, 0.12))
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        batch = fut.result()
                    except Exception:
                        batch = []
                    indexed_batches.append((idx, batch))
        except Exception:
            indexed_batches = []

        rows = merge_in_order(indexed_batches)
        if rows:
            return rows
        if not DDG_SEQUENTIAL_FALLBACK:
            return []
        # Fall through: sequential retry if parallel returned nothing (e.g. rate limits).

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for _, q in cleaned:
        batch = _ddg_text_once(
            q, region=region, timelimit=timelimit, per_query=per_query
        )
        for r in batch:
            href = (r.get("href") or "").strip()
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            key = href or (title + body)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({"title": title, "body": body, "href": href})
            if len(rows) >= max_total:
                return rows
        time.sleep(pause_s)
    return rows


def _ddg_batch_wait_cap(wall_deadline_abs: float | None) -> float:
    """Seconds to allow for this batch: min(per-batch cap, time left until total budget)."""
    cap = DDG_BATCH_TIMEOUT_S
    if wall_deadline_abs is not None:
        cap = min(cap, max(0.0, wall_deadline_abs - time.monotonic()))
    return cap


async def _run_ddg_queries_timed(
    queries: list[str],
    *,
    max_total: int,
    per_query: int,
    pause_s: float,
    region: str,
    timelimit: str | None,
    wall_deadline_abs: float | None = None,
) -> list[dict[str, str]]:
    """Run DDG in a thread with a wall-clock cap (avoids multi-minute hangs)."""
    if not queries:
        return []
    wait_cap = _ddg_batch_wait_cap(wall_deadline_abs)
    if wait_cap <= 0:
        return []
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _run_ddg_queries,
                queries,
                max_total=max_total,
                per_query=per_query,
                pause_s=pause_s,
                region=region,
                timelimit=timelimit,
            ),
            timeout=wait_cap,
        )
    except asyncio.TimeoutError:
        return []


def _fallback_queries(seed: str, user_message: str) -> list[str]:
    year, month = _utc_year_month()
    lm = user_message.lower()
    s = (seed or user_message).strip()[:280]
    fb: list[str] = []
    au_trade = bool(
        re.search(r"\b(australia|australian)\b", lm)
        and re.search(r"\b(export|import|trade|coal|statistic|tonne)\b", lm)
    )
    if au_trade:
        fb.extend(
            [
                f"{s} {year} Australia",
                f"Australia coal export tonnes OR value {year}",
                f"Australia coal exports million tonnes financial year site:industry.gov.au",
                f"Australia international trade goods {year} site:abs.gov.au",
            ]
        )
    fb.append(
        f"{s} {year} {month} breaking site:nbcnews.com OR site:theguardian.com OR site:reuters.com"
    )
    fb.append(
        f"{s} {year} site:apnews.com OR site:bbc.com OR site:axios.com OR site:7news.com.au"
    )
    fb.append(
        f"{s} OR official statement {year} site:whitehouse.gov OR site:state.gov OR "
        f"site:justice.gov OR site:congress.gov"
    )
    if "trump" in lm:
        fb.append(
            f"Donald Trump Truth Social coverage {year} nbcnews OR theguardian OR reuters OR apnews"
        )
    if "truth" in lm and "social" in lm:
        fb.append(
            f"Truth Social Trump {year} news site:nbcnews.com OR site:theguardian.com OR site:reuters.com"
        )
        fb.append(
            f"Trump Truth Social {year} site:news.com.au OR site:9news.com.au OR site:cnn.com OR "
            f"site:bbc.com OR site:truthsocial.com"
        )
    fb.append(f"{s} {year} discussion site:reddit.com")
    return fb[:8]


def _discard_fetch_target(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if not host:
            return True
        if host in (
            "duckduckgo.com",
            "google.com",
            "www.google.com",
            "bing.com",
            "www.bing.com",
        ):
            return True
        if "/search?" in url or "duckduckgo.com/l/?" in url:
            return True
        return False
    except Exception:
        return True


def _fetch_url_bucket(host: str) -> int:
    h = (host or "").lower()
    if (
        h.endswith(".gov")
        or h.endswith(".gov.uk")
        or h.endswith(".gov.au")
        or ".gov." in h
    ):
        return 0
    if any(
        x in h
        for x in (
            "theguardian.com",
            "nbcnews.com",
            "reuters.com",
            "apnews.com",
            "7news.com.au",
            "9news.com.au",
            "9news.com",
            "news.com.au",
            "bbc.com",
            "bbc.co.uk",
            "axios.com",
            "politico.com",
            "cnn.com",
            "nytimes.com",
            "washingtonpost.com",
            "foxnews.com",
            "news.yahoo.com",
        )
    ):
        return 0
    if "truthsocial.com" in h:
        return 1
    if "reddit.com" in h:
        return 2
    if any(
        x in h
        for x in (
            "forum",
            "discussion",
            "boards.",
            "stackoverflow",
            "stackexchange",
        )
    ):
        return 3
    if any(x in h for x in ("twitter.com", "x.com")):
        return 6
    return 4


def _order_fetch_urls(urls: list[str]) -> list[str]:
    indexed = list(enumerate(urls))

    def sort_key(item: tuple[int, str]) -> tuple[int, int]:
        i, u = item
        try:
            h = urlparse(u).hostname or ""
            return (_fetch_url_bucket(h), i)
        except Exception:
            return (99, i)

    indexed.sort(key=sort_key)
    return [u for _, u in indexed]


def _pick_urls_from_rows(rows: list[dict[str, str]], limit: int = 18) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        h = (r.get("href") or "").strip()
        if (
            not h
            or h in seen
            or not _url_allowed(h)
            or _discard_fetch_target(h)
        ):
            continue
        seen.add(h)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _is_bad_extraction(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 100:
        return True
    if s.startswith("(URL blocked"):
        return True
    if s.startswith("(Fetch failed"):
        return True
    if s.startswith("(No readable"):
        return True
    if s.startswith("(Page too large"):
        return True
    return False


def _expand_search_queries(user_message: str, seed: str) -> list[str]:
    um = (user_message or "").strip()
    lm = um.lower()
    seed = (seed or "").strip()[:400]
    out: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()[:400]
        if len(q) < 2 or q in seen:
            return
        seen.add(q)
        out.append(q)

    add(seed)

    social_hint = bool(
        re.search(
            r"\b(twitter|tweet(?:s)?|tweets|x\.com|posted on x|\bx post\b|mastodon)\b",
            lm,
        )
    ) or bool(re.search(r"@\w{2,30}\b", um))
    if social_hint or "latest post" in lm:
        add(f"{seed} site:x.com")
    if "elon" in lm and "musk" in lm:
        add("Elon Musk latest post site:x.com")
    if m := re.search(r"@(\w{2,30})\b", um):
        add(f"@{m.group(1)} latest site:x.com")

    if re.search(r"\btruth\s*social\b|truthsocial", lm, re.I):
        add(f"{seed} Truth Social latest")
        add("site:truthsocial.com latest post OR news")
        if "trump" in lm:
            add("Donald Trump Truth Social latest post news")
        add("Trump Truth Social reddit OR forum discussion")

    if _suggests_live_or_news_lookup(um) or "trump" in lm:
        add(f"{seed} reddit OR forum OR discussion")

    return out[:7]


def _suggests_live_or_news_lookup(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 6:
        return False
    return bool(
        re.search(
            r"\b(latest|most recent|newest|recent post|right now|currently|as of|today|this morning|"
            r"this afternoon|this week|just (posted|said)|breaking|who (posted|said)|what did .{0,48} "
            r"(post|say|tweet)|truth social|truthsocial|twitter|tweet\b|\bx post\b|reddit|forum|"
            r"people saying|being discussed|how much|how many|exported|imported|statistics|statistic\b|"
            r"tonne|tonnes)\b",
            t,
        )
    )


def _grep_relevant(extracted: str, question: str, max_chars: int = 10000) -> str:
    text = re.sub(r"\s+", " ", (extracted or "").strip())
    if not text:
        return ""
    keys = [
        w
        for w in re.findall(r"[A-Za-z]{4,}", (question or "").lower())
        if w not in _STOP_GREP
    ]
    keys = list(dict.fromkeys(keys))[:16]
    if not keys or len(text) < 400:
        return text[:max_chars]
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for k in keys:
        pos = 0
        while True:
            i = lowered.find(k, pos)
            if i == -1:
                break
            spans.append((max(0, i - 180), min(len(text), i + len(k) + 220)))
            pos = i + max(1, len(k))
    if not spans:
        return text[:max_chars]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    chunks = [text[a:b] for a, b in merged]
    joined = "\n…\n".join(chunks)
    return joined[:max_chars]


def _grep_relevant_or_lead(
    extracted: str, question: str, *, max_chars: int = 14000
) -> str:
    """Prefer keyword-focused spans; if too little survives (common on statistico pages), keep document lead."""
    g = _grep_relevant(extracted, question, max_chars=max_chars)
    if len(g) >= 500:
        return g
    text = re.sub(r"\s+", " ", (extracted or "").strip())
    return text[:max_chars]


def _extract_readable_html(html: str) -> str:
    import trafilatura

    t = trafilatura.extract(html) or ""
    return re.sub(r"\s+", " ", t).strip()


def _fetch_url_text_sync(url: str) -> str:
    if not _url_allowed(url):
        return (
            f"(URL blocked for safety (local/private hosts are not fetched): {url})"
        )
    try:
        with httpx.Client(
            timeout=22.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-AU,en;q=0.9",
            },
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            if len(r.content) > 2_000_000:
                return "(Page too large to fetch.)"
            html = r.text
    except Exception as e:
        return f"(Fetch failed: {e})"
    text = _extract_readable_html(html)
    if not text:
        return (
            "(No readable article text extracted; the page may require login or block automated access.)"
        )
    return text


async def _fetch_url_text(url: str) -> str:
    return await asyncio.to_thread(_fetch_url_text_sync, url)


async def _gather_au_trade_direct_sections(user_message: str) -> list[str]:
    """Fetch ABS/DFAT/industry landing pages in parallel (primary text when DDG returns no snippets)."""
    um = user_message.strip()
    timeout = DIRECT_FETCH_TIMEOUT_S

    async def one(url: str) -> str | None:
        try:
            raw = await asyncio.wait_for(_fetch_url_text(url), timeout=timeout)
            if _is_bad_extraction(raw):
                return None
            fl = _grep_relevant_or_lead(raw, um)
            if _is_bad_extraction(fl):
                return None
            return f"## Official page ({url})\n{fl}"
        except Exception:
            return None

    return [
        p
        for p in await asyncio.gather(*[one(u) for u in _au_trade_seed_urls(um)])
        if p
    ]


def _site_label(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").strip()
        return h or url[:80]
    except Exception:
        return url[:80]


async def gather_web_events(
    last_user: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yields progress + partial_context SSE payloads; ends with gather_complete."""
    um = last_user.strip()
    seed = um[:400]

    yield {
        "type": "progress",
        "category": "step",
        "text": "Step 1: Preparing web search…",
    }

    if REFINE_SEARCH_QUERY and not _au_trade_or_stats_intent(um):
        try:
            if OPENAI_KEY:
                seed = await _openai_refine_search_query(um)
            else:
                seed = await _ollama_refine_search_query(um)
            yield {
                "type": "progress",
                "category": "step",
                "text": f"Step 1 (refine): Using condensed search terms — {seed[:140]}{'…' if len(seed) > 140 else ''}",
            }
        except (httpx.HTTPError, httpx.RequestError):
            seed = um[:400]
    elif REFINE_SEARCH_QUERY and _au_trade_or_stats_intent(um):
        yield {
            "type": "progress",
            "category": "step",
            "text": "Step 1: Skipping query refine (Australia trade/stats — using direct ABS/DFAT-oriented searches).",
        }

    qbroad = (
        _broad_au_trade_leading_queries(um, seed)
        if _au_trade_or_stats_intent(um)
        else []
    )
    queries = _dedupe_queries_in_order(
        qbroad
        + _contextual_site_queries(um, seed)
        + _priority_this_year_queries(seed, um)
        + _expand_search_queries(um, seed)
        + _authority_recency_queries(um, seed)
    )[:DDG_MAX_QUERIES]
    nq = len(queries)
    ddg_region, ddg_time = _infer_ddg_region_timelimit(um)
    mode = (
        f"parallel (~{min(DDG_MAX_WORKERS, min(DDG_CHUNK_SIZE, max(1, nq)))} workers)"
        if DDG_PARALLEL and nq > 1
        else "sequential"
    )
    tl = f", timelimit={ddg_time}" if ddg_time else ""
    yield {
        "type": "progress",
        "category": "step",
        "text": (
            f"Step 2: DuckDuckGo + sources — up to {nq} search variant(s) in chunks of {DDG_CHUNK_SIZE}, "
            f"region={ddg_region}{tl}, {mode}. (ABS/DFAT pages load in parallel for AU trade questions.)"
        ),
    }

    rows: list[dict[str, str]] = []
    direct_sections: list[str] = []
    snippet_goal = WEB_SNIPPET_GOAL
    au_trade = _au_trade_or_stats_intent(um)
    effective_nq = nq

    _ddg_budget_end: list[float | None] = [None]

    def _ddg_wall_deadline_abs() -> float | None:
        if DDG_TOTAL_BUDGET_S <= 0:
            return None
        if _ddg_budget_end[0] is None:
            _ddg_budget_end[0] = time.monotonic() + DDG_TOTAL_BUDGET_S
        return _ddg_budget_end[0]

    def _ddg_budget_exhausted() -> bool:
        if DDG_TOTAL_BUDGET_S <= 0:
            return False
        end = _ddg_budget_end[0]
        if end is None:
            return False
        return time.monotonic() >= end

    if au_trade and queries:
        # Official pages first — yields SSE before slow DuckDuckGo (avoids long "stuck" on Step 2a).
        yield {
            "type": "progress",
            "category": "step",
            "text": (
                f"Step 2a (1/2): Loading up to {AU_DIRECT_FETCH_MAX} official page(s) "
                f"(timeout {DIRECT_FETCH_TIMEOUT_S:.0f}s each)…"
            ),
        }
        direct_sections = await _gather_au_trade_direct_sections(um)
        dchars = sum(len(s) for s in direct_sections)
        skip_ddg0 = (
            SKIP_FIRST_DDG_IF_DIRECT_CHARS > 0
            and dchars >= SKIP_FIRST_DDG_IF_DIRECT_CHARS
        )
        if skip_ddg0:
            yield {
                "type": "progress",
                "category": "step",
                "text": (
                    f"Step 2a (1/2): {len(direct_sections)} government page(s) ready "
                    f"({dchars} chars). Skipping first DuckDuckGo batch "
                    f"(≥{SKIP_FIRST_DDG_IF_DIRECT_CHARS} chars of direct text — faster, still accurate)."
                ),
            }
            b0: list[dict[str, str]] = []
        else:
            yield {
                "type": "progress",
                "category": "step",
                "text": (
                    f"Step 2a (1/2): {len(direct_sections)} government page(s) ready. "
                    f"(2/2): First DuckDuckGo batch ({DDG_CHUNK_SIZE} queries, max "
                    f"{DDG_BATCH_TIMEOUT_S:.0f}s)…"
                ),
            }
            await asyncio.sleep(0)
            c0 = queries[0 : DDG_CHUNK_SIZE]
            b0 = await _run_ddg_queries_timed(
                c0,
                max_total=snippet_goal,
                per_query=4,
                pause_s=DDG_PAUSE_S,
                region=ddg_region,
                timelimit=ddg_time,
                wall_deadline_abs=_ddg_wall_deadline_abs(),
            )
            if not b0 and c0:
                yield {
                    "type": "progress",
                    "category": "step",
                    "text": (
                        f"Step 2a: First DuckDuckGo batch stopped after {DDG_BATCH_TIMEOUT_S:.0f}s "
                        f"(timeout or empty) — continuing with direct text and later steps."
                    ),
                }
        rows = _merge_ddg_rows(rows, b0)
        if dchars > 1200:
            effective_nq = min(nq, DDG_CHUNK_SIZE + DDG_EXTRA_QUERY_SLOTS_AFTER_DIRECT)
        yield {
            "type": "progress",
            "category": "step",
            "text": (
                f"Step 2a: {len(direct_sections)} government page(s) in context; "
                f"DuckDuckGo batch 1 — {len(b0)} snippet(s)"
                + (" (batch skipped)" if skip_ddg0 else "")
                + "."
                + (
                    " Skipping extra search passes (enough primary text)."
                    if effective_nq < nq
                    else ""
                )
            ),
        }
        await asyncio.sleep(0)
        start_from = DDG_CHUNK_SIZE
    else:
        start_from = 0

    for start in range(start_from, effective_nq, DDG_CHUNK_SIZE):
        if _ddg_budget_exhausted():
            yield {
                "type": "progress",
                "category": "step",
                "text": (
                    "Step 2: DuckDuckGo total time budget reached "
                    f"({DDG_TOTAL_BUDGET_S:.0f}s) — stopping further search batches."
                ),
            }
            break
        chunk = queries[start : start + DDG_CHUNK_SIZE]
        remaining = snippet_goal - len(rows)
        if remaining <= 0:
            break
        batch = await _run_ddg_queries_timed(
            chunk,
            max_total=remaining,
            per_query=4,
            pause_s=DDG_PAUSE_S,
            region=ddg_region,
            timelimit=ddg_time,
            wall_deadline_abs=_ddg_wall_deadline_abs(),
        )
        rows = _merge_ddg_rows(rows, batch)
        done_q = min(start + len(chunk), effective_nq)
        yield {
            "type": "progress",
            "category": "step",
            "text": (
                f"Step 2: DuckDuckGo {done_q}/{effective_nq} queries — {len(rows)} snippet(s) collected "
                f"(stops early at {DDG_EARLY_EXIT_SNIPPETS} if enough material; "
                f"max {DDG_BATCH_TIMEOUT_S:.0f}s per batch)."
            ),
        }
        await asyncio.sleep(0)
        if len(rows) >= DDG_EARLY_EXIT_SNIPPETS:
            break

    yield {
        "type": "progress",
        "category": "step",
        "text": f"Step 3: Collected {len(rows)} search snippets (deduped, ranked by recency).",
    }

    direct_chars = sum(len(s) for s in direct_sections)
    need_fallback = len(rows) < 10 and not (
        direct_sections and direct_chars > 1500
    )
    if need_fallback and not _ddg_budget_exhausted():
        fb = _fallback_queries(seed, um)[: max(6, DDG_MAX_QUERIES // 2)]
        yield {
            "type": "progress",
            "category": "step",
            "text": f"Step 3b: Running extra fallback searches ({len(fb)} backup query groups)…",
        }
        fb_rows: list[dict[str, str]] = []
        for start in range(0, len(fb), DDG_CHUNK_SIZE):
            if _ddg_budget_exhausted():
                yield {
                    "type": "progress",
                    "category": "step",
                    "text": (
                        "Step 3b: DuckDuckGo total time budget reached "
                        f"({DDG_TOTAL_BUDGET_S:.0f}s) — skipping remaining fallback batches."
                    ),
                }
                break
            chunk = fb[start : start + DDG_CHUNK_SIZE]
            remaining = 16 - len(fb_rows)
            if remaining <= 0:
                break
            batch = await _run_ddg_queries_timed(
                chunk,
                max_total=remaining,
                per_query=3,
                pause_s=max(DDG_PAUSE_S, 0.35),
                region=ddg_region,
                timelimit=ddg_time,
                wall_deadline_abs=_ddg_wall_deadline_abs(),
            )
            fb_rows = _merge_ddg_rows(fb_rows, batch)
            yield {
                "type": "progress",
                "category": "step",
                "text": f"Step 3b: Fallback batch — {len(fb_rows)} snippet(s)…",
            }
            await asyncio.sleep(0)
        rows = _merge_ddg_rows(rows, fb_rows)
        yield {
            "type": "progress",
            "category": "step",
            "text": f"Step 3c: After fallbacks — {len(rows)} snippet rows total.",
        }

    rows = _sort_rows_by_recency(rows)
    ddg_block = _format_ddg_rows(rows)
    parts: list[str] = []
    if direct_sections:
        parts.append(
            "## Direct government sources (ABS / DFAT — fetched for reliability)\n"
            + "\n\n".join(direct_sections)
        )
    parts.append(
        "## Web search (DuckDuckGo — recency-biased + major outlets / .gov)\n" + ddg_block
    )
    partial_blob = "\n\n".join(parts)
    yield {"type": "partial_context", "web_blob": partial_blob}

    user_urls = [u for u in _extract_urls(um) if _url_allowed(u)][:3]
    from_rows = _pick_urls_from_rows(rows, limit=14)
    ordered = _order_fetch_urls(
        [u for u in user_urls + from_rows if not _discard_fetch_target(u)]
    )

    meaningful: list[tuple[str, str]] = []
    total_chars = 0
    max_urls = 8
    max_sections = 4
    char_cap = 4000

    fetch_list = ordered[:max_urls]
    n_fetch = len(fetch_list)
    if n_fetch == 0:
        yield {
            "type": "progress",
            "category": "step",
            "text": (
                "Step 4: No extra article URLs to scrape — using direct ABS/DFAT text and/or search snippets."
                if direct_sections
                else "Step 4: No article URLs to scrape (using search snippets only)."
            ),
        }
    else:
        yield {
            "type": "progress",
            "category": "step",
            "text": f"Step 4: Scraping up to {n_fetch} article pages (hostname + full URL shown per site)…",
        }

    for idx, url in enumerate(fetch_list, 1):
        host = _site_label(url)
        yield {
            "type": "progress",
            "category": "fetch",
            "text": f"Step 4 ({idx}/{n_fetch}): Scraping **{host}** — {url}",
        }
        try:
            raw = await asyncio.wait_for(_fetch_url_text(url), timeout=18.0)
        except TimeoutError:
            yield {
                "type": "progress",
                "category": "fetch",
                "text": f"Step 4 ({idx}/{n_fetch}): **{host}** — timed out (skipped).",
            }
            continue
        if _is_bad_extraction(raw):
            yield {
                "type": "progress",
                "category": "fetch",
                "text": f"Step 4 ({idx}/{n_fetch}): **{host}** — no usable text (skipped).",
            }
            continue
        filtered = _grep_relevant(raw, um)
        if _is_bad_extraction(filtered):
            yield {
                "type": "progress",
                "category": "fetch",
                "text": f"Step 4 ({idx}/{n_fetch}): **{host}** — extract too short after filter (skipped).",
            }
            continue
        meaningful.append((url, filtered))
        total_chars += len(filtered)
        parts.append(f"## Extracted page text ({url})\n{filtered}")
        partial_blob = "\n\n".join(parts)
        yield {"type": "partial_context", "web_blob": partial_blob}
        yield {
            "type": "progress",
            "category": "step",
            "text": (
                f"Step 4 ({idx}/{n_fetch}): **{host}** — extracted {len(filtered)} characters "
                f"(added to context)."
            ),
        }
        if total_chars >= char_cap or len(meaningful) >= max_sections:
            break

    final = "\n\n".join(parts)
    source_urls = _collect_source_urls(direct_sections, rows, meaningful)
    yield {
        "type": "gather_complete",
        "web_blob": final,
        "source_urls": source_urls,
    }


async def assemble_web_context(last_user: str) -> str:
    final = ""
    async for ev in gather_web_events(last_user):
        if ev.get("type") == "gather_complete":
            final = ev.get("web_blob") or ""
    return final


def _web_grounded_from_messages(messages: list[dict[str, str]]) -> bool:
    if not messages or messages[0].get("role") != "system":
        return False
    s = messages[0].get("content") or ""
    return "Retrieved web material" in s or "Direct government sources" in s


async def _openai_chat(messages: list[dict[str, str]]) -> str:
    payload: dict[str, Any] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
    }
    if _web_grounded_from_messages(messages):
        payload["temperature"] = 0.35
    to = httpx.Timeout(OPENAI_HTTP_TIMEOUT, connect=30.0)
    async with httpx.AsyncClient(timeout=to) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


def _clean_refined_query(text: str) -> str:
    t = text.strip().strip('"').strip("'")
    t = re.sub(r"\s+", " ", t)
    return t[:400]


async def _ollama_refine_search_query(user_message: str) -> str:
    y = datetime.now(timezone.utc).year
    prompt = (
        f"Write ONE short web search query (max 14 words) for the user message. "
        f"If they want latest or current news, include **{y}** or 'latest {y}' in the query. "
        f"Output only the query, no quotes or explanation.\n\nUser message:\n"
        + user_message.strip()[:2000]
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 64, "temperature": 0.3},
            },
        )
        r.raise_for_status()
        data = r.json()
        out = (data.get("response") or "").strip()
    return _clean_refined_query(out) or user_message.strip()[:400]


async def _openai_refine_search_query(user_message: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Reply with only a short web search query (max 14 words), no quotes. "
                            f"If the user wants recent or latest news, include the year {datetime.now(timezone.utc).year}."
                        ),
                    },
                    {"role": "user", "content": user_message.strip()[:2000]},
                ],
                "max_tokens": 40,
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        data = r.json()
        out = data["choices"][0]["message"]["content"].strip()
    return _clean_refined_query(out) or user_message.strip()[:400]


def _ollama_temperature(messages: list[dict[str, str]]) -> float:
    """Lower temperature when the system prompt includes retrieved web pages (fewer invented numbers)."""
    return 0.36 if _web_grounded_from_messages(messages) else 0.72


def _ollama_httpx_timeout() -> httpx.Timeout:
    return httpx.Timeout(OLLAMA_HTTP_TIMEOUT, connect=min(60.0, OLLAMA_HTTP_TIMEOUT))


async def _ollama_chat(messages: list[dict[str, str]]) -> str:
    temp = _ollama_temperature(messages)
    async with httpx.AsyncClient(timeout=_ollama_httpx_timeout()) as client:
        r = await client.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "repeat_penalty": 1.18,
                    "temperature": temp,
                },
            },
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()


def _build_messages_with_web(
    req: ChatRequest, web_blob: str | None
) -> list[dict[str, str]]:
    conv = [{"role": m.role, "content": m.content} for m in req.messages]
    system_text = SYSTEM_BASE + _calendar_context_block()
    if web_blob:
        system_text += (
            SYSTEM_WEB_OUTPUT_RULES
            + SYSTEM_WEB_GROUNDING
            + "\n\n--- Retrieved web material (may be incomplete or from news mirrors) ---\n"
            + web_blob
        )
    return [{"role": "system", "content": system_text}, *conv]


def _should_use_web(req: ChatRequest, last_user: str) -> bool:
    if not last_user.strip():
        return False
    if req.use_web:
        return True
    return AUTO_SEARCH_RECENCY and _suggests_live_or_news_lookup(last_user)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.perf_counter()
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    web_blob: str | None = None
    if _should_use_web(req, last_user):
        web_blob = await assemble_web_context(last_user)

    msgs = _build_messages_with_web(req, web_blob)
    su = _source_urls_from_context_blob(web_blob) if web_blob else []

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            elapsed = int((time.perf_counter() - t0) * 1000)
            return ChatResponse(
                reply=reply,
                source="openai",
                web_used=bool(web_blob),
                source_urls=su,
                elapsed_ms=elapsed,
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI error: {e.response.text[:500]}",
            ) from e
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        reply = await _ollama_chat(msgs)
        if not reply:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty reply. Is the model pulled? Try: ollama pull "
                + OLLAMA_MODEL,
            )
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ChatResponse(
            reply=reply,
            source="ollama",
            web_used=bool(web_blob),
            source_urls=su,
            elapsed_ms=elapsed,
        )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach Ollama. Install from https://ollama.com, run `ollama serve`, "
                f"then `ollama pull {OLLAMA_MODEL}`. Or set OPENAI_API_KEY in a .env file."
            ),
        ) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error: {e.response.text[:500]}",
        ) from e


@app.post("/api/chat/reply-partial", response_model=ChatResponse)
async def reply_partial(req: ReplyPartialRequest):
    """Complete the reply using the client's web_blob (may be partial from an aborted stream)."""
    t0 = time.perf_counter()
    wb = (req.web_blob or "").strip() or None
    msgs = _build_messages_with_web(
        ChatRequest(messages=req.messages, use_web=False), wb
    )
    su = _source_urls_from_context_blob(wb) if wb else []

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            elapsed = int((time.perf_counter() - t0) * 1000)
            return ChatResponse(
                reply=reply,
                source="openai",
                web_used=bool(wb),
                source_urls=su,
                elapsed_ms=elapsed,
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI error: {e.response.text[:500]}",
            ) from e
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        reply = await _ollama_chat(msgs)
        if not reply:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty reply. Is the model pulled? Try: ollama pull "
                + OLLAMA_MODEL,
            )
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ChatResponse(
            reply=reply,
            source="ollama",
            web_used=bool(wb),
            source_urls=su,
            elapsed_ms=elapsed,
        )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach Ollama. Install from https://ollama.com, run `ollama serve`, "
                f"then `ollama pull {OLLAMA_MODEL}`. Or set OPENAI_API_KEY in a .env file."
            ),
        ) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error: {e.response.text[:500]}",
        ) from e


def _sse_data(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _sse_stream_with_keepalive(
    source: AsyncIterator[str],
    *,
    interval_s: float = SSE_KEEPALIVE_INTERVAL_S,
) -> AsyncIterator[str]:
    """Emit periodic SSE comment lines so idle connections are not closed mid-request."""
    queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue(maxsize=512)

    async def pump() -> None:
        try:
            async for chunk in source:
                await queue.put(("chunk", chunk))
            await queue.put(("end", None))
        except Exception as e:
            await queue.put(("err", str(e)))

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval_s)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if kind == "end":
                break
            if kind == "err":
                yield _sse_data(
                    {
                        "type": "error",
                        "detail": payload or "Stream failed",
                    }
                )
                break
            yield payload
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _chat_event_stream(req: ChatRequest) -> AsyncIterator[str]:
    t_stream = time.perf_counter()
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    web_blob: str | None = None
    source_urls: list[str] = []
    if _should_use_web(req, last_user):
        yield _sse_data({"type": "status", "phase": "searching"})
        try:
            async for ev in gather_web_events(last_user):
                yield _sse_data(ev)
                if ev.get("type") == "gather_complete":
                    web_blob = ev.get("web_blob")
                    source_urls = list(ev.get("source_urls") or [])
        except Exception as ex:
            yield _sse_data(
                {
                    "type": "progress",
                    "category": "error",
                    "text": f"Gather step failed ({repr(ex)[:200]}) — continuing.",
                }
            )

    yield _sse_data({"type": "status", "phase": "thinking"})
    yield _sse_data(
        {
            "type": "progress",
            "category": "think",
            "text": "Step 5: Calling the local language model with gathered context…",
        }
    )

    msgs = _build_messages_with_web(req, web_blob)

    def _done_sse(
        reply: str, source: str, *, extra_urls: list[str] | None = None
    ) -> dict[str, Any]:
        urls = _dedupe_http_urls(list(source_urls) + (extra_urls or []))
        return {
            "type": "done",
            "reply": reply,
            "source": source,
            "web_used": bool(web_blob),
            "elapsed_ms": int((time.perf_counter() - t_stream) * 1000),
            "source_urls": urls,
        }

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            yield _sse_data(_done_sse(reply, "openai"))
        except httpx.HTTPStatusError as e:
            yield _sse_data(
                {
                    "type": "error",
                    "detail": f"OpenAI error: {e.response.text[:500]}",
                }
            )
        except httpx.TimeoutException as e:
            yield _sse_data(
                {
                    "type": "error",
                    "detail": f"OpenAI request timed out ({e}). Try OPENAI_HTTP_TIMEOUT in .env.",
                }
            )
        except httpx.RequestError as e:
            yield _sse_data({"type": "error", "detail": str(e)})
        return

    try:
        reply = await _ollama_chat(msgs)
        if not reply:
            yield _sse_data(
                {
                    "type": "error",
                    "detail": "Ollama returned an empty reply. Is the model pulled? Try: ollama pull "
                    + OLLAMA_MODEL,
                }
            )
            return
        yield _sse_data(_done_sse(reply, "ollama"))
    except httpx.ConnectError:
        yield _sse_data(
            {
                "type": "error",
                "detail": (
                    "Cannot reach Ollama. Install from https://ollama.com, run `ollama serve`, "
                    f"then `ollama pull {OLLAMA_MODEL}`. Or set OPENAI_API_KEY in a .env file."
                ),
            }
        )
    except httpx.HTTPStatusError as e:
        yield _sse_data(
            {"type": "error", "detail": f"Ollama error: {e.response.text[:500]}"}
        )
    except httpx.TimeoutException as e:
        yield _sse_data(
            {
                "type": "error",
                "detail": (
                    f"Ollama HTTP timed out after {OLLAMA_HTTP_TIMEOUT:.0f}s ({e}). "
                    "Try a faster/smaller model, reduce context, or set OLLAMA_HTTP_TIMEOUT in .env."
                ),
            }
        )
    except httpx.RequestError as e:
        yield _sse_data({"type": "error", "detail": f"Ollama request failed: {e}"})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _sse_stream_with_keepalive(_chat_event_stream(req)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
