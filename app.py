"""
Local chat web app. Serves the UI and proxies messages to an LLM.
Set OPENAI_API_KEY to use OpenAI; otherwise uses Ollama at http://127.0.0.1:11434.
Optional online mode uses DuckDuckGo (network), extra queries for social-style questions,
and safe HTTP fetch + text extraction for URLs in the user message.
"""
import asyncio
import ipaddress
import json
import os
import re
import time
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

SYSTEM_BASE = """You are a helpful assistant. Be direct and readable.

Do not refuse informational questions with boilerplate about being an AI, lacking real-time access,
being unable to browse, or being unable to provide up-to-date information as your whole answer.

Never make rate limits, CAPTCHAs, robots, or search-index limits the center of your reply. Do not say
you "could not find anything because of rate limiting" or similar. If material below is thin, still
synthesize the best supported story from titles, snippets, news mirrors, Reddit, or forum lines.

Do not tell the user to "check the official website" or "check reputable news sources" as the main
answer. If you mention a place to verify, it must come after you have already summarized concrete
claims from the excerpts below (who said what, what outlets reported, approximate timing).

**Banned patterns (do not use as your whole reply):** opening with "I couldn't find" then only
suggestions; bullet lists of "sources to check", "possible sources", "you can try", or "check the
Truth Social app/website"; telling the user to look at Twitter/X instead of answering; generic
homework about "reputable news" with no summary from the excerpts.

Also banned: "Unfortunately, I don't have direct access to [Truth Social / current content]";
"based on my previous knowledge" or "my knowledge up to [month year]"; "up to December 2023" or any
**pretraining cutoff year** as your timeline frame; recommending CNN/BBC/Fox as a substitute for
summarizing the excerpts. If web excerpts exist, they override your training cutoff.

For **statistics, exports, trade volumes, or census-style figures**: prioritize lines from **official
statistical agencies** (e.g. Australian Bureau of Statistics **abs.gov.au**, DFAT trade data) when they
appear in the excerpts. Do not substitute a 2021–22 fact as “current” if the user asked for a later
calendar year—say what year the figure refers to.

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
and Truth Social–related news lines. You must ground your answer in it. Prefer newer-looking items
when dates conflict. If one URL failed, other snippets and pages below remain valid."""

# Prepended only when web_blob is non-empty (before the retrieved text).
SYSTEM_WEB_OUTPUT_RULES = """
OUTPUT RULES FOR THIS TURN (web material is present):
Your FIRST sentences must summarize what the retrieved lines below actually say—topics, rough dates,
and outlet or domain names (CNN, BBC, 9News, news.com.au, Truth Social mirrors, etc.) when visible.
Do not substitute a list of places for the user to look. If snippets are vague, still paraphrase the
strongest concrete detail present; only then add a brief caveat—not the reverse order.

Do **not** illustrate "recent" with December 2023 (or other old years) unless that date explicitly
appears in the retrieved lines below. Prefer any line that looks like **this year** or last year."""


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
        out.extend(
            [
                f"Australia coal export {y_focus} site:abs.gov.au",
                f"Australian Bureau of Statistics international trade goods {y_focus} site:abs.gov.au",
                f"Australia merchandise trade exports {y_focus} site:abs.gov.au OR site:dfat.gov.au",
                f"{core} site:abs.gov.au OR site:dfat.gov.au OR site:industry.gov.au",
                f"ABS international trade in goods and services Australia {y_focus}",
            ]
        )

    if au and "population" in lm:
        out.append(f"Australia population {y_focus} site:abs.gov.au")

    if re.search(r"\b(unemployment|labour|labor statistics|cpi|inflation)\b", lm) and au:
        out.append(f"Australia {y_focus} labour force site:abs.gov.au")

    return out[:8]


def _priority_this_year_queries(seed: str, user_message: str) -> list[str]:
    """Front-load searches aimed at the current calendar year."""
    year, month = _utc_year_month()
    lm = (user_message or "").lower()
    core = (seed or user_message).strip()[:220]
    pq: list[str] = [
        f"Trump {year} site:cnn.com OR site:bbc.com OR site:nbcnews.com OR site:news.com.au",
        f"Trump Truth Social {year} {month} news site:cnn.com OR site:9news.com.au OR site:bbc.com",
        f"{core} {year} breaking site:reuters.com OR site:apnews.com OR site:theguardian.com",
    ]
    if "truth" in lm or "trump" in lm:
        pq.insert(
            0,
            f"Donald Trump statement {year} site:cnn.com OR site:nbcnews.com OR site:bbc.com",
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


class ReplyPartialRequest(BaseModel):
    """Finish with whatever web context was gathered (client may send partial blob)."""

    messages: list[ChatMessage] = Field(min_length=1)
    web_blob: str = ""


URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

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


def _run_ddg_queries(
    queries: list[str],
    *,
    max_total: int = 24,
    per_query: int = 5,
    pause_s: float = 0.5,
) -> list[dict[str, str]]:
    from duckduckgo_search import DDGS

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    if not queries:
        return rows
    try:
        with DDGS() as ddgs:
            for q in queries:
                q = (q or "").strip()[:400]
                if len(q) < 2:
                    continue
                try:
                    for r in ddgs.text(q, max_results=per_query):
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
                except Exception:
                    pass
                time.sleep(pause_s)
    except Exception:
        pass
    return rows


def _fallback_queries(seed: str, user_message: str) -> list[str]:
    year, month = _utc_year_month()
    lm = user_message.lower()
    s = (seed or user_message).strip()[:280]
    fb: list[str] = []
    fb.append(
        f"{s} {year} {month} breaking site:nbcnews.com OR site:theguardian.com OR site:reuters.com"
    )
    fb.append(
        f"{s} {year} site:apnews.com OR site:bbc.com OR site:axios.com OR site:7news.com.au"
    )
    if re.search(r"\b(australia|australian)\b", lm) and re.search(
        r"\b(export|import|trade|coal|statistic|tonne)\b", lm
    ):
        fb.insert(
            0,
            f"Australia trade exports {year} site:abs.gov.au OR site:dfat.gov.au OR ABS",
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
                "User-Agent": "Mozilla/5.0 (compatible; LocalChatBot/1.0)"
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
        "text": "Step 1: Preparing search (optional query refine)…",
    }

    if REFINE_SEARCH_QUERY:
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

    queries = _dedupe_queries_in_order(
        _contextual_site_queries(um, seed)
        + _priority_this_year_queries(seed, um)
        + _expand_search_queries(um, seed)
        + _authority_recency_queries(um, seed)
    )[:22]
    nq = len(queries)
    yield {
        "type": "progress",
        "category": "step",
        "text": (
            f"Step 2: Searching the web index (DuckDuckGo) — running {nq} query "
            f"{'group' if nq == 1 else 'groups'} in one batch (no per-query wait in the UI)."
        ),
    }

    rows = await asyncio.to_thread(
        _run_ddg_queries,
        queries,
        max_total=22,
        per_query=4,
        pause_s=0.45,
    )
    yield {
        "type": "progress",
        "category": "step",
        "text": f"Step 3: Collected {len(rows)} search snippets (deduped, ranked by recency).",
    }

    if len(rows) < 10:
        fb = _fallback_queries(seed, um)
        yield {
            "type": "progress",
            "category": "step",
            "text": f"Step 3b: Running extra fallback searches ({len(fb)} backup query groups)…",
        }
        rows = _merge_ddg_rows(
            rows,
            await asyncio.to_thread(
                _run_ddg_queries,
                fb,
                max_total=16,
                per_query=3,
                pause_s=0.5,
            ),
        )
        yield {
            "type": "progress",
            "category": "step",
            "text": f"Step 3c: After fallbacks — {len(rows)} snippet rows total.",
        }

    rows = _sort_rows_by_recency(rows)
    ddg_block = _format_ddg_rows(rows)
    parts: list[str] = [
        "## Web search (DuckDuckGo — recency-biased + major outlets / .gov)\n" + ddg_block
    ]
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
            "text": "Step 4: No article URLs to scrape (using search snippets only).",
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
    yield {"type": "gather_complete", "web_blob": final}


async def assemble_web_context(last_user: str) -> str:
    final = ""
    async for ev in gather_web_events(last_user):
        if ev.get("type") == "gather_complete":
            final = ev.get("web_blob") or ""
    return final


async def _openai_chat(messages: list[dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": messages,
            },
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


async def _ollama_chat(messages: list[dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "repeat_penalty": 1.18,
                    "temperature": 0.72,
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
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    web_blob: str | None = None
    if _should_use_web(req, last_user):
        web_blob = await assemble_web_context(last_user)

    msgs = _build_messages_with_web(req, web_blob)

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            return ChatResponse(
                reply=reply, source="openai", web_used=bool(web_blob)
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
        return ChatResponse(reply=reply, source="ollama", web_used=bool(web_blob))
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
    wb = (req.web_blob or "").strip() or None
    msgs = _build_messages_with_web(
        ChatRequest(messages=req.messages, use_web=False), wb
    )

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            return ChatResponse(reply=reply, source="openai", web_used=bool(wb))
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
        return ChatResponse(reply=reply, source="ollama", web_used=bool(wb))
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


async def _chat_event_stream(req: ChatRequest) -> AsyncIterator[str]:
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    web_blob: str | None = None
    if _should_use_web(req, last_user):
        yield _sse_data({"type": "status", "phase": "searching"})
        try:
            async for ev in gather_web_events(last_user):
                yield _sse_data(ev)
                if ev.get("type") == "gather_complete":
                    web_blob = ev.get("web_blob")
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

    if OPENAI_KEY:
        try:
            reply = await _openai_chat(msgs)
            yield _sse_data(
                {
                    "type": "done",
                    "reply": reply,
                    "source": "openai",
                    "web_used": bool(web_blob),
                }
            )
        except httpx.HTTPStatusError as e:
            yield _sse_data(
                {
                    "type": "error",
                    "detail": f"OpenAI error: {e.response.text[:500]}",
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
        yield _sse_data(
            {
                "type": "done",
                "reply": reply,
                "source": "ollama",
                "web_used": bool(web_blob),
            }
        )
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


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _chat_event_stream(req),
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
