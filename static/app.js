const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const useWeb = document.getElementById("useWeb");

/** @type {{ role: string, content: string }[]} */
const history = [];

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Build resolver so only hosts present in sourceUrls get linked (official / gathered pages).
 * @param {string[]} sourceUrls
 */
function buildSourceResolver(sourceUrls) {
  const urls = Array.isArray(sourceUrls) ? sourceUrls.filter(Boolean) : [];

  function norm(u) {
    try {
      const x = new URL(String(u).trim().replace(/&amp;/g, "&"));
      if (x.protocol !== "http:" && x.protocol !== "https:") return null;
      x.hash = "";
      return x.href;
    } catch {
      return null;
    }
  }

  /** @type {Map<string, string>} */
  const byHost = new Map();
  for (const u of urls) {
    const n = norm(u);
    if (!n) continue;
    try {
      const h = new URL(n).hostname.replace(/^www\./i, "").toLowerCase();
      if (!byHost.has(h)) byHost.set(h, n);
    } catch {
      /* skip */
    }
  }

  /**
   * @param {string} hrefRaw
   * @returns {string | null}
   */
  function resolve(hrefRaw) {
    if (!urls.length) return null;
    const n = norm(hrefRaw.replace(/&amp;/g, "&"));
    if (!n) return null;
    let host;
    try {
      host = new URL(n).hostname.replace(/^www\./i, "").toLowerCase();
    } catch {
      return null;
    }
    if (!byHost.has(host)) {
      for (const u of urls) {
        const nu = norm(u);
        if (nu && (n === nu || n.startsWith(nu.replace(/\/$/, "") + "/"))) return n;
      }
      return null;
    }
    let best = byHost.get(host);
    let bestLen = best ? best.length : 0;
    for (const u of urls) {
      const nu = norm(u);
      if (!nu) continue;
      let uh;
      try {
        uh = new URL(nu).hostname.replace(/^www\./i, "").toLowerCase();
      } catch {
        continue;
      }
      if (uh !== host) continue;
      if (n === nu || n.startsWith(nu.replace(/\/$/, "") + "/")) {
        if (nu.length > bestLen) {
          best = nu;
          bestLen = nu.length;
        }
      }
    }
    return best || null;
  }

  return { resolve, byHost };
}

/**
 * @param {string} str
 * @returns {{ out: string, chunks: string[] }}
 */
function maskAnchorTags(str) {
  const chunks = [];
  const out = str.replace(/<a\b[^>]*>[\s\S]*?<\/a>/gi, (block) => {
    chunks.push(block);
    return `<<<A${chunks.length - 1}>>>`;
  });
  return { out, chunks };
}

function unmaskAnchorTags(str, chunks) {
  return str.replace(/<<<A(\d+)>>>/g, (_, i) => chunks[Number(i)] || "");
}

/**
 * Turn plain-text reply into safe HTML: Markdown links, bare URLs, domain mentions, paragraphs.
 * @param {string} raw
 * @param {string[]} sourceUrls
 */
function renderAssistantHtml(raw, sourceUrls) {
  const { resolve, byHost } = buildSourceResolver(sourceUrls);
  let s = escapeHtml(raw);

  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, (full, label, hrefEsc) => {
    const target = resolve(hrefEsc.replace(/&amp;/g, "&"));
    if (!target) return full;
    return `<a href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  s = s.replace(/\b(https?:\/\/[^\s<]+)/g, (full) => {
    const trimmed = full.replace(/[.,;:!?)]+$/, "");
    const suffix = full.slice(trimmed.length);
    const target = resolve(trimmed);
    if (!target) return full;
    return `<a href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer">${escapeHtml(trimmed)}</a>${suffix}`;
  });

  let masked = maskAnchorTags(s);
  s = masked.out;

  const hosts = [...byHost.keys()].sort((a, b) => b.length - a.length);
  for (const h of hosts) {
    const canon = byHost.get(h);
    if (!canon) continue;
    const pat = h.replace(/\./g, "\\.");
    const re = new RegExp(`\\b(?:www\\.)?(${pat})\\b`, "gi");
    s = s.replace(re, (match, _g1, offset, whole) => {
      const before = whole.slice(Math.max(0, offset - 24), offset);
      if (/https?:\/\//i.test(before)) return match;
      return `<a href="${escapeHtml(canon)}" target="_blank" rel="noopener noreferrer">${match}</a>`;
    });
  }

  s = unmaskAnchorTags(s, masked.chunks);

  const blocks = s.split(/\n\n+/);
  return blocks
    .map((block) => {
      const inner = block.replace(/\n/g, "<br>");
      return `<p>${inner}</p>`;
    })
    .join("");
}

function addBubble(role, content, extraClass = "", meta = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role} ${extraClass}`.trim();
  const label = document.createElement("div");
  label.className = "role";
  let labelText = role === "user" ? "You" : role === "assistant" ? "Assistant" : "Notice";
  if (meta) labelText += ` · ${meta}`;
  label.textContent = labelText;
  wrap.appendChild(label);
  const text = document.createElement("div");
  text.textContent = content;
  wrap.appendChild(text);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

/**
 * @param {{
 *   text: string,
 *   elapsedMs?: number | null,
 *   sourceUrls?: string[],
 *   thinkingLines?: string[],
 * }} opts
 */
function addAssistantReply(opts) {
  const { text, elapsedMs, sourceUrls = [], thinkingLines = [] } = opts;
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const label = document.createElement("div");
  label.className = "role";
  const sec =
    elapsedMs != null && elapsedMs >= 0
      ? ` · ${(elapsedMs / 1000).toFixed(1)}s total`
      : "";
  label.textContent = `Assistant${sec}`;
  wrap.appendChild(label);

  const body = document.createElement("div");
  body.className = "assistant-body assistant-html";
  body.innerHTML = renderAssistantHtml(text, sourceUrls);
  wrap.appendChild(body);

  if (sourceUrls.length > 0) {
    const srcWrap = document.createElement("div");
    srcWrap.className = "msg-sources";
    const h = document.createElement("div");
    h.className = "msg-sources-title";
    h.textContent = "Sources (href)";
    const ul = document.createElement("ul");
    ul.className = "msg-sources-list";
    for (const u of sourceUrls) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = u;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = u;
      li.appendChild(a);
      ul.appendChild(li);
    }
    srcWrap.appendChild(h);
    srcWrap.appendChild(ul);
    wrap.appendChild(srcWrap);
  }

  if (thinkingLines.length > 0) {
    const det = document.createElement("details");
    det.className = "thinking-trace";
    const sum = document.createElement("summary");
    sum.textContent = `Thinking & activity (${thinkingLines.length} steps)`;
    const pre = document.createElement("pre");
    pre.className = "thinking-trace-pre";
    pre.textContent = thinkingLines.join("\n");
    det.appendChild(sum);
    det.appendChild(pre);
    wrap.appendChild(det);
  }

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

/**
 * @returns {{
 *   wrap: HTMLElement,
 *   setPhase: (p: "searching"|"thinking") => void,
 *   logLine: (s: string) => void,
 *   setLatestBlob: (b: string) => void,
 *   getLatestBlob: () => string,
 *   onReplyNow: (fn: () => void) => void,
 *   stopTimer: () => void,
 *   getThinkingSteps: () => string[],
 * }}
 */
function addPendingBubble() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant pending";
  wrap.setAttribute("role", "status");
  wrap.setAttribute("aria-live", "polite");

  const label = document.createElement("div");
  label.className = "role";
  label.textContent = "Assistant";

  const tStart = performance.now();
  let phaseBase = "Working…";
  const thinkingSteps = [];

  const row = document.createElement("div");
  row.className = "pending-row";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const status = document.createElement("span");
  status.className = "pending-status";
  status.textContent = "Working…";
  const timerChip = document.createElement("span");
  timerChip.className = "pending-timer";
  timerChip.textContent = "0.0s";
  row.appendChild(spinner);
  row.appendChild(status);
  row.appendChild(timerChip);

  const detail = document.createElement("div");
  detail.className = "pending-detail";
  detail.hidden = true;

  const elapsedRow = document.createElement("div");
  elapsedRow.className = "thinking-elapsed";
  elapsedRow.appendChild(document.createTextNode("Running for "));
  const elapsedSpan = document.createElement("span");
  elapsedSpan.className = "elapsed-val";
  elapsedSpan.textContent = "0.0s";
  elapsedRow.appendChild(elapsedSpan);
  const hint = document.createElement("div");
  hint.className = "thinking-hint";
  hint.textContent =
    "Step-by-step log below. Elapsed time updates while searching and generating.";

  const log = document.createElement("ul");
  log.className = "activity-log";

  const timerId = window.setInterval(() => {
    const s = (performance.now() - tStart) / 1000;
    timerChip.textContent = `${s.toFixed(1)}s`;
    status.textContent = `${phaseBase} (${s.toFixed(1)}s)`;
    elapsedSpan.textContent = `${s.toFixed(1)}s`;
    chat.scrollTop = chat.scrollHeight;
  }, 250);

  const toggleRow = document.createElement("div");
  toggleRow.className = "pending-toggle-row";
  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "pending-toggle";
  toggleBtn.setAttribute("aria-expanded", "false");
  toggleBtn.innerHTML =
    '<span class="pending-arrow" aria-hidden="true">▸</span> Activity, thinking &amp; timing';

  const replyNow = document.createElement("button");
  replyNow.type = "button";
  replyNow.className = "reply-now-btn";
  replyNow.textContent = "Reply now (use gathered info so far)";

  let latestBlob = "";
  let replyHandler = () => {};

  toggleBtn.addEventListener("click", () => {
    const open = detail.hidden;
    detail.hidden = !open;
    toggleBtn.setAttribute("aria-expanded", open ? "true" : "false");
    const ar = toggleBtn.querySelector(".pending-arrow");
    if (ar) ar.textContent = open ? "▾" : "▸";
  });

  replyNow.addEventListener("click", () => replyHandler());

  detail.appendChild(elapsedRow);
  detail.appendChild(hint);
  detail.appendChild(log);
  detail.appendChild(replyNow);

  toggleRow.appendChild(toggleBtn);
  wrap.appendChild(label);
  wrap.appendChild(row);
  wrap.appendChild(toggleRow);
  wrap.appendChild(detail);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;

  function stopTimer() {
    window.clearInterval(timerId);
  }

  return {
    wrap,
    setPhase(phase) {
      if (phase === "searching") phaseBase = "Searching the web…";
      else if (phase === "thinking") phaseBase = "Thinking…";
      const s = (performance.now() - tStart) / 1000;
      status.textContent = `${phaseBase} (${s.toFixed(1)}s)`;
      chat.scrollTop = chat.scrollHeight;
    },
    logLine(s) {
      thinkingSteps.push(s);
      const li = document.createElement("li");
      li.textContent = s;
      log.appendChild(li);
      chat.scrollTop = chat.scrollHeight;
    },
    setLatestBlob(b) {
      latestBlob = b || "";
    },
    getLatestBlob: () => latestBlob,
    onReplyNow(fn) {
      replyHandler = fn;
    },
    stopTimer,
    getThinkingSteps: () => [...thinkingSteps],
  };
}

/**
 * @param {Response} res
 * @param {{
 *   onPhase?: (phase: string) => void,
 *   onProgress?: (data: Record<string, unknown>) => void,
 *   onPartial?: (webBlob: string) => void,
 * }} handlers
 */
async function readChatSse(res, handlers) {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const block of parts) {
      const line = block.trim();
      if (!line.startsWith("data: ")) continue;
      let data;
      try {
        data = JSON.parse(line.slice(6));
      } catch {
        continue;
      }
      if (data.type === "status" && data.phase) handlers.onPhase?.(data.phase);
      if (data.type === "progress") handlers.onProgress?.(data);
      if (data.type === "partial_context" && data.web_blob != null) {
        handlers.onPartial?.(String(data.web_blob));
      }
      if (data.type === "done")
        return {
          reply: data.reply,
          web_used: data.web_used === true,
          source: data.source,
          elapsed_ms:
            typeof data.elapsed_ms === "number" ? data.elapsed_ms : null,
          source_urls: Array.isArray(data.source_urls) ? data.source_urls : [],
        };
      if (data.type === "error")
        throw new Error(data.detail || "Request failed");
    }
  }
  throw new Error("Stream ended without a reply");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  addBubble("user", text);
  history.push({ role: "user", content: text });

  const pending = addPendingBubble();
  const abortController = new AbortController();
  let earlyFinish = false;

  pending.onReplyNow(() => {
    earlyFinish = true;
    abortController.abort();
  });

  sendBtn.disabled = true;
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        messages: history,
        use_web: useWeb.checked,
      }),
      signal: abortController.signal,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      const detail = errBody.detail || res.statusText || "Request failed";
      const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
      pending.stopTimer();
      pending.wrap.remove();
      addBubble("assistant", msg, "error");
      history.pop();
      return;
    }

    const dataPromise = readChatSse(res, {
      onPhase: (phase) => {
        if (phase === "searching" || phase === "thinking") pending.setPhase(phase);
      },
      onProgress: (data) => {
        const cat = data.category || "";
        const t = data.text || "";
        if (!t) return;
        const tag =
          ["step", "fetch", "think"].includes(cat) || t.startsWith("Step ")
            ? ""
            : `[${cat}] `;
        pending.logLine(tag + t);
      },
      onPartial: (blob) => {
        pending.setLatestBlob(blob);
      },
    });

    let data;
    try {
      data = await dataPromise;
    } catch (err) {
      if (earlyFinish || err?.name === "AbortError") {
        const blob = pending.getLatestBlob();
        const thinkingLines = pending.getThinkingSteps();
        pending.stopTimer();
        pending.wrap.remove();
        const r = await fetch("/api/chat/reply-partial", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history,
            web_blob: blob,
          }),
        });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) {
          const detail = body.detail || r.statusText || "Request failed";
          const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
          addBubble("assistant", msg, "error");
          history.pop();
          return;
        }
        addAssistantReply({
          text: body.reply,
          elapsedMs: body.elapsed_ms ?? null,
          sourceUrls: body.source_urls || [],
          thinkingLines,
        });
        history.push({ role: "assistant", content: body.reply });
        return;
      }
      throw err;
    }

    pending.stopTimer();
    pending.wrap.remove();
    addAssistantReply({
      text: data.reply,
      elapsedMs: data.elapsed_ms,
      sourceUrls: data.source_urls || [],
      thinkingLines: pending.getThinkingSteps(),
    });
    history.push({ role: "assistant", content: data.reply });
  } catch (err) {
    if (earlyFinish || err?.name === "AbortError") {
      return;
    }
    pending.stopTimer();
    pending.wrap.remove();
    const raw = String(err?.message || err || "Unknown error");
    const looksNet =
      /network|failed to fetch|load failed|networkerror|aborted|reset/i.test(raw);
    const msg = looksNet
      ? `${raw}\n\nIf this happened after a long “Searching / Thinking” phase, the browser or OS may have closed the idle connection. The server now sends keep-alive pings; reload the page and try again. You can also set a shorter run (e.g. turn off “Search online” for a quick test) or increase OLLAMA_HTTP_TIMEOUT in .env.`
      : raw;
    addBubble("assistant", msg, "error");
    history.pop();
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.focus();
