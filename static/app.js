const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const useWeb = document.getElementById("useWeb");

/** @type {{ role: string, content: string }[]} */
const history = [];

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
 * @returns {{
 *   wrap: HTMLElement,
 *   setPhase: (p: "searching"|"thinking") => void,
 *   logLine: (s: string) => void,
 *   setLatestBlob: (b: string) => void,
 *   getLatestBlob: () => string,
 *   onReplyNow: (fn: () => void) => void,
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

  const row = document.createElement("div");
  row.className = "pending-row";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const status = document.createElement("span");
  status.className = "pending-status";
  status.textContent = "Working…";
  row.appendChild(spinner);
  row.appendChild(status);

  const toggleRow = document.createElement("div");
  toggleRow.className = "pending-toggle-row";
  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "pending-toggle";
  toggleBtn.setAttribute("aria-expanded", "false");
  toggleBtn.innerHTML =
    '<span class="pending-arrow" aria-hidden="true">▸</span> Activity &amp; sources';

  const detail = document.createElement("div");
  detail.className = "pending-detail";
  detail.hidden = true;

  const log = document.createElement("ul");
  log.className = "activity-log";

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

  detail.appendChild(log);
  detail.appendChild(replyNow);

  toggleRow.appendChild(toggleBtn);
  wrap.appendChild(label);
  wrap.appendChild(row);
  wrap.appendChild(toggleRow);
  wrap.appendChild(detail);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;

  return {
    wrap,
    setPhase(phase) {
      if (phase === "searching") status.textContent = "Searching the web…";
      else if (phase === "thinking") status.textContent = "Thinking…";
      chat.scrollTop = chat.scrollHeight;
    },
    logLine(s) {
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
        addBubble("assistant", body.reply);
        history.push({ role: "assistant", content: body.reply });
        return;
      }
      throw err;
    }

    pending.wrap.remove();
    addBubble("assistant", data.reply);
    history.push({ role: "assistant", content: data.reply });
  } catch (err) {
    if (earlyFinish || err?.name === "AbortError") {
      /* handled above when readChatSse throws after reply now */
      return;
    }
    pending.wrap.remove();
    addBubble("assistant", String(err.message || err), "error");
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
