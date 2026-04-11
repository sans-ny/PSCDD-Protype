# Local Chat (FastAPI + Ollama)

A small web app that chats with a **local** model through [Ollama](https://ollama.com), with optional DuckDuckGo search and page fetching. The UI runs in your browser; the LLM runs on your machine.

## Prerequisites

- **Python 3.10+** (3.11+ recommended) — [python.org](https://www.python.org/downloads/) or your OS package manager
- **A terminal**: PowerShell or Command Prompt (Windows), Terminal (macOS), or your Linux distro’s terminal
- On Linux/macOS, use `python3` if `python` is not available

---

## 1. Install Ollama

Install **Ollama** using the method that matches your OS, then continue to **Pull a model** (same for everyone).

### Windows

1. Download the installer from **[ollama.com/download](https://ollama.com/download)** and run it, **or** in **PowerShell** run the official install script:

   ```powershell
   irm https://ollama.com/install.ps1 | iex
   ```

2. Close and reopen the terminal if the `ollama` command is not found.

### macOS

1. Download the app from **[ollama.com/download](https://ollama.com/download)**, **or** with [Homebrew](https://brew.sh):

   ```bash
   brew install ollama
   ```

2. Start Ollama from Applications or run `ollama serve` in a terminal if needed.

### Linux

1. In a terminal:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Follow any prompts. You may need `sudo` depending on your setup.

---

### Pull a model (all platforms)

In a terminal:

```bash
ollama pull llama3.2:3b
```

Check that the CLI works:

```bash
ollama list
```

You should see `llama3.2:3b` (or the tag you pulled).

> **Note:** The app reads `OLLAMA_MODEL` from `.env` if present; otherwise it defaults to `llama3.2:3b`.

---

## 2. Install Python dependencies

Go to the project folder (the directory that contains `app.py` and `requirements.txt`).

**Windows (PowerShell or CMD):**

```powershell
cd C:\Users\YourName\Desktop\Hackathon
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
cd /path/to/Hackathon
python3 -m pip install -r requirements.txt
```

If `pip` fails, try: `python -m ensurepip --upgrade` or install pip via your OS docs.

---

## 3. Run the app

From the same project folder:

**Windows:**

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**macOS / Linux:**

```bash
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open a browser:

**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Stop the server with **Ctrl+C** in the terminal.

---

## Optional: environment variables

Create a **`.env`** file next to `app.py` to customize behavior:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_MODEL` | Model name, e.g. `llama3.2:3b` (default). |
| `OLLAMA_HOST` | Ollama API URL (default `http://127.0.0.1:11434`). |
| `OPENAI_API_KEY` | If set, the app uses OpenAI instead of Ollama for chat. |
| `OPENAI_MODEL` | OpenAI model name (default `gpt-4o-mini`). |
| `OLLAMA_REFINE_SEARCH_QUERY` | `1` / `true` to refine search queries with an extra local call. |
| `AUTO_SEARCH_RECENCY` | `0` to disable automatic web search on “latest/news” style questions when the checkbox is off. |

Example:

```env
OLLAMA_MODEL=llama3.2:3b
```

---

## Troubleshooting

- **`ollama` command not found**  
  Reopen the terminal, confirm Ollama is installed, and on Windows check that Ollama was added to PATH (reinstall if needed).

- **Cannot reach Ollama / HTTP 503 from the app**  
  Ensure Ollama is running (menu bar on macOS, tray on Windows, or run `ollama serve` in a terminal). Match `OLLAMA_HOST` to your Ollama listen address.

- **`python` / `python3` not found**  
  Install Python from [python.org](https://www.python.org/downloads/) or use your package manager (`brew`, `apt`, etc.).

- **Empty or slow replies**  
  Smaller models are faster; large models need more RAM/VRAM. Try a smaller model tag or close other applications.

- **Search / web features**  
  Online search uses the network (DuckDuckGo and optional page fetches). Leave **Search online** unchecked if you do not want that traffic.

---

## Project layout

| Path | Role |
|------|------|
| `app.py` | FastAPI server, chat API, web gather pipeline |
| `static/` | Frontend (HTML, CSS, JS) |
| `requirements.txt` | Python dependencies |
