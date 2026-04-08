# Agents

## Project Overview

BIMI Preview Generator -- a Flask web app that converts logos into BIMI-compliant SVG Tiny-PS files and renders realistic Gmail, Apple Mail, and Yahoo Mail inbox mockups showing how a verified sender avatar appears.

## Architecture

- **app.py** -- Flask routes and request handling. The `/preview` POST endpoint validates input, runs BIMI conversion and LLM content generation in parallel via `ThreadPoolExecutor`, computes timezone-aware email times (`status_bar_time`, `inbox_time`, `email_time`) from a hidden `tz` form field, then renders the preview template. GET `/preview` redirects to `/`.
- **bimi.py** -- Image-to-BIMI SVG conversion. Handles raster images (PNG, JPG, etc.) and SVG input. Outputs SVG Tiny-PS with square viewBox, solid background, circle-crop clearance, and forbidden elements stripped. Uses PIL color quantization and potrace for multi-color raster-to-SVG tracing with smooth Bézier curves, plus CairoSVG for SVG rasterization.
- **llm.py** -- LLM provider abstraction supporting Anthropic, OpenAI, and Google Gemini. Generates realistic email content (subject line, body, and an `other_senders` array of inbox neighbors) plus translated `ui_labels` for mail client chrome, all as structured JSON. `_parse_response` flattens arrays into numbered template keys (`other_sender_1_name`, etc.) and `ui_labels` into prefixed keys (`ui_inbox`, `ui_reply`, etc.). `get_supported_languages()` returns the language list for the configured provider. `_PROVIDER_LANGUAGES` defines per-provider language support (Anthropic: 15, OpenAI/Gemini: 50). Returns `{}` on any failure so Jinja `| default()` filters activate gracefully.
- **templates/** -- Jinja2 templates:
  - `index.jinja2.html` -- Upload form (company, domain, industry, language, logo file)
  - `preview.jinja2.html` -- Gmail, Apple Mail, and Yahoo Mail inbox + open email mockups with BIMI avatar
  - `prompt.jinja2.html` -- LLM prompt documentation page

## Key Design Decisions

- LLM content generation is optional. If no API key is configured or the LLM call fails (including credit exhaustion), the preview renders with hardcoded English default text via Jinja `| default()` filters -- no crash, no broken page.
- The language dropdown shows only languages supported by the configured LLM provider. The selected language applies to both email content and mail client UI labels (Inbox, Reply, Delete, etc.). When English is selected, `ui_labels` is omitted from the prompt to save tokens -- the template's `| default()` fallbacks provide the English UI strings.
- Email addresses are constructed in client-side JS (not rendered in HTML) to prevent Cloudflare email obfuscation from mangling them.
- The BIMI SVG is served via URL (`/bimi-svg/<job_id>`) and displayed in the preview using `<img>` tags. A download button links to `/download/<job_id>`.
- A background daemon thread scans `uploads/` every 60 seconds and deletes files older than 10 minutes. Original uploads are deleted immediately after conversion.
- The `email_body` field from LLM output is sanitized server-side -- only `<p>`, `<br>`, `<strong>`, and `<em>` tags are allowed.
- User inputs (company, industry) are stripped of control characters, capped at 100 chars, and wrapped in XML tags in the LLM prompt with an explicit data-only instruction to defend against prompt injection.

## Tech Stack

- Python 3.11+, Flask, Jinja2
- Pillow, CairoSVG, NumPy (image processing); potrace (system dependency for tracing)
- anthropic / openai / google-genai SDKs (LLM providers)
- Waitress (production WSGI server)
- nginx (reverse proxy with rate limiting)

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # add your API key
python app.py          # runs on http://localhost:5000
```

## Testing

```bash
pip install -e ".[test]"
pytest
```

## CI Requirements

After every Python code change, run linting and tests before committing to avoid CI failures:

```bash
ruff check .
ruff format --check .
pytest -v
```

If `ruff format --check` fails, run `ruff format .` to auto-fix, then re-stage the changed files.

## Style Conventions

- No type stubs or `.pyi` files -- type hints go inline.
- HTML templates use Jinja2 with `.jinja2.html` double extension.
- Environment variables for all configuration (see `.env.example`).
- Keep LLM provider implementations self-contained in `llm.py` -- each provider is a private `_call_<provider>()` function.
