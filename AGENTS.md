# Agents

## Project Overview

BIMI Preview Generator -- a Flask web app that converts logos into BIMI-compliant SVG Tiny-PS files and renders realistic Gmail, Apple Mail, and Yahoo Mail inbox mockups showing how a verified sender avatar appears.

## Architecture

- **app.py** -- Flask routes and request handling. The `/preview` POST endpoint validates input, runs BIMI conversion and LLM content generation in parallel via `ThreadPoolExecutor`, computes timezone-aware email times (`status_bar_time`, `inbox_time`, `email_time`) from a hidden `tz` form field, then renders the preview template. GET `/preview` redirects to `/`.
- **bimi.py** -- Image-to-BIMI SVG conversion. Handles raster images (PNG, JPG, etc.) and SVG input. Outputs SVG Tiny-PS with square viewBox, solid background, circle-crop clearance, and forbidden elements stripped. Uses Pillow, CairoSVG, scikit-image, and optionally `potrace`.
- **llm.py** -- LLM provider abstraction supporting Anthropic, OpenAI, and Google Gemini. Generates realistic email content (subject line, body, and an `other_senders` array of inbox neighbors) as structured JSON. `_parse_response` flattens the array into numbered template keys (`other_sender_1_name`, etc.) and derives `_initial` from each sender's name. Email times and `email_subject` are not in the prompt -- times are computed server-side in `app.py` and `email_subject` falls back to `inbox_subject` in the template. Returns `{}` on any failure so Jinja `| default()` filters activate gracefully.
- **templates/** -- Jinja2 templates:
  - `index.jinja2.html` -- Upload form (company, domain, industry, logo file)
  - `preview.jinja2.html` -- Gmail, Apple Mail, and Yahoo Mail inbox + open email mockups with BIMI avatar
  - `prompt.jinja2.html` -- LLM prompt documentation page

## Key Design Decisions

- LLM content generation is optional. If no API key is configured or the LLM call fails (including credit exhaustion), the preview renders with hardcoded default text via Jinja `| default()` filters -- no crash, no broken page.
- Email addresses are constructed in client-side JS (not rendered in HTML) to prevent Cloudflare email obfuscation from mangling them.
- The BIMI SVG is served via URL (`/bimi-svg/<job_id>`) and displayed in the preview using `<img>` tags. A download button links to `/download/<job_id>`.
- A background daemon thread scans `uploads/` every 60 seconds and deletes files older than 10 minutes. Original uploads are deleted immediately after conversion.
- The `email_body` field from LLM output is sanitized server-side -- only `<p>`, `<br>`, `<strong>`, and `<em>` tags are allowed.

## Tech Stack

- Python 3.11+, Flask, Jinja2
- Pillow, CairoSVG, NumPy, SciPy, scikit-image (image processing)
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
