# BIMI Preview Generator

A Flask web app that converts a logo into a BIMI-compliant SVG Tiny-PS file and renders realistic Gmail, Apple Mail, and Yahoo Mail inbox previews showing how the verified sender avatar would appear.

## What is BIMI?

[Brand Indicators for Message Identification (BIMI)](https://bimigroup.org/) is an email standard that lets organizations display their verified logo next to authenticated messages in supporting email clients. BIMI requires [DMARC alignment](https://seanthegeek.net/459/demystifying-dmarc/) and one of two certificate types:

- **VMC (Verified Mark Certificate)**: Requires a registered trademark. Shows the logo with a verified checkmark in most desktop clients.
- **CMC (Common Mark Certificate)**: No trademark required. Shows the logo in Gmail only, without a checkmark. Yahoo Mail and Apple Mail do not display CMC logos.

| Client | VMC Logo | VMC Checkmark | CMC Logo |
| --- | --- | --- | --- |
| Gmail (mobile & desktop) | Yes | Yes | Yes (no checkmark) |
| Apple Mail (iOS) | Yes | No | No |
| Apple Mail (macOS) | Yes | Yes | No |
| Yahoo Mail (Mobile) | Yes | No | No |
| Yahoo Mail (desktop) | Yes | Yes | No |

## How It Works

1. **Upload** a logo image (PNG, JPG, SVG, WebP, GIF, BMP, or TIFF) along with your company name, email domain, industry, and preferred email language.
2. **Convert** -- the app converts the image to a BIMI-compliant SVG Tiny-PS file using PIL color quantization and potrace for multi-color tracing with smooth Bezier curves.
3. **Generate** -- a small LLM generates industry-appropriate email content and translated UI labels in the selected language.
4. **Preview** -- see pixel-accurate Gmail, Apple Mail, and Yahoo Mail mockups (mobile and desktop) with your BIMI avatar, all rendered in the chosen language.
5. **Download** -- click the download button to grab the BIMI-compliant SVG to publish on your domain and configure in DNS. Files are automatically cleaned up after 10 minutes.

## Prerequisites

- Python 3.11+
- An API key for **Anthropic**, **OpenAI**, or **Google Gemini** (for LLM-generated email content; optional -- defaults are used if no key is set)
- `potrace` (required for raster-to-SVG tracing)
  
  ```bash
  # Debian/Ubuntu
  sudo apt install potrace

  # macOS
  brew install potrace
  ```

## Installation

```bash
git clone https://github.com/seanthegeek/bimi-preview.git
cd bimi-preview

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e .
# or: pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API key
```

## Running

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Production Deployment (Waitress + systemd + nginx)

#### 1. Create a system user and install the app

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin bimi
sudo -u bimi git clone https://github.com/seanthegeek/bimi-preview.git /home/bimi/bimi-preview
cd /home/bimi/bimi-preview
sudo -u bimi python3 -m venv /home/bimi/bimi-preview/.venv
sudo -u bimi /home/bimi/bimi-preview/.venv/bin/pip install -e .
sudo -u bimi /home/bimi/bimi-preview/.venv/bin/pip install waitress
```

Copy and edit the environment file:

```bash
sudo -u bimi cp /home/bimi/bimi-preview/.env.example /home/bimi/bimi-preview/.env
sudo -u bimi editor /home/bimi/bimi-preview/.env
```

Set `SECRET_KEY` to a long random string (e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`) and add your API key.

#### 2. Create a systemd service

Create `/etc/systemd/system/bimi-preview.service`:

```ini
[Unit]
Description=BIMI Preview Generator
After=network.target

[Service]
Type=simple
User=bimi
Group=bimi
WorkingDirectory=/home/bimi/bimi-preview
EnvironmentFile=/home/bimi/bimi-preview/.env
ExecStart=/home/bimi/bimi-preview/.venv/bin/waitress-serve \
    --host=127.0.0.1 \
    --port=8000 \
    --threads=4 \
    --channel-timeout=120 \
    --recv-bytes=65536 \
    app:app
Restart=on-failure
RestartSec=5

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/bimi/bimi-preview/uploads
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bimi-preview
sudo systemctl status bimi-preview
```

#### 3. Configure nginx as a reverse proxy

Create `/etc/nginx/sites-available/bimi-preview`:

```nginx
# Rate limiting zones — above all server blocks, so they're in the http context
limit_req_zone $binary_remote_addr zone=bimi_limit:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=bimi_llm:10m   rate=1r/s;

server {
    listen 80;
    listen [::]:80;
    server_name bimi.example.com;

    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name bimi.example.com;

    ssl_certificate     /etc/letsencrypt/live/bimi.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bimi.example.com/privkey.pem;
    # certbot manages TLS protocol and cipher settings in this file
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # --- Upload and body limits ---
    client_max_body_size 16m;

    # --- Timeouts ---
    proxy_connect_timeout 10s;
    proxy_send_timeout    30s;
    proxy_read_timeout    120s;   # LLM calls can take a few seconds
    send_timeout          30s;

    # --- Headers ---
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Return 429 for all rate-limited requests
    limit_req_status 429;

    # --- Stricter rate limit on the preview endpoint (LLM cost) ---
    location = /preview {
        limit_req zone=bimi_llm burst=3 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }

    # --- Default proxy with general rate limit ---
    location / {
        limit_req zone=bimi_limit burst=10 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/bimi-preview /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Use [certbot](https://certbot.eff.org/) to obtain TLS certificates:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d bimi.example.com
```

## Configuration

All configuration is via environment variables (or a `.env` file):

| Variable | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | Random | Flask session secret key |
| `LLM_PROVIDER` | Auto-detect | `"anthropic"`, `"openai"`, or `"gemini"` |
| `LLM_MODEL` | `claude-haiku-4-5` / `gpt-5.4-nano` / `gemini-2.5-flash-lite` | Model ID for the chosen provider |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key |
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `GEMINI_API_KEY` | -- | Google Gemini API key (also accepts `GOOGLE_API_KEY`) |

If no API key is configured, the preview renders with generic placeholder content.

## BIMI SVG Compliance

The generated SVG conforms to the SVG Tiny-PS profile required by BIMI:

- `version="1.2"` and `baseProfile="tiny-ps"` on the root `<svg>` element
- Square `viewBox` (equal width and height)
- Solid background `<rect>` covering the full canvas
- `<title>` element with the brand name
- No forbidden elements (`<script>`, `<animate>`, `<filter>`, `<mask>`, `<image>`, etc.)
- No external `href` references
- File size under 32 KB
- Circle-crop clearance with 85% scale factor for safe rendering in circular avatar contexts

## Project Structure

```text
bimi-preview/
  app.py              Flask routes and request handling
  bimi.py             Image-to-BIMI SVG conversion (PIL quantization + potrace)
  llm.py              LLM provider abstraction (Anthropic / OpenAI / Gemini) with multilingual support
  templates/
    index.jinja2.html        Upload form
    preview.jinja2.html      Gmail/Apple Mail/Yahoo Mail mockups (Jinja2)
    prompt.jinja2.html       LLM prompt documentation
  .env.example        Example environment configuration
  pyproject.toml      Python project metadata and dependencies
  requirements.txt    Pip-compatible dependency list
```

## License

See [LICENSE](LICENSE) for details.
