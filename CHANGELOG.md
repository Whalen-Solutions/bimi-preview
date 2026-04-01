# Changelog

All notable changes to this project will be documented in this file.

## [0.3.1] - 2026-04-01

### Changed

- LLM prompt now includes today's date so generated emails reference current time periods
- LLM prompt uses company's fiscal quarter/year for financial content when known, calendar dates for events and meetings

## [0.3.0] - 2026-03-31

### Added

- Gmail mobile inbox preview with realistic phone frame, Dynamic Island, search bar, and bottom navigation
- Gmail mobile open email preview with toolbar, sender header, BIMI verified checkmark, email body, and reply bar
- Side-by-side responsive layout for mobile phone previews on desktop
- VSCode task for running Flask dev server with auto-reload

### Changed

- Replaced Gunicorn with Waitress in project dependencies and Dockerfile
- Removed light blue background from unread desktop inbox rows to match current Gmail styling
- Renamed "Gmail Inbox" section to "Gmail Desktop"
- Removed "Open Email" heading above the desktop email view
- Renamed download button to "Download BIMI SVG"
- Added spacing between desktop inbox and open email sections

## [0.2.1] - 2026-03-31

### Changed

- Renamed `docker-compose.yml` to `docker-compose.dev.yml` for development use
- New production `docker-compose.yml` using Waitress instead of Gunicorn, with no published ports
- Added `docker-compose.override.example.yml` documenting how to connect to an external nginx reverse proxy stack
- Added `docker-compose.override.yml` to `.gitignore`

## [0.2.0] - 2026-03-31

### Added

- Background cleanup daemon thread that deletes uploaded files older than 10 minutes, removing the need for request-triggered cleanup
- Manual download button on the preview page

### Changed

- BIMI SVG avatars in preview are now served via URL (`/bimi-svg/<job_id>`) using `<img>` tags instead of inline SVG markup
- Download is no longer automatic on page load; users click the download button when ready
- Files persist in `uploads/` until the background cleanup removes them (previously deleted immediately after download)
- Sharper BIMI SVG output from raster images: higher upsample factors, reduced blur, tighter potrace and Bezier fitting tolerances

### Removed

- Auto-download JavaScript on preview page load
- `@after_this_request` file deletion in the download route

## [0.1.0] - 2026-03-30

### Added

- Flask web app for BIMI SVG generation and Gmail inbox preview
- Raster-to-SVG conversion with Otsu thresholding, contour tracing, and cubic Bezier fitting
- Optional potrace integration for higher-quality tracing
- SVG-to-BIMI conversion preserving existing artwork
- LLM-generated email content via Anthropic, OpenAI, or Google Gemini
- Graceful fallback to default text when no LLM is configured or calls fail
- BIMI SVG Tiny-PS compliance: square viewBox, solid background, circle-crop clearance, forbidden element removal, 32 KB size limit
- Gmail inbox mockup with verified sender checkmark
- Auto-download of generated BIMI SVG
- Cloudflare-safe email address rendering via client-side JS
- XSS sanitization of LLM-generated HTML
- CI workflow with ruff linting, pytest, and CodeQL code scanning
- Production deployment guide: Waitress systemd service, nginx reverse proxy with rate limiting
- Dockerfile and docker-compose configuration
- `AGENTS.md` and `CLAUDE.md` for AI-assisted development context
- Dark code blocks with copy-to-clipboard buttons on the LLM prompt documentation page
