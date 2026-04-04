# Changelog

All notable changes to this project will be documented in this file.

## [0.5.1] - 2026-04-04

### Fixed

- Email subjects in Apple Mail desktop message list are no longer bold, matching the real macOS Mail UI

### Changed

- Updated README with plain-language VMC/CMC explanation and DMARC guide link

## [0.5.0] - 2026-04-04

### Added

- Apple Mail for iOS inbox and open email previews with iOS 26.4 UI: category filter pills, white floating pill buttons with shadow, mailing list unsubscribe banner, Summarize bar, and floating bottom toolbar overlaying email content
- Apple Mail for macOS desktop preview with three-pane layout: sidebar (Favorites/iCloud mailboxes), message list with category pills, and reading pane with toolbar, mailing list banner, sender header with verified checkmark, and email body
- VMC/CMC certificate handling for Apple Mail: VMC shows BIMI logo, CMC shows initial (Apple Mail does not display CMC logos)
- Dark mode support for all Apple Mail previews (true black iOS backgrounds, dark gray macOS surfaces)
- SEO meta tags on the index page: description, Open Graph, and Twitter Card tags for social sharing

### Changed

- Updated certificate hint to clarify VMC checkmark support varies by client (desktop only for Apple/Yahoo) and that Yahoo Mail and Apple Mail do not display CMC logos
- Removed `www.` prefix from domain display in Apple Mail and Yahoo Mail desktop previews
- Preview page footer now mentions Apple Mail alongside Gmail and Yahoo Mail

## [0.4.5] - 2026-04-02

### Fixed

- Yahoo mobile text-size (AA) icon now turns white in dark mode instead of staying gray

## [0.4.4] - 2026-04-02

### Changed

- Clarified BIMI certificate type hint: VMC shows checkmarks in Gmail and Yahoo, CMC shows logo only in Gmail, Yahoo does not support CMC

## [0.4.3] - 2026-04-02

### Added

- Gmail desktop toolbar with action icons (back, archive, spam, delete, mark read, snooze, move, more), "1 of 52" pagination, and keyboard shortcut icon
- Gmail desktop email subject line now shows "Inbox x" label badge, print, and open-in-new-window icons
- Gmail desktop email header now shows sender address and "Unsubscribe" link inline, "to me" dropdown arrow, and action icons (star, emoji, reply, more)
- Yahoo desktop toolbar now includes previous/next navigation arrows and close button
- Yahoo desktop inbox unread row highlight in dark mode (brighter purple background with white text)
- Clickable "BIMI Preview" heading linking to home page

### Changed

- Gmail desktop dark mode now only darkens the toolbar chrome; email content stays light, matching real Gmail behavior
- Yahoo desktop dark mode switched from gray to Yahoo's signature purple palette for both inbox and toolbar
- Yahoo Mail section label renamed to "Yahoo Mail Mobile" for clarity
- Yahoo desktop open email checkmark color updated to Yahoo brand purple
- Yahoo reply/reply-all/forward icons switched from stroked outlines to filled icons matching the real Yahoo UI
- Yahoo mobile text-size icon changed from "TT" to stacked "AA" matching the real Yahoo UI
- All vertical three-dot (more) icons replaced with horizontal three-dot icons across Gmail and Yahoo mockups
- Removed unread dot from Yahoo desktop open email subject line

### Fixed

- Dark mode no longer fills the entire phone battery icon by excluding `fill="none"` outline rects from the dark mode fill override

## [0.4.2] - 2026-04-01

### Added

- Dark mode toggle on the preview page that switches mobile and desktop Gmail mockups to Gmail's dark theme colors

## [0.4.1] - 2026-04-01

### Fixed

- Palette-mode images (GIF, indexed PNG) with transparency now correctly detected and converted using alpha channel instead of Otsu thresholding
- Multi-frame formats (GIF, WebP, TIFF) now select the largest frame by pixel area instead of silently using the first frame
- ICO multi-frame selection no longer returns the wrong frame when no frame is larger than the first

## [0.4.0] - 2026-04-01

### Added

- Proper SVG Tiny-PS compliance conversion that preserves visual appearance: inlines CSS `<style>` blocks and `style` attributes as presentation attributes, resolves `<use>` references into inline content, strips forbidden elements (`clipPath`, `filter`, `pattern`, `symbol`, `marker`, `image`, `foreignObject`, animations), removes non-SVG namespace elements/attributes (Inkscape, Sodipodi, etc.), and cleans forbidden attributes (`clip-path`, `mask`, `filter`, `data-*`, event handlers)
- Rasterize-and-retrace fallback for SVGs containing `<text>` elements or malformed XML, using CairoSVG rendering at 1024px
- ICO file format support — selects the largest embedded frame from multi-size favicon files

### Changed

- Rewrote `svg_to_bimi_svg` from regex-based string manipulation to `xml.etree.ElementTree`-based parsing for reliable XML handling

## [0.3.4] - 2026-04-01

### Changed

- Reduced LLM prompt token usage (~45%) by consolidating other senders into an array, dropping `email_subject` (template already falls back to `inbox_subject`), removing `_initial` fields (derived server-side from sender name), and trimming verbose rule descriptions
- Updated default OpenAI model from `gpt-4o-mini` to `gpt-5.4-nano`
- Updated default Gemini model from `gemini-2.0-flash-lite` to `gemini-2.5-flash-lite`
- Switched Anthropic default to model alias `claude-haiku-4-5` for automatic minor version updates

## [0.3.3] - 2026-04-01

### Changed

- Phone mockup status bars now show the actual time the page was loaded in the user's timezone
- Main email inbox and open email times are computed server-side with accurate relative timestamps
- Removed `inbox_time` and `email_time` from LLM prompt (now computed server-side)

## [0.3.2] - 2026-04-01

### Changed

- Added note about BIMI multi-provider support on form and preview pages with link to BIMI Group
- GET requests to `/preview` now redirect to `/` instead of returning a 405 error

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
