"""Flask app for BIMI preview generation."""

import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from flask import (  # noqa: E402
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    abort,
    session,
)
from bimi import convert_to_bimi  # noqa: E402
from llm import generate_email_content, get_supported_languages  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

UPLOAD_MAX_AGE = 600  # seconds (10 minutes)


def _cleanup_uploads(directory: Path, max_age: int) -> None:
    """Delete files older than max_age seconds. Runs in a daemon thread."""
    while True:
        try:
            now = time.time()
            for f in directory.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > max_age:
                    f.unlink(missing_ok=True)
        except Exception:
            pass
        time.sleep(60)


threading.Thread(
    target=_cleanup_uploads,
    args=(UPLOAD_DIR, UPLOAD_MAX_AGE),
    daemon=True,
).start()

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "tiff",
    "tif",
    "ico",
    "svg",
}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

DOMAIN_RE = re.compile(
    r"^(?!-)"  # no leading hyphen
    r"(?:[a-zA-Z0-9-]{1,63}\.)*"  # subdomains
    r"[a-zA-Z]{2,63}$"  # TLD
)


def validate_domain(domain: str) -> str | None:
    """Return cleaned domain or None if invalid."""
    domain = domain.strip().lower()
    # strip leading protocol / trailing slashes
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.rstrip("/")
    if not domain or len(domain) > 253:
        return None
    if not DOMAIN_RE.match(domain):
        return None
    return domain


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.jinja2.html", languages=get_supported_languages())


@app.route("/preview", methods=["GET"])
def preview_redirect():
    return redirect(url_for("index"), 302)


@app.route("/preview", methods=["POST"])
def preview():
    # --- Validate inputs ---
    domain = request.form.get("domain", "")
    clean_domain = validate_domain(domain)
    if not clean_domain:
        flash("Invalid domain name. Enter a domain like example.com", "error")
        return redirect(url_for("index"))

    company = re.sub(
        r"[\x00-\x1f\x7f-\x9f]", "", request.form.get("company", "")
    ).strip()
    if not company or len(company) > 100:
        flash("Company name is required (max 100 characters).", "error")
        return redirect(url_for("index"))

    industry = re.sub(
        r"[\x00-\x1f\x7f-\x9f]", "", request.form.get("industry", "")
    ).strip()
    if not industry or len(industry) > 100:
        flash("Industry is required (max 100 characters).", "error")
        return redirect(url_for("index"))

    language = request.form.get("language", "English").strip()
    supported = get_supported_languages()
    if language not in supported:
        language = "English"

    file = request.files.get("logo")
    if not file or not file.filename:
        flash("Please upload a logo image.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            "error",
        )
        return redirect(url_for("index"))

    # --- Save upload ---
    job_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_name = f"{job_id}.{ext}"
    upload_path = UPLOAD_DIR / upload_name
    file.save(upload_path)

    # --- Convert to BIMI SVG + generate email content in parallel ---
    with ThreadPoolExecutor(max_workers=2) as pool:
        bimi_future = pool.submit(convert_to_bimi, str(upload_path), company)
        llm_future = pool.submit(
            generate_email_content,
            company,
            clean_domain,
            industry,
            language,
        )

        try:
            bimi_svg = bimi_future.result()
        except Exception as e:
            flash(f"BIMI conversion failed: {e}", "error")
            upload_path.unlink(missing_ok=True)
            return redirect(url_for("index"))

        llm_content = llm_future.result()

    # Save BIMI SVG
    bimi_path = UPLOAD_DIR / f"{job_id}-bimi.svg"
    bimi_path.write_text(bimi_svg)

    # Clean up the original upload — only the BIMI SVG is needed now
    upload_path.unlink(missing_ok=True)

    # Store job_id in session for download
    session["job_id"] = job_id

    cert_type = request.form.get("cert_type", "vmc")
    show_checkmark = cert_type == "vmc"

    # Infer sender address
    sender_email_parts = ["info", "@", clean_domain]

    # --- Compute realistic times in user's timezone ---
    tz_name = request.form.get("tz", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except (KeyError, ValueError):
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    email_arrived = now - timedelta(minutes=random.randint(30, 90))

    status_bar_time = now.strftime("%-I:%M")
    inbox_time = email_arrived.strftime("%-I:%M %p")

    delta_min = int((now - email_arrived).total_seconds() // 60)
    if delta_min < 60:
        tmpl = llm_content.get("ui_minutes_ago", "NUM minutes ago")
        relative = tmpl.replace("NUM", str(delta_min))
    else:
        hours = delta_min // 60
        tmpl = llm_content.get("ui_hours_ago", "NUM hours ago")
        relative = tmpl.replace("NUM", str(hours))
    email_time = f"{inbox_time} ({relative})"

    return render_template(
        "preview.jinja2.html",
        company=company,
        domain=clean_domain,
        industry=industry,
        job_id=job_id,
        sender_email_parts=sender_email_parts,
        show_checkmark=show_checkmark,
        status_bar_time=status_bar_time,
        inbox_time=inbox_time,
        email_time=email_time,
        **llm_content,
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    # Sanitize job_id to prevent path traversal
    if not re.match(r"^[a-f0-9]{12}$", job_id):
        abort(404)
    bimi_path = UPLOAD_DIR / f"{job_id}-bimi.svg"
    if not bimi_path.exists():
        abort(404)
    return send_file(
        bimi_path,
        mimetype="image/svg+xml",
        as_attachment=True,
        download_name="bimi-mark.svg",
    )


@app.route("/bimi-svg/<job_id>")
def bimi_svg_inline(job_id: str):
    """Serve the BIMI SVG inline (for embedding in img tags)."""
    if not re.match(r"^[a-f0-9]{12}$", job_id):
        abort(404)
    bimi_path = UPLOAD_DIR / f"{job_id}-bimi.svg"
    if not bimi_path.exists():
        abort(404)
    return send_file(bimi_path, mimetype="image/svg+xml")


@app.route("/prompt")
def prompt_page():
    """Show the LLM prompt template for generating email content."""
    return render_template("prompt.jinja2.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
