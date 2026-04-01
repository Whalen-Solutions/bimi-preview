"""LLM provider abstraction for generating email preview content."""

import json
import logging
import os
import re
from datetime import date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (mirrors templates/prompt.jinja2.html)
# ---------------------------------------------------------------------------

LLM_PROMPT = """\
You are generating realistic email preview content for a BIMI (Brand Indicators \
for Message Identification) inbox mockup. This mockup demonstrates how a \
company's verified logo appears in a Gmail inbox.

Given the following inputs:
- Company name: {company_name}
- Email domain: {domain}
- Industry: {industry}

Generate realistic, professional email content that this company might \
plausibly send. Return ONLY a JSON object with exactly these fields:

{{
  "inbox_subject": "Subject line as it appears in the inbox list (50-80 chars)",
  "inbox_preview": "Preview text snippet shown after the subject in the inbox (60-100 chars)",

  "email_subject": "Full subject line for the open email view (can match inbox_subject)",
  "email_body": "2-3 short paragraphs of HTML. Use <p> tags. Write in a professional \
but warm tone appropriate for the industry. Reference the company by name. Include a \
specific, plausible reason for the email (product update, security notice, appointment \
reminder, order confirmation, etc.) that fits the industry.",

  "other_sender_1_name": "A plausible colleague or contact name",
  "other_sender_1_initial": "First letter of their name",
  "other_sender_1_subject": "A realistic personal/work email subject",
  "other_sender_1_preview": "Preview text for this email",
  "other_sender_1_time": "Time like '9:15 AM'",

  "other_sender_2_name": "Another plausible contact name",
  "other_sender_2_initial": "First letter",
  "other_sender_2_subject": "Another realistic subject",
  "other_sender_2_preview": "Preview text",
  "other_sender_2_time": "Time like 'Yesterday'",

  "other_sender_3_name": "A lesser-known service, internal tool, or noreply sender that would NOT have its own logo in Gmail (e.g. Teamwork, Gusto, DocuSign)",
  "other_sender_3_initial": "First letter",
  "other_sender_3_subject": "A notification-style subject",
  "other_sender_3_preview": "Preview text",
  "other_sender_3_time": "A date like 'Mar 28'"
}}

Rules:
- Consider what the specific company actually does within its industry — use your \
knowledge of the company to generate content that reflects their real products or \
services, not generic industry stereotypes (e.g. a healthcare distributor should \
send a supply chain or account update, not a pharmacy pickup reminder)
- The main email (from the company) should be industry-appropriate and specific
- The other senders should look like normal inbox neighbors -- a mix of personal \
messages and notifications
- The email body HTML should use only <p> tags, no other HTML elements
- Do not include any email addresses as literal text in the output (they will be \
constructed separately)
- Keep all text professional and realistic
- Today's date is {today}. If referencing a time period (quarter, month, year), it must be \
current — do NOT reference a past quarter, month, or year unless today is near its end. \
If you know the company's fiscal calendar, use their fiscal quarter/year for financial \
content (earnings, reports, billing), but use calendar dates for events and meetings
- Return ONLY valid JSON, no markdown fences, no explanation"""

EXPECTED_KEYS = {
    "inbox_subject",
    "inbox_preview",
    "email_subject",
    "email_body",
    "other_sender_1_name",
    "other_sender_1_initial",
    "other_sender_1_subject",
    "other_sender_1_preview",
    "other_sender_1_time",
    "other_sender_2_name",
    "other_sender_2_initial",
    "other_sender_2_subject",
    "other_sender_2_preview",
    "other_sender_2_time",
    "other_sender_3_name",
    "other_sender_3_initial",
    "other_sender_3_subject",
    "other_sender_3_preview",
    "other_sender_3_time",
}

# Tags allowed in the LLM-generated email_body to prevent XSS.
_ALLOWED_TAGS_RE = re.compile(r"<(?!/?(p|br|strong|em)\b)[^>]+>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_anthropic(prompt: str) -> str:
    import anthropic

    model = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic(timeout=30.0)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("No text block in Anthropic response")


def _call_openai(prompt: str) -> str:
    import openai

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    client = openai.OpenAI(timeout=30.0)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("No content in OpenAI response")
    return text


def _call_gemini(prompt: str) -> str:
    from google import genai

    model = os.environ.get("LLM_MODEL", "gemini-2.0-flash-lite")
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    if response.text is None:
        raise ValueError("No content in Gemini response")
    return response.text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: str) -> dict:
    """Parse the LLM JSON response, validate keys, sanitize HTML."""
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try to extract a JSON object if there's surrounding text
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    data = json.loads(cleaned)

    missing = EXPECTED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in LLM response: {missing}")

    # Sanitize email_body — allow only safe tags
    if "email_body" in data:
        data["email_body"] = _ALLOWED_TAGS_RE.sub("", data["email_body"])

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _detect_provider() -> str | None:
    """Auto-detect provider from env vars."""
    provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if provider in ("anthropic", "openai", "gemini"):
        return provider
    # Infer from whichever API key is set
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return None


def generate_email_content(company: str, domain: str, industry: str) -> dict:
    """Generate realistic email preview content via LLM.

    Returns a dict of template variables on success, or an empty dict on
    failure (so that Jinja ``| default()`` filters activate).
    """
    provider = _detect_provider()
    if provider is None:
        logger.info("No LLM provider configured — using template defaults")
        return {}

    prompt = LLM_PROMPT.format(
        company_name=company,
        domain=domain,
        industry=industry,
        today=date.today().strftime("%B %-d, %Y"),
    )

    try:
        if provider == "anthropic":
            raw = _call_anthropic(prompt)
        elif provider == "gemini":
            raw = _call_gemini(prompt)
        else:
            raw = _call_openai(prompt)

        return _parse_response(raw)

    except Exception:
        logger.warning("LLM content generation failed — using defaults", exc_info=True)
        return {}
