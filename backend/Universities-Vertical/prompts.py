"""
prompts.py — Thomas Scientific University Intel
Edit this file to tune search behavior, recency, and output fields.
"""

from datetime import datetime, timedelta

# ── Recency window ────────────────────────────────────────────────
DAYS_BACK = 30
cutoff = (datetime.today() - timedelta(days=DAYS_BACK)).strftime("%B %d, %Y")
TODAY = datetime.today().strftime("%B %d, %Y")

RECENCY_INSTRUCTION = (
    f"Only include results from the last {DAYS_BACK} days (on or after {cutoff}). "
    f"Today's date is {TODAY}. "
    f"Focus on news articles, press releases, university announcements, and official publications. "
    f"Ignore results older than {DAYS_BACK} days."
)

# ── Shared context ────────────────────────────────────────────────
ROLE = (
    "You are a market intelligence analyst for Thomas Scientific, "
    "a B2B scientific supply distributor serving R1 research universities and hospital research institutions."
)

JSON_INSTRUCTION = "Return ONLY a raw JSON array with no markdown, no explanation, no preamble."

# ── Trigger prompts ───────────────────────────────────────────────
PROMPTS = {
    "grant": lambda uni: (
        f"{ROLE} "
        f"Search for recent news and announcements about NIH or NSF grant awards to professors or research labs at {uni}. "
        f"{RECENCY_INSTRUCTION} "
        f"{JSON_INSTRUCTION} "
        f"Each object must use these exact keys: "
        f"summary, professor, department, amount, agency, why_it_matters, source_url. "
        f"'why_it_matters' = one sentence on why a scientific supply sales rep should act on this. "
        f"If no results within the recency window, return []."
    ),

    "faculty": lambda uni: (
        f"{ROLE} "
        f"Search for recent news and announcements about new faculty hires, incoming professors, or newly appointed researchers at {uni} "
        f"in biology, chemistry, immunology, biochemistry, neuroscience, or medical research. "
        f"{RECENCY_INSTRUCTION} "
        f"{JSON_INSTRUCTION} "
        f"Each object must use these exact keys: "
        f"summary, name, department, start_date, why_it_matters, source_url. "
        f"'why_it_matters' = one sentence noting new faculty need to outfit new labs with supplies and equipment. "
        f"If no results within the recency window, return []."
    ),

    "capital": lambda uni: (
        f"{ROLE} "
        f"Search for recent news, press releases, or announcements about new research buildings, laboratory facilities, "
        f"biomedical research centers, or major construction projects at {uni} valued at $100 million or more. "
        f"{RECENCY_INSTRUCTION} "
        f"{JSON_INSTRUCTION} "
        f"Each object must use these exact keys: "
        f"summary, project_name, value, timeline, why_it_matters, source_url. "
        f"'why_it_matters' = one sentence on the lab supply opportunity from new facility build-out. "
        f"Only include projects at or above $100M. If none, return []."
    ),

    "contract": lambda uni: (
        f"{ROLE} "
        f"Search for recent news, announcements, or postings about open RFPs, bid opportunities, or expiring procurement contracts "
        f"for laboratory supplies, scientific equipment, or research consumables at {uni}. "
        f"{RECENCY_INSTRUCTION} "
        f"{JSON_INSTRUCTION} "
        f"Each object must use these exact keys: "
        f"summary, contract_name, estimated_value, deadline_or_expiration, why_it_matters, source_url. "
        f"'why_it_matters' = one sentence on the bid or re-compete opportunity for Thomas Scientific. "
        f"If no results within the recency window, return []."
    ),
}

# ── Output field display order ─────────────────────────────────────
FIELD_MAPS = {
    "grant":    ["professor", "department", "agency", "amount"],
    "faculty":  ["name", "department", "start_date"],
    "capital":  ["project_name", "value", "timeline"],
    "contract": ["contract_name", "estimated_value", "deadline_or_expiration"],
}