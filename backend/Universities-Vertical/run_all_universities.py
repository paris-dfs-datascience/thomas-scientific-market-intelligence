"""
Batch University Intel — Thomas Scientific
Runs all triggers across all target universities from the April 1 discovery call.

Usage:
    python run_all_universities.py
    python run_all_universities.py --trigger grant
    python run_all_universities.py --output results.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, HttpOptions, Tool
from prompts import PROMPTS, FIELD_MAPS, DAYS_BACK

MODEL = "gemini-2.5-flash"

UNIVERSITIES = [
    # Priority accounts — named by Andrew
    "Yale University",
    "University of Pennsylvania",
    # UC System (all 10 campuses)
    "UCLA",
    "UC San Diego",
    "UC Berkeley",
    "UC San Francisco",
    "UC Davis",
    "UC Santa Barbara",
    "UC Irvine",
    "UC Santa Cruz",
    "UC Riverside",
    "UC Merced",
    # Hospital research institutions
    "Cleveland Clinic",
    "Boston Children's Hospital",
]

COLORS = {
    "grant":    "\033[94m",
    "faculty":  "\033[92m",
    "capital":  "\033[91m",
    "contract": "\033[95m",
    "reset":    "\033[0m",
    "dim":      "\033[90m",
    "bold":     "\033[1m",
    "yellow":   "\033[93m",
    "header":   "\033[97m",
}


def parse_signals(raw: str) -> list:
    cleaned = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return []


def print_signals(trigger: str, university: str, signals: list):
    c = COLORS[trigger]
    r = COLORS["reset"]
    d = COLORS["dim"]
    y = COLORS["yellow"]
    b = COLORS["bold"]

    if not signals:
        print(f"  {d}[{trigger}] No signals found.{r}")
        return

    print(f"  {c}{b}[{trigger.upper()}] {len(signals)} signal(s){r}")
    for i, s in enumerate(signals, 1):
        print(f"    {b}[{i}] {s.get('summary', '')}{r}")
        meta = " · ".join(str(s[f]) for f in FIELD_MAPS.get(trigger, []) if s.get(f))
        if meta:
            print(f"        {d}{meta}{r}")
        if s.get("why_it_matters"):
            print(f"        {y}↳ {s['why_it_matters']}{r}")
        if s.get("source_url"):
            print(f"        {d}{s['source_url']}{r}")


def run_university(client, university: str, triggers: list) -> dict:
    b = COLORS["bold"]
    h = COLORS["header"]
    r = COLORS["reset"]
    d = COLORS["dim"]

    print(f"\n{b}{'═'*60}{r}")
    print(f"{h}{b}  {university}{r}")
    print(f"{b}{'═'*60}{r}")

    results = {"university": university, "triggers": {}, "timestamp": datetime.now().isoformat()}

    for trigger in triggers:
        print(f"  {d}Searching [{trigger}]...{r}", end="", flush=True)
        try:
            prompt = PROMPTS[trigger](university)
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                    temperature=0.2,
                ),
            )
            signals = parse_signals(response.text)
            print(f"\r", end="")
            print_signals(trigger, university, signals)
            results["triggers"][trigger] = signals
        except Exception as e:
            print(f"\r  \033[91mERROR [{trigger}]: {e}\033[0m")
            results["triggers"][trigger] = []

    return results


def print_summary(all_results: list):
    b = COLORS["bold"]
    r = COLORS["reset"]
    d = COLORS["dim"]
    y = COLORS["yellow"]

    print(f"\n{b}{'═'*60}{r}")
    print(f"{b}  SUMMARY{r}")
    print(f"{b}{'═'*60}{r}")

    total_signals = 0
    for result in all_results:
        uni = result["university"]
        counts = {t: len(s) for t, s in result["triggers"].items()}
        total = sum(counts.values())
        total_signals += total
        if total > 0:
            count_str = "  ".join(f"{t}:{n}" for t, n in counts.items() if n > 0)
            print(f"  {b}{uni}{r}  {y}{count_str}{r}")
        else:
            print(f"  {d}{uni} — no signals{r}")

    print(f"\n  {b}Total signals: {total_signals}{r} across {len(all_results)} universities\n")


def main():
    parser = argparse.ArgumentParser(description="Thomas Scientific — Batch University Intel")
    parser.add_argument("--trigger", "-t", default="all",
                        choices=["grant", "faculty", "capital", "contract", "all"])
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file (e.g. results.json)")
    parser.add_argument("--api-key", "-k", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Add GEMINI_API_KEY=your_key to your .env file.")
        sys.exit(1)

    client = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(api_version="v1alpha")
    )

    triggers = list(PROMPTS.keys()) if args.trigger == "all" else [args.trigger]

    print(f"\n\033[1mThomas Scientific // Batch University Intel\033[0m")
    print(f"\033[90m{len(UNIVERSITIES)} universities | Triggers: {', '.join(triggers)} | Last {DAYS_BACK} days\033[0m")

    all_results = []
    for university in UNIVERSITIES:
        result = run_university(client, university, triggers)
        all_results.append(result)

    print_summary(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\033[90mResults saved to {args.output}\033[0m\n")


if __name__ == "__main__":
    main()