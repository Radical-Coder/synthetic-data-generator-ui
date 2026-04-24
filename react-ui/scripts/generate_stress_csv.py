#!/usr/bin/env python3
"""Generate a broad CSV seed file for stress-testing the React UI.

The data is deterministic by default, includes mixed column types, and injects
intentional numeric outliers so quality-dashboard box plots have something to
surface.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "public" / "demo" / "stress-test-synthetic-seed.csv"
DEFAULT_ROWS = 750
DEFAULT_SEED = 20260424

USE_CASES = [
    "AI-Powered Code Analysis",
    "Annotation/Tagging - Matching App Enhancement",
    "Automated Semantic Data Labeling",
    "DetectQ",
    "Drift Watch Containerization",
    "Synthetic Data Generation",
    "Claims Review Automation",
    "Document Extraction",
]

TASKS = [
    "Add Data Type Override Feature",
    "Add Quality Dashboard",
    "Build React UI",
    "Create Container Deployment",
    "Improve Chart Interactivity",
    "Implement Sortable Tables",
    "Load Seed Dataset",
    "Validate Synthetic Distribution",
    "Wire API Contract",
    "Write Review Notes",
]

RESOURCES = [
    "Amit",
    "Anastasia",
    "Faraz",
    "Jason",
    "Niloy",
    "Praful",
    "Riken",
    "Robel",
    "Ryan",
    "Shaymaa",
    "Sunny",
]

REGIONS = ["Northeast", "Midwest", "South", "West", "Remote", "International"]
STATES = ["CA", "FL", "GA", "IL", "MD", "NC", "NY", "OH", "PA", "TX", "VA", "WA"]
CITIES = ["Arlington", "Atlanta", "Austin", "Buffalo", "Chicago", "Dallas", "Raleigh", "Seattle"]
DEPARTMENTS = ["Engineering", "Innovation", "Operations", "Product", "QA", "Sales", "Security"]
STATUSES = ["Backlog", "Blocked", "Done", "In Review", "In Progress", "Ready", "Waiting"]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
PLANS = ["Free", "Starter", "Professional", "Enterprise", "Government"]
FIRST_NAMES = ["Alex", "Ari", "Casey", "Dana", "Devon", "Emery", "Jordan", "Morgan", "Quinn", "Riley", "Taylor"]
LAST_NAMES = ["Baker", "Chen", "Ghosh", "Johnson", "Khan", "Oberoi", "Patel", "Rivera", "Smith", "Walker"]
NOTES = [
    "baseline row",
    "requires manual review",
    "edge case from meeting notes",
    "candidate for dashboard validation",
    "includes delayed handoff",
    "sample imported from client workflow",
    "contains missing optional fields",
]


def weighted_choice(rng: random.Random, weighted_values: list[tuple[str, int]]) -> str:
    total = sum(weight for _, weight in weighted_values)
    pick = rng.randint(1, total)
    running = 0
    for value, weight in weighted_values:
        running += weight
        if pick <= running:
            return value
    return weighted_values[-1][0]


def maybe_blank(rng: random.Random, value: object, chance: float = 0.025) -> object:
    return "" if rng.random() < chance else value


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def currency(value: float) -> str:
    return f"{value:.2f}"


def phone_number(rng: random.Random) -> str:
    return f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def zip_code(rng: random.Random) -> str:
    return f"{rng.randint(1000, 99999):05d}"


def build_row(index: int, rng: random.Random, base_date: datetime) -> dict[str, object]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    created_at = base_date - timedelta(days=rng.randint(0, 540), hours=rng.randint(0, 23))
    last_seen = created_at + timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23))
    due_date = created_at.date() + timedelta(days=rng.randint(3, 90))

    status = weighted_choice(
        rng,
        [
            ("In Progress", 26),
            ("Done", 24),
            ("Ready", 16),
            ("In Review", 14),
            ("Backlog", 10),
            ("Waiting", 6),
            ("Blocked", 4),
        ],
    )
    priority = weighted_choice(rng, [("Medium", 45), ("High", 25), ("Low", 20), ("Critical", 10)])
    region = weighted_choice(
        rng,
        [
            ("Northeast", 23),
            ("South", 22),
            ("West", 20),
            ("Midwest", 16),
            ("Remote", 14),
            ("International", 5),
        ],
    )

    age = int(clamp(rng.gauss(39, 11), 18, 76))
    income = max(18000, rng.gauss(88000, 28500))
    score = clamp(rng.gauss(78, 13), 18, 100)
    orders = max(0, int(rng.gauss(14, 7)))
    tickets_open = max(0, int(rng.gauss(3, 2.2)))
    quantity = max(1, int(rng.lognormvariate(2.15, 0.55)))
    team_size = max(1, int(rng.gauss(8, 4)))
    revenue = max(0, rng.lognormvariate(8.7, 0.75))
    latency_ms = max(20, rng.lognormvariate(4.8, 0.55))
    completion_pct = clamp(rng.gauss(72, 20), 0, 100)
    risk_score = clamp(rng.betavariate(2.3, 5.5) * 100, 0, 100)
    defect_count = max(0, int(rng.gauss(2, 2)))
    satisfaction_rating = clamp(rng.gauss(4.1, 0.65), 1, 5)

    if index % 137 == 0:
        income *= 4.4
        revenue *= 6.2
        latency_ms *= 14
        orders += 95
        tickets_open += 28
        defect_count += 17
        risk_score = clamp(risk_score + 55, 0, 100)
    elif index % 211 == 0:
        income = 12000 + rng.randint(0, 3500)
        revenue = rng.uniform(0, 500)
        latency_ms = rng.uniform(3, 12)
        completion_pct = rng.uniform(0, 12)
        score = rng.uniform(12, 25)
        satisfaction_rating = rng.uniform(1, 1.6)

    return {
        "Record ID": f"REC-{index:05d}",
        "Account ID": f"ACCT-{rng.randint(1000, 9999)}",
        "Customer Name": name,
        "Email": f"{first.lower()}.{last.lower()}{index}@example.com",
        "Phone": phone_number(rng),
        "Postal Code": zip_code(rng),
        "Region": region,
        "State": rng.choice(STATES),
        "City": rng.choice(CITIES),
        "Use Case": rng.choice(USE_CASES),
        "Task": rng.choice(TASKS),
        "Resource": rng.choice(RESOURCES),
        "Department": rng.choice(DEPARTMENTS),
        "Status": status,
        "Priority": priority,
        "Plan": rng.choice(PLANS),
        "Is Active": rng.choice(["TRUE", "FALSE"]),
        "Needs Review": "TRUE" if status in {"Blocked", "Waiting", "In Review"} or rng.random() < 0.14 else "FALSE",
        "Created At": created_at.isoformat(timespec="seconds"),
        "Last Seen": maybe_blank(rng, last_seen.isoformat(timespec="seconds"), 0.04),
        "Due Date": due_date.isoformat(),
        "Age": maybe_blank(rng, age),
        "Income": maybe_blank(rng, currency(income)),
        "Score": f"{score:.1f}",
        "Orders": orders,
        "Tickets Open": maybe_blank(rng, tickets_open),
        "Quantity": quantity,
        "Team Size": team_size,
        "Revenue": currency(revenue),
        "Latency MS": f"{latency_ms:.1f}",
        "Completion Percent": f"{completion_pct:.1f}",
        "Risk Score": f"{risk_score:.1f}",
        "Defect Count": defect_count,
        "Satisfaction Rating": f"{satisfaction_rating:.2f}",
        "Notes": rng.choice(NOTES),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mixed-type CSV for React UI stress testing.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Number of rows to write. Default: {DEFAULT_ROWS}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed. Default: {DEFAULT_SEED}")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    rng = random.Random(args.seed)
    base_date = datetime(2026, 4, 24, 12, 0, 0)
    rows = [build_row(index, rng, base_date) for index in range(1, args.rows + 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows and {len(rows[0])} columns to {args.output}")


if __name__ == "__main__":
    main()
