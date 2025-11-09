"""Utility script to enrich studenten.csv with dummy assignment descriptions."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDENTS_PATH = REPO_ROOT / "DATASETS" / "studenten.csv"
OPDRACHTEN_PATH = REPO_ROOT / "dummy_opdrachten_dataset.csv"
OUTPUT_PATH = REPO_ROOT / "DATASETS" / "studenten_met_omschrijving.csv"


def main() -> None:
    with OPDRACHTEN_PATH.open("r", newline="", encoding="utf-8") as assignments_file:
        assignments_reader = csv.DictReader(assignments_file)
        assignments = [row["omschrijving"] for row in assignments_reader if row.get("omschrijving")]

    if not assignments:
        raise RuntimeError("Geen omschrijvingen gevonden in dummy_opdrachten_dataset.csv")

    with STUDENTS_PATH.open("r", newline="", encoding="utf-8") as students_file:
        students_reader = csv.DictReader(students_file)
        fieldnames = list(students_reader.fieldnames or [])

        if "Onderwerp" not in fieldnames:
            raise RuntimeError("Kolom 'Onderwerp' ontbreekt in studenten.csv")

        if "omschrijving" not in fieldnames:
            fieldnames.append("omschrijving")

        rows = []
        for idx, row in enumerate(students_reader):
            row["Onderwerp"] = ""
            try:
                row["omschrijving"] = assignments[idx]
            except IndexError as exc:
                raise RuntimeError(
                    "Onvoldoende dummy omschrijvingen beschikbaar voor alle studenten"
                ) from exc
            rows.append(row)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Bestand geschreven: {OUTPUT_PATH} ({len(rows)} rijen)")


if __name__ == "__main__":
    main()
