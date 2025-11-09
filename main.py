from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from log_reg_library import load_classifier
from matching import MatchingEngine


def run_matching(
    *,
    students_input_path: str,
    mentors_type1_path: str,
    mentors_type2_path: str | None = None,
    n_type1: int = 4,
    n_type2: int | None = None,
    export_path: str = "./DATASETS/matches.csv",
    timeout_seconds: int = 120,
    show_progress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame | None:
    classifier = load_classifier(
        model_path="nlp_model_logreg_embeddings.pkl",
        label_encoder_path="label_encoder_log_reg.pkl",
        embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
    )

    students_df = pd.read_csv(students_input_path)
    mentors_type1_df = pd.read_csv(mentors_type1_path)
    mentors_type2_df = pd.read_csv(mentors_type2_path) if mentors_type2_path else None

    classified_students = classifier.annotate_dataframe(
        students_df,
        description_column="omschrijving",
        fill_column="Onderwerp",
        show_progress=show_progress,
    )

    if verbose:
        print(classified_students)

    engine_kwargs = {
        "students_df": classified_students,
        "mentors_type1_df": mentors_type1_df,
        "n_type1": n_type1,
        "verbose": verbose,
    }

    if mentors_type2_df is not None:
        engine_kwargs["mentors_type2_df"] = mentors_type2_df

    if n_type2 is not None:
        engine_kwargs["n_type2"] = n_type2

    engine = MatchingEngine(**engine_kwargs)

    matches = engine.solve_matches(timeout_seconds=timeout_seconds)
    df_matches = engine.export_matches(matches, filename=export_path)

    if df_matches is not None and verbose:
        print(df_matches.head())

    return df_matches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mentor matching pipeline.")
    parser.add_argument(
        "--students-input-path",
        required=True,
        help="Path to the CSV file containing student descriptions.",
    )
    parser.add_argument(
        "--mentors-type1-path",
        required=True,
        help="Path to the CSV file containing Type 1 mentor data.",
    )
    parser.add_argument(
        "--mentors-type2-path",
        help="Optional path to the CSV file containing Type 2 mentor data.",
    )
    parser.add_argument(
        "--type1-n",
        type=int,
        default=4,
        help="Number of Type 1 mentors to match per student (default: 4).",
    )
    parser.add_argument(
        "--type2-n",
        type=int,
        help="Number of Type 2 mentors to match per student (requires --mentors-type2-path).",
    )
    parser.add_argument(
        "--export-path",
        default="./DATASETS/matches.csv",
        help="Destination path for the exported matches CSV (default: ./DATASETS/matches.csv).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Solver timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display during classification.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress DataFrame previews in the console output.",
    )

    args = parser.parse_args(argv)

    if args.mentors_type2_path is None and args.type2_n is not None:
        parser.error("--type2-n requires --mentors-type2-path")

    for label, path_value in (
        ("students", args.students_input_path),
        ("mentors type1", args.mentors_type1_path),
    ):
        path = Path(path_value)
        if not path.exists():
            parser.error(f"Missing {label} file: {path}")

    if args.mentors_type2_path is not None:
        path = Path(args.mentors_type2_path)
        if not path.exists():
            parser.error(f"Missing mentors type2 file: {path}")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    export_path = Path(args.export_path)
    if not export_path.parent.exists():
        export_path.parent.mkdir(parents=True, exist_ok=True)

    run_matching(
        students_input_path=str(Path(args.students_input_path)),
        mentors_type1_path=str(Path(args.mentors_type1_path)),
        mentors_type2_path=str(Path(args.mentors_type2_path)) if args.mentors_type2_path else None,
        n_type1=args.type1_n,
        n_type2=args.type2_n,
        export_path=str(export_path),
        timeout_seconds=args.timeout_seconds,
        show_progress=not args.no_progress,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()