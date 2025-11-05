from matching import MatchingEngine


def main() -> None:
    engine = MatchingEngine(
        students_path="./DATASETS/studenten.csv",
        mentors_type1_path="./DATASETS/mentoren.csv",
        mentors_type2_path="./DATASETS/mentorenB.csv",
        n_type1=3,
        n_type2=2,
        verbose=True,
    )

    matches = engine.solve_matches(timeout_seconds=120)
    df = engine.export_matches(matches, filename="./DATASETS/matches.csv")

    if df is not None:
        print(df.head())


if __name__ == "__main__":
    main()