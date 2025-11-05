from matching import MatchingEngine
from log_reg_library import NLPMinorPredictor

def main() -> None:
    classificator = NLPMinorPredictor(
        model_path="nlp_model_logreg_embeddings.pkl",
        label_encoder_path="label_encoder_log_reg.pkl",
        embed_model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    classificator.predict_csv
    (
        csv_path="./DATASETS/opdrachten_te_classificeren.csv",
        output_path="./DATASETS/voorspellingen_output_log_reg.csv"
    )

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