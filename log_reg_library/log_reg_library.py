"""Utilities for running the logistic-regression embedding classifier.

This module factors the original script into reusable building blocks so the
prediction workflow can be imported from other modules or notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


@dataclass
class LogRegEmbeddingClassifier:
	"""Wraps the trained logistic-regression pipeline and label encoder."""

	model: object
	label_encoder: object
	embed_model: SentenceTransformer

	@classmethod
	def from_files(
		cls,
		model_path: Path | str,
		label_encoder_path: Path | str,
		embedding_model_name: str = DEFAULT_EMBED_MODEL,
		*,
		embed_model: Optional[SentenceTransformer] = None,
	) -> "LogRegEmbeddingClassifier":
		"""Construct an instance by loading artefacts from disk."""

		model = joblib.load(Path(model_path))
		label_encoder = joblib.load(Path(label_encoder_path))
		embedder = embed_model or SentenceTransformer(embedding_model_name)
		return cls(model=model, label_encoder=label_encoder, embed_model=embedder)

	def encode(self, descriptions: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
		"""Generate sentence embeddings for the provided descriptions."""

		if not descriptions:
			return np.empty((0, self.embed_model.get_sentence_embedding_dimension()))
		return self.embed_model.encode(
			list(descriptions), show_progress_bar=show_progress, convert_to_numpy=True
		)

	def predict_descriptions(
		self,
		descriptions: Sequence[str],
		*,
		show_progress: bool = False,
	) -> tuple[List[str], np.ndarray]:
		"""Predict labels and return per-sample probability distributions."""

		embeddings = self.encode(descriptions, show_progress=show_progress)
		if embeddings.size == 0:
			return [], np.empty((0, 0))

		pred_indices = self.model.predict(embeddings)
		predicted_labels = self.label_encoder.inverse_transform(pred_indices)
		probabilities = self.model.predict_proba(embeddings)
		return list(predicted_labels), probabilities

	def predict_single(self, description: str) -> tuple[str, float]:
		"""Predict a label for one description and return (label, confidence)."""

		labels, probas = self.predict_descriptions([description])
		if not labels:
			raise ValueError("Geen omschrijving opgegeven voor voorspelling.")
		confidence = float(probas[0].max())
		return labels[0], confidence

	def annotate_dataframe(
		self,
		df: pd.DataFrame,
		*,
		description_column: str = "omschrijving",
		prediction_column: str = "voorspeld_onderwerp",
		confidence_column: str = "zekerheid_%",
		fill_column: Optional[str] = None,
		show_progress: bool = False,
	) -> pd.DataFrame:
		"""Return a copy of *df* with predictions and confidence columns added."""

		if description_column not in df.columns:
			raise ValueError(f"CSV mist verplichte kolom '{description_column}'.")

		descriptions = df[description_column].astype(str).tolist()
		labels, probas = self.predict_descriptions(descriptions, show_progress=show_progress)
		scores = probas.max(axis=1) if probas.size else np.array([])

		enriched = df.copy()
		enriched[prediction_column] = labels
		enriched[confidence_column] = np.round(scores * 100, 2)
		if fill_column:
			enriched[fill_column] = enriched[prediction_column]
		return enriched


def load_classifier(
	model_path: Path | str,
	label_encoder_path: Path | str,
	embedding_model_name: str = DEFAULT_EMBED_MODEL,
	*,
	embed_model: Optional[SentenceTransformer] = None,
) -> LogRegEmbeddingClassifier:
	"""Helper that mirrors the original script's artefact loading logic."""

	return LogRegEmbeddingClassifier.from_files(
		model_path=model_path,
		label_encoder_path=label_encoder_path,
		embedding_model_name=embedding_model_name,
		embed_model=embed_model,
	)


def predict_to_csv(
	classifier: LogRegEmbeddingClassifier,
	csv_path: Path | str,
	*,
	output_path: Path | str = "voorspellingen_output_log_reg.csv",
	description_column: str = "omschrijving",
	show_progress: bool = True,
) -> pd.DataFrame:
	"""Load a CSV, annotate it with predictions, and persist the result."""

	df = pd.read_csv(csv_path)
	enriched = classifier.annotate_dataframe(
		df,
		description_column=description_column,
		show_progress=show_progress,
	)
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(output_path, index=False)
	return enriched


__all__ = [
	"DEFAULT_EMBED_MODEL",
	"LogRegEmbeddingClassifier",
	"load_classifier",
	"predict_to_csv",
]
