# This code is for testing the Information Retrieval Optimization algorithms.
# It should loop through the test queries and calculate the precision@K, recall@K , and F2 and nDCG score for each embedding model.
# At the end it should plot the results in a bar chart for each metric.

#The code should be structured as follows:
# 1. Load the test queries and relevance judgments.
# 2. Loop through the test queries and calculate the precision@K, recall@K , and F2 and nDCG score for each embedding model.
# 3. Store the results in a dictionary.
# 4. Plot the results in a bar chart for each metric.

# The embedding models to be tested are:
#  - ManuBERT
#  - BAAI BGE-M3
#  - ColBERT v2
#  - E5-LARGE
#  - nvidia-llama-embed-nemotron-8b
#  - OpenAI embedding model (text-embedding-3-large)
#  - OpenAI embedding model (text-embedding-3-small)
#  - Qwen3-embedding-8B

#The golden queries and results are srored in a JSON file located here:

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from dotenv import load_dotenv

try:
	from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
	OpenAI = None  # type: ignore[assignment]

try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
	SentenceTransformer = None  # type: ignore[assignment]

try:
	from transformers import AutoTokenizer, AutoModel
	import torch
	import torch.nn.functional as torch_f
except Exception:  # pragma: no cover - optional dependency
	AutoTokenizer = None  # type: ignore[assignment]
	AutoModel = None  # type: ignore[assignment]
	torch = None  # type: ignore[assignment]
	torch_f = None  # type: ignore[assignment]


@dataclass
class Neo4jConfig:
	uri: str
	username: str
	password: str
	database: Optional[str]


@dataclass
class ModelSpec:
	name: str
	provider: str


_HF_ENCODER = None
_HF_MODEL_NAME = None
_HF_TRUST_REMOTE_CODE = None

_COLBERT_TOKENIZER = None
_COLBERT_MODEL = None
_COLBERT_MODEL_NAME = None
_COLBERT_DEVICE = None


def load_config_from_env() -> Neo4jConfig:
	"""Load Neo4j connection settings from .env or the environment."""
	load_dotenv()
	uri = os.getenv("NEO4J_URI")
	username = os.getenv("NEO4J_USERNAME")
	password = os.getenv("NEO4J_PASSWORD")
	database = os.getenv("NEO4J_DATABASE")

	if not uri or not username or not password:
		raise ValueError(
			"Missing Neo4j connection details. Please set NEO4J_URI, "
			"NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file."
		)

	return Neo4jConfig(uri=uri, username=username, password=password, database=database)


def sanitize_property_name(model_name: str) -> str:
	"""Convert a model name into a safe Neo4j property name."""
	base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in model_name.strip())
	base = base.strip("_") or "embedding"
	return f"{base}_embedding"


def embedding_property_for_model(model: ModelSpec) -> str:
	"""Match the property naming rules used by the embedding generator."""
	base = sanitize_property_name(model.name)
	if model.provider == "openai":
		return f"openai_{base}"
	return base


def load_golden_queries(path: str) -> List[dict]:
	"""Load the golden queries and expected app ids from JSON."""
	with open(path, "r", encoding="utf-8") as handle:
		payload = json.load(handle)
	return payload.get("golden_set", [])


def fetch_node_embeddings(driver, database: Optional[str], node_label: str, embedding_property: str) -> Dict[str, List[float]]:
	"""Fetch stored embeddings from Neo4j, keyed by app id."""
	query = (
		f"MATCH (n:{node_label}) "
		f"WHERE n.{embedding_property} IS NOT NULL "
		"RETURN coalesce(n.App_ID, n.id, n.App_Name, n.name) AS app_id, "
		f"n.{embedding_property} AS embedding"
	)
	with driver.session(database=database) as session:
		result = session.run(query)
		embeddings: Dict[str, List[float]] = {}
		for record in result:
			app_id = record.get("app_id")
			embedding = record.get("embedding")
			if app_id and embedding is not None:
				embeddings[str(app_id)] = embedding
		return embeddings


def embed_query_openai(model: str, text: str) -> List[float]:
	"""Embed a query string with OpenAI."""
	if OpenAI is None:
		raise ImportError("openai package is not installed. Install it to use OpenAI embeddings.")

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY is missing in your .env file.")

	client = OpenAI(api_key=api_key)
	response = client.embeddings.create(model=model, input=[text])
	return response.data[0].embedding


def embed_query_hf(model: str, text: str, trust_remote_code: bool) -> List[float]:
	"""Embed a query string with sentence-transformers."""
	if SentenceTransformer is None:
		raise ImportError("sentence-transformers package is not installed.")

	global _HF_ENCODER, _HF_MODEL_NAME, _HF_TRUST_REMOTE_CODE
	if _HF_ENCODER is None or _HF_MODEL_NAME != model or _HF_TRUST_REMOTE_CODE != trust_remote_code:
		_HF_ENCODER = SentenceTransformer(model, trust_remote_code=trust_remote_code)
		_HF_MODEL_NAME = model
		_HF_TRUST_REMOTE_CODE = trust_remote_code

	embedding = _HF_ENCODER.encode([text], normalize_embeddings=True)[0]
	return embedding.tolist()


def embed_query_colbert(model: str, text: str, pool: str) -> List[float]:
	"""Embed a query string using ColBERT token embeddings with pooling."""
	if AutoTokenizer is None or AutoModel is None or torch is None or torch_f is None:
		raise ImportError("transformers/torch are not installed. Install them to use ColBERT models.")

	global _COLBERT_TOKENIZER, _COLBERT_MODEL, _COLBERT_MODEL_NAME, _COLBERT_DEVICE
	if _COLBERT_MODEL is None or _COLBERT_MODEL_NAME != model:
		_COLBERT_TOKENIZER = AutoTokenizer.from_pretrained(model)
		_COLBERT_MODEL = AutoModel.from_pretrained(model)
		_COLBERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		_COLBERT_MODEL.to(_COLBERT_DEVICE)
		_COLBERT_MODEL.eval()
		_COLBERT_MODEL_NAME = model

	inputs = _COLBERT_TOKENIZER(
		[text],
		padding=True,
		truncation=True,
		max_length=512,
		return_tensors="pt",
	)
	inputs = {key: value.to(_COLBERT_DEVICE) for key, value in inputs.items()}

	with torch.no_grad():
		outputs = _COLBERT_MODEL(**inputs, return_dict=True)
		last_hidden = outputs.last_hidden_state[0]

	mask = inputs.get("attention_mask")[0].bool()
	tokens = last_hidden[mask]
	tokens = torch_f.normalize(tokens, p=2, dim=1)

	pool = pool.lower()
	if pool == "cls":
		pooled = tokens[0]
	elif pool == "max":
		pooled = tokens.max(dim=0).values
	else:
		pooled = tokens.mean(dim=0)

	return pooled.cpu().tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
	"""Compute cosine similarity between two vectors."""
	vec_a = np.array(a, dtype=np.float32)
	vec_b = np.array(b, dtype=np.float32)
	norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
	if norm == 0.0:
		return 0.0
	return float(np.dot(vec_a, vec_b) / norm)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
	"""Compute precision@k for a single query."""
	retrieved_k = retrieved[:k]
	hits = sum(1 for item in retrieved_k if item in relevant)
	return hits / max(k, 1)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
	"""Compute recall@k for a single query."""
	retrieved_k = retrieved[:k]
	hits = sum(1 for item in retrieved_k if item in relevant)
	return hits / max(len(relevant), 1)


def f2_score(precision: float, recall: float, beta: float = 2.0) -> float:
	"""Compute F2 score."""
	denom = (beta ** 2) * precision + recall
	if denom == 0.0:
		return 0.0
	return (1 + beta ** 2) * (precision * recall) / denom


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
	"""Compute nDCG@k for a single query (binary relevance)."""
	def dcg(items: List[str]) -> float:
		score = 0.0
		for idx, item in enumerate(items):
			rel = 1.0 if item in relevant else 0.0
			score += rel / math.log2(idx + 2)
		return score

	retrieved_k = retrieved[:k]
	dcg_value = dcg(retrieved_k)
	ideal_k = min(k, len(relevant))
	ideal_list = list(relevant)[:ideal_k]
	idcg_value = dcg(ideal_list)
	if idcg_value == 0.0:
		return 0.0
	return dcg_value / idcg_value


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for evaluation settings."""
	default_query_path = os.path.join(
		os.path.dirname(__file__),
		"..",
		"Queries&Truths",
		"GQ&T.json",
	)
	parser = argparse.ArgumentParser(description="Evaluate retrieval metrics for embedding models.")
	parser.add_argument("--k", type=int, default=5, help="Cutoff K for @K metrics.")
	parser.add_argument("--node-label", default="App", help="Neo4j node label containing apps.")
	parser.add_argument("--query-file", default=default_query_path, help="Path to golden queries JSON file.")
	parser.add_argument(
		"--trust-remote-code",
		action="store_true",
		help="Allow Hugging Face models that require custom code.",
	)
	parser.add_argument(
		"--colbert-pool",
		choices=["mean", "cls", "max"],
		default="mean",
		help="Pooling strategy for ColBERT token embeddings.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run evaluation over all specified models and plot metric summaries."""
	args = parse_args()
	config = load_config_from_env()
	queries = load_golden_queries(args.query_file)

	models = [
		ModelSpec("akumar33/ManuBERT", "hf"),
		ModelSpec("BAAI/bge-m3", "hf"),
		ModelSpec("colbert-ir/colbertv2.0", "colbert"),
		ModelSpec("intfloat/multilingual-e5-large", "hf"),
		ModelSpec("nvidia/llama-embed-nemotron-8b", "hf"),
		ModelSpec("text-embedding-3-large", "openai"),
		ModelSpec("text-embedding-3-small", "openai"),
		ModelSpec("Qwen/Qwen3-Embedding-8B", "hf"),
	]

	results: Dict[str, Dict[str, float]] = {}

	with GraphDatabase.driver(config.uri, auth=(config.username, config.password)) as driver:
		for model in models:
			property_name = embedding_property_for_model(model)
			node_embeddings = fetch_node_embeddings(driver, config.database, args.node_label, property_name)
			if not node_embeddings:
				print(f"No embeddings found for model '{model.name}'. Skipping.")
				continue

			precision_scores: List[float] = []
			recall_scores: List[float] = []
			f2_scores: List[float] = []
			ndcg_scores: List[float] = []

			for query in queries:
				query_text = query.get("query_text", "")
				expected = query.get("expected_app_ids", [])
				if not query_text or not expected:
					continue

				if model.provider == "openai":
					query_embedding = embed_query_openai(model.name, query_text)
				elif model.provider == "hf":
					query_embedding = embed_query_hf(model.name, query_text, args.trust_remote_code)
				else:
					query_embedding = embed_query_colbert(model.name, query_text, args.colbert_pool)

				# Score all apps by cosine similarity.
				scored = [
					(app_id, cosine_similarity(query_embedding, embedding))
					for app_id, embedding in node_embeddings.items()
				]
				ranked = [app_id for app_id, _ in sorted(scored, key=lambda item: item[1], reverse=True)]

				p = precision_at_k(ranked, expected, args.k)
				r = recall_at_k(ranked, expected, args.k)
				f2 = f2_score(p, r)
				ndcg = ndcg_at_k(ranked, expected, args.k)

				precision_scores.append(p)
				recall_scores.append(r)
				f2_scores.append(f2)
				ndcg_scores.append(ndcg)

			results[model.name] = {
				"precision@k": float(np.mean(precision_scores) if precision_scores else 0.0),
				"recall@k": float(np.mean(recall_scores) if recall_scores else 0.0),
				"f2": float(np.mean(f2_scores) if f2_scores else 0.0),
				"ndcg@k": float(np.mean(ndcg_scores) if ndcg_scores else 0.0),
			}

	# Plot metrics in bar charts for quick comparison.
	if not results:
		print("No results to plot.")
		return

	model_names = list(results.keys())
	metrics = ["precision@k", "recall@k", "f2", "ndcg@k"]
	for metric in metrics:
		values = [results[name][metric] for name in model_names]
		plt.figure(figsize=(10, 4))
		plt.bar(model_names, values, color="#3A6EA5")
		plt.title(f"{metric} across models")
		plt.ylabel(metric)
		plt.xticks(rotation=45, ha="right")
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	main()



