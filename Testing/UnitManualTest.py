# This code will be used to conduct manual tests for the different models.
# The purpose is to give more transparency into the metrics rather than just bulk graph generation.

# 4 different fucntions for each test
# Each fucntion will carry out test for different values of k 
# The overall script will only run the test of one model at a time. 

"""
Manual recall testing for multiple models and K values.

This script focuses on recall@K for K in [1, 3, 5, 10, 15] and writes
the results to a CSV file for external plotting (e.g., Excel).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from xml.parsers.expat import model

import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

try:
	from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
	OpenAI = None  # type: ignore[assignment]

try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
	SentenceTransformer = None  # type: ignore[assignment]

try:
	import voyageai
except Exception:  # pragma: no cover - optional dependency
	voyageai = None  # type: ignore[assignment]

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
	if model.name == "voyage-code-3":
		return "voyage_code_3"
	if model.name == "voyage-4":
		return "voyage_4"
	if model.name == "Qodo/Qodo-Embed-1-1.5B":
		return "Qodo_Qodo_Embed_1_1_5B_embedding"
	if model.name == "Qodo/Qodo-Embed-1-7B":
		return "Qodo_Qodo_Embed_1_7B_embedding"
	if model.name == "LightOnAI/LateOn-Code":
		return "lightonai_LateOn_Code_embedding"
	if model.name == "Qwen/Qwen3-embedding-0.6B":
		return "Qwen_Qwen3_Embedding_0_6B_embedding"
	if model.name == "Qwen/Qwen3-embedding-8B":
		return "Qwen_Qwen3_Embedding_8B_embedding"
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
		
def fetch_text_BM25(driver, database: Optional[str], node_label: str, text_property: str) -> Dict[str, str]:
	query = (
		f"MATCH (n:{node_label}) "
		f"WHERE n.{text_property} IS NOT NULL "
		"RETURN coalesce(n.App_ID, n.id, n.App_Name, n.name) AS app_id, "
		f"n.{text_property} AS description"
	)
	with driver.session(database=database) as session:
		result = session.run(query)
		texts: Dict[str, str] = {}
		for record in result:
			app_id = record.get("app_id")
			description = record.get("description")
			if app_id and description is not None:
				texts[str(app_id)] = description
		return texts


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


def embed_query_voyage(model: str, text: str) -> List[float]:
	"""Embed a query string with Voyage AI."""
	if voyageai is None:
		raise ImportError("voyageai package is not installed. Install it to use Voyage embeddings.")

	api_key = os.getenv("VOYAGE_API_KEY")
	if not api_key:
		raise ValueError("VOYAGE_API_KEY is missing in your .env file.")

	client = voyageai.Client(api_key=api_key)
	response = client.embed([text], model=model)
	return response.embeddings[0]


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



def simple_tokenize(text: str) -> List[str]:
    """A very basic tokenizer that splits on whitespace and lowercases."""
    return text.lower().split()




def cosine_similarity(a: List[float], b: List[float]) -> float:
	"""Compute cosine similarity between two vectors."""
	vec_a = np.array(a, dtype=np.float32)
	vec_b = np.array(b, dtype=np.float32)
	norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
	if norm == 0.0:
		return 0.0
	return float(np.dot(vec_a, vec_b) / norm)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
	"""Compute recall@k for a single query."""
	retrieved_k = retrieved[:k]
	hits = sum(1 for item in retrieved_k if item in relevant)
	return hits / max(len(relevant), 1)


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for manual recall testing."""
	default_query_path = os.path.join(
		os.path.dirname(__file__),
		"..",
		"Queries&Truths",
		"GQ&T.json",
	)
	parser = argparse.ArgumentParser(description="Manual recall@K evaluation for embedding models.")
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
	parser.add_argument(
		"--output",
		default="manual_recall_results.csv",
		help="Path to CSV output file.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-query recall details.",
	)
	parser.add_argument(
		"--bm25-only",
		action="store_true",
		help="Run only the BM25 baseline (skip embedding models).",
	)
	return parser.parse_args()


def main() -> None:
	"""Run recall@K evaluation for each model and store results to CSV."""
	args = parse_args()
	config = load_config_from_env()
	queries = load_golden_queries(args.query_file)
	ks = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	text_property = "App_Metadescription"

	models = [
		ModelSpec("akumar33/ManuBERT", "hf"),
		ModelSpec("BAAI/bge-m3", "hf"),
		ModelSpec("colbert-ir/colbertv2.0", "colbert"),
		ModelSpec("intfloat/multilingual-e5-large", "hf"),
		ModelSpec("nvidia/llama-embed-nemotron-8b", "hf"),
		ModelSpec("text-embedding-3-large", "openai"),
		ModelSpec("text-embedding-3-small", "openai"),
		ModelSpec("Qwen/Qwen3-embedding-8B", "hf"),
		ModelSpec("LightOnAI/LateOn-Code", "hf"),
		ModelSpec("voyageai/voyage-4-nano", "hf"),
		ModelSpec("voyage-code-3", "voyage"),
		ModelSpec("voyage-4", "voyage"),
		#ModelSpec("Qodo/Qodo-Embed-1-1.5B", "hf"),
        #ModelSpec("Qodo/Qodo-Embed-1-7B", "hf"),    
		ModelSpec("Qwen/Qwen3-embedding-0.6B", "hf"),
	]

	results: Dict[str, Dict[int, float]] = {}

	with GraphDatabase.driver(config.uri, auth=(config.username, config.password)) as driver:
		text_by_id = fetch_text_BM25(driver, config.database, args.node_label, text_property)
		app_ids = list(text_by_id.keys())
		docs = [text_by_id[app_id] for app_id in app_ids]
		tokenized_docs = [simple_tokenize(doc) for doc in docs]
		bm25 = BM25Okapi(tokenized_docs)
		if not args.bm25_only:
			for model in models:
				property_name = embedding_property_for_model(model)
				node_embeddings = fetch_node_embeddings(driver, config.database, args.node_label, property_name)
				if not node_embeddings:
					print(f"No embeddings found for model '{model.name}'. Skipping.")
					continue

				per_k_scores: Dict[int, List[float]] = {k: [] for k in ks}
				for query in queries:
					query_text = query.get("query_text", "")
					expected = query.get("expected_app_ids", [])
					if not query_text or not expected:
						continue

					if model.provider == "openai":
						query_embedding = embed_query_openai(model.name, query_text)
					elif model.provider == "voyage":
						query_embedding = embed_query_voyage(model.name, query_text)
					elif model.provider == "hf":
						query_embedding = embed_query_hf(model.name, query_text, args.trust_remote_code)
					else:
						query_embedding = embed_query_colbert(model.name, query_text, args.colbert_pool)

					scored = [
						(app_id, cosine_similarity(query_embedding, embedding))
						for app_id, embedding in node_embeddings.items()
					]
					ranked = [app_id for app_id, _ in sorted(scored, key=lambda item: item[1], reverse=True)]

					if args.verbose:
						print(f"Model: {model.name} | Query: {query_text}")

					for k in ks:
						recall = recall_at_k(ranked, expected, k)
						per_k_scores[k].append(recall)
						if args.verbose:
							print(f"  recall@{k}: {recall:.3f}")

				results[model.name] = {k: float(np.mean(per_k_scores[k]) if per_k_scores[k] else 0.0) for k in ks}

		bm25_per_k: Dict[int, List[float]] = {k: [] for k in ks}
		for query in queries:
			query_text = query.get("query_text", "")
			expected = query.get("expected_app_ids", [])
			if not query_text or not expected:
				continue

			query_tokens = simple_tokenize(query_text)
			scores = bm25.get_scores(query_tokens)
			ranked_indices = scores.argsort()[::-1]
			ranked_app_ids = [app_ids[i] for i in ranked_indices]

			for k in ks:
				recall = recall_at_k(ranked_app_ids, expected, k)
				bm25_per_k[k].append(recall)

		results["BM25"] = {k: float(np.mean(bm25_per_k[k]) if bm25_per_k[k] else 0.0) for k in ks}

	if not results:
		print("No results to write.")
		return
	
	with open(args.output, "w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["Model"] + [str(k) for k in ks])
		for model_name, model_results in results.items():
			row = [model_name] + [f"{model_results.get(k, 0.0):.6f}" for k in ks]
			writer.writerow(row)

	print(f"Recall results saved to {args.output}")


if __name__ == "__main__":
	main()