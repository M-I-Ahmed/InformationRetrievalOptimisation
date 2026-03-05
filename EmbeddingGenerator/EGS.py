# This is the script used to create the embeddings for the descriptions of the nodes retrieved from neo4j. 
# As it is for testing and optimisation purposes, the script has to be general enough to be used for different embedding models.

#The models that will be tested are:
# Baseline (lexical matching)
#  - BM 25

# Semantic matching 
#  - OpenAI embedding model (text-embedding-3-small) - Small and large to be tested to see the difference in performance and computational cost.
#  - OpenAI embedding model (text-embedding-3-large)
#  - Qwen3-embedding-8B
#  - BAAI BGE-M3
#  - nvidia-llama-embed-nemotron-8b
#  - Jina Embeddings v3
#  - ManuBERT
#  - SciDeBERTa
#  - ColBERT v2
#  - E5-LARGE
#  - llama-3-8b
#  - Gemma-2B

# code structure:
# 1. Load the data from neo4j - Retrieve all the app IDs and their metadescriptions and store them in a dataframe.
# 2. Run the model and create the embeddings for the descriptions of the nodes retrieved from neo4j. 
# 3. Store the embeddings back on the node with the id - the naming convention should be "embedding_model_name_embedding" (e.g. "text-embedding-3-small_embedding") to avoid confusion and to be able to easily retrieve the embeddings later on for the retrieval process.

"""
Embedding Generator Script (EGS)

Goal:
1) Load node descriptions from Neo4j.
2) Embed them using a selected model.
3) Store embeddings back on the nodes using a consistent property name.

Notes:
- API keys must be kept outside of code. Use a .env file (ignored by git).
- This script is intentionally model-agnostic: you choose the provider and model at runtime.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

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


@dataclass
class Neo4jConfig:
	uri: str
	username: str
	password: str
	database: Optional[str]


@dataclass
class EmbeddingJobConfig:
	model: str
	provider: str
	description_property: str
	node_label: Optional[str]
	embedding_property: Optional[str]
	batch_size: int
	limit: Optional[int]


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
	base = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name.strip())
	base = base.strip("_") or "embedding"
	return f"{base}_embedding"


def build_read_query(description_property: str, node_label: Optional[str]) -> str:
	"""Build the query that pulls text to embed from Neo4j."""
	label_clause = f":{node_label}" if node_label else ""
	return (
		f"MATCH (n{label_clause}) "
		f"WHERE n[$desc_prop] IS NOT NULL "
		f"RETURN id(n) AS node_id, n[$desc_prop] AS text"
	)


def build_write_query(embedding_property: str) -> str:
	"""Build the query that stores embeddings back on each node."""
	if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", embedding_property):
		raise ValueError(
			"Embedding property name is invalid. Use only letters, numbers, and underscores, "
			"and do not start with a number."
		)
	return (
		"UNWIND $rows AS row "
		"MATCH (n) WHERE id(n) = row.node_id "
		f"SET n.{embedding_property} = row.embedding"
	)


def fetch_nodes(driver, database: Optional[str], read_query: str, desc_prop: str, limit: Optional[int]) -> List[dict]:
	"""Fetch node ids and description text from Neo4j."""
	params = {"desc_prop": desc_prop}
	if limit is not None:
		read_query = f"{read_query} LIMIT $limit"
		params["limit"] = limit

	with driver.session(database=database) as session:
		result = session.run(read_query, params)
		return [record.data() for record in result]


def chunked(iterable: List[dict], size: int) -> Iterable[List[dict]]:
	"""Yield fixed-size chunks from a list (for batching)."""
	for i in range(0, len(iterable), size):
		yield iterable[i : i + size]


def embed_texts_openai(model: str, texts: List[str]) -> List[List[float]]:
	"""Embed text using OpenAI's embedding API."""
	if OpenAI is None:
		raise ImportError("openai package is not installed. Install it to use OpenAI embeddings.")

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY is missing in your .env file.")

	client = OpenAI(api_key=api_key)
	response = client.embeddings.create(model=model, input=texts)
	return [item.embedding for item in response.data]


def embed_texts_sentence_transformers(model: str, texts: List[str], batch_size: int) -> List[List[float]]:
	"""Embed text using Hugging Face sentence-transformers models."""
	if SentenceTransformer is None:
		raise ImportError(
			"sentence-transformers package is not installed. Install it to use Hugging Face models."
		)

	encoder = SentenceTransformer(model)
	embeddings = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
	return [embedding.tolist() for embedding in embeddings]


def embed_texts(provider: str, model: str, texts: List[str], batch_size: int) -> List[List[float]]:
	"""Route embedding requests to the chosen provider."""
	if provider == "openai":
		return embed_texts_openai(model, texts)
	if provider == "hf":
		return embed_texts_sentence_transformers(model, texts, batch_size)
	raise ValueError("Unsupported provider. Use 'openai' or 'hf'.")


def write_embeddings(driver, database: Optional[str], write_query: str, rows: List[dict]) -> None:
	"""Write embeddings back to Neo4j in a single batch."""
	with driver.session(database=database) as session:
		session.run(write_query, {"rows": rows})


def parse_args() -> EmbeddingJobConfig:
	"""Parse CLI arguments to configure the embedding job."""
	parser = argparse.ArgumentParser(description="Generate and store embeddings for Neo4j nodes.")
	parser.add_argument("--provider", choices=["openai", "hf"], required=True, help="Embedding provider.")
	parser.add_argument("--model", required=True, help="Model name (e.g., text-embedding-3-small or BAAI/bge-m3).")
	parser.add_argument(
		"--description-property",
		default="App_Metadescription",
		help="Node property containing the text to embed.",
	)
	parser.add_argument("--node-label", default=None, help="Optional label to limit nodes.")
	parser.add_argument(
		"--embedding-property",
		default=None,
		help="Override embedding property name (default is <model>_embedding).",
	)
	parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding and writing.")
	parser.add_argument("--limit", type=int, default=None, help="Limit number of nodes for testing.")

	args = parser.parse_args()
	return EmbeddingJobConfig(
		model=args.model,
		provider=args.provider,
		description_property=args.description_property,
		node_label=args.node_label,
		embedding_property=args.embedding_property,
		batch_size=args.batch_size,
		limit=args.limit,
	)


def main() -> None:
	"""Run the full embedding pipeline: read -> embed -> write."""
	config = load_config_from_env()
	job = parse_args()

	# Determine which node property will store the embeddings.
	embedding_property = job.embedding_property or sanitize_property_name(job.model)

	# Build Cypher queries for reading and writing.
	read_query = build_read_query(job.description_property, job.node_label)
	write_query = build_write_query(embedding_property)

	driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
	try:
		# Step 1: Fetch nodes and their description text.
		nodes = fetch_nodes(driver, config.database, read_query, job.description_property, job.limit)
		if not nodes:
			print("No nodes found with the specified description property.")
			return

		# Step 2: Embed in batches to avoid excessive memory usage.
		for batch in chunked(nodes, job.batch_size):
			texts = [item["text"] for item in batch]
			embeddings = embed_texts(job.provider, job.model, texts, job.batch_size)

			# Step 3: Write embeddings back to the corresponding nodes.
			rows = [
				{"node_id": item["node_id"], "embedding": embedding}
				for item, embedding in zip(batch, embeddings)
			]
			write_embeddings(driver, config.database, write_query, rows)

		print(f"Stored embeddings in property '{embedding_property}' for {len(nodes)} nodes.")
	finally:
		driver.close()


if __name__ == "__main__":
	main()
