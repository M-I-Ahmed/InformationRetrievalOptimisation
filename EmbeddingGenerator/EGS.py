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
