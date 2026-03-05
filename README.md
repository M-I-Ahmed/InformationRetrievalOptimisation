# InformationRetrievalOptimisation

The aim of this repository is to create a generalised way of evaluating the efficacy of different embedding models.

To do this, the inital step is to create the embeddings. The script for this is located in the EmbeddingGenerator file. 
After this, the set of queries and truths should be listed in the Queries&Truths file.

The scripts for running the tests are located within the Testing folder. 
The evaluated metrics are as follows:
- Recall@K
- Precision@K
- F2 Score
- nDCG



Embedding Generator Script
1) Retrieve descriptions from neo4j.
2) Embed them.
3) Store the embedding on the graph node.

The aim is to create a generalised script that can handle multiple different embedding models with minimal intervention.


