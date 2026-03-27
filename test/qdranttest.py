from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://fst3-0.gcp.cloud.qdrant.io:6333", 
    api_key="",
)

print(qdrant_client.get_collections())