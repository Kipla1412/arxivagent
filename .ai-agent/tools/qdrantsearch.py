from __future__ import annotations
import json
from typing import List
from pydantic import BaseModel, Field
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from config.config import Config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantSearchParams(BaseModel):

    vector: List[float] = Field(
        ..., description="Embedding vector used for similarity search"
    )

    limit: int = Field(
        default=5,
        description="Number of results to return"
    )


class QdrantSearchTool(Tool):

    name = "qdrant_search"
    description = "Searches Qdrant vector database for similar documents"
    kind = ToolKind.NETWORK

    def __init__(self, config: Config):
        super().__init__(config)

        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=10
        )

        self.collection = config.qdrant_collection

    @property
    def schema(self):
        return QdrantSearchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        try:
            params = QdrantSearchParams(**invocation.params)

            results = self.client.query_points(
                collection_name=self.collection,
                query=params.vector,
                limit=params.limit,
                score_threshold=0.3,
                with_payload=True,
                with_vectors=False
            )

            formatted = []

            for r in results.points:

                payload = r.payload or {}

                formatted.append({
                    "score": r.score,
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "arxiv_id": payload.get("arxiv_id"),
                    "section": payload.get("section")
                })

            return ToolResult.success_result(
                output=json.dumps(formatted, indent=2),
                metadata={
                    "result_count": len(formatted)
                }
            )

        except Exception as e:
            return ToolResult.error_result(
                error=f"Qdrant search failed: {str(e)}"
            )