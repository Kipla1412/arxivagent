from __future__ import annotations
import json
from pydantic import BaseModel, Field
from typing import Literal
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from config.config import Config
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingParams(BaseModel):

    text: str = Field(..., description="Text to convert into embedding")

    task: Literal["retrieval.query", "retrieval.passage"] = Field(
        default="retrieval.query",
        description="Type of embedding task"
    )


class SentenceTransformerEmbeddingTool(Tool):

    name = "sentence_transformer_embedding"
    description = "Generates 384-dimension embedding using sentence-transformers/all-MiniLM-L6-v2"
    kind = ToolKind.NETWORK

    def __init__(self, config: Config):
        super().__init__(config)

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    @property
    def schema(self):
        return SentenceEmbeddingParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        try:
            params = SentenceEmbeddingParams(**invocation.params)

            embedding = self.model.encode(params.text).tolist()

            return ToolResult.success_result(
                output=json.dumps({
                    "vector": embedding
                }),
                metadata={
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            )

        except Exception as e:
            return ToolResult.error_result(
                error=f"SentenceTransformer embedding failed: {str(e)}"
            )