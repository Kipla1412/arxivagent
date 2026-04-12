from __future__ import annotations
import json
import httpx
from typing import Any, Literal, List
from pydantic import BaseModel, Field
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from config.config import Config
from knowledgebase.embedding import EmbeddingConnector

class JinaEmbeddingParams(BaseModel):
    
    text: str = Field(..., description="The text string to convert into a vector.")
    task: Literal["retrieval.query", "retrieval.passage"] = Field(
        default="retrieval.query", 
        description="Task type: 'retrieval.query' for searching, 'retrieval.passage' for saving data."
    )

class JinaEmbeddingTool(Tool):
    name = "jina_embedding"
    description = "MANDATORY first step. Converts query into vector for retrieval. MUST be called before search."
    kind = ToolKind.NETWORK


    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def schema(self) -> type[BaseModel]:
        return JinaEmbeddingParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = JinaEmbeddingParams(**invocation.params)
            
            # Use session-level connector if available
            session = getattr(self.config, "_session", None)

            if session:
                client = session.embedding_connector.connect()
            else:
                client = EmbeddingConnector(self.config).connect()

            response = await client.post(
                self.config.jina_api_url,
                json={
                    "model": self.config.jina_model,
                    "task": params.task,
                    "dimensions": self.config.jina_dimensions,
                    "input": [params.text]
                }
            )
            
            response.raise_for_status()
            data = response.json()
            vector = data["data"][0]["embedding"]
                
            return ToolResult.success_result(
                output=json.dumps({"vector": vector}),
                metadata={
                    "model": self.config.jina_model,
                    "dimensions": self.config.jina_dimensions,
                    "task": params.task
                }
            )

        except Exception as e:
            return ToolResult.error_result(error=f"Jina Embedding failed: {str(e)}")