import logging
from dataclasses import dataclass
from logging import Logger
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing_extensions import override

from azarrot.common_data import Model, ReranksGenerationRequest
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.base import Frontend
from azarrot.models.model_manager import ModelManager


class CohereRerankRequest(BaseModel):
    model: str
    query: str
    top_n: int | None = None
    documents: list[str]
    return_documents: bool = False
    max_tokens_per_doc: int | None = None


@dataclass
class CohereRerankResultTextItem:
    text: str


@dataclass
class CohereRerankResultItem:
    index: int
    document: CohereRerankResultTextItem | None
    relevance_score: float


class CohereFrontend(Frontend):
    _log: Logger = logging.getLogger(__name__)

    _api: FastAPI
    _model_manager: ModelManager
    _backend_pipe: BackendPipe

    def __init__(self, api: FastAPI, model_manager: ModelManager, backend_pipe: BackendPipe) -> None:
        self._api = api
        self._model_manager = model_manager
        self._backend_pipe = backend_pipe

        self.__init_routes()

    @override
    def id(self) -> str:
        return "Cohere"

    def __init_routes(self) -> None:
        router = APIRouter()

        # Rerank (v2) API
        router.add_api_route("/v2/rerank", self.rerank, methods=["POST"])

        self._api.include_router(router, prefix="/cohere")

    def __get_model(self, model_id: str) -> Model:
        model = self._model_manager.get_model(model_id)

        if model is None:
            raise ValueError(f"Requested model {model_id} is not loaded!")

        return model

    def rerank(self, request: CohereRerankRequest) -> dict[str, Any]:
        model = self.__get_model(request.model)

        results, gen_stats = self._backend_pipe.generate_reranks(
            model,
            ReranksGenerationRequest(
                model_id=model.id,
                query=request.query,
                documents=request.documents,
                max_count=request.top_n,
                max_tokens_per_document=request.max_tokens_per_doc,
            ),
        )

        self._log.info(gen_stats.to_stats_text())

        return {
            "results": [
                CohereRerankResultItem(
                    index=r.input_index,
                    document=CohereRerankResultTextItem(text=request.documents[r.input_index])
                    if request.return_documents
                    else None,
                    relevance_score=r.score,
                )
                for r in results
            ],
            "id": str(uuid4()),
            "meta": {
                "api_version": {"version": 2, "is_experimental": False},
                "billed_units": {"search_units": gen_stats.completion_tokens},
            },
        }
