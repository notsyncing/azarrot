import logging
from dataclasses import dataclass
from logging import Logger
from typing import override

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from azarrot.common_data import Model, ReranksGenerationRequest
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.base import Frontend
from azarrot.models.model_manager import ModelManager


class JinaRerankRequest(BaseModel):
    model: str
    query: str
    top_n: int | None = None
    documents: list[str]


@dataclass
class JinaUsageInfo:
    total_tokens: int


@dataclass
class JinaRerankResultTextItem:
    text: str


@dataclass
class JinaRerankResultItem:
    index: int
    document: JinaRerankResultTextItem
    relevance_score: float


@dataclass
class JinaRerankResult:
    model: str
    usage: JinaUsageInfo
    results: list[JinaRerankResultItem]


class JinaFrontend(Frontend):
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
        return "Jina"

    def __init_routes(self) -> None:
        router = APIRouter()

        # Reranker API
        router.add_api_route("/v1/rerank", self.rerank, methods=["POST"])

        self._api.include_router(router, prefix="/jina")

    def __get_model(self, model_id: str) -> Model:
        model = self._model_manager.get_model(model_id)

        if model is None:
            raise ValueError(f"Requested model {model_id} is not loaded!")

        return model

    def rerank(self, request: JinaRerankRequest) -> JinaRerankResult:
        model = self.__get_model(request.model)

        results, gen_stats = self._backend_pipe.generate_reranks(
            model,
            ReranksGenerationRequest(
                model_id=model.id, query=request.query, documents=request.documents, max_count=request.top_n
            ),
        )

        self._log.info(gen_stats.to_stats_text())

        return JinaRerankResult(
            model=model.id,
            usage=JinaUsageInfo(total_tokens=gen_stats.total_tokens()),
            results=[
                JinaRerankResultItem(
                    index=r.input_index,
                    document=JinaRerankResultTextItem(text=request.documents[r.input_index]),
                    relevance_score=r.score,
                )
                for r in results
            ],
        )
