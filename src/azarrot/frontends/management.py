from fastapi import APIRouter, FastAPI

from azarrot.models.model_manager import ModelManager


class ManagementFrontend:
    _model_manager: ModelManager

    def __init__(self, model_manager: ModelManager, api: FastAPI) -> None:
        self._model_manager = model_manager
        router = APIRouter()
        router.add_api_route("/models/refresh", self.refresh_models, methods=["POST"])
        api.include_router(router)


    def refresh_models(self) -> None:
        self._model_manager.refresh_models()
