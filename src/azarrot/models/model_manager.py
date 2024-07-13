import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import yaml

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.common_data import Model
from azarrot.config import ServerConfig


class ModelManager:
    _log = logging.getLogger(__name__)
    _config: ServerConfig
    _backends: dict[str, BaseBackend]
    _models: ClassVar[list[Model]] = []

    def __init__(self, config: ServerConfig, backends: list[BaseBackend]) -> None:
        self._config = config

        self._backends = {}

        for backend in backends:
            self._backends[backend.id()] = backend
            self._log.info("Registered backend %s", backend.id())

        self.refresh_models()

    def __parse_model_file(self, file: Path) -> Model:
        with file.open() as f:
            model_info = yaml.safe_load(f)

            return Model(
                id=model_info["id"],
                backend=model_info.get("backend", BACKEND_ID_OPENVINO),
                path=self._config.models_dir / Path(model_info["path"]),
                task=model_info["task"],
                create_time=datetime.fromtimestamp(file.stat().st_mtime),
            )

    def refresh_models(self) -> None:
        new_models = [self.__parse_model_file(file) for file in self._config.models_dir.glob("*.model.yml")]

        for model in self._models:
            backend = self._backends[model.backend]

            if model not in new_models:
                backend.unload_model(model.id)
                self._models.remove(model)

        for model in new_models:
            backend = self._backends[model.backend]

            if model not in self._models:
                backend.load_model(model)
                self._models.append(model)

    def get_models(self) -> list[Model]:
        return self._models

    def get_model(self, model_id: str) -> Model | None:
        try:
            return next(m for m in self._models if m.id == model_id)
        except StopIteration:
            return None
