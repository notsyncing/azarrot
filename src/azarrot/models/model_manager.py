from datetime import datetime
from pathlib import Path
from typing import ClassVar

import yaml

from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.common_data import Model
from azarrot.config import ServerConfig


class ModelManager:
    _config: ServerConfig
    _backend: OpenVINOBackend
    _models: ClassVar[list[Model]] = []

    def __init__(self, config: ServerConfig, backend: OpenVINOBackend) -> None:
        self._config = config
        self._backend = backend
        self.refresh_models()


    def __parse_model_file(self, file: Path) -> Model:
        with file.open() as f:
            model_info = yaml.safe_load(f)

            return Model(
                id=model_info["id"],
                path=self._config.models_dir / Path(model_info["path"]),
                task=model_info["task"],
                create_time=datetime.fromtimestamp(file.stat().st_mtime)
            )


    def refresh_models(self) -> None:
        new_models = [
            self.__parse_model_file(file) for file in self._config.models_dir.glob("*.model.yml")
        ]

        for model in self._models:
            if model not in new_models:
                self._backend.unload_model(model.id)
                self._models.remove(model)

        for model in new_models:
            if model not in self._models:
                self._backend.load_model(model)
                self._models.append(model)


    def get_models(self) -> list[Model]:
        return self._models


    def get_model(self, model_id: str) -> Model:
        return next(m for m in self._models if m.id == model_id)
