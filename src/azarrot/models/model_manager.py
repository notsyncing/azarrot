import json
import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import yaml

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.ipex_llm_backend import BACKEND_ID_IPEX_LLM
from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.common_data import IPEXLLMModelConfig, Model, ModelPreset
from azarrot.config import ServerConfig
from azarrot.models.chat_templates import DEFAULT_LOCALE

DEFAULT_MODEL_PRESET = ModelPreset(
    preferred_locale=DEFAULT_LOCALE,
    supports_tool_calling=False
)

DEFAULT_MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen2": ModelPreset(
        preferred_locale=DEFAULT_LOCALE,
        supports_tool_calling=True
    )
}


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

    def __determine_model_generation_variant(self, model_path: Path) -> str:
        hf_config_file = model_path / "config.json"

        if hf_config_file.exists():
            try:
                with hf_config_file.open() as f:
                    hf_config = json.load(f)

                hf_model_archs: list[str] = hf_config.get("architectures", [])

                if "InternVLChatModel" in hf_model_archs:
                    return "internvl2"
                elif "Qwen2ForCausalLM" in hf_model_archs:
                    return "qwen2"
            except:
                self._log.warn("Failed to parse config %s as JSON", hf_config_file)

        return "normal"

    def __parse_model_file(self, file: Path) -> Model:
        with file.open() as f:
            model_info = yaml.safe_load(f)
            model_path = self._config.models_dir / Path(model_info["path"])
            model_backend = model_info.get("backend", BACKEND_ID_OPENVINO)

            model_generation_variant=model_info.get(
                "generation_variant",
                self.__determine_model_generation_variant(model_path)
            )

            ipex_llm = None

            if model_backend == BACKEND_ID_IPEX_LLM:
                ipex_llm_config = model_info.get("ipex_llm", {})

                ipex_llm = IPEXLLMModelConfig(
                    use_cache=ipex_llm_config.get("use_cache", False)
                )

            model_preset_data = model_info.get("preset", None)
            model_preset: ModelPreset

            if model_preset_data is not None:
                model_preset = ModelPreset(
                    preferred_locale=model_preset_data.get("preferred_locale", None),
                    supports_tool_calling=model_preset_data.get("support_tool_calling", False),
                )
            else:
                model_preset = DEFAULT_MODEL_PRESETS.get(model_generation_variant, DEFAULT_MODEL_PRESET)

            return Model(
                id=model_info["id"],
                backend=model_backend,
                path=model_path,
                task=model_info["task"],

                generation_variant=model_generation_variant,
                preset=model_preset,

                ipex_llm=ipex_llm,

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
