import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import huggingface_hub
import yaml

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.ipex_llm_backend import BACKEND_ID_IPEX_LLM
from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.common_data import IPEXLLMModelConfig, Model, ModelPreset
from azarrot.config import ServerConfig
from azarrot.models.chat_templates import DEFAULT_LOCALE

DEFAULT_MODEL_PRESET = ModelPreset(
    preferred_locale=DEFAULT_LOCALE,  # type: ignore[arg-type]
    supports_tool_calling=False,
    enable_internal_tools=False,
)

DEFAULT_MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen2": ModelPreset(preferred_locale=DEFAULT_LOCALE, supports_tool_calling=True, enable_internal_tools=False)  # type: ignore[arg-type]
}

MODEL_PATH_HUGGINGFACE = "huggingface://"


class ModelManager:
    _log = logging.getLogger(__name__)
    _config: ServerConfig
    _backends: dict[str, BaseBackend]
    _models: dict[str, Model]
    _dynamic_loaded_models: dict[str, Model]

    def __init__(self, config: ServerConfig, backends: list[BaseBackend]) -> None:
        self._config = config

        self._models = {}
        self._dynamic_loaded_models = {}
        self._backends = {}

        for backend in backends:
            self._backends[backend.id()] = backend
            self._log.info("Registered backend %s", backend.id())

        self.refresh_models()

    def __determine_model_generation_variant(self, model_path: Path) -> Literal["normal", "internvl2", "qwen2"]:
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
                self._log.warning("Failed to parse config %s as JSON", hf_config_file)

        return "normal"

    def __parse_model_file(self, file: Path) -> Model:
        with file.open() as f:
            model_info = yaml.safe_load(f)

            raw_model_path = model_info["path"]
            model_path: Path

            if raw_model_path.startswith(MODEL_PATH_HUGGINGFACE):
                hf_model_id = raw_model_path[len(MODEL_PATH_HUGGINGFACE) :]
                hf_model_path = huggingface_hub.snapshot_download(hf_model_id)
                model_path = Path(hf_model_path)
            else:
                model_path = self._config.models_dir / Path(model_info["path"])

            model_backend = model_info.get("backend", BACKEND_ID_OPENVINO)

            model_generation_variant = model_info.get(
                "generation_variant", self.__determine_model_generation_variant(model_path)
            )

            ipex_llm = None

            if model_backend == BACKEND_ID_IPEX_LLM:
                ipex_llm_config = model_info.get("ipex_llm", {})

                ipex_llm = IPEXLLMModelConfig(use_cache=ipex_llm_config.get("use_cache", False))

            default_model_preset = DEFAULT_MODEL_PRESETS.get(model_generation_variant, DEFAULT_MODEL_PRESET)
            model_preset_data = model_info.get("preset", None)
            model_preset: ModelPreset

            if model_preset_data is not None:
                model_preset = ModelPreset(
                    preferred_locale=model_preset_data.get("preferred_locale", default_model_preset.preferred_locale),
                    supports_tool_calling=model_preset_data.get(
                        "support_tool_calling", default_model_preset.supports_tool_calling
                    ),
                    enable_internal_tools=model_preset_data.get(
                        "enable_internal_tools", default_model_preset.enable_internal_tools
                    ),
                )
            else:
                model_preset = default_model_preset

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

        for model in self._models.values():
            backend = self._backends[model.backend]

            if model.id not in new_models and model.id not in self._dynamic_loaded_models:
                backend.unload_model(model.id)
                del self._models[model.id]

        for model in new_models:
            backend = self._backends[model.backend]

            if model.id not in self._models:
                backend.load_model(model)
                self._models[model.id] = model

    def get_models(self) -> list[Model]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Model | None:
        return self._models.get(model_id)

    def load_huggingface_model(
        self,
        huggingface_id: str,
        backend_id: str,
        for_task: str,
        skip_if_loaded: bool = False,
        model_preset: ModelPreset | None = None,
    ) -> None:
        backend = self._backends.get(backend_id)

        if backend is None:
            raise ValueError(f"Unknown backend {backend_id}")

        if huggingface_id in self._models:
            if not skip_if_loaded:
                raise ValueError(f"Model {huggingface_id} from huggingface is already loaded!")

            self._log.warning("Model %s from huggingface is already loaded, skip loading.", huggingface_id)

        self._log.info("Downloading model %s from huggingface...", huggingface_id)

        model_path = Path(huggingface_hub.snapshot_download(huggingface_id))
        model_generation_variant = self.__determine_model_generation_variant(model_path)

        preset: ModelPreset

        if model_preset is not None:
            preset = model_preset
        else:
            preset = DEFAULT_MODEL_PRESETS.get(model_generation_variant, DEFAULT_MODEL_PRESET)

        model = Model(
            id=huggingface_id,
            backend=backend_id,
            path=model_path,
            task=for_task,
            generation_variant=model_generation_variant,
            preset=preset,
            ipex_llm=None,
            create_time=datetime.now(),
        )

        backend.load_model(model)

        self._models[model.id] = model
        self._dynamic_loaded_models[model.id] = model
