import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import huggingface_hub
import yaml
from optimum.intel.openvino.configuration import OVQuantizationMethod

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.backends.pytorch_backend import BACKEND_ID_PYTORCH
from azarrot.common_data import (
    Model,
    ModelPreset,
    OpenVINOModelConfig,
    OpenVINOQuantizationConfigs,
    PyTorchModelConfig,
    TransformersModelConfig,
)
from azarrot.config import ServerConfig
from azarrot.models.chat_templates import DEFAULT_LOCALE

HF_MODEL_PRESET_MAPPING: dict[str, str] = {
    "InternVLChatModel": "internvl2",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
}

DEFAULT_MODEL_PRESET = ModelPreset(
    preferred_locale=DEFAULT_LOCALE,  # type: ignore[arg-type]
    supports_tool_calling=False,
    enable_internal_tools=False,
)

DEFAULT_MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen2": ModelPreset(preferred_locale=DEFAULT_LOCALE, supports_tool_calling=True, enable_internal_tools=False),  # type: ignore[arg-type]
    "qwen3": ModelPreset(preferred_locale=DEFAULT_LOCALE, supports_tool_calling=True, enable_internal_tools=False),  # type: ignore[arg-type]
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

    def __determine_model_generation_variant(self, model_path: Path) -> str:
        hf_config_file = model_path / "config.json"

        if hf_config_file.exists():
            try:
                with hf_config_file.open() as f:
                    hf_config = json.load(f)

                hf_model_archs: list[str] = hf_config.get("architectures", [])

                for k, v in HF_MODEL_PRESET_MAPPING.items():
                    if k in hf_model_archs:
                        return v
            except:
                self._log.warning("Failed to parse config %s as JSON", hf_config_file)

        return "normal"

    def __download_from_huggingface(self, hf_model_id: str) -> Path:
        if self._config.huggingface_download_to_home:
            hf_local_dir = None
        else:
            hf_local_dir = self._config.models_dir / Path(f"huggingface/{hf_model_id.replace('/', '_')}")

            if not hf_local_dir.exists():
                hf_local_dir.mkdir(parents=True)

        self._log.info("Downloading model %s from huggingface...", hf_model_id)

        hf_model_path = huggingface_hub.snapshot_download(hf_model_id, local_dir=hf_local_dir)
        return Path(hf_model_path)

    def __parse_openvino_model_config(self, config_data: Any) -> OpenVINOModelConfig:
        quant_config = config_data.get("quantization_configs")

        if quant_config is not None:
            qc = OpenVINOQuantizationConfigs(
                bits=quant_config.get("bits", 8),
                sym=quant_config.get("sym", False),
                group_size=quant_config.get("group_size"),
                ratio=quant_config.get("ratio", 1.0),
                all_layers=quant_config.get("all_layers"),
                quant_method=quant_config.get("quant_method", OVQuantizationMethod.DEFAULT),
                weight_format=quant_config.get("weight_format"),
            )
        else:
            qc = None

        return OpenVINOModelConfig(quantization_configs=qc)

    def __parse_model_file(self, file: Path) -> Model:
        with file.open() as f:
            model_info = yaml.safe_load(f)

            raw_model_path: str = model_info["path"]
            model_path: Path

            if raw_model_path.startswith(MODEL_PATH_HUGGINGFACE):
                hf_model_id = raw_model_path[len(MODEL_PATH_HUGGINGFACE) :]
                model_path = self.__download_from_huggingface(hf_model_id)
            else:
                model_path = self._config.models_dir / Path(model_info["path"])

            hf_cache_files = model_path.glob("*")
            latest_file = max(hf_cache_files, key=os.path.getmtime)
            model_revision = str(latest_file.lstat().st_mtime)

            model_backend = model_info.get("backend", BACKEND_ID_OPENVINO)

            model_generation_variant = model_info.get(
                "generation_variant", self.__determine_model_generation_variant(model_path)
            )

            transformers = None

            if model_backend in (BACKEND_ID_OPENVINO, BACKEND_ID_PYTORCH):
                transformers_config = model_info.get("transformers", {})

                transformers = TransformersModelConfig(assistant_model=transformers_config.get("assistant_model"))

            openvino = None

            if model_backend == BACKEND_ID_OPENVINO:
                openvino_config = model_info.get("openvino", {})
                openvino = self.__parse_openvino_model_config(openvino_config)

            pytorch = None

            if model_backend == BACKEND_ID_PYTORCH:
                pytorch_config = model_info.get("pytorch", {})

                pytorch = PyTorchModelConfig(
                    compile=pytorch_config.get("compile", False),
                    compile_backend=pytorch_config.get("compile_backend", "inductor"),
                )

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
                revision=model_revision,
                task=model_info["task"],
                generation_variant=model_generation_variant,
                preset=model_preset,
                use_original_precision=model_info.get("use_original_precision", False),
                is_for_raw_completion=model_info.get("is_for_raw_completion", False),
                is_reasoning_model=model_info.get("is_reasoning_model", False),
                transformers=transformers,
                openvino=openvino,
                pytorch=pytorch,
                info=None,
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
                model_info = backend.load_model(model)
                model.info = model_info
                self._models[model.id] = model

    def get_models(self) -> list[Model]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Model | None:
        return self._models.get(model_id)

    def add_model(self, model: Model) -> None:
        self._models[model.id] = model

    def load_huggingface_model(
        self,
        huggingface_id: str,
        backend_id: str,
        for_task: str,
        skip_if_loaded: bool = False,
        model_preset: ModelPreset | None = None,
        use_original_precision: bool = False,
        is_for_raw_completion: bool = False,
        is_reasoning_model: bool = False,
        override_model_id: str | None = None,
    ) -> None:
        backend = self._backends.get(backend_id)

        if backend is None:
            raise ValueError(f"Unknown backend {backend_id}")

        if (override_model_id or huggingface_id) in self._models:
            if not skip_if_loaded:
                raise ValueError(f"Model {huggingface_id} from huggingface is already loaded!")

            self._log.warning("Model %s from huggingface is already loaded, skip loading.", huggingface_id)
            return

        model_path = self.__download_from_huggingface(huggingface_id)
        model_generation_variant = self.__determine_model_generation_variant(model_path)

        preset: ModelPreset

        if model_preset is not None:
            preset = model_preset
        else:
            preset = DEFAULT_MODEL_PRESETS.get(model_generation_variant, DEFAULT_MODEL_PRESET)

        hf_cache_files = model_path.glob("*")
        latest_file = max(hf_cache_files, key=os.path.getmtime)

        model = Model(
            id=override_model_id or huggingface_id,
            backend=backend_id,
            path=model_path,
            revision=str(latest_file.lstat().st_mtime),
            task=for_task,
            generation_variant=model_generation_variant,
            preset=preset,
            use_original_precision=use_original_precision,
            is_for_raw_completion=is_for_raw_completion,
            is_reasoning_model=is_reasoning_model,
            transformers=None,
            openvino=None,
            pytorch=None,
            info=None,
            create_time=datetime.now(),
        )

        model_info = backend.load_model(model)
        model.info = model_info

        self._models[model.id] = model
        self._dynamic_loaded_models[model.id] = model
