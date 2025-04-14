import logging
import threading
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, cast, override

import openvino
import psutil
import torch
from openvino import properties as ov_props
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction, OVWeightQuantizationConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationStatistics,
    Model,
)
from azarrot.config import ServerConfig

OPENVINO_TASK_MODEL_MAP = {
    "text-generation": OVModelForCausalLM,
    "text-generation-with-past": OVModelForCausalLM,
    "feature-extraction": OVModelForFeatureExtraction,
}

BACKEND_ID_OPENVINO = "openvino"


class ThreadLocalAwareInferRequest:
    _log = logging.getLogger(__name__)
    _model: openvino.CompiledModel
    _request_holder = threading.local()

    def __init__(self, model: openvino.CompiledModel) -> None:
        self._model = model

    def __get_request(self) -> openvino.InferRequest:
        if not hasattr(self._request_holder, "request"):
            self._request_holder.request = self._model.create_infer_request()

        return self._request_holder.request

    def reset_state(self) -> None:
        req = self.__get_request()
        req.reset_state()

    def start_async(self, inputs: Any | None = None, userdata: Any | None = None, share_inputs: bool = False) -> None:
        req = self.__get_request()
        req.start_async(inputs, userdata, share_inputs)

    def wait(self) -> None:
        req = self.__get_request()
        req.wait()

    def get_tensor(self, *args, **kwargs) -> openvino.Tensor:  # type: ignore[no-untyped-def]    # noqa: ANN002, ANN003
        req = self.__get_request()
        return req.get_tensor(*args, **kwargs)

    def __call__(self, inputs: Any) -> Any:
        req = self.__get_request()
        req.start_async(inputs)
        req.wait()
        return req.results


def patched_compile(self) -> None:  # type: ignore[no-untyped-def]    # noqa: ANN001
    if self.request is None:
        super(type(self), self).compile()  # type: ignore[unused-ignore]

        if isinstance(self.request, openvino.InferRequest):
            self.compiled_model = self.request.get_compiled_model()
        else:
            self.compiled_model = self.request

        self.request = ThreadLocalAwareInferRequest(self.compiled_model)


class OpenVINOBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _default_device: str = "CPU"
    _auto_use_igpu: bool = True
    _cpu_phy_core_count: int = 0

    def __init__(self, config: ServerConfig, auto_use_igpu: bool = True) -> None:
        self.__patch_openvino()

        self._auto_use_igpu = auto_use_igpu

        super().__init__(config)

        self._cpu_phy_core_count = psutil.cpu_count(logical=False) or 0
        self._log.info("CPU has %d physical cores.", self._cpu_phy_core_count)

    def __patch_openvino(self) -> None:
        from openvino.frontend.pytorch import gptq as ov_pt_gptq
        ov_pt_gptq.supported_quant_types.append("ipex")

    @override
    def id(self) -> str:
        return BACKEND_ID_OPENVINO

    @override
    def _print_device_list(self) -> int:
        self._log.info("OpenVINO Available devices:")

        for device in self._ov.available_devices:
            self._log.info(
                "%s (%s): %s",
                device,
                self._ov.get_property(device, "DEVICE_TYPE"),
                self._ov.get_property(device, "FULL_DEVICE_NAME"),
            )

        return len(self._ov.available_devices)

    @override
    def _determine_default_device(self, accel_device_count: int) -> str:
        for device in self._ov.available_devices:
            device_type = self._ov.get_property(device, "DEVICE_TYPE")

            if device_type == openvino.properties.device.Type.DISCRETE:
                return device

        if self._auto_use_igpu:
            for device in self._ov.available_devices:
                device_type = self._ov.get_property(device, "DEVICE_TYPE")

                if device_type == openvino.properties.device.Type.INTEGRATED and device == "GPU":
                    return device
        else:
            self._log.info("No discrete device found, and iGPU usage is disabled. Will default to CPU.")

        return "CPU"

    def __patch_model(self, original_model: Any) -> Any:
        cast("Any", original_model).compiled_model = None
        original_model.compile = MethodType(patched_compile, original_model)
        original_model._is_stateful = original_model.stateful   # noqa: SLF001
        return original_model

    @override
    def _get_model_class(self, task: str) -> Any | None:
        return OPENVINO_TASK_MODEL_MAP.get(task)

    def __make_openvino_export_path(self, model: Model) -> Path:
        return self._server_config.models_dir / "openvino_exports" / f"{model.id}-{model.revision}"

    @override
    def _customize_model_and_kwargs(self, model: Model, model_config: Any, model_kwargs: dict[str, Any]) -> None:
        model_path = model.path.absolute()
        openvino_model_file_path = model_path / Path("openvino_model.xml")
        need_export = not openvino_model_file_path.exists()

        if need_export:
            openvino_export_path = self.__make_openvino_export_path(model)

            if (openvino_export_path / "openvino_model.xml").exists():
                self._log.info("Will load previous exported OpenVINO model from %s", openvino_export_path)
                model.path = openvino_export_path
                need_export = False
            else:
                self._log.info(
                    "OpenVINO model file does not exist at %s. Will export it to %s",
                    openvino_model_file_path,
                    openvino_export_path,
                )

        need_load_in_4bit = need_export and not model.use_original_precision
        model_kwargs["export"] = need_export

        if model.openvino is not None:
            if need_export and model.openvino.quantization_configs is not None:
                model_kwargs["quantization_config"] = OVWeightQuantizationConfig(
                    bits=model.openvino.quantization_configs.bits,
                    sym=model.openvino.quantization_configs.sym,
                    group_size=model.openvino.quantization_configs.group_size,
                    ratio=model.openvino.quantization_configs.ratio,
                    all_layers=model.openvino.quantization_configs.all_layers,
                    quant_method=model.openvino.quantization_configs.quant_method,
                    weight_format=model.openvino.quantization_configs.weight_format,
                )

        if "quantization_config" not in model_kwargs and need_load_in_4bit:
            model_kwargs["quantization_config"] = OVWeightQuantizationConfig(bits=4)

        ov_config = {
            "PERFORMANCE_HINT": ov_props.hint.PerformanceMode.LATENCY,
        }

        if model.device is not None and model.device.upper() == "CPU":
            ov_config["INFERENCE_NUM_THREADS"] = self._cpu_phy_core_count
            ov_config["SCHEDULING_CORE_TYPE"] = ov_props.hint.SchedulingCoreType.PCORE_ONLY
            ov_config["ENABLE_HYPER_THREADING"] = False
            ov_config["ENABLE_CPU_PINNING"] = True

        self._log.info("Using OpenVINO configs for device %s: %s", model.device, ov_config)

        model_kwargs["ov_config"] = ov_config

        model_kwargs["use_cache"] = model.task == "text-generation-with-past"

    @override
    def _customize_loaded_model(
        self,
        model: Model,
        loaded_model: PreTrainedModel,
        loaded_tokenizer: PreTrainedTokenizer,
        model_kwargs: dict[str, Any],
    ) -> PreTrainedModel:
        if model_kwargs.get("export", False):
            ov_model_export_path = self.__make_openvino_export_path(model)
            loaded_model.save_pretrained(ov_model_export_path)
            loaded_tokenizer.save_pretrained(ov_model_export_path)
            self._log.info("Exported OpenVINO model to %s", ov_model_export_path)

        ov_model = self.__patch_model(loaded_model)
        ov_model.compile()
        return ov_model

    @override
    def _should_move_inputs_to_device(self) -> bool:
        return False

    @override
    def _parse_device_str(self, device_str: str) -> list[str]:
        def sanitize_device(device: str) -> str:
            if device != "CPU" and "." not in device:
                return device + ".0"
            else:
                return device

        if device_str is None or device_str == "":
            return []

        return [sanitize_device(d.strip().upper()) for d in device_str.split(",")]

    @override
    def generate_embeddings(
        self, request: EmbeddingsGenerationRequest
    ) -> tuple[list[list[float]], GenerationStatistics]:
        loaded_model = self._get_model(request.model_id)

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=0,
            completion_tokens=0,
        )

        pipe = pipeline(
            "feature-extraction", loaded_model.model, tokenizer=loaded_model.tokenizer, trust_remote_code=True
        )

        outputs: Any = pipe(request.text, return_tensors=True)
        result = []

        if not isinstance(outputs, list):
            outputs = [outputs]

        for output in outputs:
            normalized_embeddings = torch.nn.functional.normalize(output, dim=-1)
            embeddings = normalized_embeddings[0][0]
            result.append(embeddings.tolist())

            gen_stats.prompt_tokens = gen_stats.prompt_tokens + output.size()[1]
            gen_stats.completion_tokens = gen_stats.completion_tokens + output.size()[2]

        gen_stats.end_time = datetime.now()

        return result, gen_stats
