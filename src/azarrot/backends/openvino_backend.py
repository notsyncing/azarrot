import logging
import sys
from datetime import datetime
from pathlib import Path
from types import MethodType, ModuleType
from typing import TYPE_CHECKING, Any, cast, override

import gptqmodel
import openvino
import psutil
import torch
from openvino import properties as ov_props
from openvino._pyopenvino import VariableState
from optimum.intel import (
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForVisualCausalLM,
    OVWeightQuantizationConfig,
)
from transformers.cache_utils import DynamicCache
from transformers.pipelines import pipeline

from azarrot.backends.backend_base import BackendGenerationTask
from azarrot.backends.caching import PreparedCache
from azarrot.backends.common import CompletionChunkStreamer, CustomTextIteratorStreamer
from azarrot.backends.transformers_based_backend import LoadedTransformersModel, TransformersBasedBackend
from azarrot.backends.transformers_common import TransformersGenerationMethods, TransformersModelPrefixCache
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationStatistics,
    Model,
    ModelQuirks,
    TextGenerationRequest,
)
from azarrot.config import ModelPrefixCacheConfig, ServerConfig
from azarrot.models.model_quirks import MODEL_GENERATION_QUIRKS

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer

OPENVINO_TASK_MODEL_MAP = {
    "text-generation": OVModelForCausalLM,
    "text-generation-with-past": OVModelForCausalLM,
    "feature-extraction": OVModelForFeatureExtraction,
    "image-text-to-text": OVModelForVisualCausalLM,
    "image-text-to-text-with-past": OVModelForVisualCausalLM,
}

BACKEND_ID_OPENVINO = "openvino"


class OpenVINOGenerationMethod(TransformersGenerationMethods):
    def __init__(
        self,
        model: "PreTrainedModel",
        streamer: CustomTextIteratorStreamer,
        prefix_cache: TransformersModelPrefixCache | None,
        seed: int | None,
        generation_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(model, streamer, prefix_cache, seed, generation_kwargs)

    @override
    def generate(self) -> tuple[bool, list[CustomTextIteratorStreamer]]:
        if self.prefix_cache is not None:
            current_cache: DynamicCache = self.generation_kwargs["past_key_values"]
            # del self.generation_kwargs["past_key_values"]

            if len(current_cache) == 0:
                del self.generation_kwargs["past_key_values"]
            else:
                self.generation_kwargs["past_key_values"] = current_cache.to_legacy_cache()
                request_wrapper = cast("CustomInferRequestWrapper", self.model.request)
                request_wrapper.bind_prefix_cache(current_cache)

            del self.generation_kwargs["cache_position"]

        return super().generate()

    def __convert_to_hf_transformers_cache(self) -> DynamicCache:
        request_wrapper = cast("CustomInferRequestWrapper", self.model.request)
        request = request_wrapper.get_request()

        new_cache = DynamicCache()
        key_caches = {}
        value_caches = {}

        for var_state in request.query_state():
            if "past_key_values." in var_state.name:
                name_parts = var_state.name.split(".")

                if len(name_parts) != 5:  # noqa: PLR2004
                    continue

                layer = int(name_parts[1])

                if name_parts[4] == "key":
                    key_caches[layer] = torch.from_numpy(var_state.state.data)
                elif name_parts[4] == "value":
                    value_caches[layer] = torch.from_numpy(var_state.state.data)
                else:
                    continue

                if layer in key_caches and layer in value_caches:
                    new_cache.update(key_caches[layer], value_caches[layer], layer)

        return new_cache

    @override
    def split_from_batch(self, others: list["TransformersGenerationMethods"]) -> None:
        if self.prefix_cache is not None:
            self.generation_kwargs["past_key_values"] = self.__convert_to_hf_transformers_cache()

        return super().split_from_batch(others)

    @override
    def on_execution_successful(self, index_in_batch: int) -> None:
        if self.prefix_cache is not None:
            new_cache = self.__convert_to_hf_transformers_cache()
            self.generation_kwargs["past_key_values"] = new_cache

        super().on_execution_successful(index_in_batch)


class CustomInferRequestWrapper:
    _log = logging.getLogger(__name__)
    _model: openvino.CompiledModel
    _hf_model: "PreTrainedModel"
    _request: openvino.InferRequest | None = None

    def __init__(self, model: openvino.CompiledModel, hf_model: "PreTrainedModel") -> None:
        self._model = model
        self._hf_model = hf_model

    def __get_request(self) -> openvino.InferRequest:
        if self._request is None:
            self._request = self._model.create_infer_request()

        return self._request

    def get_request(self) -> openvino.InferRequest:
        assert self._request is not None
        return self._request

    def bind_prefix_cache(self, cache: DynamicCache) -> None:
        req = self.__get_request()
        req.reset_state()

        for var_state in req.query_state():
            if "past_key_values." in var_state.name:
                name_parts = var_state.name.split(".")

                if len(name_parts) != 5:  # noqa: PLR2004
                    continue

                layer = int(name_parts[1])

                if layer >= len(cache):
                    continue

                if name_parts[4] == "key":
                    cached_tensor = cache.key_cache[layer]
                elif name_parts[4] == "value":
                    cached_tensor = cache.value_cache[layer]
                else:
                    continue

                var_state.state = openvino.Tensor(cached_tensor.numpy())

        self._hf_model._past_length = cache.get_seq_length()  # type: ignore[assignment]  # noqa: SLF001

    def reset_state(self) -> None:
        req = self.__get_request()
        req.reset_state()

    def start_async(self, inputs: Any | None = None, userdata: Any | None = None, share_inputs: bool = False) -> None:
        req = self.__get_request()
        req.start_async(inputs, userdata, share_inputs)

    def wait(self) -> None:
        req = self.__get_request()
        req.wait()

    def get_tensor(self, *args: Any, **kwargs: Any) -> openvino.Tensor:
        req = self.__get_request()
        return req.get_tensor(*args, **kwargs)

    def query_state(self) -> list[VariableState]:
        req = self.__get_request()
        return req.query_state()

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

        self.request = CustomInferRequestWrapper(self.compiled_model, self)


class OpenVINOBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _default_device: str = "CPU"
    _auto_use_igpu: bool = True
    _cpu_phy_core_count: int = 0
    _auto_gptq_faked: bool = False

    def __init__(self, config: ServerConfig, auto_use_igpu: bool = True) -> None:
        self.__patch_openvino()

        self._auto_use_igpu = auto_use_igpu

        super().__init__(config)

        self._cpu_phy_core_count = psutil.cpu_count(logical=False) or 0
        self._log.info("CPU has %d physical cores.", self._cpu_phy_core_count)

    def __patch_openvino(self) -> None:
        from openvino.frontend.pytorch import gptq as ov_pt_gptq  # noqa: PLC0415

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

    def __patch_model(self, original_model: Any, model_quirks: ModelQuirks | None) -> Any:
        should_patch_compile = not model_quirks.openvino_dont_patch_model_compile if model_quirks is not None else True

        if should_patch_compile:
            cast("Any", original_model).compiled_model = None
            original_model.compile = MethodType(patched_compile, original_model)

        return original_model

    @override
    def _get_model_class(self, task: str) -> Any | None:
        return OPENVINO_TASK_MODEL_MAP.get(task)

    def __make_openvino_export_path(self, model: Model) -> Path:
        return self._server_config.models_dir / "openvino_exports" / f"{model.id}-{model.revision}"

    def __workaround_optimum_intel_gptqmodel_export_begin(self) -> None:
        if "auto_gptq" in sys.modules:
            return

        fake_auto_gptq = ModuleType("auto_gptq")
        cast("Any", fake_auto_gptq).exllama_set_max_input_length = gptqmodel.exllama_set_max_input_length
        sys.modules["auto_gptq"] = fake_auto_gptq
        self._auto_gptq_faked = True

    def __workaround_optimum_intel_gptqmodel_export_end(self) -> None:
        if not self._auto_gptq_faked:
            return

        del sys.modules["auto_gptq"]

    @override
    def _customize_model_and_kwargs(
        self,
        model: Model,
        model_config: Any,
        model_kwargs: dict[str, Any],
        *,
        device: str,
        prefix_cache_config: ModelPrefixCacheConfig | None = None,
    ) -> None:
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

        model_kwargs["use_cache"] = model.task.endswith("-with-past")

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

        ov_config: dict[str, Any] = {
            "PERFORMANCE_HINT": ov_props.hint.PerformanceMode.LATENCY,
        }

        if prefix_cache_config is not None and model_kwargs["use_cache"] and "GPU" in device.upper():
            self._log.info("Prefix caching enabled, and device is GPU, forcing KV_CACHE_PRECISION to undefined.")
            ov_config["KV_CACHE_PRECISION"] = "undefined"

        if device.upper() == "CPU":
            ov_config["INFERENCE_NUM_THREADS"] = self._cpu_phy_core_count
            ov_config["SCHEDULING_CORE_TYPE"] = ov_props.hint.SchedulingCoreType.PCORE_ONLY
            ov_config["ENABLE_HYPER_THREADING"] = False
            ov_config["ENABLE_CPU_PINNING"] = True

        self._log.info("Using OpenVINO configs for device %s: %s", device, ov_config)

        model_kwargs["ov_config"] = ov_config

        if need_export:
            self.__workaround_optimum_intel_gptqmodel_export_begin()

    @override
    def _customize_loaded_model(
        self,
        model: Model,
        loaded_model: "PreTrainedModel",
        loaded_tokenizer: "PreTrainedTokenizer | None",
        loaded_processor: "ProcessorMixin | None",
        model_kwargs: dict[str, Any],
    ) -> "PreTrainedModel":
        if model_kwargs.get("export", False):
            self.__workaround_optimum_intel_gptqmodel_export_end()

            ov_model_export_path = self.__make_openvino_export_path(model)
            loaded_model.save_pretrained(ov_model_export_path)

            if loaded_tokenizer is not None:
                loaded_tokenizer.save_pretrained(ov_model_export_path)

            if loaded_processor is not None:
                loaded_processor.save_pretrained(ov_model_export_path)

            self._log.info("Exported OpenVINO model to %s", ov_model_export_path)

        model_quirks = MODEL_GENERATION_QUIRKS.get(model.generation_variant)

        ov_model = self.__patch_model(loaded_model, model_quirks)
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
    def _bind_prefix_cache_to_generation(
        self,
        loaded_model: LoadedTransformersModel,
        prepared_cache: PreparedCache[DynamicCache],
        generation_kwargs: dict[str, Any],
    ) -> None:
        super()._bind_prefix_cache_to_generation(loaded_model, prepared_cache, generation_kwargs)

        input_ids: torch.Tensor = generation_kwargs["input_ids"]
        cache_positions: torch.Tensor = generation_kwargs["cache_position"]
        cache_pos = int(cache_positions[0])

        if "position_ids" in generation_kwargs:
            position_ids = generation_kwargs["position_ids"]
            generation_kwargs["position_ids"] = position_ids[:, cache_pos:]
        else:
            generation_kwargs["position_ids"] = torch.Tensor([list(range(cache_pos, len(input_ids[0])))])

        generation_kwargs["cache_position"] = torch.Tensor([0])

    @override
    def _prefix_cache_need_copying(self) -> bool:
        return False

    @override
    def _generate(
        self, request: TextGenerationRequest
    ) -> tuple[BackendGenerationTask, CompletionChunkStreamer, GenerationStatistics]:
        task, streamer, stats = super()._generate(request)

        original_methods = cast("TransformersGenerationMethods", task.methods)

        task.methods = OpenVINOGenerationMethod(
            model=original_methods.model,
            streamer=original_methods.streamer,
            prefix_cache=original_methods.prefix_cache,
            seed=original_methods.seed,
            generation_kwargs=original_methods.generation_kwargs,
        )

        return task, streamer, stats

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
            cached_prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
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
