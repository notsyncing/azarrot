import gc
import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, pipeline

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CustomTextIteratorStreamer,
    GenerationHandlers,
    GenerationMethods,
    TransformersGenerationMethods,
    to_transformers_chat_messages,
)
from azarrot.backends.internvl2_support import (
    InternVL2TransformersGenerationMethods,
    internvl2_apply_chat_template,
    internvl2_patch_model,
)
from azarrot.common_data import (
    EmbeddingModelInfo,
    EmbeddingsGenerationRequest,
    GenerationStatistics,
    Model,
    ModelInfo,
    TextGenerationRequest,
)
from azarrot.config import ServerConfig
from azarrot.models.model_quirks import MODEL_GENERATION_QUIRKS

TRANSFORMERS_TASK_MODEL_MAP = {
    "text-generation": AutoModelForCausalLM,
}

MODEL_PYTORCH_QUIRKS = {"internvl2": {"use_cache": False}}


@dataclass
class LoadedTransformersModel:
    info: Model
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: str


class TransformersBasedBackend(BaseBackend, ABC):
    _log = logging.getLogger(__name__)
    _models: dict[str, LoadedTransformersModel]
    _default_device: str = "xpu"

    _generation_variants: dict[
        str,
        Callable[
            [
                LoadedTransformersModel,
                TextGenerationRequest,
                dict[str, Any],
                CustomTextIteratorStreamer,
                GenerationStatistics,
            ],
            GenerationMethods,
        ],
    ]

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)

        self._server_config = config
        self._models = {}

        self._generation_variants = {
            "normal": self.__generate_normal,
            "internvl2": self.__generate_internvl2,
        }

        accel_device_count = self._print_device_list()
        self._default_device = self._determine_default_device(accel_device_count)

        self._log.info("Using default device: %s", self._default_device)

    def _determine_default_device(self, accel_device_count: int) -> str:
        if accel_device_count <= 0:
            return "cpu"
        else:
            return "xpu"

    def _print_device_list(self) -> int:
        self._log.info("%s Available devices:", self.id())
        xpu_count = torch.xpu.device_count()

        for i in range(xpu_count):
            self._log.info("XPU #%s: %s", i, str(torch.xpu.get_device_properties(i)))

        return xpu_count

    def __extract_model_info(self, transformers_model: PreTrainedModel, task: str) -> ModelInfo:
        if task == "feature-extraction":
            return EmbeddingModelInfo(dimension=transformers_model.config.hidden_size)
        else:
            return ModelInfo()

    def _get_model_class(self, task: str) -> Any | None:
        return TRANSFORMERS_TASK_MODEL_MAP.get(task)

    def _customize_model_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        pass

    def _customize_loaded_model(self, model: PreTrainedModel) -> PreTrainedModel:
        return model

    def load_model(self, model: Model) -> ModelInfo:
        model_class = self._get_model_class(model.task)

        if model_class is None:
            raise ValueError(f"Model {model.id} ({model.path}) wants task {model.task}, which is not supported!")

        if model.id in self._models:
            self._log.warning("Model %s is already loaded, will skip it.", model.id)
            assert model.info is not None
            return model.info

        model_path = model.path.absolute()

        device = self._server_config.model_device_map.get(model.id, self._default_device)

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        model_kwargs: dict[str, Any] = {}

        self._customize_model_kwargs(model, model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        generation_variant = model.generation_variant

        model_kwargs["use_cache"] = True

        model_kwargs_quirks = MODEL_PYTORCH_QUIRKS.get(generation_variant, {})
        model_kwargs.update(model_kwargs_quirks)

        if "use_cache" in model_kwargs and not model_kwargs.get("use_cache"):
            del model_kwargs["use_cache"]

        transformers_model: Any = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs,
        ).to(device)

        transformers_model.eval()

        transformers_model = self._customize_loaded_model(transformers_model)

        self._models[model.id] = LoadedTransformersModel(model, transformers_model, tokenizer, device)

        self._log.info("Loaded model %s", model.id)

        return self.__extract_model_info(transformers_model, model.task)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warning("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        torch.xpu.empty_cache()
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def __get_model(self, model_id: str) -> LoadedTransformersModel:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} is not loaded!")

        return self._models[model_id]

    def _parse_device_str(self, device_str: str) -> list[str]:
        def sanitize_device(device: str) -> str:
            if device.isdigit():
                return device
            elif device != "cpu" and ":" not in device:
                return device + ":0"
            else:
                return device

        if device_str is None or device_str == "":
            return []

        return [sanitize_device(d.strip().lower()) for d in device_str.split(",")]

    def __generate_normal(
        self,
        loaded_model: LoadedTransformersModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> GenerationMethods:
        result = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt", return_dict=True
        )

        result = cast(dict[str, Any], result)

        inputs = result["input_ids"]
        attention_mask = result.get("attention_mask")

        gen_stats.prompt_tokens = len(cast(torch.Tensor, inputs[0]))

        generation_kwargs = common_generation_kwargs.copy()

        generation_kwargs.update(
            {
                "input_ids": inputs.to(loaded_model.device),
                "attention_mask": attention_mask.to(loaded_model.device) if attention_mask is not None else None,
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
            }
        )

        return TransformersGenerationMethods(
            model=loaded_model.model, streamer=streamer, seed=request.seed, generation_kwargs=generation_kwargs
        )

    def __generate_internvl2(
        self,
        loaded_model: LoadedTransformersModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> GenerationMethods:
        internvl2_patch_model(loaded_model.model, loaded_model.tokenizer)

        inputs, pixel_values = internvl2_apply_chat_template(
            loaded_model.model, loaded_model.tokenizer, request.messages
        )

        text_input_length = len(cast(torch.Tensor, inputs[0]))
        image_input_length = len(pixel_values) if pixel_values is not None else 0
        gen_stats.prompt_tokens = text_input_length + image_input_length

        # token id 2 is from tokenizer.json ('</s>')
        attention_mask = loaded_model.model._prepare_attention_mask_for_generation(  # noqa: SLF001
            inputs,
            torch.Tensor([2]),  # pyright: ignore[reportArgumentType]
            torch.Tensor([2]),  # pyright: ignore[reportArgumentType]
        )

        if pixel_values is not None:
            pixel_values = pixel_values.to(loaded_model.device)

        generation_kwargs = common_generation_kwargs.copy()

        generation_kwargs.update(
            {
                "input_ids": cast(torch.Tensor, inputs).to(loaded_model.device),
                "attention_mask": attention_mask.to(loaded_model.device),
                "pixel_values": pixel_values,
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
                # token id list is taken from https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/conversation.py#368
                "eos_token_id": [2, 92543, 92542],
            }
        )

        return InternVL2TransformersGenerationMethods(
            model=loaded_model.model, streamer=streamer, seed=request.seed, generation_kwargs=generation_kwargs
        )

    def _generate(
        self, request: TextGenerationRequest, generation_handlers: GenerationHandlers
    ) -> tuple[BackendGenerationTask, CustomTextIteratorStreamer, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)
        generation_variant = loaded_model.info.generation_variant
        generation_method = self._generation_variants.get(generation_variant, self.__generate_normal)

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.max,
            end_time=datetime.max,
            prompt_tokens=0,
            completion_tokens=0,
        )

        model_quirks = MODEL_GENERATION_QUIRKS.get(loaded_model.info.generation_variant)

        streamer = CustomTextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer),
            gen_stats,
            skip_prompt=True,
            timeout=self._server_config.single_token_generation_timeout / 1000,
            skip_special_tokens=True,
            model_quirks=model_quirks,
            generation_handlers=generation_handlers,
        )

        common_generation_kwargs = {"do_sample": True, "temperature": request.temperature, "top_p": request.top_p}

        m = generation_method(loaded_model, request, common_generation_kwargs, streamer, gen_stats)

        task = BackendGenerationTask(
            model_id=loaded_model.info.id,
            model_quirks=model_quirks,
            backend_id=self.id(),
            methods=m,
            device=loaded_model.device,
            seed=request.seed,
        )

        return task, streamer, gen_stats

    def generate_embeddings(
        self, request: EmbeddingsGenerationRequest
    ) -> tuple[list[list[float]], GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)

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
