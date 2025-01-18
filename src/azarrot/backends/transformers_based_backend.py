import gc
import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing_extensions import override

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CustomTextIteratorStreamer,
    GenerationHandlers,
    GenerationMethods,
)
from azarrot.backends.internvl2_support import (
    InternVL2TransformersGenerationMethods,
    internvl2_apply_chat_template,
    internvl2_patch_model,
)
from azarrot.backends.pytorch_common import (
    determine_pytorch_default_device,
    parse_pytorch_device_str,
    print_pytorch_device_list,
)
from azarrot.backends.transformers_common import TransformersGenerationMethods, to_transformers_chat_messages
from azarrot.common_data import (
    EmbeddingModelInfo,
    GenerationMessage,
    GenerationStatistics,
    Model,
    ModelInfo,
    TextGenerationMessageContent,
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
    data: Model
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
        return determine_pytorch_default_device(accel_device_count)

    def _print_device_list(self) -> int:
        return print_pytorch_device_list(self._log, self.id())

    def __extract_model_info(self, transformers_model: PreTrainedModel, task: str) -> ModelInfo:
        if task == "feature-extraction":
            return EmbeddingModelInfo(dimension=transformers_model.config.hidden_size)
        else:
            return ModelInfo()

    def _get_model_class(self, task: str) -> Any | None:
        return TRANSFORMERS_TASK_MODEL_MAP.get(task)

    def _customize_model_and_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        pass

    def _customize_loaded_model(
        self,
        model: Model,  # noqa: ARG002
        loaded_model: PreTrainedModel,
        loaded_tokenizer: PreTrainedTokenizer,  # noqa: ARG002
        model_kwargs: dict[str, Any],  # noqa: ARG002
    ) -> PreTrainedModel:
        return loaded_model

    def _should_move_inputs_to_device(self) -> bool:
        return True

    @override
    def load_model(self, model: Model) -> ModelInfo:
        model_class = self._get_model_class(model.task)

        if model_class is None:
            raise ValueError(f"Model {model.id} ({model.path}) wants task {model.task}, which is not supported!")

        if model.id in self._models:
            self._log.warning("Model %s is already loaded, will skip it.", model.id)
            assert model.info is not None
            return model.info

        device = self._determine_device_for_model(model.id)
        model.device = device

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        model_kwargs: dict[str, Any] = {
            "use_cache": True
        }

        self._customize_model_and_kwargs(model, model_kwargs)

        model_path = model.path.absolute()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        generation_variant = model.generation_variant

        model_kwargs_quirks = MODEL_PYTORCH_QUIRKS.get(generation_variant, {})
        model_kwargs.update(model_kwargs_quirks)

        transformers_model: Any = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs,
        ).to(device)

        transformers_model.eval()

        transformers_model = self._customize_loaded_model(model, transformers_model, tokenizer, model_kwargs)

        self._models[model.id] = LoadedTransformersModel(model, transformers_model, tokenizer, device)

        self._log.info("Loaded model %s", model.id)

        return self.__extract_model_info(transformers_model, model.task)

    @override
    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warning("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        torch.xpu.empty_cache()
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def _get_model(self, model_id: str) -> LoadedTransformersModel:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} is not loaded!")

        return self._models[model_id]

    @override
    def _parse_device_str(self, device_str: str) -> list[str]:
        return parse_pytorch_device_str(device_str)

    def __get_first_text_message_with_role(self, expected_role: str, messages: list[GenerationMessage]) -> str | None:
        for msg in messages:
            if msg.role == expected_role:
                if isinstance(msg.contents[0], TextGenerationMessageContent):
                    return msg.contents[0].text

        return None

    def __generate_normal(
        self,
        loaded_model: LoadedTransformersModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> GenerationMethods:
        if not loaded_model.data.is_for_raw_completion:
            result = loaded_model.tokenizer.apply_chat_template(
                to_transformers_chat_messages(request.messages), return_tensors="pt", return_dict=True
            )

            result = cast(dict[str, Any], result)
        else:
            first_user_msg = self.__get_first_text_message_with_role("user", request.messages)

            if first_user_msg is None:
                raise ValueError(
                    f"This model {loaded_model.data.id} is for raw completion, but no user text message was found in "
                    "this request!"
                )

            result = loaded_model.tokenizer(first_user_msg, return_tensors="pt")

        inputs: Any = result["input_ids"]
        attention_mask = result.get("attention_mask")

        gen_stats.prompt_tokens = len(cast(torch.Tensor, inputs[0]))

        generation_kwargs = common_generation_kwargs.copy()

        if self._should_move_inputs_to_device():
            inputs = inputs.to(loaded_model.device)

            if attention_mask is not None:
                attention_mask = attention_mask.to(loaded_model.device)

        generation_kwargs.update(
            {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
            }
        )

        seed = request.seed

        if seed is None:
            seed = self._server_config.default_seed

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

        inputs, attention_mask, pixel_values = internvl2_apply_chat_template(
            loaded_model.model, loaded_model.tokenizer, request.messages
        )

        text_input_length = len(cast(torch.Tensor, inputs[0]))
        image_input_length = len(pixel_values) if pixel_values is not None else 0
        gen_stats.prompt_tokens = text_input_length + image_input_length

        if self._should_move_inputs_to_device():
            inputs = inputs.to(loaded_model.device)

            if attention_mask is not None:
                attention_mask = attention_mask.to(loaded_model.device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(loaded_model.device)

        generation_kwargs = common_generation_kwargs.copy()

        generation_kwargs.update(
            {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
            }
        )

        return InternVL2TransformersGenerationMethods(
            model=loaded_model.model, streamer=streamer, seed=request.seed, generation_kwargs=generation_kwargs
        )

    @override
    def _generate(
        self, request: TextGenerationRequest, generation_handlers: GenerationHandlers
    ) -> tuple[BackendGenerationTask, CustomTextIteratorStreamer, GenerationStatistics]:
        loaded_model = self._get_model(request.model_id)
        generation_variant = loaded_model.data.generation_variant
        generation_method = self._generation_variants.get(generation_variant, self.__generate_normal)

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.max,
            end_time=datetime.max,
            prompt_tokens=0,
            completion_tokens=0,
        )

        model_quirks = MODEL_GENERATION_QUIRKS.get(loaded_model.data.generation_variant)

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
            model_id=loaded_model.data.id,
            model_quirks=model_quirks,
            backend_id=self.id(),
            methods=m,
            device=loaded_model.device,
            seed=request.seed,
        )

        return task, streamer, gen_stats
