import gc
import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CompletionChunkStreamer,
    CustomTextIteratorStreamer,
    GenerationMethods,
)
from azarrot.backends.internvl_support import (
    InternVLTransformersGenerationMethods,
    internvl_apply_chat_template,
    internvl_patch_model,
)
from azarrot.backends.pytorch_common import (
    determine_pytorch_default_device,
    parse_pytorch_device_str,
    print_pytorch_device_list,
)
from azarrot.backends.transformers_common import (
    ProcessorToTokenizerAdapter,
    TransformersGenerationMethods,
    to_transformers_chat_messages,
)
from azarrot.common_data import (
    EmbeddingModelInfo,
    GenerationMessage,
    GenerationStatistics,
    Model,
    ModelInfo,
    TextGenerationMessageContent,
    TextGenerationRequest,
)
from azarrot.config import DEFAULT_MAX_TOKENS, DEFAULT_REASONING_MAX_TOKENS, ServerConfig
from azarrot.models.chat_templates import MODEL_TOOL_CALL_CONFIGS
from azarrot.models.model_quirks import MODEL_GENERATION_QUIRKS
from azarrot.models.supports.default_chat_support import DEFAULT_MODEL_TOOL_CALL_CONFIG
from azarrot.tools.tool import convert_tool_descriptions_to_json_schema

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer

TRANSFORMERS_TASK_MODEL_MAP = {
    "text-generation": AutoModelForCausalLM,
    "text-generation-with-past": AutoModelForCausalLM,
}

TRANSFORMERS_TASK_NEED_PROCESSOR_MAP = {
    "image-text-to-text": True,
    "image-text-to-text-with-past": True,
}

MODEL_PYTORCH_QUIRKS = {}


@dataclass
class LoadedTransformersModel:
    data: Model
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizer | None"
    processor: "ProcessorMixin | None"
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
            "internvl": self.__generate_internvl,
        }

        accel_device_count = self._print_device_list()
        self._default_device = self._determine_default_device(accel_device_count)

        self._log.info("Using default device: %s", self._default_device)

    def _determine_default_device(self, accel_device_count: int) -> str:
        return determine_pytorch_default_device(accel_device_count)

    def _print_device_list(self) -> int:
        return print_pytorch_device_list(self._log, self.id())

    def __extract_model_info(self, transformers_model: "PreTrainedModel", task: str) -> ModelInfo:
        if task == "feature-extraction":
            return EmbeddingModelInfo(dimension=transformers_model.config.hidden_size)
        else:
            return ModelInfo()

    def _get_model_class(self, task: str) -> Any | None:
        return TRANSFORMERS_TASK_MODEL_MAP.get(task)

    def _is_model_need_processor(self, model: Model) -> bool:
        return TRANSFORMERS_TASK_NEED_PROCESSOR_MAP.get(model.task, False)

    def _customize_model_and_kwargs(self, model: Model, model_config: Any, model_kwargs: dict[str, Any]) -> None:
        pass

    def _customize_loaded_model(
        self,
        model: Model,  # noqa: ARG002
        loaded_model: "PreTrainedModel",
        loaded_tokenizer: "PreTrainedTokenizer | None",  # noqa: ARG002
        loaded_processor: "ProcessorMixin | None",  # noqa: ARG002
        model_kwargs: dict[str, Any],  # noqa: ARG002
    ) -> "PreTrainedModel":
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

        model_config = AutoConfig.from_pretrained(model.path.absolute(), trust_remote_code=True)

        model_kwargs: dict[str, Any] = {}

        self._customize_model_and_kwargs(model, model_config, model_kwargs)

        model_path = model.path.absolute()

        tokenizer: PreTrainedTokenizer | None = None
        processor: ProcessorMixin | None = None

        if self._is_model_need_processor:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        generation_variant = model.generation_variant

        model_kwargs_quirks = MODEL_PYTORCH_QUIRKS.get(generation_variant, {})
        model_kwargs.update(model_kwargs_quirks)

        for k, v in dict(model_kwargs).items():
            if v is None:
                del model_kwargs[k]

        transformers_model: Any = model_class.from_pretrained(
            model_path,
            config=model_config,
            trust_remote_code=True,
            **model_kwargs,
        ).to(device)

        transformers_model.eval()

        transformers_model = self._customize_loaded_model(model, transformers_model, tokenizer, processor, model_kwargs)

        self._models[model.id] = LoadedTransformersModel(model, transformers_model, tokenizer, processor, device)

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

    def __determine_default_max_tokens(self, model: Model) -> int:
        if model.is_reasoning_model:
            return DEFAULT_REASONING_MAX_TOKENS
        else:
            return DEFAULT_MAX_TOKENS

    def __generate_normal(
        self,
        loaded_model: LoadedTransformersModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> GenerationMethods:
        if not loaded_model.data.is_for_raw_completion:
            transformers_tools_desc = cast(
                "Any",
                convert_tool_descriptions_to_json_schema(request.tools_info.tools)
                if request.tools_info is not None
                else None,
            )

            if loaded_model.tokenizer is not None:
                result = loaded_model.tokenizer.apply_chat_template(
                    to_transformers_chat_messages(request.messages),
                    tools=transformers_tools_desc,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            elif loaded_model.processor is not None:
                result = loaded_model.processor.apply_chat_template(
                    to_transformers_chat_messages(request.messages),
                    tools=transformers_tools_desc,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    processor_kwargs={},
                    mm_load_kwargs={},
                    template_kwargs={},
                )
            else:
                raise ValueError(f"This model {loaded_model.data.id} has neither tokenizer nor processor!")

            result = cast("dict[str, Any]", result)
        else:
            if loaded_model.tokenizer is None:
                raise ValueError(f"This model {loaded_model.data.id} is not supported for raw completion!")

            first_user_msg = self.__get_first_text_message_with_role("user", request.messages)

            if first_user_msg is None:
                raise ValueError(
                    f"This model {loaded_model.data.id} is for raw completion, but no user text message was found in "
                    "this request!"
                )

            result = loaded_model.tokenizer(first_user_msg, return_tensors="pt")

        inputs: Any = result["input_ids"]
        attention_mask = result.get("attention_mask")

        gen_stats.prompt_tokens = len(cast("torch.Tensor", inputs[0]))

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
                "max_new_tokens": request.max_tokens or self.__determine_default_max_tokens(loaded_model.data),
            }
        )

        if loaded_model.data.transformers is not None:
            assistant_model_id = loaded_model.data.transformers.assistant_model

            if assistant_model_id is not None:
                assistant_model = self._models.get(assistant_model_id)

                if assistant_model is None:
                    raise ValueError(
                        f"Model {loaded_model.data.id} wants to use assistant model {assistant_model_id}, "
                        f"but it is not loaded in current backend {self.id()}"
                    )

                generation_kwargs["tokenizer"] = loaded_model.tokenizer
                generation_kwargs["assistant_model"] = assistant_model.model
                generation_kwargs["assistant_tokenizer"] = assistant_model.tokenizer

        seed = request.seed

        if seed is None:
            seed = self._server_config.default_seed

        return TransformersGenerationMethods(
            model=loaded_model.model, streamer=streamer, seed=seed, generation_kwargs=generation_kwargs
        )

    def __generate_internvl(
        self,
        loaded_model: LoadedTransformersModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> GenerationMethods:
        # The processor is actually a Qwen2TokenizerFast
        internvl_patch_model(loaded_model.model, loaded_model.processor)  # type: ignore[reportArgumentType]

        inputs, attention_mask, pixel_values = internvl_apply_chat_template(
            loaded_model.model,
            cast("PreTrainedTokenizer", loaded_model.processor),
            request.messages,  # type: ignore[reportArgumentType]
        )

        text_input_length = len(cast("torch.Tensor", inputs[0]))
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
                "max_new_tokens": request.max_tokens or self.__determine_default_max_tokens(loaded_model.data),
                "eos_token_id": [151645],
            }
        )

        return InternVLTransformersGenerationMethods(
            model=loaded_model.model, streamer=streamer, seed=request.seed, generation_kwargs=generation_kwargs
        )

    def __make_streamer_tokenizer(self, loaded_model: LoadedTransformersModel) -> Any:
        if loaded_model.tokenizer is not None:
            return loaded_model.tokenizer
        else:
            if loaded_model.processor is None:
                raise ValueError(f"Model {loaded_model.data.id} has neither tokenizer nor processor!")

            return ProcessorToTokenizerAdapter(loaded_model.processor)

    @override
    def _generate(
        self, request: TextGenerationRequest
    ) -> tuple[BackendGenerationTask, CompletionChunkStreamer, GenerationStatistics]:
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
            cast("AutoTokenizer", self.__make_streamer_tokenizer(loaded_model)),
            gen_stats,
            skip_prompt=True,
            timeout=self._server_config.single_token_generation_timeout / 1000,
            skip_special_tokens=True,
            model_quirks=model_quirks,
        )

        chunk_streamer = CompletionChunkStreamer(
            text_streamer=streamer,
            model_quirks=model_quirks,
            model_tool_call_config=MODEL_TOOL_CALL_CONFIGS.get(generation_variant, DEFAULT_MODEL_TOOL_CALL_CONFIG),
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

        return task, chunk_streamer, gen_stats
