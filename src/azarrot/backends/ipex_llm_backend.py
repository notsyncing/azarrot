import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, set_seed

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CustomTextIteratorStreamer,
    GenerationHandlers,
    StopGenerationError,
    to_transformers_chat_messages,
)
from azarrot.backends.ipex_llm_support.internvl2_processor import (
    internvl2_apply_chat_template,
    internvl2_patch_model,
)
from azarrot.common_data import EmbeddingsGenerationRequest, GenerationStatistics, Model, TextGenerationRequest
from azarrot.config import ServerConfig
from azarrot.models.model_quirks import MODEL_GENERATION_QUIRKS

TASK_MODEL_MAP = {
    "text-generation": AutoModelForCausalLM,
}

BACKEND_ID_IPEX_LLM = "ipex-llm"

MODEL_IPEXLLM_QUIRKS = {"internvl2": {"use_cache": False}}


@dataclass
class LoadedModel:
    info: Model
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: str


class IPEXLLMBackend(BaseBackend):
    _log = logging.getLogger(__name__)
    _server_config: ServerConfig
    _models: dict[str, LoadedModel]

    _generation_variants: dict[
        str,
        Callable[
            [LoadedModel, TextGenerationRequest, dict[str, Any], CustomTextIteratorStreamer, GenerationStatistics],
            Callable[[], None],
        ],
    ]

    def __init__(self, config: ServerConfig) -> None:
        super().__init__()

        self._server_config = config
        self._models = {}

        self._generation_variants = {
            "normal": self.__generate_normal,
            "internvl2": self.__generate_internvl2,
        }

        self.__print_device_list()

    def id(self) -> str:
        return BACKEND_ID_IPEX_LLM

    def __print_device_list(self) -> None:
        self._log.info("IPEX-LLM Available devices:")

        for i in range(torch.xpu.device_count()):
            self._log.info("XPU #%s: %s", i, str(torch.xpu.get_device_properties(i)))

    def load_model(self, model: Model) -> None:
        if model.task not in TASK_MODEL_MAP:
            self._log.error("Model %s (%s) wants task %s, which is not supported!", model.id, model.path, model.task)

            return

        if model.id in self._models:
            self._log.warning("Model %s is already loaded, will skip it.", model.id)
            return

        model_class = TASK_MODEL_MAP[model.task]
        model_path = model.path.absolute()

        device = self._server_config.model_device_map.get(model.id, "xpu")

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model_kwargs = {}
        generation_variant = model.generation_variant

        if model.ipex_llm is not None:
            if model.ipex_llm.use_cache:
                model_kwargs["use_cache"] = True

        model_quirks = MODEL_IPEXLLM_QUIRKS.get(generation_variant, {})
        model_kwargs.update(model_quirks)

        if "use_cache" in model_kwargs and not model_kwargs.get("use_cache"):
            del model_kwargs["use_cache"]

        ipex_model: Any = model_class.from_pretrained(
            model_path, load_in_4bit=True, optimize_model=True, trust_remote_code=True, **model_kwargs
        ).to(device)

        self._models[model.id] = LoadedModel(model, ipex_model, tokenizer, device)

        self._log.info("Loaded model %s", model.id)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warning("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        torch.xpu.empty_cache()
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def __get_model(self, model_id: str) -> LoadedModel:
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

    def __make_generation_method(
        self,
        loaded_model: LoadedModel,
        streamer: CustomTextIteratorStreamer,
        seed: int | None,
        generation_kwargs: dict[str, Any]
    ) -> Callable[[], None]:
        def generate_method() -> None:
            if seed is not None:
                set_seed(seed)

            try:
                loaded_model.model.generate(**generation_kwargs)
            except StopGenerationError:
                return
            except:
                streamer.set_failed()
                self._log.exception("An exception occurred in generation thread")

            if seed is not None:
                set_seed(int(datetime.now().timestamp()))

        return generate_method

    def __generate_normal(
        self,
        loaded_model: LoadedModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> Callable[[], None]:
        inputs = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt"
        )

        gen_stats.prompt_tokens = len(cast(torch.Tensor, inputs[0]))

        generation_kwargs = common_generation_kwargs.copy()

        generation_kwargs.update(
            {
                "inputs": cast(torch.Tensor, inputs).to(loaded_model.device),
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
            }
        )

        return self.__make_generation_method(loaded_model, streamer, request.seed, generation_kwargs)

    def __generate_internvl2(
        self,
        loaded_model: LoadedModel,
        request: TextGenerationRequest,
        common_generation_kwargs: dict[str, Any],
        streamer: CustomTextIteratorStreamer,
        gen_stats: GenerationStatistics,
    ) -> Callable[[], None]:
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
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "streamer": streamer,
                "max_new_tokens": request.max_tokens,
                # token id list is taken from https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/conversation.py#368
                "eos_token_id": [2, 92543, 92542],
            }
        )

        return self.__make_generation_method(loaded_model, streamer, request.seed, generation_kwargs)

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

        streamer = CustomTextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer),
            gen_stats,
            skip_prompt=True,
            skip_special_tokens=True,
            model_quirks=MODEL_GENERATION_QUIRKS.get(loaded_model.info.generation_variant),
            generation_handlers=generation_handlers,
        )

        common_generation_kwargs = {"do_sample": True, "temperature": request.temperature, "top_p": request.top_p}

        m = generation_method(loaded_model, request, common_generation_kwargs, streamer, gen_stats)

        task = BackendGenerationTask(method=m, device=loaded_model.device)

        return task, streamer, gen_stats

    def generate_embeddings(self, request: EmbeddingsGenerationRequest) -> tuple[list[float], GenerationStatistics]:
        raise NotImplementedError
