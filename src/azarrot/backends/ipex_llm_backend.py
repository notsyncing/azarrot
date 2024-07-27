import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Any, ClassVar, cast

import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.common import CountedTextIteratorStreamer, to_transformers_chat_messages
from azarrot.backends.ipex_llm_support.internvl2_processor import (
    internvl2_apply_chat_template,
    internvl2_patch_model,
)
from azarrot.common_data import EmbeddingsGenerationRequest, GenerationStatistics, Model, TextGenerationRequest
from azarrot.config import ServerConfig

TASK_MODEL_MAP = {
    "text-generation": AutoModelForCausalLM,
}

BACKEND_ID_IPEX_LLM = "ipex-llm"


@dataclass
class LoadedModel:
    info: Model
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: str


class IPEXLLMBackend(BaseBackend):
    _log = logging.getLogger(__name__)
    _server_config: ServerConfig
    _models: ClassVar[dict[str, LoadedModel]] = {}

    _generation_variants: dict[
        str, Callable[[LoadedModel, TextGenerationRequest], tuple[TextIteratorStreamer, GenerationStatistics]]
    ]

    def __init__(self, config: ServerConfig) -> None:
        self._server_config = config
        self.__print_device_list()

        self._generation_variants = {
            "normal": self.__generate_normal,
            "internvl2": self.__generate_internvl2,
        }

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
            self._log.warn("Model %s is already loaded, will skip it.", model.id)
            return

        model_class = TASK_MODEL_MAP[model.task]
        model_path = model.path.absolute()

        device = self._server_config.model_device_map.get(model.id, "xpu")

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model_kwargs = {}

        if model.ipex_llm is not None:
            if model.ipex_llm.use_cache:
                model_kwargs["use_cache"] = True

        ipex_model: Any = model_class.from_pretrained(
            model_path,
            load_in_4bit=True,
            optimize_model=True,
            trust_remote_code=True,
            **model_kwargs
        ).to(device)

        self._models[model.id] = LoadedModel(model, ipex_model, tokenizer, device)

        self._log.info("Loaded model %s", model.id)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warn("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        torch.xpu.empty_cache()
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def __get_model(self, model_id: str) -> LoadedModel:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} is not loaded!")

        return self._models[model_id]

    def __generate_normal(
        self,
        loaded_model: LoadedModel,
        request: TextGenerationRequest
    ) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        inputs = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt"
        )

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=len(cast(torch.Tensor, inputs[0])),
            completion_tokens=0
        )

        streamer = CountedTextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer),
            gen_stats,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = {
            "inputs": cast(torch.Tensor, inputs).to(loaded_model.device),
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
        }

        def generate_method() -> None:
            try:
                loaded_model.model.generate(**generation_kwargs)
            except:
                streamer.set_failed()
                self._log.exception("An exception occurred in generation thread")

        thread = Thread(target=generate_method)
        thread.start()

        return streamer, gen_stats

    def __generate_internvl2(
        self,
        loaded_model: LoadedModel,
        request: TextGenerationRequest
    ) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        internvl2_patch_model(loaded_model.model, loaded_model.tokenizer)

        inputs, pixel_values = internvl2_apply_chat_template(
            loaded_model.model,
            loaded_model.tokenizer,
            request.messages
        )

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=len(cast(torch.Tensor, inputs[0])) + (len(pixel_values) if pixel_values is not None else 0),
            completion_tokens=0
        )

        streamer = CountedTextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer),
            gen_stats,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # token id 2 is from tokenizer.json ('</s>')
        attention_mask = loaded_model.model._prepare_attention_mask_for_generation(   # noqa: SLF001
            inputs, torch.Tensor([2]), torch.Tensor([2])
        )

        if pixel_values is not None:
            pixel_values = pixel_values.to(loaded_model.device)

        generation_kwargs = {
            "input_ids": cast(torch.Tensor, inputs).to(loaded_model.device),
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,

            # token id list is taken from https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/conversation.py#368
            "eos_token_id": [2, 92543, 92542],
        }

        def generate_method() -> None:
            try:
                loaded_model.model.generate(**generation_kwargs)
            except:
                streamer.set_failed()
                self._log.exception("An exception occurred in generation thread")

        thread = Thread(target=generate_method)
        thread.start()

        return streamer, gen_stats

    def generate(self, request: TextGenerationRequest) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)
        generation_variant = "normal"

        if loaded_model.info.ipex_llm is not None:
            generation_variant = loaded_model.info.ipex_llm.generation_variant

        generation_method = self._generation_variants[generation_variant]
        return generation_method(loaded_model, request)

    def generate_embeddings(self, request: EmbeddingsGenerationRequest) -> tuple[list[float], GenerationStatistics]:
        raise NotImplementedError
