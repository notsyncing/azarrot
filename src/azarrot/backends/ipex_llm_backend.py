import gc
import logging
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

    def __init__(self, config: ServerConfig) -> None:
        self._server_config = config
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

        ipex_model: Any = model_class.from_pretrained(
            model_path,
            load_in_4bit=True,
            use_cache=True,
            optimize_model=True,
            trust_remote_code=True,
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

    def generate(self, request: TextGenerationRequest) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)

        inputs = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt"
        )

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=len(cast(torch.Tensor, inputs[0])),
            total_tokens=0
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

        thread = Thread(target=loaded_model.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer, gen_stats

    def generate_embeddings(self, request: EmbeddingsGenerationRequest) -> tuple[list[float], GenerationStatistics]:
        raise NotImplementedError
