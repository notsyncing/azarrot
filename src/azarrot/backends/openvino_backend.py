import gc
import logging
from dataclasses import asdict, dataclass
from threading import Thread
from typing import Any, ClassVar, cast

import openvino
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from azarrot.common_data import Model
from azarrot.config import ServerConfig

TASK_MODEL_MAP = {
    "text-generation": OVModelForCausalLM,
    "text-generation-with-past": OVModelForCausalLM,
    "feature-extraction": OVModelForFeatureExtraction
}


@dataclass
class LoadedModel:
    info: Model
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


@dataclass
class GenerationMessage:
    role: str
    content: str


@dataclass
class GenerationRequest:
    model_id: str
    messages: list[GenerationMessage]


class OpenVINOBackend:
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _server_config: ServerConfig
    _models: ClassVar[dict[str, LoadedModel]] = {}


    def __print_device_list(self) -> None:
        self._log.info("Available devices:")

        for device in self._ov.available_devices:
            self._log.info(f"{device}: {self._ov.get_property(device, "FULL_DEVICE_NAME")}")


    def __init__(self, config: ServerConfig) -> None:
        self._server_config = config
        self.__print_device_list()


    def load_model(self, model: Model) -> None:
        if model.task not in TASK_MODEL_MAP:
            self._log.error("Model %s (%s) wants task %s, which is not supported!", model.id, model.path,
                model.task)

            return

        if model.id in self._models:
            self._log.warn("Model %s is already loaded, will skip it.", model.id)
            return

        model_class = TASK_MODEL_MAP[model.task]
        model_path = model.path.absolute()

        self._log.info("Loading model %s from %s", model.id, model.path)

        use_cache = model.task == "text-generation-with-past"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        ov_model: Any = model_class.from_pretrained(model_path, use_cache=use_cache)
        self._models[model.id] = LoadedModel(model, ov_model, tokenizer)

        self._log.info("Loaded model %s from %s", model.id, model.path)


    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warn("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)


    def __to_transformers_chat_messages(self, messages: list[GenerationMessage]) -> list[dict[str, str]]:
        return [asdict(m) for m in messages]


    def generate(self, request: GenerationRequest) -> tuple[Model, TextIteratorStreamer]:
        if request.model_id not in self._models:
            raise ValueError(f"Model {request.model_id} is not loaded!")

        loaded_model = self._models[request.model_id]

        inputs = loaded_model.tokenizer.apply_chat_template(
            self.__to_transformers_chat_messages(request.messages),
            return_tensors="pt"
        )

        streamer = TextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer),
            skip_prompt=True,
        )

        generation_kwargs = {
            "inputs": inputs,
            "streamer": streamer,
            "max_new_tokens": 2048,
        }

        thread = Thread(target=loaded_model.model.generate, kwargs=generation_kwargs)
        thread.start()

        return loaded_model.info, streamer
