import gc
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from types import MethodType
from typing import Any, ClassVar, cast

import openvino
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.common import CountedTextIteratorStreamer, to_transformers_chat_messages
from azarrot.common_data import GenerationRequest, GenerationStatistics, Model
from azarrot.config import ServerConfig

TASK_MODEL_MAP = {
    "text-generation": OVModelForCausalLM,
    "text-generation-with-past": OVModelForCausalLM,
    "feature-extraction": OVModelForFeatureExtraction,
}

BACKEND_ID_OPENVINO = "openvino"


@dataclass
class LoadedModel:
    info: Model
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


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

    def start_async(self, inputs: Any | None = None, userdata: Any | None = None, share_inputs: bool = False) -> None:    # noqa: ANN401
        req = self.__get_request()
        req.start_async(inputs, userdata, share_inputs)

    def wait(self) -> None:
        req = self.__get_request()
        req.wait()

    def get_tensor(self, *args, **kwargs) -> openvino.runtime.Tensor:   # type: ignore[no-untyped-def]    # noqa: ANN002, ANN003
        req = self.__get_request()
        return req.get_tensor(*args, **kwargs)


def patched_compile(self) -> None:    # type: ignore[no-untyped-def]    # noqa: ANN001
    if self.request is None:
        super(type(self), self).compile()   # type: ignore[unused-ignore]

        if isinstance(self.request, openvino.runtime.InferRequest):
            self.compiled_model = self.request.get_compiled_model()
        else:
            self.compiled_model = self.request

        self.request = ThreadLocalAwareInferRequest(self.compiled_model)


class OpenVINOBackend(BaseBackend):
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _server_config: ServerConfig
    _models: ClassVar[dict[str, LoadedModel]] = {}

    def __init__(self, config: ServerConfig) -> None:
        self._server_config = config
        self.__print_device_list()

    def id(self) -> str:
        return BACKEND_ID_OPENVINO

    def __print_device_list(self) -> None:
        self._log.info("OpenVINO Available devices:")

        for device in self._ov.available_devices:
            self._log.info("%s: %s", device, self._ov.get_property(device, "FULL_DEVICE_NAME"))

    def __patch_model(self, original_model: Any) -> Any:    # noqa: ANN401
        original_model.compiled_model = None
        original_model.compile = MethodType(patched_compile, original_model)
        return original_model

    def load_model(self, model: Model) -> None:
        if model.task not in TASK_MODEL_MAP:
            self._log.error("Model %s (%s) wants task %s, which is not supported!", model.id, model.path, model.task)

            return

        if model.id in self._models:
            self._log.warn("Model %s is already loaded, will skip it.", model.id)
            return

        model_class = TASK_MODEL_MAP[model.task]
        model_path = model.path.absolute()

        device = self._server_config.model_device_map.get(model.id, "CPU")

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        ov_config = {
            "PERFORMANCE_HINT": "THROUGHPUT"
        }

        use_cache = model.task == "text-generation-with-past"
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        ov_model: Any = self.__patch_model(
            model_class.from_pretrained(model_path, device=device, use_cache=use_cache,
                ov_config=ov_config, compile=False)
        )

        ov_model.compile()

        self._models[model.id] = LoadedModel(model, ov_model, tokenizer)

        self._log.info("Loaded model %s", model.id)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warn("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def generate(self, request: GenerationRequest) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        if request.model_id not in self._models:
            raise ValueError(f"Model {request.model_id} is not loaded!")

        loaded_model = self._models[request.model_id]

        inputs = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt"
        )

        gen_stats = GenerationStatistics(
            start_time=datetime.now(), end_time=datetime.max, prompt_tokens=len(cast(Tensor, inputs[0])), total_tokens=0
        )

        streamer = CountedTextIteratorStreamer(
            cast(AutoTokenizer, loaded_model.tokenizer), gen_stats, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            "inputs": inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
        }

        thread = Thread(target=loaded_model.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer, gen_stats
