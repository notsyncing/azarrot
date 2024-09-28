import gc
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, cast

import openvino
import torch
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction
from torch import Tensor
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CustomTextIteratorStreamer,
    GenerationHandlers,
    TransformersGenerationMethods,
    to_transformers_chat_messages,
)
from azarrot.common_data import EmbeddingsGenerationRequest, GenerationStatistics, Model, TextGenerationRequest
from azarrot.config import ServerConfig
from azarrot.models.model_quirks import MODEL_GENERATION_QUIRKS

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
    tokenizer: PreTrainedTokenizer
    device: str


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

    def get_tensor(self, *args, **kwargs) -> openvino.runtime.Tensor:  # type: ignore[no-untyped-def]    # noqa: ANN002, ANN003
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

        if isinstance(self.request, openvino.runtime.InferRequest):
            self.compiled_model = self.request.get_compiled_model()
        else:
            self.compiled_model = self.request

        self.request = ThreadLocalAwareInferRequest(self.compiled_model)


class OpenVINOBackend(BaseBackend):
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _models: dict[str, LoadedModel]
    _default_device: str = "CPU"

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)

        self._server_config = config
        self._models = {}

        self.__print_device_list()

        for device in self._ov.available_devices:
            device_type = self._ov.get_property(device, "DEVICE_TYPE")

            if device_type == openvino.properties.device.Type.DISCRETE:
                self._default_device = device
                break

        self._log.info("Using default device: %s", self._default_device)

    def id(self) -> str:
        return BACKEND_ID_OPENVINO

    def __print_device_list(self) -> None:
        self._log.info("OpenVINO Available devices:")

        for device in self._ov.available_devices:
            self._log.info(
                "%s (%s): %s",
                device,
                self._ov.get_property(device, "DEVICE_TYPE"),
                self._ov.get_property(device, "FULL_DEVICE_NAME"),
            )

    def __patch_model(self, original_model: Any) -> Any:
        cast(Any, original_model).compiled_model = None
        original_model.compile = MethodType(patched_compile, original_model)
        return original_model

    def load_model(self, model: Model) -> None:
        if model.task not in TASK_MODEL_MAP:
            self._log.error("Model %s (%s) wants task %s, which is not supported!", model.id, model.path, model.task)

            return

        if model.id in self._models:
            self._log.warning("Model %s is already loaded, will skip it.", model.id)
            return

        model_class = TASK_MODEL_MAP[model.task]
        model_path = model.path.absolute()
        openvino_model_file_path = model_path / Path("openvino_model.xml")
        need_export = not openvino_model_file_path.exists()
        need_load_in_4bit = need_export

        device = self._server_config.model_device_map.get(model.id, self._default_device)

        self._log.info("Loading model %s from %s to device %s", model.id, model.path, device)

        ov_config = {"PERFORMANCE_HINT": "THROUGHPUT"}

        use_cache = model.task == "text-generation-with-past"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        ov_model: Any = self.__patch_model(
            model_class.from_pretrained(
                model_path,
                device=device,
                use_cache=use_cache,
                ov_config=ov_config,
                compile=False,
                export=need_export,
                load_in_4bit=need_load_in_4bit,
                trust_remote_code=True,
            )
        )

        ov_model.eval()
        ov_model.compile()

        self._models[model.id] = LoadedModel(model, ov_model, tokenizer, device)

        self._log.info("Loaded model %s", model.id)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warning("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def __get_model(self, model_id: str) -> LoadedModel:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} is not loaded!")

        return self._models[model_id]

    def _parse_device_str(self, device_str: str) -> list[str]:
        def sanitize_device(device: str) -> str:
            if device != "CPU" and "." not in device:
                return device + ".0"
            else:
                return device

        if device_str is None or device_str == "":
            return []

        return [sanitize_device(d.strip().upper()) for d in device_str.split(",")]

    def _generate(
        self, request: TextGenerationRequest, generation_handlers: GenerationHandlers
    ) -> tuple[BackendGenerationTask, CustomTextIteratorStreamer, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)

        inputs = loaded_model.tokenizer.apply_chat_template(
            to_transformers_chat_messages(request.messages), return_tensors="pt"
        )

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.max,
            end_time=datetime.max,
            prompt_tokens=len(cast(Tensor, inputs[0])),
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

        generation_kwargs = {
            "inputs": inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
            "do_sample": True,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        task = BackendGenerationTask(
            model_id=loaded_model.info.id,
            model_quirks=model_quirks,
            backend_id=BACKEND_ID_OPENVINO,
            methods=TransformersGenerationMethods(
                model=loaded_model.model, streamer=streamer, seed=request.seed, generation_kwargs=generation_kwargs
            ),
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
