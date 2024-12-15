import logging
import threading
from pathlib import Path
from types import MethodType
from typing import Any, cast

import openvino
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction
from transformers import (
    PreTrainedModel,
)
from typing_extensions import override

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import (
    Model,
)

OPENVINO_TASK_MODEL_MAP = {
    "text-generation": OVModelForCausalLM,
    "text-generation-with-past": OVModelForCausalLM,
    "feature-extraction": OVModelForFeatureExtraction,
}

BACKEND_ID_OPENVINO = "openvino"


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


class OpenVINOBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()
    _default_device: str = "CPU"

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

        return "CPU"

    def __patch_model(self, original_model: Any) -> Any:
        cast(Any, original_model).compiled_model = None
        original_model.compile = MethodType(patched_compile, original_model)
        return original_model

    @override
    def _get_model_class(self, task: str) -> Any | None:
        return OPENVINO_TASK_MODEL_MAP.get(task)

    @override
    def _customize_model_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        model_path = model.path.absolute()
        openvino_model_file_path = model_path / Path("openvino_model.xml")
        need_export = not openvino_model_file_path.exists()
        need_load_in_4bit = need_export and not model.use_original_precision
        model_kwargs["export"] = need_export
        model_kwargs["load_in_4bit"] = need_load_in_4bit

        model_kwargs["ov_config"] = {"PERFORMANCE_HINT": "THROUGHPUT"}

        model_kwargs["use_cache"] = model.task == "text-generation-with-past"

    @override
    def _customize_loaded_model(self, model: PreTrainedModel) -> PreTrainedModel:
        ov_model = self.__patch_model(model)
        ov_model.compile()
        return ov_model

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
