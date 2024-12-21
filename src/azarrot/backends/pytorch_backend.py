import logging
from typing import Any

from typing_extensions import override

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import Model

BACKEND_ID_PYTORCH = "pytorch"


class PyTorchBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)

    @override
    def id(self) -> str:
        return BACKEND_ID_PYTORCH

    @override
    def _customize_model_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["low_cpu_mem_usage"] = True

        # TODO: Enable this when bitsandbytes is usable
        # if not model.use_original_precision:
        #     model_kwargs["quantization_config"] = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.bfloat16
        #     )
