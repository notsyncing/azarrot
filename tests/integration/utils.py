import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from openai import OpenAI

from azarrot.common_data import EmbeddingModelInfo, Model, ModelPreset
from azarrot.file_store import FileStore
from azarrot.server import Server

if TYPE_CHECKING:
    from azarrot.frontends.openai_frontend import OpenAIFrontend


def create_openai_client(server: Server) -> OpenAI:
    return OpenAI(base_url=f"http://{server.config.host}:{server.config.port}/openai/v1", api_key="__TEST__")


def create_temp_file(content: str):  # type: ignore[no-untyped-def]    # noqa: ANN201
    file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt")  # noqa: SIM115
    file.write(content.encode("utf-8"))
    return file


def get_file_store(server: Server) -> FileStore:
    return cast("OpenAIFrontend", server.frontends[0])._openai_files._file_store


def create_fake_embedding_model(model_id: str) -> Model:
    return Model(
        id=model_id,
        backend="openvino",
        path=Path("./"),
        revision=model_id,
        task="feature-extraction",
        generation_variant="normal",
        use_original_precision=True,
        is_for_raw_completion=False,
        preset=ModelPreset(preferred_locale=None, supports_tool_calling=False, enable_internal_tools=False),
        transformers=None,
        openvino=None,
        pytorch=None,
        info=EmbeddingModelInfo(1024),
        create_time=datetime.now(),
    )
