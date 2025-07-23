import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from openai import OpenAI
from pydantic.fields import FieldInfo

from azarrot.common_data import EmbeddingModelInfo, LoadedModel, ModelPreset
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


def create_fake_embedding_model(model_id: str) -> LoadedModel:
    return LoadedModel(
        id=model_id,
        backend="openvino",
        path=Path("./"),
        revision=model_id,
        task="feature-extraction",
        generation_variant="normal",
        use_original_precision=True,
        is_for_raw_completion=False,
        is_reasoning_model=False,
        preset=ModelPreset(preferred_locale=None, supports_tool_calling=False, enable_internal_tools=False),
        transformers=None,
        openvino=None,
        pytorch=None,
        info=EmbeddingModelInfo(1024),
        device="cpu",
        create_time=datetime.now(),
        loaded_time=datetime.now(),
    )


def add_fields_to_pydantic_model(cls: Any, **field_definitions: Any) -> None:
    new_fields: dict[str, FieldInfo] = {}
    new_annotations: dict[str, type | None] = {}

    for f_name, f_def in field_definitions.items():
        if isinstance(f_def, tuple):
            try:
                f_annotation, f_value = f_def
            except ValueError as e:
                raise Exception(  # noqa: TRY002
                    "field definitions should either be a tuple of (<type>, <default>) or just a "
                    "default value, unfortunately this means tuples as "
                    "default values are not allowed"
                ) from e
        else:
            f_annotation, f_value = None, f_def

        if f_annotation:
            new_annotations[f_name] = f_annotation

        new_fields[f_name] = FieldInfo(annotation=f_annotation, default=f_value)

    cls.model_fields.update(new_fields)
    cls.model_rebuild(force=True)
