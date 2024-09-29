import tempfile

from openai import OpenAI

from azarrot.server import Server


def create_openai_client(server: Server) -> OpenAI:
    return OpenAI(
        base_url=f"http://{server.config.host}:{server.config.port}/v1", api_key="__TEST__"
    )


def create_temp_file(content: str):     # noqa: ANN201
    file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt")
    file.write(content.encode("utf-8"))
    return file
