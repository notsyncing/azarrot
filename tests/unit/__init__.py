import tempfile


def create_temp_file(content: str):  # noqa: ANN201
    file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt")
    file.write(content.encode("utf-8"))
    file.flush()
    return file
