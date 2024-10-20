from pathlib import Path

from azarrot.utils.merged_file import MergedReadOnlyBinaryFile
from tests.unit import create_temp_file


def test_read_from_single_file() -> None:
    base_file1 = create_temp_file("test content1")
    merged_file = MergedReadOnlyBinaryFile([Path(base_file1.name)])

    with merged_file:
        assert merged_file.read() == b"test content1"


def test_read_from_multiple_files() -> None:
    base_file1 = create_temp_file("content1")
    base_file2 = create_temp_file("content2")
    base_file3 = create_temp_file("content3")

    merged_file = MergedReadOnlyBinaryFile([Path(base_file1.name), Path(base_file2.name), Path(base_file3.name)])

    with merged_file:
        assert merged_file.read() == b"content1content2content3"


def test_read_from_multiple_files_another_order() -> None:
    base_file1 = create_temp_file("content1")
    base_file2 = create_temp_file("content2")
    base_file3 = create_temp_file("content3")

    merged_file = MergedReadOnlyBinaryFile([Path(base_file2.name), Path(base_file1.name), Path(base_file3.name)])

    with merged_file:
        assert merged_file.read() == b"content2content1content3"
