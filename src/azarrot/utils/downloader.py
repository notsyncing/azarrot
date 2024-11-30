import logging
import os
import urllib.request
import uuid
from os import environ
from pathlib import Path

from azarrot.config import ENV_AZARROT_TEST_MODE, ENV_AZARROT_TEST_RESOURCES_ROOT

_log = logging.getLogger(__name__)


def __check_path(path: Path, base: Path) -> None:
    prefix = Path(os.path.commonpath([path, base]))

    if prefix != base:
        raise ValueError("Target path %s is out of working directory %s", path, base)


def download_file_to_store(
    url: str,
    target_directory: Path,
    target_directory_is_full_path: bool = False,
    file_extension: str = ".file"
) -> Path:
    is_test_mode = environ.get(ENV_AZARROT_TEST_MODE) == "True"
    test_resources_root = environ.get(ENV_AZARROT_TEST_RESOURCES_ROOT)

    local_file: Path

    if url.startswith(("http://", "https://")):
        if target_directory_is_full_path:
            local_file = target_directory
        else:
            local_file = (target_directory / (str(uuid.uuid4()) + file_extension)).resolve()

        _log.info("Downloading image from %s to %s", url, local_file)
        urllib.request.urlretrieve(url, local_file)  # noqa: S310
    elif url.startswith("test-resources://") and is_test_mode:
        if test_resources_root is None:
            raise ValueError("Test mode enabled, but test resources root path is not specified!")

        local_file = (Path(test_resources_root) / Path(url[len("test-resources://") :])).resolve()
    else:
        local_file = (target_directory / url).resolve()
        __check_path(local_file, target_directory)

    return local_file
