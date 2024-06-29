import logging
import openvino


class OpenVINOBackend:
    _log = logging.getLogger(__name__)
    _ov = openvino.Core()


    def __print_device_list(self) -> None:
        self._log.info("Available devices:")

        for device in self._ov.available_devices:
            self._log.info(f"{device}: {self._ov.get_property(device, "FULL_DEVICE_NAME")}")


    def __init__(self):
        self.__print_device_list()