from logging import Logger

import torch


def parse_pytorch_device_str(device_str: str) -> list[str]:
    def sanitize_device(device: str) -> str:
        if device.isdigit():
            return device
        elif device != "cpu" and ":" not in device:
            return device + ":0"
        else:
            return device

    if device_str is None or device_str == "":
        return []

    return [sanitize_device(d.strip().lower()) for d in device_str.split(",")]


def print_pytorch_device_list(logger: Logger, backend_id: str) -> int:
    logger.info("%s Available devices:", backend_id)
    xpu_count = torch.xpu.device_count()

    for i in range(xpu_count):
        logger.info("XPU #%s: %s", i, str(torch.xpu.get_device_properties(i)))

    return xpu_count


def determine_pytorch_default_device(accel_device_count: int) -> str:
    if accel_device_count <= 0:
        return "cpu"
    else:
        return "xpu"
