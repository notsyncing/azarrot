import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from queue import Empty, Queue
from typing import ClassVar

from azarrot.backends.common import CustomTextIteratorStreamer, GenerationHandlers
from azarrot.common_data import EmbeddingsGenerationRequest, GenerationStatistics, Model, TextGenerationRequest


@dataclass
class BackendGenerationTask:
    method: Callable[[], None]
    device: str


class TaskReference:
    task: BackendGenerationTask
    dependencies: list["TaskReference"]

    _ready: threading.Event
    _done: threading.Event

    def __init__(self, task: BackendGenerationTask) -> None:
        self.dependencies = []
        self.task = task

        self._ready = threading.Event()
        self._done = threading.Event()

    def add_dependency(self, dependency: "TaskReference") -> None:
        self.dependencies.append(dependency)

    def mark_self_as_ready(self) -> None:
        self._ready.set()

    def is_ready(self) -> bool:
        return self._ready.is_set() and all(d.is_ready() for d in self.dependencies)

    def wait_all_ready(self) -> None:
        self._ready.wait()

        for dep in self.dependencies:
            dep.wait_all_ready()

    def mark_self_as_done(self) -> None:
        self._done.set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait_done(self) -> None:
        self._done.wait()


class BubbleTaskReference(TaskReference):
    original_ref: TaskReference

    def __init__(self, original_ref: TaskReference) -> None:
        super().__init__(BackendGenerationTask(method=self.__empty_method, device="__BUBBLE__"))

        self.original_ref = original_ref
        self.original_ref.add_dependency(self)

    def __empty_method(self) -> None:
        pass

    def is_done(self) -> bool:
        return self.original_ref.is_done()

    def wait_done(self) -> None:
        self.original_ref.wait_done()


class DeviceWorker:
    _log = logging.getLogger(__name__)
    _device: str
    _queue: Queue[TaskReference]
    _thread: threading.Thread
    _stop = False

    def __init__(self, device: str) -> None:
        self._device = device
        self._queue = Queue()
        self._thread = threading.Thread(target=self.__worker_loop, daemon=True)
        self._thread.start()

        self._log.info(f"Device worker {self} for {device} has started.")

    def __worker_loop(self) -> None:
        while True:
            if self._stop:
                break

            try:
                task_ref = self._queue.get(timeout=1)
            except Empty:
                continue

            task_ref.mark_self_as_ready()

            self._log.info(f"Device worker {self._device} {self} is executing task {task_ref}")

            if isinstance(task_ref, BubbleTaskReference):
                task_ref.wait_done()
            else:
                task = task_ref.task
                task.method()

            task_ref.mark_self_as_done()

            self._log.info(f"Device worker {self._device} {self} has done executing task {task_ref}")

    def put(self, task: TaskReference) -> None:
        self._log.info(f"Device worker {self._device} {self} has received task {task}")
        self._queue.put(task)

    def stop(self) -> None:
        self._stop = True


class BaseBackend(ABC):
    _device_queues: ClassVar[dict[str, DeviceWorker]] = {}
    _device_queue_lock: ClassVar[threading.Lock] = threading.Lock()

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def load_model(self, model: Model) -> None:
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        pass

    @abstractmethod
    def _parse_device_str(self, device_str: str) -> list[str]:
        pass

    @staticmethod
    def __get_device_worker(device: str) -> DeviceWorker:
        with BaseBackend._device_queue_lock:
            if device not in BaseBackend._device_queues:
                worker = DeviceWorker(device)
                BaseBackend._device_queues[device] = worker
                return worker
            else:
                return BaseBackend._device_queues[device]

    def generate(
        self, request: TextGenerationRequest, generation_handlers: GenerationHandlers
    ) -> tuple[CustomTextIteratorStreamer, GenerationStatistics]:
        task, streamer, gen_stats = self._generate(request, generation_handlers)
        target_devices = self._parse_device_str(task.device)

        if len(target_devices) < 0:
            raise ValueError(f"Unknown device {task.device} in generation task!")

        first_device = target_devices[0]
        first_device_task_ref = TaskReference(task)
        first_device_worker = BaseBackend.__get_device_worker(first_device)
        first_device_worker.put(first_device_task_ref)

        if len(target_devices) > 1:
            for device in islice(target_devices, 1, None):
                device_task = BubbleTaskReference(first_device_task_ref)
                device_worker = BaseBackend.__get_device_worker(device)
                device_worker.put(device_task)

        return streamer, gen_stats

    @abstractmethod
    def _generate(
        self, request: TextGenerationRequest, generation_handlers: GenerationHandlers
    ) -> tuple[BackendGenerationTask, CustomTextIteratorStreamer, GenerationStatistics]:
        pass

    @abstractmethod
    def generate_embeddings(self, request: EmbeddingsGenerationRequest) -> tuple[list[float], GenerationStatistics]:
        pass
