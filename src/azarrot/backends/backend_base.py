import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from queue import Empty, Queue
from typing import ClassVar

from azarrot.backends.common import CustomTextIteratorStreamer, GenerationHandlers, GenerationMethods
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationStatistics,
    Model,
    ModelQuirks,
    TextGenerationRequest,
)
from azarrot.config import ServerConfig


@dataclass
class BackendGenerationTask:
    model_id: str
    model_quirks: ModelQuirks | None
    backend_id: str
    methods: GenerationMethods | None
    device: str
    seed: int | None

    def is_batchable_with(self, other: "BackendGenerationTask") -> bool:
        return (
            (self.model_quirks is None or not self.model_quirks.does_not_support_batching)
            and self.model_id == other.model_id
            and self.backend_id == other.backend_id
            and self.device == other.device
            and self.seed == other.seed
        )


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

    def is_batchable_with(self, other: "TaskReference") -> bool:
        return not isinstance(other, BubbleTaskReference) and self.task.is_batchable_with(other.task)

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

    def update_start_generation_time(self, time: datetime) -> None:
        if self.task.methods is not None:
            self.task.methods.streamer.update_start_generation_time(time)

    def execution_failed(self) -> None:
        if self.task.methods is not None:
            self.task.methods.streamer.set_failed()


class BubbleTaskReference(TaskReference):
    original_ref: TaskReference

    def __init__(self, original_ref: TaskReference) -> None:
        super().__init__(
            BackendGenerationTask(
                model_id=original_ref.task.model_id,
                model_quirks=original_ref.task.model_quirks,
                backend_id=original_ref.task.backend_id,
                methods=None,
                device="__BUBBLE__",
                seed=original_ref.task.seed,
            )
        )

        self.original_ref = original_ref
        self.original_ref.add_dependency(self)

    def is_batchable_with(self, other: "TaskReference") -> bool:  # noqa: ARG002
        return False

    def is_done(self) -> bool:
        return self.original_ref.is_done()

    def wait_done(self) -> None:
        self.original_ref.wait_done()


@dataclass
class DeviceWorkerConfig:
    device: str
    auto_batch_threshold: int = 100
    auto_batch_max_size: int = 8


class DeviceWorker:
    _log = logging.getLogger(__name__)
    _config: DeviceWorkerConfig
    _queue: Queue[TaskReference]
    _thread: threading.Thread
    _stop = False

    def __init__(self, config: DeviceWorkerConfig) -> None:
        self._config = config
        self._queue = Queue()
        self._thread = threading.Thread(target=self.__worker_loop, daemon=True)
        self._thread.start()

        self._log.info(f"Device worker {self} for {config.device} has started.")

    def __fetch_tasks(self) -> list[TaskReference]:
        try:
            first_task_ref = self._queue.get(timeout=1)
            tasks = [first_task_ref]

            if self._config.auto_batch_threshold > 0:
                time.sleep(self._config.auto_batch_threshold / 1000)

                batched_count = 1
                return_list = []

                while batched_count < self._config.auto_batch_max_size:
                    try:
                        next_task_ref = self._queue.get_nowait()

                        if next_task_ref.is_batchable_with(first_task_ref):
                            tasks.append(next_task_ref)
                            batched_count = batched_count + 1
                        else:
                            return_list.append(next_task_ref)
                    except Empty:
                        break

                for t in return_list:
                    self._queue.put(t)
        except Empty:
            return []
        else:
            return tasks

    def __worker_loop(self) -> None:
        while True:
            if self._stop:
                break

            task_ref_list = self.__fetch_tasks()
            fetched_count = len(task_ref_list)

            if fetched_count <= 0:
                continue

            if fetched_count == 1:
                self._log.info(f"Device worker {self._config.device} is executing task {task_ref_list[0]}")
            else:
                self._log.info(f"Device worker {self._config.device} is batching {fetched_count} tasks together.")

            for task_ref in task_ref_list:
                task_ref.update_start_generation_time(datetime.now())
                task_ref.mark_self_as_ready()

            first_task_methods = task_ref_list[0].task.methods
            assert first_task_methods is not None

            if fetched_count > 1:
                other_task_methods = [t.task.methods for t in task_ref_list[1:]]
                first_task_methods.merge_into_batch(other_task_methods)

            success = first_task_methods.generate()

            if not success:
                for task_ref in task_ref_list:
                    task_ref.execution_failed()

            for task_ref in task_ref_list:
                task_ref.mark_self_as_done()

            if fetched_count == 1:
                self._log.info(f"Device worker {self._config.device} has done task {task_ref_list[0]}")
            else:
                self._log.info(f"Device worker {self._config.device} has done a batch of {fetched_count} tasks.")

    def put(self, task: TaskReference) -> None:
        self._log.info(f"Device worker {self._config.device} has received task {task}")
        self._queue.put(task)

    def stop(self) -> None:
        self._stop = True


class BaseBackend(ABC):
    _device_queues: ClassVar[dict[str, DeviceWorker]] = {}
    _device_queue_lock: ClassVar[threading.Lock] = threading.Lock()

    _server_config: ServerConfig

    def __init__(self, server_config: ServerConfig) -> None:
        self._server_config = server_config

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

    def __get_device_worker(self, device: str) -> DeviceWorker:
        with BaseBackend._device_queue_lock:
            if device not in BaseBackend._device_queues:
                worker = DeviceWorker(
                    DeviceWorkerConfig(
                        device,
                        auto_batch_threshold=self._server_config.auto_batch_threshold,
                        auto_batch_max_size=self._server_config.auto_batch_max_size,
                    )
                )

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
        first_device_worker = self.__get_device_worker(first_device)
        first_device_worker.put(first_device_task_ref)

        if len(target_devices) > 1:
            for device in islice(target_devices, 1, None):
                device_task = BubbleTaskReference(first_device_task_ref)
                device_worker = self.__get_device_worker(device)
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
