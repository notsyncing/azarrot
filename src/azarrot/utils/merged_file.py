import io
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class MergedReadOnlyBinaryFile(BinaryIO):
    _file_readers: list[io.BufferedReader]
    _file_sizes: list[int]
    _total_size: int
    _current_file_idx: int
    _position_in_current_file: int

    def __init__(self, file_paths: list[Path]) -> None:
        super().__init__()

        self._file_readers = []
        self._file_sizes = []
        self._total_size = 0
        self._current_file_idx = 0
        self._position_in_current_file = 0

        for file_name in file_paths:
            r = file_name.open("rb")
            self._file_readers.append(r)

            file_size = file_name.stat().st_size
            self._file_sizes.append(file_size)
            self._total_size += file_size

    def tell(self) -> int:
        return self._position_in_current_file + sum(self._file_sizes[: self._current_file_idx])

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.tell() + offset
        elif whence == io.SEEK_END:
            new_position = self._total_size + offset
        else:
            raise ValueError("Unknown whence value")

        if new_position < 0:
            new_position = 0
        elif new_position > self._total_size:
            new_position = self._total_size

        cumulative_size = 0
        self._current_file_idx = 0
        for i, file_size in enumerate(self._file_sizes):
            cumulative_size += file_size
            if new_position < cumulative_size:
                self._current_file_idx = i
                self._position_in_current_file = new_position - (cumulative_size - file_size)
                break

        for i, r in enumerate(self._file_readers):
            if i < self._current_file_idx:
                r.seek(0, io.SEEK_END)
            elif i == self._current_file_idx:
                r.seek(self._position_in_current_file)
            else:
                r.seek(0)

        return new_position

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        data = b""
        total_read = 0

        if size < 0:
            size = self._total_size

        while total_read < size and self._current_file_idx < len(self._file_readers):
            remaining_to_read = size - total_read

            if remaining_to_read <= 0:
                break

            file_reader = self._file_readers[self._current_file_idx]
            file_reader_remaining_length = len(file_reader.peek())

            if file_reader_remaining_length >= remaining_to_read:
                data += file_reader.read(remaining_to_read)
                total_read += remaining_to_read
            else:
                data += file_reader.read()
                total_read += file_reader_remaining_length
                self._current_file_idx += 1

        if total_read == 0:
            return b""

        return data

    def close(self) -> None:
        for r in self._file_readers:
            r.close()

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        b = self.read(1)

        if b == b"":
            raise StopIteration

        return b
