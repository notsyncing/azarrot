# ruff: noqa: SLF001

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from torch import Size, Tensor

# def (value: T, next_part_start_pos: int) -> tuple[T@PREV_PART, T@NEXT_PART]
type TensorRadixTreeValueSplitter[T] = Callable[[T, int], tuple[T, T]]

# def (values: list[T]) -> T@MERGED
type TensorRadixTreeValueMerger[T] = Callable[[list[T]], T]

# def (value: T) -> int
type TensorRadixTreeValueSizing[T] = Callable[[T], int]

# def (value: T) -> torch.Size:
type TensorRadixTreeValueShaping[T] = Callable[[T], Size]


class TensorRadixTree[T]:
    """Radix tree of tensors.

    Only support 1D tensors.
    """

    _parent: "TensorRadixTree[T] | None" = None
    _children: "list[TensorRadixTree[T]]"
    _key: Tensor | None = None
    _value: T | None = None
    _value_splitter: TensorRadixTreeValueSplitter[T]
    _value_merger: TensorRadixTreeValueMerger[T]
    _value_sizing: TensorRadixTreeValueSizing[T]
    _value_shaping: TensorRadixTreeValueShaping[T]
    _estimated_size: int = 0
    _last_access_time: datetime

    def __init__(
        self,
        value_splitter: TensorRadixTreeValueSplitter[T],
        value_merger: TensorRadixTreeValueMerger[T],
        value_sizing: TensorRadixTreeValueSizing[T],
        value_shaping: TensorRadixTreeValueShaping[T],
        key: Tensor | None = None,
        value: T | None = None,
        parent: "TensorRadixTree[T] | None" = None,
    ) -> None:
        self._parent = parent
        self._children = []
        self._key = key
        self._value = value
        self._value_splitter = value_splitter
        self._value_merger = value_merger
        self._value_sizing = value_sizing
        self._value_shaping = value_shaping
        self._last_access_time = datetime.now()

        self._estimated_size = 0

        self.__increase_estimated_size(1, [key] if key is not None else None, [value] if value is not None else None)

    @property
    def parent(self) -> "TensorRadixTree[T] | None":
        return self._parent

    @property
    def children(self) -> "list[TensorRadixTree[T]]":
        return self._children

    @property
    def key(self) -> Tensor | None:
        return self._key

    @property
    def value(self) -> T | None:
        return self._value

    @property
    def estimated_size(self) -> int:
        return self._estimated_size

    def get_total_estimated_size(self) -> int:
        total_size = self._estimated_size

        for child in self._children:
            total_size += child.get_total_estimated_size()

        return total_size

    def __match_tensors(self, first: Tensor, second: Tensor) -> tuple[bool, int]:
        first_len = len(first)
        second_len = len(second)
        compare_len = min(first_len, second_len)

        latest_match_pos = -1
        mismatch_found = False

        for i in range(compare_len):
            if first[i] != second[i]:
                mismatch_found = True
                break

            latest_match_pos = i

        if not mismatch_found and first_len == second_len:
            return True, -1
        else:
            return False, latest_match_pos

    def __calculate_estimated_size(
        self, new_node_count: int = 0, new_keys: list[Tensor] | None = None, new_values: list[T] | None = None
    ) -> int:
        # 32 is for node overhead
        # 16 is for tensor overhead

        estimated_size = new_node_count * 32

        if new_keys is not None:
            estimated_size += sum([k.element_size() * k.nelement() for k in new_keys]) + 16 * len(new_keys)

        if new_values is not None:
            estimated_size += sum([self._value_sizing(v) for v in new_values])

        return estimated_size

    def __increase_estimated_size(
        self, new_node_count: int = 0, new_keys: list[Tensor] | None = None, new_values: list[T] | None = None
    ) -> None:
        self._estimated_size += self.__calculate_estimated_size(new_node_count, new_keys, new_values)

    def __create_self(
        self, parent: "TensorRadixTree[T]", key: Tensor | None = None, value: T | None = None
    ) -> "TensorRadixTree[T]":
        return TensorRadixTree(
            self._value_splitter,
            self._value_merger,
            self._value_sizing,
            self._value_shaping,
            key,
            value,
            parent,
        )

    def estimate_size_for(self, key: Tensor, value: T | None) -> int:
        incoming_size: int

        if value is None:
            incoming_size = self.__calculate_estimated_size(1, [key])
        else:
            incoming_size = self.__calculate_estimated_size(1, [key], [value])

        existing_parts = self.find_unmerged(key)
        existing_key_length = sum([part_len for _, part_len in existing_parts])

        return round(existing_key_length / len(key) * incoming_size)

    def __add(self, key: Tensor, value: T | None) -> "TensorRadixTree[T] | None":
        for i, child in enumerate(self._children):
            child_key = child.key
            assert child_key is not None

            matched, match_pos = self.__match_tensors(child_key, key)

            if matched:
                return None

            if match_pos < 0:
                continue

            if match_pos == len(child_key) - 1:
                new_key = key[match_pos + 1 :]
                _, new_value = self._value_splitter(value, match_pos + 1) if value is not None else (None, None)
                return child.__add(new_key, new_value)

            next_part_start_pos = match_pos + 1
            common_key = child_key[:next_part_start_pos]
            new_original_key = child_key[next_part_start_pos:]
            new_key = key[next_part_start_pos:]
            child_value = child.value

            common_value, new_original_value = (
                self._value_splitter(child_value, next_part_start_pos) if child_value is not None else (None, None)
            )

            _, new_value = self._value_splitter(value, next_part_start_pos) if value is not None else (None, None)

            common_node = self.__create_self(self, common_key, common_value)
            original_children = self._children[i].children
            self._children[i] = common_node

            new_original_node = common_node.__add(new_original_key, new_original_value)
            assert new_original_node is not None
            new_original_node._children = original_children

            common_node.add(new_key, new_value)
            return new_original_node

        new_node = self.__create_self(self, key, value)
        self._children.append(new_node)
        return new_node

    def add(self, key: Tensor, value: T | None) -> bool:
        return self.__add(key, value) is not None

    def __refresh_access_time(self) -> None:
        self._last_access_time = datetime.now()

    def find_unmerged(self, key: Tensor, accept_partial_match: bool = True) -> list[tuple[T, int]]:
        for child in self._children:
            child_key = child.key
            assert child_key is not None

            matched, match_pos = self.__match_tensors(child_key, key)
            child_value = child.value
            assert child_value is not None

            if matched:
                child.__refresh_access_time()
                return [(child_value, len(child_key))]

            if match_pos < 0:
                continue

            if len(child_key) >= len(key):
                if accept_partial_match:
                    child.__refresh_access_time()
                    matched_value_part, _ = self._value_splitter(child_value, match_pos + 1)
                    return [(matched_value_part, match_pos + 1)]
                else:
                    return []

            new_key = key[match_pos + 1 :]
            next_values = child.find_unmerged(new_key)
            values = [(child_value, len(child_key))]
            values.extend(next_values)
            child.__refresh_access_time()
            return values

        return []

    def find(self, key: Tensor, accept_partial_match: bool = True) -> tuple[T | None, int]:
        unmerged_result = self.find_unmerged(key, accept_partial_match)

        if len(unmerged_result) <= 0:
            return None, 0

        total_cached_length = sum([length for _, length in unmerged_result])
        caches = [c for c, _ in unmerged_result]

        if len(caches) == 1:
            return caches[0], total_cached_length

        return self._value_merger(caches), total_cached_length

    def __find_leaf_nodes(self) -> "list[TensorRadixTree[T]]":
        if len(self._children) <= 0:
            return [self]

        leaf_nodes = []

        for child in self._children:
            child_leaf_nodes = child.__find_leaf_nodes()
            leaf_nodes.extend(child_leaf_nodes)

        return leaf_nodes

    def __remove_node(self, node: "TensorRadixTree[T]") -> "list[TensorRadixTree[T]]":
        removed_nodes = []

        if len(node.children) > 0:
            for child in node.children:
                child_removed_nodes = node.__remove_node(child)
                removed_nodes.extend(child_removed_nodes)

        parent = node.parent

        if parent is None:
            return removed_nodes

        parent.children.remove(node)
        removed_nodes.append(node)

        return removed_nodes

    def evict_lru(self, expected_free_space: int) -> bool:
        freed_space = 0

        while freed_space < expected_free_space:
            leaf_nodes = self.__find_leaf_nodes()

            if len(leaf_nodes) <= 0 or (len(leaf_nodes) == 1 and leaf_nodes[0].parent is None):
                return False

            leaf_nodes.sort(key=lambda n: n._last_access_time, reverse=True)

            if len(leaf_nodes) <= 0:
                break

            to_remove = leaf_nodes.pop()

            removed_nodes = self.__remove_node(to_remove)
            freed_space += sum([n.estimated_size for n in removed_nodes])

        return freed_space >= expected_free_space

    def print_structure(self, indent: str = "") -> str:
        content = ""

        if self.key is None:
            content += f"(root) total estimated_size: {self.get_total_estimated_size()}\n"
        else:
            content += indent + f"key ({len(self.key)}): " + str(self.key)

            if self.value is not None:
                content += ", value shape: " + str(self._value_shaping(self.value))

            content += f", estimated size {self.estimated_size}, last access {self._last_access_time}\n"

        for child in self.children:
            content += child.print_structure(indent + "  ")

        return content

    def __str__(self) -> str:
        return f"{self.__class__}@{self.__hash__()}\n{self.print_structure()}"


@dataclass
class PreparedCache[T]:
    cache_id: str
    cache: T
    cached_token_count: int


class ModelPrefixCache[T](ABC):
    _max_size: int
    _radix_tree: TensorRadixTree[T]
    _log: logging.Logger = logging.getLogger(__name__)

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size

        self._radix_tree = TensorRadixTree(
            self._cache_splitter,
            self._cache_merger,
            self._cache_sizing,
            self._cache_shaping,
        )

    @abstractmethod
    def _cache_splitter(self, cache: T, next_part_start_pos: int) -> tuple[T, T]:
        pass

    @abstractmethod
    def _cache_merger(self, caches: list[T]) -> T:
        pass

    @abstractmethod
    def _cache_sizing(self, cache: T) -> int:
        pass

    @abstractmethod
    def _cache_shaping(self, cache: T) -> Size:
        pass

    @abstractmethod
    def _create_empty_cache(self, to_device: str) -> T:
        pass

    @abstractmethod
    def _copy_cache(self, original_cache: T) -> T:
        pass

    @property
    def current_estimated_size(self) -> int:
        return self._radix_tree.get_total_estimated_size()

    def retrieve_cache(self, model_input: Tensor, to_device: str, copy: bool = False) -> PreparedCache[T]:
        cached_data, cached_length = self._radix_tree.find(model_input)
        cache_id = str(uuid.uuid4())

        if cached_data is None:
            return PreparedCache(cache_id, self._create_empty_cache(to_device), 0)
        else:
            new_cached_data = self._copy_cache(cached_data) if copy else cached_data
            return PreparedCache(cache_id, new_cached_data, cached_length)

    def update_cache(self, new_cache: T, new_cache_tokens: Tensor) -> bool:
        new_estimated_size = self._radix_tree.estimate_size_for(new_cache_tokens, new_cache)

        if new_estimated_size >= self._max_size:
            self._log.warning(
                "New cache estimated size %d is larger than max prefix cache size %d, it will be skipped!",
                new_estimated_size,
                self._max_size,
            )

            return True

        current_estimated_size = self._radix_tree.get_total_estimated_size()

        if current_estimated_size + new_estimated_size >= self._max_size:
            if not self._radix_tree.evict_lru(new_estimated_size):
                return False

        self._radix_tree.add(new_cache_tokens, new_cache)
        return True

    def is_prefix_in_cache(self, prefix_tokens: Tensor, accept_partial_match: bool = True) -> bool:
        r = self._radix_tree.find_unmerged(prefix_tokens, accept_partial_match)
        return len(r) > 0
