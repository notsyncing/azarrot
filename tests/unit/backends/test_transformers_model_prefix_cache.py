import logging

import torch
from transformers.cache_utils import DynamicCache

from azarrot.backends.transformers_common import TransformersModelPrefixCache

log = logging.getLogger(__name__)


def test_update_and_retrieve_simple_cache() -> None:
    mpc = TransformersModelPrefixCache(1048576)

    source_cache = DynamicCache()
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 0)
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 1)
    source_tokens = torch.rand(128)

    mpc.update_cache(source_cache, source_tokens)

    target_cache_p = mpc.retrieve_cache(source_tokens, "cpu")
    assert target_cache_p.cache == source_cache
    assert target_cache_p.cache.get_seq_length() == 128
    assert target_cache_p.cached_token_count == 128

    target_cache_copy_p = mpc.retrieve_cache(source_tokens, "cpu", copy=True)
    assert target_cache_copy_p.cache != source_cache

    for layer in range(len(source_cache)):
        assert torch.equal(target_cache_copy_p.cache.key_cache[layer], source_cache.key_cache[layer])
        assert torch.equal(target_cache_copy_p.cache.value_cache[layer], source_cache.value_cache[layer])

    assert target_cache_copy_p.cache.get_seq_length() == 128


def test_update_and_retrieve_by_prefix() -> None:
    mpc = TransformersModelPrefixCache(1048576)

    prefix_tokens = torch.rand(32)

    source_cache = DynamicCache()
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 0)
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 1)
    source_tokens = torch.cat([prefix_tokens, torch.rand(96)])

    mpc.update_cache(source_cache, source_tokens)

    target_cache_p = mpc.retrieve_cache(prefix_tokens, "cpu")
    assert target_cache_p.cache != source_cache
    assert target_cache_p.cache.get_seq_length() == 32
    assert target_cache_p.cached_token_count == 32


def test_update_and_retrieve_by_prefix_partial() -> None:
    mpc = TransformersModelPrefixCache(1048576)

    prefix_tokens = torch.rand(32)

    source_cache = DynamicCache()
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 0)
    source_cache.update(torch.rand(1, 8, 128, 16), torch.rand(1, 8, 128, 32), 1)
    source_tokens = torch.cat([prefix_tokens, torch.rand(96)])

    mpc.update_cache(source_cache, source_tokens)

    target_tokens = torch.cat([prefix_tokens, torch.rand(32)])

    target_cache_p = mpc.retrieve_cache(target_tokens, "cpu")
    assert target_cache_p.cache != source_cache
    assert target_cache_p.cache.get_seq_length() == 32
    assert target_cache_p.cached_token_count == 32


def test_update_and_retrieve_by_prefix_with_multiple_branches() -> None:
    mpc = TransformersModelPrefixCache(1048576)

    part1_tokens = torch.rand(4)
    part2_tokens = torch.rand(6)
    part3_tokens = torch.rand(2)

    log.info("Part1 key: %s", part1_tokens)
    log.info("Part2 key: %s", part2_tokens)
    log.info("Part3 key: %s", part3_tokens)

    source1_cache = DynamicCache()
    source1_cache.update(torch.rand(1, 2, 16, 4), torch.rand(1, 2, 16, 4), 0)
    source1_tokens = torch.cat([part1_tokens, torch.rand(12)])
    assert mpc.update_cache(source1_cache, source1_tokens)

    log.info("Tree after source1: \n%s", mpc._radix_tree)

    source2_cache = DynamicCache()
    source2_cache.update(torch.rand(1, 2, 16, 4), torch.rand(1, 2, 16, 4), 0)
    source2_tokens = torch.cat([part1_tokens, part2_tokens, torch.rand(6)])
    assert mpc.update_cache(source2_cache, source2_tokens)

    log.info("Tree after source2: \n%s", mpc._radix_tree)

    source3_cache = DynamicCache()
    source3_cache.update(torch.rand(1, 2, 16, 4), torch.rand(1, 2, 16, 4), 0)
    source3_tokens = torch.cat([part1_tokens, part2_tokens, part3_tokens, torch.rand(4)])
    assert mpc.update_cache(source3_cache, source3_tokens)

    log.info("Final tree: \n%s", mpc._radix_tree)

    target_tokens = torch.cat([part1_tokens, part2_tokens, torch.rand(8)])

    target_cache_p = mpc.retrieve_cache(target_tokens, "cpu")
    assert target_cache_p.cache.get_seq_length() == 10
    assert target_cache_p.cached_token_count == 10

    merged_key12 = torch.cat([source1_cache.key_cache[0][..., :4, :], source2_cache.key_cache[0][..., 4:10, :]], dim=-2)

    log.info(
        "Expected key cache shape %s, actual key cache shape %s",
        merged_key12.size(),
        target_cache_p.cache.key_cache[0].size(),
    )

    assert torch.equal(target_cache_p.cache.key_cache[0], merged_key12)

    merged_value12 = torch.cat(
        [source1_cache.value_cache[0][..., :4, :], source2_cache.value_cache[0][..., 4:10, :]], dim=-2
    )

    log.info(
        "Expected value cache shape %s, actual value cache shape %s",
        merged_value12.size(),
        target_cache_p.cache.value_cache[0].size(),
    )

    assert torch.equal(target_cache_p.cache.value_cache[0], merged_value12)


def test_evict_lru() -> None:
    mpc = TransformersModelPrefixCache(max_size=2048)

    part1_tokens = torch.rand(4)
    part2_tokens = torch.rand(4)
    part3_tokens = torch.rand(4)

    log.info("Part1 key: %s", part1_tokens)
    log.info("Part2 key: %s", part2_tokens)
    log.info("Part3 key: %s", part3_tokens)

    source1_cache = DynamicCache()
    source1_cache.update(torch.rand(1, 2, 12, 4), torch.rand(1, 2, 12, 4), 0)
    source1_tokens = torch.cat([part1_tokens, part2_tokens, torch.rand(4)])
    assert mpc.update_cache(source1_cache, source1_tokens)

    log.info("Tree after source1: %s", mpc._radix_tree)

    source2_cache = DynamicCache()
    source2_cache.update(torch.rand(1, 2, 12, 4), torch.rand(1, 2, 12, 4), 0)
    source2_tokens = torch.cat([part1_tokens, part2_tokens, torch.rand(4)])
    assert mpc.update_cache(source2_cache, source2_tokens)

    log.info("Tree after source2: %s", mpc._radix_tree)

    source3_cache = DynamicCache()
    source3_cache.update(torch.rand(1, 2, 8, 4), torch.rand(1, 2, 8, 4), 0)
    source3_tokens = torch.cat([part1_tokens, torch.rand(4)])
    assert mpc.update_cache(source3_cache, source3_tokens)

    log.info("Tree after source3: %s", mpc._radix_tree)

    assert mpc.current_estimated_size == 1952

    new_cache = DynamicCache()
    new_cache.update(torch.rand(1, 2, 12, 4), torch.rand(1, 2, 12, 4), 0)
    new_cache_tokens = torch.cat([part1_tokens, part2_tokens, part3_tokens])

    assert mpc.update_cache(new_cache, new_cache_tokens)

    log.info("Tree after new: %s", mpc._radix_tree)

    source1_cache_now = mpc.retrieve_cache(source1_tokens, "cpu")
    assert source1_cache_now.cached_token_count < len(source1_tokens)
