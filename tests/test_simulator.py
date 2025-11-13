"""Test core simulator functionality."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "vllm_prefill_model",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'vllm-prefill-model.py')
)
vllm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vllm_module)

KVBlock = vllm_module.KVBlock
SimulatedRequest = vllm_module.SimulatedRequest
PrefixCacheManager = vllm_module.PrefixCacheManager


def test_kvblock_creation():
    """Test KVBlock can be created."""
    block = KVBlock(block_id=0, block_size=4)
    assert block.block_id == 0
    assert block.block_size == 4
    assert block.ref_count == 0
    assert len(block.token_ids) == 0


def test_kvblock_is_full():
    """Test KVBlock full detection."""
    block = KVBlock(block_id=0, block_size=4)
    assert not block.is_full()
    
    block.token_ids = [1, 2, 3, 4]
    assert block.is_full()


def test_simulated_request_creation():
    """Test SimulatedRequest can be created."""
    req = SimulatedRequest(
        request_id="test_req",
        prompt_tokens=[1, 2, 3, 4, 5]
    )
    assert req.request_id == "test_req"
    assert req.prompt_tokens == [1, 2, 3, 4, 5]
    assert req.block_table == []


def test_prefix_cache_manager_creation():
    """Test PrefixCacheManager can be created."""
    manager = PrefixCacheManager(total_blocks=10, block_size=4)
    assert manager.total_blocks == 10
    assert manager.block_size == 4
    assert len(manager.block_pool) == 10
    assert len(manager.free_block_ids) == 10


def test_simple_request_processing():
    """Test processing a simple request."""
    manager = PrefixCacheManager(total_blocks=10, block_size=4)
    
    # Create a request
    req = SimulatedRequest(
        request_id="req1",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    
    # Process it
    cache_hits = manager.process_request(req)
    
    # First request should have no cache hits
    assert cache_hits == 0
    
    # Request should have blocks allocated
    assert len(req.block_table) > 0
    
    # Free the request
    manager.free_request(req)
    
    # All blocks should be free again
    assert len(manager.free_block_ids) == 10


def test_prefix_caching():
    """Test that prefix caching works."""
    manager = PrefixCacheManager(total_blocks=20, block_size=4)
    
    # First request with prefix [1,2,3,4,5,6,7,8]
    req1 = SimulatedRequest(
        request_id="req1",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    cache_hits1 = manager.process_request(req1)
    assert cache_hits1 == 0  # No cache hits on first request
    
    # Free first request
    manager.free_request(req1)
    
    # Second request with same prefix [1,2,3,4,5,6,7,8] + new tokens
    req2 = SimulatedRequest(
        request_id="req2",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    cache_hits2 = manager.process_request(req2)
    
    # Should have cache hits on the prefix
    assert cache_hits2 > 0
    assert cache_hits2 == 8  # All 8 prefix tokens cached
    
    manager.free_request(req2)

