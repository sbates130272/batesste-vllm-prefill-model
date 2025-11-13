# vLLM Prefill Model Simulator

A Python simulation of vLLM's PagedAttention and Prefix Caching mechanisms for understanding how Key-Value (KV) cache blocks are allocated, reused, and managed during LLM inference.

## Quick Start

```bash
# Run with default settings
python3 src/vllm-prefill-model.py

# Show all available options
python3 src/vllm-prefill-model.py --help

# Configure block size and total blocks
python3 src/vllm-prefill-model.py --block-size 8 --total-blocks 20

# Test with custom prompts
python3 src/vllm-prefill-model.py --prompts "1,2,3,4,5" "1,2,3,6,7"

# Run in quiet mode for cleaner output
python3 src/vllm-prefill-model.py --quiet
```

## Features

- 🔧 **Configurable Parameters**: Adjust block size and total blocks via command-line arguments
- 🎯 **Custom Prompts**: Test your own token sequences and prefix patterns
- 📊 **Detailed Logging**: Observe cache hits/misses and block allocation in real-time
- 🤫 **Quiet Mode**: Reduce verbosity for cleaner output
- 🎓 **Educational**: Clear visualization of vLLM's memory management internals
- 🐍 **Pure Python**: No external dependencies required

## Overview

This project provides an educational implementation of vLLM's memory management system, specifically focusing on:

- **PagedAttention**: A memory-efficient attention mechanism that divides KV cache into fixed-size blocks
- **Prefix Caching**: A technique to reuse computed KV cache blocks across requests with shared prompt prefixes
- **Reference Counting**: Memory management through reference counting to enable safe block sharing and deallocation

## Key Concepts

### KV Cache Blocks

The KV (Key-Value) cache stores attention keys and values for previously processed tokens. In this simulation:

- Blocks are fixed-size containers (default: 4 tokens per block)
- Each block tracks:
  - `token_ids`: The tokens stored in this block
  - `ref_count`: Number of active requests using this block
  - `is_cached`: Whether the block is saved in the prefix cache

### Prefix Caching

Prefix caching allows multiple requests with shared prompt prefixes to reuse the same KV cache blocks:

- Full blocks (completely filled) are cached for reuse
- Partial blocks (not full) are not cached
- Cache keys are generated from the token sequence
- Blocks can be shared across multiple concurrent requests through reference counting

### Memory Management

The system uses a free block pool and reference counting:

- **Free Queue**: List of available block IDs for allocation
- **Reference Counting**: Tracks how many requests use each block
- **Deallocation**: Blocks with `ref_count == 0` are returned to the free queue

## Architecture

### Classes

#### `KVBlock`

Represents a single KV cache block in physical GPU memory.

- `block_id`: Unique identifier for the block
- `token_ids`: List of token IDs stored in this block
- `ref_count`: Number of requests currently using this block
- `is_cached`: Flag indicating if the block is in the prefix cache

#### `SimulatedRequest`

Represents an inference request with a prompt.

- `request_id`: Unique identifier for the request
- `prompt_tokens`: List of token IDs in the prompt
- `block_table`: Maps logical block indices to physical block IDs

#### `PrefixCacheManager`

Main orchestrator for the caching and memory management system.

**Key Methods:**

- `process_request()`: Processes a new request, attempting to reuse cached blocks
- `free_request()`: Frees blocks associated with a finished request
- `get_status()`: Displays current system state
- `_generate_cache_key()`: Generates hash keys for prefix cache lookup

## Configuration

The simulation can be configured via command-line arguments:

- `--block-size`, `-b`: Number of tokens per KV cache block (default: 4)
- `--total-blocks`, `-t`: Total number of blocks in the cache pool (default: 10)
- `--prompts`, `-p`: Custom prompt token sequences (space or comma-separated integers)
- `--quiet`, `-q`: Reduce output verbosity (hide status after each operation)

## Running the Simulation

### Prerequisites

- Python 3.6 or higher
- No external dependencies required (uses only standard library)

### Basic Usage

```bash
# Run with default settings
python3 src/vllm-prefill-model.py

# Show help and all options
python3 src/vllm-prefill-model.py --help
```

### Configuration Options

```bash
# Configure block size and total blocks
python3 src/vllm-prefill-model.py --block-size 8 --total-blocks 20

# Run with custom prompts (test your own scenarios)
python3 src/vllm-prefill-model.py --prompts "1,2,3,4,5" "1,2,3,6,7" "1,2,3,4,5"

# Quiet mode (less verbose output)
python3 src/vllm-prefill-model.py --quiet

# Combine options
python3 src/vllm-prefill-model.py --block-size 4 --prompts "1,2,3,4" "1,2,3,5" --quiet
```

### Example Output

#### Default Simulation

The default simulation (no custom prompts) demonstrates three scenarios:

1. **Request A**: Initial prompt allocation (11 tokens)
   - Allocates 3 blocks (2 full + 1 partial)
   - Full blocks are cached

2. **Request B**: Exact prefix match (same prompt as A)
   - Reuses cached blocks from Request A
   - Shows 100% cache hit for full blocks

3. **Request C**: Partial prefix match
   - Reuses first 2 blocks (8 tokens) from A/B
   - Allocates new blocks for divergent suffix

The simulation then demonstrates proper cleanup by freeing requests and showing how reference counting prevents premature deallocation of shared blocks.

#### Custom Prompts

When using `--prompts`, you can define your own test scenarios with custom token sequences. Each prompt should be a sequence of comma or space-separated integers representing token IDs:

```bash
# Test prefix caching with custom token sequences
python3 src/vllm-prefill-model.py --prompts "1,2,3,4,5,6" "1,2,3,7,8,9" "10,11,12,13"
```

This is useful for:
- Testing specific cache hit/miss patterns
- Exploring different prompt overlap scenarios
- Understanding how block boundaries affect caching

## Simulation Flow

```
Request Processing:
1. Iterate through prompt tokens in block-sized chunks
2. For each block:
   a. Generate cache key from token sequence
   b. Check prefix cache for existing block
   c. If HIT and full block: increment ref_count, reuse
   d. If MISS: allocate new block, compute (simulate), cache if full
3. Build block table mapping logical to physical blocks

Request Cleanup:
1. Iterate through block table in reverse
2. Decrement ref_count for each block
3. If ref_count reaches 0:
   - Return block to free queue
   - Clear block data
   - Keep cache entry (for potential future reuse)
```

## Key Observations

### Cache Efficiency

- **Full Block Caching**: Only complete blocks are cached to simplify key generation
- **Prefix Matching**: Requests with shared prefixes benefit from cached blocks
- **Memory Reuse**: Reference counting enables safe sharing across concurrent requests

### Design Decisions

1. **Simplified Hashing**: Uses tuple of all tokens as cache key (real vLLM uses more sophisticated hashing)
2. **No Eviction Policy**: Simulation assumes sufficient memory (production systems implement LRU eviction)
3. **Synchronous Processing**: Real vLLM handles concurrent requests asynchronously

## Differences from Production vLLM

This is a simplified educational model. Production vLLM includes:

- **Advanced Hash Functions**: Parent hash + block tokens + position encoding
- **LRU Eviction**: Automatic eviction when memory pressure occurs
- **GPU Memory Management**: Actual CUDA memory allocation and management
- **Async Scheduling**: Sophisticated request batching and scheduling
- **Partial Block Handling**: More nuanced handling of partial blocks

## Use Cases

This simulation is useful for:

- Understanding vLLM's memory management internals
- Learning about prefix caching benefits
- Visualizing block allocation and reuse patterns
- Educational purposes for LLM inference optimization

## Extending the Simulation

Potential enhancements:

1. **Add LRU Eviction**: Implement proper eviction when free queue is empty
2. **Metrics Collection**: Track cache hit rates, memory utilization
3. **Visualization**: Add graphical representation of block allocation
4. **Concurrent Requests**: Simulate multiple simultaneous requests
5. **Variable Block Sizes**: Test impact of different block sizes

## License

This is an educational project for understanding vLLM internals.

## References

- [vLLM Project](https://github.com/vllm-project/vllm)
- PagedAttention: Memory-Efficient Attention for Large Language Models
- vLLM Prefix Caching Documentation

## Author

Created as a learning resource for understanding vLLM's PagedAttention and Prefix Caching mechanisms.
