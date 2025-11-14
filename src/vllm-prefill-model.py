import argparse
import hashlib
import uuid
from typing import List, Dict, Optional, Tuple

# --- Debug Configuration ---
# Set to True to enable verbose debug logging (WARNING: very slow!)
DEBUG = False

# --- Default Configuration ---
DEFAULT_BLOCK_SIZE = 4
DEFAULT_TOTAL_BLOCKS = 10


class KVBlock:
    """
    Represents a single KV Cache block in physical GPU memory.
    
    Matches vLLM's KVCacheBlock schema from:
    https://docs.vllm.ai/en/latest/design/prefix_caching/
    """

    def __init__(self, block_id: int, block_size: int):
        # The block ID (immutable)
        self.block_id: int = block_id
        
        # Block size - needed for simulation to know when blocks are full
        self.block_size: int = block_size
        
        # The block hash (assigned when block is full, reset on eviction)
        # In vLLM: BlockHash type, here we use Optional[Tuple[int, ...]]
        self.block_hash: Optional[Tuple[int, ...]] = None
        
        # The number of requests using this block now
        # (vLLM calls this ref_cnt)
        self.ref_count: int = 0
        
        # Pointers to form a doubly linked list for the free queue
        self.prev_free_block: Optional["KVBlock"] = None
        self.next_free_block: Optional["KVBlock"] = None
        
        # Token IDs stored in this block (for simulation/visualization)
        # Note: Real vLLM doesn't store token IDs in the block metadata
        self.token_ids: List[int] = []

    def is_full(self) -> bool:
        """Check if this block is full of tokens."""
        return len(self.token_ids) == self.block_size
    
    def is_cached(self) -> bool:
        """
        Check if this block is cached.
        In vLLM, a block is cached if block_hash is set.
        """
        return self.block_hash is not None

    def __repr__(self) -> str:
        status = "CACHED" if self.is_cached() else "NOT CACHED"
        return (
            f"Block(ID={self.block_id}, Tokens={self.token_ids}, "
            f"Refs={self.ref_count}, Status={status})"
        )


class SimulatedRequest:
    """Represents a single inference request."""

    def __init__(self, request_id: str, prompt_tokens: List[int]):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        # Maps logical token index to physical block ID
        # (the Block Table).
        self.block_table: List[int] = []

    def __repr__(self):
        return (
            f"Request(ID={self.request_id}, "
            f"Prompt_Len={len(self.prompt_tokens)}, "
            f"Block_Table={self.block_table})"
        )


class PrefixCacheManager:
    """
    Simulates vLLM's PagedAttention and Prefix Caching mechanisms.
    """

    def __init__(self, total_blocks: int, block_size: int, 
                 visualizer=None):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.visualizer = visualizer

        # Physical Block Pool (simulated GPU memory)
        self.block_pool: Dict[int, KVBlock] = {
            i: KVBlock(i, block_size) for i in range(total_blocks)
        }

        # Free Queue (list of available block IDs - acts as LRU/FIFO
        # if used as a queue)
        self.free_block_ids: List[int] = list(range(total_blocks))

        # Prefix Cache (Hash Map):
        # Key: Tuple of tokens (simulated hash key:
        #      tuple[prefix_tokens, block_tokens])
        # Value: KVBlock ID
        self.prefix_cache: Dict[Tuple[int, ...], int] = {}

        # List of active requests
        self.active_requests: List[SimulatedRequest] = []
        
        # Track current conversation ID for cache hit reporting
        self.current_conv_id: str = ""

    def get_status(self):
        """Prints the current state of the system."""
        if DEBUG:
            print("--- Cache Manager Status ---")
        if DEBUG:
            print(
            f"Total Blocks: {self.total_blocks}, "
            f"Free Blocks: {len(self.free_block_ids)}"
        )
        if DEBUG:
            print(f"Prefix Cache Size: {len(self.prefix_cache)}")
        if DEBUG:
            print("--- Block Pool Summary (Active Blocks) ---")
        for block in self.block_pool.values():
            if block.ref_count > 0 or block.is_cached():
                if DEBUG:
                    print(f"  {block}")
        if DEBUG:
            print("----------------------------")

    def _generate_cache_key(
        self, prefix_tokens: List[int], block_tokens: List[int]
    ) -> Tuple[int, ...]:
        """
        Generates the simplified hash key:
        (token_id_1, token_id_2, ..., token_id_N).
        In real vLLM, this is more complex (parent hash + block tokens
        + extra hashes).
        We use all tokens up to this block to ensure uniqueness.
        """
        # The key is the sequence of all tokens from the start of the
        # prompt up to and including the current block's tokens.
        return tuple(prefix_tokens + block_tokens)
    
    def _generate_hash_string(self, cache_key: Tuple[int, ...]) -> str:
        """
        Generate an MD5 hash string from the cache key tuple.
        Similar to vLLM's approach for cache key hashing.
        """
        # Convert tuple of token IDs to bytes and hash
        key_bytes = str(cache_key).encode('utf-8')
        return hashlib.md5(key_bytes).hexdigest()

    def allocate_block(self, block: KVBlock) -> int:
        """Increments ref count and allocates a physical block ID."""
        if not self.free_block_ids:
            # Simple eviction policy: just say memory is full for
            # this demo
            if DEBUG:
                print("ERROR: KV Cache is full. Cannot allocate new block.")
            return -1

        # Get an available physical block ID from the free queue
        # (usually FIFO/LRU)
        physical_block_id = self.free_block_ids.pop(0)

        # Update the block in the pool
        block_to_use = self.block_pool[physical_block_id]
        block_to_use.ref_count += 1

        return physical_block_id

    def process_request(self, request: SimulatedRequest):
        """
        Handles a new request, attempting to reuse cached prefixes.
        """
        if DEBUG:
            print(f"\n--- Processing New Request: {request.request_id} ---")

        # The block table holds the mapping from logical block index
        # to physical block ID.
        block_table: List[int] = []

        prompt_tokens = request.prompt_tokens

        # Track the tokens processed so far for generating the prefix
        # hash key
        tokens_processed: List[int] = []

        current_token_idx = 0
        cache_hit_count = 0

        # Process the prompt token by token, in blocks
        while current_token_idx < len(prompt_tokens):

            # 1. Identify the tokens for the next block
            start = current_token_idx
            end = min(
                current_token_idx + self.block_size,
                len(prompt_tokens)
            )
            block_tokens = prompt_tokens[start:end]

            # 2. Generate the cache lookup key
            cache_key = self._generate_cache_key(
                tokens_processed, block_tokens
            )

            # 3. Check for prefix cache hit
            cached_block_id = self.prefix_cache.get(cache_key)

            if cached_block_id is not None:
                # --- CACHE HIT ---

                # Check for partial hit: In vLLM, partial hits on a
                # block stop caching. Here, we only cache/reuse full
                # blocks to simplify the model, as suggested by the
                # documentation: "We only cache full blocks."
                if len(block_tokens) == self.block_size:
                    # Full Block Hit - REUSE
                    hit_block = self.block_pool[cached_block_id]
                    hit_block.ref_count += 1
                    block_table.append(cached_block_id)
                    cache_hit_count += len(block_tokens)
                    
                    # Record cache hit for visualization
                    if self.visualizer and hasattr(self.visualizer, 
                                                   'record_cache_hit'):
                        # Generate MD5 hash of the cache key
                        block_hash = self._generate_hash_string(cache_key)
                        self.visualizer.record_cache_hit(
                            block_hash, 
                            list(block_tokens), 
                            self.current_conv_id
                        )
                    
                    if DEBUG:
                        print(
                        f"  [Cache Hit] Reusing Block ID "
                        f"{cached_block_id} for tokens {block_tokens}. "
                        f"Refs: {hit_block.ref_count}"
                    )

                    # Update tokens_processed for the next block's prefix
                    tokens_processed.extend(block_tokens)
                    current_token_idx = end

                else:
                    # Partial Block Hit or end of prompt. Stop
                    # caching/reuse. The rest of the prompt must be
                    # allocated anew.
                    if DEBUG:
                        print(
                        f"  [Partial Match] Block {block_tokens} found, "
                        f"but it's the end of prompt. "
                        f"Allocation begins now."
                    )
                    break  # Exit caching loop, move to allocation stage

            else:
                # --- CACHE MISS or Allocation Begins ---
                if DEBUG:
                    print(
                    f"  [Cache Miss] Block {block_tokens} not found in "
                    f"cache. Allocating new blocks."
                )
                break  # Exit caching loop, move to allocation stage

        # --- Allocation & Prefill Stage (for the rest of the prompt) ---

        # Start allocation from where the cache hit stopped
        # (current_token_idx)
        while current_token_idx < len(prompt_tokens):

            start = current_token_idx
            end = min(
                current_token_idx + self.block_size,
                len(prompt_tokens)
            )
            block_tokens = prompt_tokens[start:end]

            # 1. Allocate a new physical block
            if not self.free_block_ids:
                if DEBUG:
                    print("  [Allocation Failed] Ran out of free blocks.")
                break

            physical_block_id = self.free_block_ids.pop(0)
            new_block = self.block_pool[physical_block_id]
            
            # Evict the block if it's cached (LRU eviction)
            # This matches vLLM behavior: "If the head block is a cached
            # block, this also evicts the block so that no other requests
            # can reuse it anymore from now on."
            if new_block.is_cached():
                # Remove from cache blocks
                if new_block.block_hash in self.prefix_cache:
                    del self.prefix_cache[new_block.block_hash]
                # Reset block hash to mark as evicted
                new_block.block_hash = None

            # 2. Populate and increment reference count
            new_block.token_ids = block_tokens  # Simulate prefill
            new_block.ref_count += 1
            block_table.append(physical_block_id)
            if DEBUG:
                print(
                f"  [Allocation] Allocated Block ID {physical_block_id} "
                f"for tokens {block_tokens}."
            )

            # 3. Save to Prefix Cache (only if full)
            if new_block.is_full():
                cache_key = self._generate_cache_key(
                    tokens_processed, block_tokens
                )
                self.prefix_cache[cache_key] = physical_block_id
                # Set block_hash to mark this block as cached (vLLM style)
                new_block.block_hash = cache_key
                if DEBUG:
                    print(
                    f"  [Cache Save] Block ID {physical_block_id} "
                    f"saved to cache (Key Length: {len(cache_key)})."
                )

            # Update indices for the next iteration
            tokens_processed.extend(block_tokens)
            current_token_idx = end

        request.block_table = block_table
        self.active_requests.append(request)
        if DEBUG:
            print(
            f"  Request {request.request_id} finished processing. "
            f"Blocks used: {request.block_table}"
        )
        if DEBUG:
            print(
            f"  Total tokens computed (not cached): "
            f"{len(prompt_tokens) - cache_hit_count}"
        )
        return cache_hit_count

    def free_request(self, request: SimulatedRequest):
        """
        Frees blocks associated with a finished request by decrementing
        ref counts. Blocks with ref_count == 0 are returned to the free
        queue.
        """
        if DEBUG:
            print(f"\n--- Freeing Request: {request.request_id} ---")

        # Free blocks in reverse order, as suggested by vLLM (to
        # prioritize evicting less reusable blocks)
        for block_id in reversed(request.block_table):
            block = self.block_pool[block_id]
            block.ref_count -= 1

            if DEBUG:
                print(
                f"  Block ID {block_id}: Ref Count Decremented to "
                f"{block.ref_count}."
            )

            if block.ref_count == 0:
                # Block is no longer used by any request. Return it to
                # the free queue.

                # Check if it was a cached block. If so, its cache
                # entry must be removed *if* we wanted to strictly
                # enforce LRU eviction, but for simplicity we only
                # remove it from free_ids. The block remains *cached*
                # (is_cached=True) but is available for reuse or
                # eviction if needed later.

                # The vLLM docs show blocks being added to the free
                # queue even if cached. Eviction (LRU) happens only
                # when memory pressure requires freeing a block from
                # the head of the free queue *and* removing it from
                # the cache.

                # We simply add it to the free queue.
                self.free_block_ids.append(block_id)

                # Reset for next use (but keep block_hash for potential
                # reuse unless evicted). According to vLLM docs, block_hash
                # is reset on eviction, not when just added to free queue.
                block.token_ids = []
                if DEBUG:
                    print(f"  Block ID {block_id} added back to free queue.")

        # Remove request from active list
        self.active_requests.remove(request)
        if DEBUG:
            print(f"Request {request.request_id} freed successfully.")


def run_simulation(
    block_size=DEFAULT_BLOCK_SIZE,
    total_blocks=DEFAULT_TOTAL_BLOCKS,
    custom_prompts=None,
    verbose=True
):
    """
    Run the vLLM prefix caching simulation.

    Args:
        block_size: Number of tokens per block
        total_blocks: Total number of blocks in the pool
        custom_prompts: Optional list of custom prompt token lists
        verbose: Whether to print detailed status information
    """
    manager = PrefixCacheManager(
        total_blocks=total_blocks, block_size=block_size
    )
    if DEBUG:
        print(f"Starting simulation with {total_blocks} blocks, size "
          f"{block_size}.")
    if verbose:
        manager.get_status()

    # Use custom prompts if provided, otherwise use default simulation
    if custom_prompts:
        requests = []
        for i, prompt_tokens in enumerate(custom_prompts):
            if DEBUG:
                print(f"\n--- Processing Custom Request {i+1} ---")
            req = SimulatedRequest(
                request_id=str(uuid.uuid4())[:8],
                prompt_tokens=prompt_tokens
            )
            hit_count = manager.process_request(req)
            if DEBUG:
                print(f"Total Cache Hit Tokens: {hit_count}")
            if verbose:
                manager.get_status()
            requests.append(req)

        # Free all requests
        for req in requests:
            manager.free_request(req)
            if verbose:
                manager.get_status()
        return

    # --- Default Simulation ---

    # --- Time 1: Request A (Full Prompt Allocation & Caching) ---
    # 11 tokens -> 3 full blocks + 1 partial block
    prompt_a_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    req_a = SimulatedRequest(
        request_id=str(uuid.uuid4())[:8],
        prompt_tokens=prompt_a_tokens
    )
    manager.process_request(req_a)
    if verbose:
        manager.get_status()

    # --- Time 2: Request B (Exact Prefix Match) ---
    # This prompt is the same as Request A, showing 100% cache reuse.
    prompt_b_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    req_b = SimulatedRequest(
        request_id=str(uuid.uuid4())[:8],
        prompt_tokens=prompt_b_tokens
    )
    hit_count = manager.process_request(req_b)
    if DEBUG:
        print(f"Total Cache Hit Tokens for Request B: {hit_count}")
    if verbose:
        manager.get_status()

    # --- Time 3: Request C (Partial Prefix Match) ---
    # This prompt matches the first 8 tokens (2 full blocks) of A/B,
    # then forks.
    # [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14] -> B0, B1 reused.
    # B2, B3 allocated.
    prompt_c_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]  # 12 tokens
    req_c = SimulatedRequest(
        request_id=str(uuid.uuid4())[:8],
        prompt_tokens=prompt_c_tokens
    )
    hit_count = manager.process_request(req_c)
    if DEBUG:
        print(f"Total Cache Hit Tokens for Request C: {hit_count}")
    if verbose:
        manager.get_status()

    # --- Time 4: Free Request A ---
    # Blocks 0, 1 are still in use by B and C (ref_count > 0).
    # Blocks 2, 3 are only used by A. They will be freed and returned
    # to the free queue.
    manager.free_request(req_a)
    if verbose:
        manager.get_status()

    # --- Time 5: Free Request B ---
    # Blocks 0, 1, 2, 3 were used by B. Blocks 0, 1 are still used by C
    # (ref_count > 0).
    # Blocks 2, 3 are now only used by B's old ref (which is now 0) and
    # will be freed, but wait...
    # Blocks 2 and 3 have ref_count=1 (from A) after Time 4, then B
    # reused them.
    # Let's check the current ref counts:
    # B0: Refs=2 (from B, C)
    # B1: Refs=2 (from B, C)
    # B2 (A/B's B2): Refs=1 (from B)
    # B3 (A/B's B3): Refs=1 (from B)

    # After freeing A (Time 4):
    # B0: Refs=2 (B, C)
    # B1: Refs=2 (B, C)
    # B2 (A/B's B2): Refs=1 (B) -> NOT freed
    # B3 (A/B's B3): Refs=1 (B) -> NOT freed

    manager.free_request(req_b)
    if verbose:
        manager.get_status()

    # After freeing B (Time 5):
    # B0: Refs=1 (C)
    # B1: Refs=1 (C)
    # B2 (A/B's B2): Refs=0 -> Freed
    # B3 (A/B's B3): Refs=0 -> Freed
    # B4 (C's B2): Refs=1 (C)
    # B5 (C's B3): Refs=1 (C)

    # Free queue should now contain: [2, 3] (from B)

    # --- Time 6: Free Request C ---
    manager.free_request(req_c)
    if verbose:
        manager.get_status()
    # All blocks are now back in the free queue, available for the
    # next request.


def parse_prompt_arg(prompt_str):
    """
    Parse a prompt string into a list of integers.
    Accepts formats like: "1,2,3,4" or "1 2 3 4"
    """
    # Replace spaces with commas for uniform parsing
    prompt_str = prompt_str.replace(' ', ',')
    try:
        return [
            int(x.strip()) for x in prompt_str.split(',') if x.strip()
        ]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid prompt format. Use comma or space-separated "
            f"integers: {e}"
        )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Simulate vLLM PagedAttention and Prefix Caching '
                    'mechanisms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python3 vllm-prefill-model.py

  # Configure block size and total blocks
  python3 vllm-prefill-model.py --block-size 8 --total-blocks 20

  # Run with custom prompts
  python3 vllm-prefill-model.py --prompts "1,2,3,4,5" "1,2,3,6,7"

  # Quiet mode (less verbose output)
  python3 vllm-prefill-model.py --quiet

  # Combine options
  python3 vllm-prefill-model.py --block-size 4 \\
    --prompts "1,2,3,4" "1,2,3,5" --quiet
        """
    )

    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f'Number of tokens per KV cache block '
             f'(default: {DEFAULT_BLOCK_SIZE})'
    )

    parser.add_argument(
        '--total-blocks', '-t',
        type=int,
        default=DEFAULT_TOTAL_BLOCKS,
        help=f'Total number of blocks in the cache pool '
             f'(default: {DEFAULT_TOTAL_BLOCKS})'
    )

    parser.add_argument(
        '--prompts', '-p',
        nargs='+',
        type=parse_prompt_arg,
        metavar='PROMPT',
        help='Custom prompt token sequences (space or comma-separated '
             'integers). Example: --prompts "1,2,3,4" "1,2,3,5"'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity (hide status after each operation)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.block_size <= 0:
        parser.error("Block size must be positive")
    if args.total_blocks <= 0:
        parser.error("Total blocks must be positive")

    # Run simulation with parsed arguments
    run_simulation(
        block_size=args.block_size,
        total_blocks=args.total_blocks,
        custom_prompts=args.prompts,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
