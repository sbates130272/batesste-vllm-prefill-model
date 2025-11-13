"""
Multi-User vLLM Prefix Caching Simulator

Simulates multiple concurrent users hitting the system with multi-turn
conversations, inspired by vLLM's multi-turn benchmark.
"""
import argparse
import asyncio
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import from the main simulator
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vllm_prefill_model",
    os.path.join(os.path.dirname(__file__), "vllm-prefill-model.py")
)
vllm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vllm_module)

# Import classes from the module
KVBlock = vllm_module.KVBlock
SimulatedRequest = vllm_module.SimulatedRequest
PrefixCacheManager = vllm_module.PrefixCacheManager
DEFAULT_BLOCK_SIZE = vllm_module.DEFAULT_BLOCK_SIZE
DEFAULT_TOTAL_BLOCKS = vllm_module.DEFAULT_TOTAL_BLOCKS


class ConversationSampling(str, Enum):
    """Strategy for selecting which conversation to process next."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    user_tokens: List[int]
    assistant_tokens: List[int]


@dataclass
class Conversation:
    """Represents a multi-turn conversation."""
    conv_id: str
    turns: List[ConversationTurn]
    current_turn: int = 0
    
    def has_more_turns(self) -> bool:
        """Check if conversation has more turns to process."""
        return self.current_turn < len(self.turns)
    
    def get_messages_up_to_turn(self, turn_num: int) -> List[int]:
        """
        Get all tokens (user + assistant) up to the specified turn.
        This simulates the conversation context growing over time.
        """
        tokens = []
        for i in range(turn_num):
            if i < len(self.turns):
                tokens.extend(self.turns[i].user_tokens)
                if i < turn_num - 1 or i == turn_num - 1:
                    # Include assistant response for completed turns
                    tokens.extend(self.turns[i].assistant_tokens)
        # Add the current user message
        if turn_num < len(self.turns):
            tokens.extend(self.turns[turn_num].user_tokens)
        return tokens


@dataclass
class ClientStats:
    """Statistics for a single client."""
    client_id: int
    requests_sent: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_tokens: int = 0


class Distribution:
    """Base class for statistical distributions."""
    def sample(self) -> int:
        raise NotImplementedError


class UniformDist(Distribution):
    """Uniform distribution."""
    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self) -> int:
        return random.randint(self.min_val, self.max_val)


class ConstantDist(Distribution):
    """Constant value distribution."""
    def __init__(self, value: int):
        self.value = value
    
    def sample(self) -> int:
        return self.value


class LognormalDist(Distribution):
    """Lognormal distribution for realistic token counts."""
    def __init__(self, average: int, max_val: Optional[int] = None):
        self.average = average
        self.max_val = max_val
        # Use median ratio of 0.85 for realistic skew
        median_ratio = 0.85
        target_median = average * median_ratio
        self.sigma = np.sqrt(2 * np.log(average / target_median))
        self.mu = np.log(target_median)
    
    def sample(self) -> int:
        value = int(np.random.lognormal(self.mu, self.sigma))
        if self.max_val:
            value = min(value, self.max_val)
        return max(1, value)


def generate_synthetic_conversations(
    num_conversations: int,
    num_turns_dist: Distribution,
    prefix_tokens_dist: Distribution,
    user_tokens_dist: Distribution,
    assistant_tokens_dist: Distribution,
    common_prefix_tokens: int = 0
) -> List[Conversation]:
    """
    Generate synthetic multi-turn conversations with configurable
    distributions.
    
    Args:
        num_conversations: Number of conversations to generate
        num_turns_dist: Distribution for number of turns per conversation
        prefix_tokens_dist: Distribution for unique prefix tokens per conv
        user_tokens_dist: Distribution for user message token count
        assistant_tokens_dist: Distribution for assistant response tokens
        common_prefix_tokens: Common prefix shared across all conversations
    
    Returns:
        List of Conversation objects
    """
    conversations = []
    
    # Generate a common prefix (shared across all conversations)
    common_prefix = list(range(1, common_prefix_tokens + 1))
    token_counter = common_prefix_tokens + 1
    
    for conv_idx in range(num_conversations):
        conv_id = f"conv_{conv_idx:04d}"
        
        # Generate unique prefix for this conversation
        prefix_len = prefix_tokens_dist.sample()
        unique_prefix = list(range(
            token_counter, 
            token_counter + prefix_len
        ))
        token_counter += prefix_len
        
        # Determine number of turns
        num_turns = max(1, num_turns_dist.sample() // 2)  # Ensure even
        
        turns = []
        for turn_idx in range(num_turns):
            # User message tokens
            user_len = user_tokens_dist.sample()
            user_tokens = list(range(
                token_counter,
                token_counter + user_len
            ))
            token_counter += user_len
            
            # Assistant response tokens
            assistant_len = assistant_tokens_dist.sample()
            assistant_tokens = list(range(
                token_counter,
                token_counter + assistant_len
            ))
            token_counter += assistant_len
            
            # For the first turn, include prefixes
            if turn_idx == 0:
                user_tokens = common_prefix + unique_prefix + user_tokens
            
            turns.append(ConversationTurn(
                user_tokens=user_tokens,
                assistant_tokens=assistant_tokens
            ))
        
        conversations.append(Conversation(
            conv_id=conv_id,
            turns=turns
        ))
    
    return conversations


async def client_worker(
    client_id: int,
    conversations: List[Conversation],
    manager: PrefixCacheManager,
    sampling_strategy: ConversationSampling,
    request_rate: float,
    max_active_conversations: int,
    stats: ClientStats,
    verbose: bool = False
) -> None:
    """
    Simulate a single client processing multiple conversations.
    
    Args:
        client_id: Unique identifier for this client
        conversations: List of conversations to process
        manager: The KV cache manager (shared across clients)
        sampling_strategy: How to pick next conversation
        request_rate: Request rate (Poisson process, 0 = no delay)
        max_active_conversations: Max concurrent conversations per client
        stats: Statistics collector for this client
        verbose: Whether to print detailed logs
    """
    if verbose:
        print(f"[Client {client_id}] Starting with "
              f"{len(conversations)} conversations")
    
    # Active conversations queue
    active_convs: Dict[str, Conversation] = {}
    conv_queue: deque = deque(maxlen=max_active_conversations)
    
    # Add initial conversations up to max_active_conversations
    for conv in conversations[:max_active_conversations]:
        active_convs[conv.conv_id] = conv
        conv_queue.append(conv.conv_id)
    
    # Track index for round-robin through all conversations
    next_conv_idx = max_active_conversations
    
    while len(active_convs) > 0:
        # Pick a conversation to process
        if sampling_strategy == ConversationSampling.ROUND_ROBIN:
            conv_id = conv_queue.pop() if conv_queue else None
        else:  # RANDOM
            conv_id = random.choice(list(active_convs.keys())) \
                if active_convs else None
        
        if conv_id is None:
            break
        
        conv = active_convs[conv_id]
        
        # Get all tokens up to current turn (includes conversation history)
        prompt_tokens = conv.get_messages_up_to_turn(conv.current_turn)
        
        # Create and process request
        req = SimulatedRequest(
            request_id=f"{conv_id}_turn_{conv.current_turn}",
            prompt_tokens=prompt_tokens
        )
        
        if verbose:
            history_tokens = len(prompt_tokens) - \
                len(conv.turns[conv.current_turn].user_tokens)
            print(f"[Client {client_id}] {conv_id} Turn "
                  f"{conv.current_turn}: {len(prompt_tokens)} tokens "
                  f"({history_tokens} history + "
                  f"{len(conv.turns[conv.current_turn].user_tokens)} new)")
        
        # Process the request (this is synchronous in our simulation)
        cache_hit_count = manager.process_request(req)
        
        # Update stats
        stats.requests_sent += 1
        stats.total_input_tokens += len(prompt_tokens)
        stats.cached_tokens += cache_hit_count
        
        # Simulate assistant response (would be in manager in real system)
        output_tokens = conv.turns[conv.current_turn].assistant_tokens
        stats.total_output_tokens += len(output_tokens)
        
        # Calculate cache hit percentage for this request
        cached_percent = (cache_hit_count / len(prompt_tokens) * 100) \
            if len(prompt_tokens) > 0 else 0
        
        if verbose:
            print(f"[Client {client_id}] {conv_id} Turn "
                  f"{conv.current_turn} completed: "
                  f"{cached_percent:.1f}% cached")
        
        # Free the request
        manager.free_request(req)
        
        # Move to next turn
        conv.current_turn += 1
        
        if conv.has_more_turns():
            # Conversation continues, add back to queue
            if sampling_strategy == ConversationSampling.ROUND_ROBIN:
                conv_queue.appendleft(conv_id)
        else:
            # Conversation finished
            active_convs.pop(conv_id)
            if verbose:
                print(f"[Client {client_id}] Finished {conv_id}")
            
            # Add a new conversation if available
            if next_conv_idx < len(conversations):
                new_conv = conversations[next_conv_idx]
                active_convs[new_conv.conv_id] = new_conv
                conv_queue.appendleft(new_conv.conv_id)
                next_conv_idx += 1
        
        # Sleep between requests (Poisson process)
        if request_rate > 0:
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
    
    if verbose:
        print(f"[Client {client_id}] Completed all conversations")


async def run_multi_user_simulation(
    num_clients: int,
    conversations: List[Conversation],
    block_size: int,
    total_blocks: int,
    sampling_strategy: ConversationSampling,
    request_rate: float,
    max_active_conversations: int,
    verbose: bool = False
) -> Dict[int, ClientStats]:
    """
    Run the multi-user simulation with multiple clients.
    
    Returns:
        Dictionary mapping client_id to ClientStats
    """
    # Create shared KV cache manager
    manager = PrefixCacheManager(
        total_blocks=total_blocks,
        block_size=block_size
    )
    
    print(f"\n{'='*70}")
    print(f"Multi-User Simulation Starting")
    print(f"{'='*70}")
    print(f"Clients: {num_clients}")
    print(f"Total Conversations: {len(conversations)}")
    print(f"Block Size: {block_size}, Total Blocks: {total_blocks}")
    print(f"Max Active Conversations per Client: "
          f"{max_active_conversations}")
    print(f"Sampling Strategy: {sampling_strategy.value}")
    print(f"Request Rate: {request_rate} req/sec per client")
    print(f"{'='*70}\n")
    
    # Distribute conversations across clients
    convs_per_client = len(conversations) // num_clients
    client_stats = {}
    
    # Create client tasks
    tasks = []
    for client_id in range(num_clients):
        start_idx = client_id * convs_per_client
        end_idx = start_idx + convs_per_client
        if client_id == num_clients - 1:
            # Last client gets remaining conversations
            end_idx = len(conversations)
        
        client_convs = conversations[start_idx:end_idx]
        stats = ClientStats(client_id=client_id)
        client_stats[client_id] = stats
        
        task = asyncio.create_task(client_worker(
            client_id=client_id,
            conversations=client_convs,
            manager=manager,
            sampling_strategy=sampling_strategy,
            request_rate=request_rate,
            max_active_conversations=max_active_conversations,
            stats=stats,
            verbose=verbose
        ))
        tasks.append(task)
    
    # Run all clients concurrently
    start_time = time.time()
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Simulation Complete")
    print(f"{'='*70}")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"\nPer-Client Statistics:")
    print(f"{'-'*70}")
    
    total_requests = 0
    total_input_tokens = 0
    total_cached_tokens = 0
    
    for client_id, stats in sorted(client_stats.items()):
        cached_pct = (stats.cached_tokens / stats.total_input_tokens * 100) \
            if stats.total_input_tokens > 0 else 0
        print(f"Client {client_id}: "
              f"{stats.requests_sent} requests, "
              f"{stats.total_input_tokens} input tokens, "
              f"{stats.cached_tokens} cached ({cached_pct:.1f}%)")
        total_requests += stats.requests_sent
        total_input_tokens += stats.total_input_tokens
        total_cached_tokens += stats.cached_tokens
    
    print(f"{'-'*70}")
    overall_cached_pct = (total_cached_tokens / total_input_tokens * 100) \
        if total_input_tokens > 0 else 0
    print(f"Overall: {total_requests} requests, "
          f"{total_input_tokens} input tokens, "
          f"{total_cached_tokens} cached ({overall_cached_pct:.1f}%)")
    print(f"Requests/sec: {total_requests / (end_time - start_time):.2f}")
    print(f"{'='*70}\n")
    
    # Print final cache manager status
    manager.get_status()
    
    return client_stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Multi-User vLLM Prefix Caching Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 3 clients, 10 conversations each
  python3 multi_user_simulator.py --num-clients 3 \\
    --num-conversations 30

  # Use lognormal distribution for realistic token counts
  python3 multi_user_simulator.py --num-clients 5 \\
    --num-conversations 50 --user-tokens-dist lognormal \\
    --user-tokens-avg 150

  # High request rate simulation
  python3 multi_user_simulator.py --num-clients 10 \\
    --request-rate 2.0 --max-active-conversations 5

  # Verbose output for debugging
  python3 multi_user_simulator.py --num-clients 2 \\
    --num-conversations 4 --verbose
        """
    )
    
    parser.add_argument(
        '--num-clients', '-c',
        type=int,
        default=3,
        help='Number of concurrent clients/users (default: 3)'
    )
    
    parser.add_argument(
        '--num-conversations', '-n',
        type=int,
        default=20,
        help='Total number of conversations to simulate (default: 20)'
    )
    
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f'KV cache block size (default: {DEFAULT_BLOCK_SIZE})'
    )
    
    parser.add_argument(
        '--total-blocks', '-t',
        type=int,
        default=DEFAULT_TOTAL_BLOCKS,
        help=f'Total KV cache blocks (default: {DEFAULT_TOTAL_BLOCKS})'
    )
    
    parser.add_argument(
        '--num-turns',
        type=int,
        default=None,
        help='Fixed number of turns per conversation (default: uniform 4-8)'
    )
    
    parser.add_argument(
        '--common-prefix-tokens',
        type=int,
        default=100,
        help='Common prefix tokens shared across all conversations '
             '(default: 100)'
    )
    
    parser.add_argument(
        '--prefix-tokens-dist',
        choices=['constant', 'uniform', 'lognormal'],
        default='lognormal',
        help='Distribution for unique prefix tokens per conversation '
             '(default: lognormal)'
    )
    
    parser.add_argument(
        '--prefix-tokens-avg',
        type=int,
        default=200,
        help='Average prefix tokens per conversation (default: 200)'
    )
    
    parser.add_argument(
        '--user-tokens-dist',
        choices=['constant', 'uniform', 'lognormal'],
        default='uniform',
        help='Distribution for user message tokens (default: uniform)'
    )
    
    parser.add_argument(
        '--user-tokens-avg',
        type=int,
        default=50,
        help='Average user message tokens (default: 50)'
    )
    
    parser.add_argument(
        '--assistant-tokens-dist',
        choices=['constant', 'uniform', 'lognormal'],
        default='uniform',
        help='Distribution for assistant response tokens (default: uniform)'
    )
    
    parser.add_argument(
        '--assistant-tokens-avg',
        type=int,
        default=100,
        help='Average assistant response tokens (default: 100)'
    )
    
    parser.add_argument(
        '--request-rate', '-r',
        type=float,
        default=0.0,
        help='Request rate per client in requests/sec (Poisson process). '
             '0 = no delay (default: 0.0)'
    )
    
    parser.add_argument(
        '--max-active-conversations', '-k',
        type=int,
        default=3,
        help='Max active conversations per client (default: 3)'
    )
    
    parser.add_argument(
        '--sampling-strategy',
        type=ConversationSampling,
        choices=list(ConversationSampling),
        default=ConversationSampling.ROUND_ROBIN,
        help='Strategy for selecting conversations (default: round_robin)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create distributions
    if args.num_turns:
        num_turns_dist = ConstantDist(args.num_turns)
    else:
        num_turns_dist = UniformDist(4, 8)
    
    if args.prefix_tokens_dist == 'constant':
        prefix_tokens_dist = ConstantDist(args.prefix_tokens_avg)
    elif args.prefix_tokens_dist == 'uniform':
        prefix_tokens_dist = UniformDist(
            args.prefix_tokens_avg // 2,
            args.prefix_tokens_avg * 2
        )
    else:  # lognormal
        prefix_tokens_dist = LognormalDist(args.prefix_tokens_avg)
    
    if args.user_tokens_dist == 'constant':
        user_tokens_dist = ConstantDist(args.user_tokens_avg)
    elif args.user_tokens_dist == 'uniform':
        user_tokens_dist = UniformDist(
            args.user_tokens_avg // 2,
            args.user_tokens_avg + args.user_tokens_avg // 2
        )
    else:  # lognormal
        user_tokens_dist = LognormalDist(args.user_tokens_avg)
    
    if args.assistant_tokens_dist == 'constant':
        assistant_tokens_dist = ConstantDist(args.assistant_tokens_avg)
    elif args.assistant_tokens_dist == 'uniform':
        assistant_tokens_dist = UniformDist(
            args.assistant_tokens_avg - 20,
            args.assistant_tokens_avg + 20
        )
    else:  # lognormal
        assistant_tokens_dist = LognormalDist(args.assistant_tokens_avg)
    
    # Generate conversations
    print("Generating synthetic conversations...")
    conversations = generate_synthetic_conversations(
        num_conversations=args.num_conversations,
        num_turns_dist=num_turns_dist,
        prefix_tokens_dist=prefix_tokens_dist,
        user_tokens_dist=user_tokens_dist,
        assistant_tokens_dist=assistant_tokens_dist,
        common_prefix_tokens=args.common_prefix_tokens
    )
    print(f"Generated {len(conversations)} conversations")
    
    # Run the simulation
    asyncio.run(run_multi_user_simulation(
        num_clients=args.num_clients,
        conversations=conversations,
        block_size=args.block_size,
        total_blocks=args.total_blocks,
        sampling_strategy=args.sampling_strategy,
        request_rate=args.request_rate,
        max_active_conversations=args.max_active_conversations,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()

