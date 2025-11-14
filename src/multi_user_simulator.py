#!/usr/bin/env python3
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

# Import visualizer
try:
    from visualizer import SimulationVisualizer, NullVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    # Create a fallback NullVisualizer if import fails
    class NullVisualizer:
        def __init__(self, *args, **kwargs): pass
        def update_cache_state(self, *args, **kwargs): pass
        def update_request(self, *args, **kwargs): pass
        def update_active_conversations(self, *args, **kwargs): pass
        def add_event(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def keep_alive(self): pass
        def close(self): pass


class ConversationSampling(str, Enum):
    """Strategy for selecting which conversation to process next."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class ConversationTemplate(str, Enum):
    """Predefined conversation patterns for realistic simulation."""
    QUICK_QA = "quick_qa"          # 1-2 turns, fast responses
    STANDARD_CHAT = "standard"     # 4-8 turns, normal pace
    DEEP_DIVE = "deep_dive"        # 10-20 turns, longer messages
    DEBUG_SESSION = "debug"        # 15-30 turns, technical
    MIXED = "mixed"                # Random mix of above


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
    template: ConversationTemplate = ConversationTemplate.STANDARD_CHAT
    inter_turn_delay: float = 5.0  # Seconds between turns (think time)
    
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


@dataclass
class TemplateConfig:
    """Configuration for a conversation template."""
    name: str
    turns_range: tuple  # (min, max) turns
    user_tokens_range: tuple  # (min, max) tokens
    assistant_tokens_range: tuple  # (min, max) tokens
    inter_turn_delay_range: tuple  # (min, max) seconds
    description: str


# Predefined template configurations
TEMPLATE_CONFIGS = {
    ConversationTemplate.QUICK_QA: TemplateConfig(
        name="Quick Q&A",
        turns_range=(1, 2),
        user_tokens_range=(20, 50),
        assistant_tokens_range=(30, 80),
        inter_turn_delay_range=(1.0, 3.0),
        description="Fast, short questions with quick responses"
    ),
    ConversationTemplate.STANDARD_CHAT: TemplateConfig(
        name="Standard Chat",
        turns_range=(4, 8),
        user_tokens_range=(40, 80),
        assistant_tokens_range=(80, 150),
        inter_turn_delay_range=(3.0, 8.0),
        description="Normal conversation with typical think time"
    ),
    ConversationTemplate.DEEP_DIVE: TemplateConfig(
        name="Deep Dive",
        turns_range=(10, 20),
        user_tokens_range=(80, 200),
        assistant_tokens_range=(150, 400),
        inter_turn_delay_range=(5.0, 15.0),
        description="In-depth discussion with longer messages"
    ),
    ConversationTemplate.DEBUG_SESSION: TemplateConfig(
        name="Debug Session",
        turns_range=(15, 30),
        user_tokens_range=(50, 150),
        assistant_tokens_range=(100, 300),
        inter_turn_delay_range=(2.0, 10.0),
        description="Technical troubleshooting with code/logs"
    ),
}


def generate_sample_text(
    conv_id: str,
    turn: int,
    is_user: bool,
    token_count: int,
    conversation_obj=None
) -> str:
    """
    Generate sample conversational text for visualization.
    
    If conversation_obj is a TextConversation, extracts actual text.
    Otherwise generates synthetic sample text.
    
    Args:
        conv_id: Conversation ID
        turn: Turn number
        is_user: True for user message, False for assistant
        token_count: Number of tokens (affects length)
        conversation_obj: Optional Conversation or TextConversation object
    
    Returns:
        Sample text string
    """
    # Check if this is a TextConversation with real text
    if conversation_obj and hasattr(conversation_obj, 'messages'):
        try:
            # Find the message at this turn
            # In text mode, messages alternate user/assistant
            # turn index maps to message index
            if turn < len(conversation_obj.messages):
                msg = conversation_obj.messages[turn]
                if hasattr(msg, 'content'):
                    return msg.content
        except:
            pass
    
    # Fallback to synthetic text
    if is_user:
        templates = [
            f"Can you explain how {conv_id} works in detail?",
            f"I need help with {conv_id} turn {turn}",
            f"What are the best practices for {conv_id}?",
            f"How do I optimize {conv_id} performance?",
            f"Tell me more about {conv_id} implementation",
        ]
        text = random.choice(templates)
    else:
        templates = [
            f"I'd be happy to explain {conv_id}. Let me break it down...",
            f"Based on turn {turn}, here's what you need to know...",
            f"The key aspects of {conv_id} are: efficiency, reliability...",
            f"For {conv_id}, consider these best practices...",
            f"Great question! {conv_id} is implemented using...",
        ]
        text = random.choice(templates)
    
    # Adjust length based on token count
    if token_count < 20:
        # Short response
        text = text.split('.')[0] + "."
    elif token_count > 50:
        # Long response - ensure proper punctuation
        if not text.endswith(('.', '!', '?')):
            text += "."
        text += " This involves multiple steps and considerations..."
    
    return text


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


def generate_new_conversation(
    client_id: int,
    conv_counter: int,
    num_turns_dist,
    prefix_tokens_dist,
    user_tokens_dist,
    assistant_tokens_dist,
    common_prefix_tokens: int
) -> Conversation:
    """Generate a new conversation on-the-fly."""
    conv_id = f"conv_c{client_id}_{conv_counter:04d}"
    
    # Generate common and unique prefix
    common_prefix = list(range(1, common_prefix_tokens + 1))
    prefix_len = prefix_tokens_dist.sample()
    # Use client_id and conv_counter to make unique token IDs
    base_token = 10000 + (client_id * 100000) + (conv_counter * 1000)
    unique_prefix = list(range(base_token, base_token + prefix_len))
    token_counter = base_token + prefix_len
    
    # Determine number of turns
    num_turns = max(1, num_turns_dist.sample() // 2)
    
    turns = []
    for turn_idx in range(num_turns):
        # User message tokens
        user_len = user_tokens_dist.sample()
        user_tokens = list(range(token_counter, token_counter + user_len))
        token_counter += user_len
        
        # Assistant response tokens
        assistant_len = assistant_tokens_dist.sample()
        assistant_tokens = list(range(
            token_counter, token_counter + assistant_len
        ))
        token_counter += assistant_len
        
        # For the first turn, include prefixes
        if turn_idx == 0:
            user_tokens = common_prefix + unique_prefix + user_tokens
        
        turns.append(ConversationTurn(
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens
        ))
    
    return Conversation(conv_id=conv_id, turns=turns)


def generate_conversation_from_template(
    conv_id: str,
    template: ConversationTemplate,
    common_prefix_tokens: int,
    client_id: int = 0,
    conv_counter: int = 0
) -> Conversation:
    """Generate a conversation based on a template."""
    if template == ConversationTemplate.MIXED:
        # Randomly pick a non-mixed template
        template = random.choice([
            ConversationTemplate.QUICK_QA,
            ConversationTemplate.STANDARD_CHAT,
            ConversationTemplate.DEEP_DIVE,
            ConversationTemplate.DEBUG_SESSION
        ])
    
    config = TEMPLATE_CONFIGS[template]
    
    # Determine conversation parameters from template
    num_turns = random.randint(*config.turns_range)
    inter_turn_delay = random.uniform(*config.inter_turn_delay_range)
    
    # Generate common and unique prefix
    common_prefix = list(range(1, common_prefix_tokens + 1))
    prefix_len = random.randint(100, 300)
    base_token = 10000 + (client_id * 100000) + (conv_counter * 1000)
    unique_prefix = list(range(base_token, base_token + prefix_len))
    token_counter = base_token + prefix_len
    
    turns = []
    for turn_idx in range(num_turns):
        # Use template ranges for token counts
        user_len = random.randint(*config.user_tokens_range)
        user_tokens = list(range(token_counter, token_counter + user_len))
        token_counter += user_len
        
        assistant_len = random.randint(*config.assistant_tokens_range)
        assistant_tokens = list(range(
            token_counter, token_counter + assistant_len
        ))
        token_counter += assistant_len
        
        # For the first turn, include prefixes
        if turn_idx == 0:
            user_tokens = common_prefix + unique_prefix + user_tokens
        
        turns.append(ConversationTurn(
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens
        ))
    
    return Conversation(
        conv_id=conv_id,
        turns=turns,
        template=template,
        inter_turn_delay=inter_turn_delay
    )


async def client_worker(
    client_id: int,
    conversations: List[Conversation],
    manager: PrefixCacheManager,
    sampling_strategy: ConversationSampling,
    request_rate: float,
    max_active_conversations: int,
    stats: ClientStats,
    visualizer: Optional[object] = None,
    verbose: bool = False,
    conversation_generator: Optional[callable] = None
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
        visualizer: Optional visualizer for real-time display
        verbose: Whether to print detailed logs
    """
    if verbose:
        print(f"[Client {client_id}] Starting with "
              f"{len(conversations)} conversations", flush=True)
    
    # Active conversations queue
    active_convs: Dict[str, Conversation] = {}
    conv_queue: deque = deque(maxlen=max_active_conversations)
    
    # Start with random initial load (0 to max_active_conversations)
    # Add variability: use client_id as seed for different behavior per client
    rng = np.random.RandomState(seed=client_id + 42)
    initial_count = rng.randint(0, max(1, max_active_conversations // 2) + 1)
    
    for conv in conversations[:initial_count]:
        active_convs[conv.conv_id] = conv
        conv_queue.append(conv.conv_id)
    
    # Track index for round-robin through all conversations
    next_conv_idx = initial_count
    
    # Client-specific variability (some clients more/less active)
    client_activity_factor = 0.5 + rng.random()  # 0.5 to 1.5
    
    if verbose:
        print(f"[Client {client_id}] Starting with {initial_count} active "
              f"conversations (activity factor: {client_activity_factor:.2f})", 
              flush=True)
    
    # Update visualizer with initial state
    if visualizer:
        visualizer.update_active_conversations(
            client_id, len(active_convs)
        )
        visualizer.add_event(
            f"Client {client_id} started with {initial_count} active convs"
        )
    
    # Continue until we've gone through all conversations or hit stop condition
    while len(active_convs) > 0:
        # Pick a conversation to process
        if sampling_strategy == ConversationSampling.ROUND_ROBIN:
            conv_id = conv_queue.pop() if conv_queue else None
        else:  # RANDOM
            conv_id = random.choice(list(active_convs.keys())) \
                if active_convs else None
        
        if conv_id is None:
            # No active conversations, wait a bit before trying again
            if next_conv_idx < len(conversations):
                await asyncio.sleep(0.1)
                continue
            else:
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
            # Calculate history tokens for both conversation types
            if hasattr(conv, 'turns'):
                current_user_tokens = len(conv.turns[conv.current_turn].user_tokens)
            elif hasattr(conv, 'messages') and conv.current_turn < len(conv.messages):
                current_user_tokens = len(conv.messages[conv.current_turn].token_ids)
            else:
                current_user_tokens = 0
            
            history_tokens = len(prompt_tokens) - current_user_tokens
            print(f"[Client {client_id}] {conv_id} Turn "
                  f"{conv.current_turn}: {len(prompt_tokens)} tokens "
                  f"({history_tokens} history + {current_user_tokens} new)", 
                  flush=True)
        
        # Process the request (this is synchronous in our simulation)
        # Set current conversation ID for cache hit tracking
        manager.current_conv_id = conv_id
        cache_hit_count = manager.process_request(req)
        
        # Update stats
        stats.requests_sent += 1
        stats.total_input_tokens += len(prompt_tokens)
        stats.cached_tokens += cache_hit_count
        
        # Simulate assistant response (would be in manager in real system)
        # Handle both synthetic and text mode conversations
        if hasattr(conv, 'turns'):
            # Synthetic mode: Conversation with turns
            output_tokens = conv.turns[conv.current_turn].assistant_tokens
        elif hasattr(conv, 'messages') and conv.current_turn + 1 < len(conv.messages):
            # Text mode: TextConversation with messages
            # Assistant message is at current_turn + 1 (messages alternate user/assistant)
            output_tokens = conv.messages[conv.current_turn + 1].token_ids
        else:
            output_tokens = []
        
        stats.total_output_tokens += len(output_tokens)
        
        # Calculate cache hit percentage for this request
        cached_percent = (cache_hit_count / len(prompt_tokens) * 100) \
            if len(prompt_tokens) > 0 else 0
        
        # Update visualizer
        if visualizer:
            # Update cache state
            used_blocks = manager.total_blocks - len(manager.free_block_ids)
            visualizer.update_cache_state(used_blocks)
            
            # Update request stats
            visualizer.update_request(
                client_id, cache_hit_count, len(prompt_tokens)
            )
            
            # Add event
            visualizer.add_event(
                f"C{client_id} {conv_id} T{conv.current_turn}: "
                f"{cached_percent:.0f}% cached"
            )
            
            # Add conversation snippet (user message)
            # For text mode, extract actual text; for synthetic mode, generate
            if hasattr(conv, 'turns'):
                user_tokens_len = len(conv.turns[conv.current_turn].user_tokens)
            elif hasattr(conv, 'messages') and conv.current_turn < len(conv.messages):
                user_tokens_len = len(conv.messages[conv.current_turn].token_ids)
            else:
                user_tokens_len = 0
            user_text = generate_sample_text(
                conv_id=conv_id,
                turn=conv.current_turn,
                is_user=True,
                token_count=user_tokens_len,
                conversation_obj=conv
            )
            visualizer.add_conversation_snippet(
                client_id=client_id,
                conv_id=conv_id,
                turn=conv.current_turn,
                text=user_text,
                is_user=True,
                cache_hit_count=cache_hit_count,
                token_count=len(prompt_tokens)
            )
            
            # Add conversation snippet (assistant response)
            assistant_text = generate_sample_text(
                conv_id=conv_id,
                turn=conv.current_turn,
                is_user=False,
                token_count=len(output_tokens),
                conversation_obj=conv
            )
            visualizer.add_conversation_snippet(
                client_id=client_id,
                conv_id=conv_id,
                turn=conv.current_turn,
                text=assistant_text,
                is_user=False
            )
        
        if verbose:
            print(f"[Client {client_id}] {conv_id} Turn "
                  f"{conv.current_turn} completed: "
                  f"{cached_percent:.1f}% cached", flush=True)
        
        # Check if we should stop early (e.g., max turns limit reached)
        if visualizer and hasattr(visualizer, 'should_stop') and \
           visualizer.should_stop:
            if verbose:
                print(f"[Client {client_id}] Stopping due to turn limit",
                      flush=True)
            # Free current request before stopping
            manager.free_request(req)
            break
        
        # Free the request
        manager.free_request(req)
        
        # Move to next turn
        conv.current_turn += 1
        
        if conv.has_more_turns():
            # Add inter-turn delay (think time between turns in same
            # conversation)
            if hasattr(conv, 'inter_turn_delay') and conv.inter_turn_delay > 0:
                if verbose:
                    print(f"[Client {client_id}] {conv_id} pausing for "
                          f"{conv.inter_turn_delay:.1f}s (think time)",
                          flush=True)
                await asyncio.sleep(conv.inter_turn_delay)
            
            # Conversation continues, add back to queue
            if sampling_strategy == ConversationSampling.ROUND_ROBIN:
                conv_queue.appendleft(conv_id)
        else:
            # Conversation finished
            active_convs.pop(conv_id)
            if verbose:
                print(f"[Client {client_id}] Finished {conv_id}", flush=True)
            
            # Update visualizer
            if visualizer:
                # Record full conversation token sequence for similarity analysis
                full_tokens = conv.get_messages_up_to_turn(conv.current_turn)
                visualizer.record_conversation_tokens(conv_id, full_tokens)
                
                # Record conversation completion time
                visualizer.record_conversation_completion(conv_id)
                
                visualizer.update_active_conversations(
                    client_id, len(active_convs)
                )
            
            # Probabilistically add a new conversation (not always immediate replacement)
            # Probability increases if we're below max capacity
            capacity_ratio = len(active_convs) / max(1, max_active_conversations)
            # Probability ranges from 0.9 (at 0% capacity) to 0.3 (at 100% capacity)
            add_prob = (0.9 - 0.6 * capacity_ratio) * client_activity_factor
            
            should_add = rng.random() < add_prob and \
                        len(active_convs) < max_active_conversations
            
            if should_add:
                # Add a new conversation - either from original list or generate
                new_conv = None
                if next_conv_idx < len(conversations):
                    # Use conversation from original list
                    new_conv = conversations[next_conv_idx]
                    next_conv_idx += 1
                elif conversation_generator:
                    # Generate new conversation on-the-fly (continuous mode)
                    new_conv = conversation_generator()
                    if verbose:
                        print(f"[Client {client_id}] Generated new "
                              f"conversation {new_conv.conv_id}", flush=True)
                
                if new_conv:
                    active_convs[new_conv.conv_id] = new_conv
                    conv_queue.appendleft(new_conv.conv_id)
                    
                    if verbose:
                        print(f"[Client {client_id}] Started new conversation "
                              f"{new_conv.conv_id} ({len(active_convs)} active)",
                              flush=True)
                    
                    # Update visualizer
                    if visualizer:
                        visualizer.update_active_conversations(
                            client_id, len(active_convs)
                        )
            elif verbose:
                print(f"[Client {client_id}] Not starting new conversation "
                      f"({len(active_convs)} active, prob={add_prob:.2f})",
                      flush=True)
        
        # Sleep between requests (Poisson process)
        if request_rate > 0:
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
    
    if verbose:
        print(f"[Client {client_id}] Completed all conversations", flush=True)


async def update_visualization_loop(visualizer, stop_event):
    """Background task to force visualization updates."""
    while not stop_event.is_set():
        try:
            visualizer.keep_alive()
            await asyncio.sleep(0.1)  # Update 10 times per second
        except:
            break


async def run_multi_user_simulation(
    num_clients: int,
    conversations: List[Conversation],
    block_size: int,
    total_blocks: int,
    sampling_strategy: ConversationSampling,
    request_rate: float,
    max_active_conversations: int,
    visualizer: Optional[object] = None,
    verbose: bool = False,
    conversation_generator_params: Optional[Dict] = None
) -> Dict[int, ClientStats]:
    """
    Run the multi-user simulation with multiple clients.
    
    Args:
        visualizer: Optional visualizer for real-time display
    
    Returns:
        Dictionary mapping client_id to ClientStats
    """
    # Create shared KV cache manager
    manager = PrefixCacheManager(
        total_blocks=total_blocks,
        block_size=block_size,
        visualizer=visualizer
    )
    
    # Start visualization update task if visualizer is present
    stop_event = asyncio.Event()
    viz_task = None
    if visualizer:
        viz_task = asyncio.create_task(
            update_visualization_loop(visualizer, stop_event)
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
        
        # Create conversation generator if parameters provided
        generator = None
        if conversation_generator_params:
            conv_counter = [0]  # Mutable to track count
            
            def make_generator(cid):
                def gen():
                    conv_counter[0] += 1
                    conv_id = f"conv_c{cid}_{conv_counter[0]:04d}"
                    
                    # Check if using templates
                    if 'template' in conversation_generator_params:
                        return generate_conversation_from_template(
                            conv_id=conv_id,
                            client_id=cid,
                            conv_counter=conv_counter[0],
                            **conversation_generator_params
                        )
                    else:
                        # Legacy generator with distributions
                        return generate_new_conversation(
                            client_id=cid,
                            conv_counter=conv_counter[0],
                            **conversation_generator_params
                        )
                return gen
            
            generator = make_generator(client_id)
        
        task = asyncio.create_task(client_worker(
            client_id=client_id,
            conversations=client_convs,
            manager=manager,
            sampling_strategy=sampling_strategy,
            request_rate=request_rate,
            max_active_conversations=max_active_conversations,
            stats=stats,
            visualizer=visualizer,
            verbose=verbose,
            conversation_generator=generator
        ))
        tasks.append(task)
    
    # Run all clients concurrently
    start_time = time.time()
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Stop visualization update task
    if viz_task:
        stop_event.set()
        await viz_task
    
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
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable real-time visualization dashboard (requires matplotlib)'
    )
    
    # Text mode arguments
    parser.add_argument(
        '--text-mode',
        action='store_true',
        help='Use real text conversations with tokenization instead of '
             'integer token IDs'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to ShareGPT JSON dataset file (optional, will generate '
             'synthetic if not provided)'
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='HuggingFace tokenizer to use in text mode (default: gpt2)'
    )
    
    parser.add_argument(
        '--common-prefix-text',
        type=str,
        default=None,
        help='Common text prefix for all conversations (system prompt)'
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
    if args.text_mode:
        print("\n" + "="*70)
        print("TEXT MODE ENABLED")
        print("="*70)
        print(f"Tokenizer: {args.tokenizer}")
        if args.dataset_path:
            print(f"Dataset: {args.dataset_path}")
        else:
            print("Dataset: Synthetic text conversations (no file provided)")
        print("="*70 + "\n")
        
        # Import text mode module
        try:
            import text_mode
            
            # Create text dataset
            dataset = text_mode.TextModeDataset(
                dataset_path=args.dataset_path,
                tokenizer_name=args.tokenizer
            )
            
            # Load conversations
            # Load more than needed for better filtering
            dataset.load_sharegpt_dataset(
                max_conversations=args.num_conversations * 2,
                min_turns=2,
                max_turns=20
            )
            
            # Get text conversations
            conversations = dataset.get_text_conversations(
                num_conversations=args.num_conversations,
                common_prefix=args.common_prefix_text
            )
            
            # Print stats
            stats = dataset.get_stats()
            print(f"\n📊 Dataset Statistics:")
            print(f"  Conversations: {stats.get('num_conversations', 0)}")
            print(f"  Total messages: {stats.get('total_messages', 0)}")
            print(f"  Avg turns/conv: {stats.get('avg_turns_per_conv', 0):.1f}")
            print(f"  Turn range: {stats.get('min_turns', 0)}-{stats.get('max_turns', 0)}")
            print(f"  Total characters: {stats.get('total_chars', 0):,}\n")
            
        except ImportError as e:
            print(f"ERROR: Could not import text_mode module: {e}")
            print("Falling back to synthetic integer token mode...")
            args.text_mode = False
            
            # Fall back to regular mode
            conversations = generate_synthetic_conversations(
                num_conversations=args.num_conversations,
                num_turns_dist=num_turns_dist,
                prefix_tokens_dist=prefix_tokens_dist,
                user_tokens_dist=user_tokens_dist,
                assistant_tokens_dist=assistant_tokens_dist,
                common_prefix_tokens=args.common_prefix_tokens
            )
    else:
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
    
    # Create visualizer if requested
    visualizer = None
    if args.visualize:
        if not VISUALIZER_AVAILABLE:
            print("\nWARNING: matplotlib not available. "
                  "Install with: pip install matplotlib")
            print("Running without visualization...\n")
            visualizer = NullVisualizer()
        else:
            print("\nStarting visualization dashboard...")
            print("(Close the window to end visualization)\n")
            visualizer = SimulationVisualizer(
                num_clients=args.num_clients,
                total_blocks=args.total_blocks,
                update_interval=100  # Update every 100ms
            )
            visualizer.start()
    
    # Run the simulation
    try:
        asyncio.run(run_multi_user_simulation(
            num_clients=args.num_clients,
            conversations=conversations,
            block_size=args.block_size,
            total_blocks=args.total_blocks,
            sampling_strategy=args.sampling_strategy,
            request_rate=args.request_rate,
            max_active_conversations=args.max_active_conversations,
            visualizer=visualizer,
            verbose=args.verbose
        ))
        
        # Keep visualization alive after simulation
        if visualizer and args.visualize and VISUALIZER_AVAILABLE:
            print("\nSimulation complete! Visualization will remain open.")
            print("Press Ctrl+C or close the window to exit.")
            try:
                import matplotlib.pyplot as plt
                plt.show(block=True)
            except KeyboardInterrupt:
                print("\nClosing visualization...")
    finally:
        if visualizer:
            visualizer.stop()
            visualizer.close()


if __name__ == "__main__":
    main()

