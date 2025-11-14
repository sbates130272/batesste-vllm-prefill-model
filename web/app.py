#!/usr/bin/env python3
"""
Web UI for vLLM Prefill Model Simulator

FastAPI-based web interface with real-time simulation updates.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import simulator modules
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vllm_prefill_model",
    Path(__file__).parent.parent / "src" / "vllm-prefill-model.py"
)
vllm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vllm_module)

from multi_user_simulator import (
    generate_synthetic_conversations,
    run_multi_user_simulation,
    ConversationSampling,
    UniformDist,
    ConstantDist,
    LognormalDist,
    ClientStats,
    ConversationTemplate,
    generate_conversation_from_template
)

# Try to import text_mode
try:
    import text_mode
    TEXT_MODE_AVAILABLE = True
except ImportError:
    TEXT_MODE_AVAILABLE = False

# Try to load a default tokenizer
TOKENIZER = None
TOKENIZER_DECODE_CACHE = {}  # Cache for token ID -> text mapping
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    print("✓ Loaded GPT-2 tokenizer for text generation")
except ImportError:
    print("⚠ transformers not installed. Run: pip install transformers")
    print("  Using synthetic token IDs instead of real text.")
except Exception as e:
    print(f"⚠ Could not load tokenizer: {e}")
    print("  Using synthetic token IDs instead of real text.")


def decode_tokens(token_ids: List[int]) -> List[str]:
    """Decode token IDs to text strings."""
    if not TOKENIZER:
        # Return token IDs as strings if no tokenizer
        return [f"tok_{tid}" for tid in token_ids]
    
    result = []
    for tid in token_ids:
        if tid not in TOKENIZER_DECODE_CACHE:
            try:
                TOKENIZER_DECODE_CACHE[tid] = TOKENIZER.decode([tid])
            except:
                TOKENIZER_DECODE_CACHE[tid] = f"<{tid}>"
        result.append(TOKENIZER_DECODE_CACHE[tid])
    return result


app = FastAPI(
    title="vLLM Prefill Model Simulator",
    description="Web interface for simulating vLLM prefix caching",
    version="1.0.0"
)

# Store active simulations
active_simulations: Dict[str, dict] = {}


class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""
    num_clients: int = 3
    num_turns: Optional[int] = None
    block_size: int = 16
    total_blocks: int = 500
    common_prefix_tokens: int = 100
    prefix_tokens_avg: int = 200
    user_tokens_avg: int = 50
    assistant_tokens_avg: int = 100
    request_rate: float = 0.0
    max_active_conversations: int = 3
    max_total_turns: int = 0  # 0 = unlimited (per simulation)
    time_limit: int = 60  # seconds, -1 = infinite
    sampling_strategy: str = "round_robin"
    conversation_template: str = "standard"  # Template for conversation patterns
    text_mode: bool = False
    dataset_path: Optional[str] = None
    tokenizer: str = "gpt2"
    common_prefix_text: Optional[str] = None


class WebSimulationCollector:
    """Collects simulation data for web display."""
    
    def __init__(self, sim_id: str):
        self.sim_id = sim_id
        self.events: List[Dict] = []
        self.metrics: Dict = {
            'cache_occupancy': [],
            'cache_hit_rate': [],
            'requests_per_second': [],
            'active_conversations': [],
            'timestamps': []
        }
        self.client_stats: Dict[int, dict] = {}
        self.conversations: Dict[str, Dict] = {}
        self.start_time = time.time()
        self.status = "running"
        self.active_conv_history: List[int] = []  # Track active conv counts
        
        # Advanced analytics data
        self.heatmap_data: List[Dict] = []  # Time-series cache hit data
        self.conv_effectiveness: Dict[str, Dict] = {}  # Per-conversation stats
        self.prefix_stats = {
            'common_prefix_tokens': 0,
            'unique_tokens': 0,
            'total_cached_tokens': 0
        }
        
        # Cache state log
        self.cache_state_log: List[Dict] = []
        self.last_cache_hit: Dict = {}
        self.recent_cache_hits: List[Dict] = []  # Last 5 hits
        
        # Conversation duration tracking
        self.conversation_durations: List[float] = []  # In seconds
        self.conversation_start_times: Dict[str, float] = {}  # conv_id -> start time
        
        # Conversation similarity tracking
        self.conversation_tokens: Dict[str, List[int]] = {}  # conv_id -> token sequence
        self.jaccard_timeline: List[Dict] = []  # Average Jaccard over time
        
        # Initialize heatmap with starting point at t=0
        self.heatmap_data.append({
            'time': 0.0,
            'hit_rate': 0.0,
            'active_convs': 0
        })
        
        # Initialize Jaccard timeline
        self.jaccard_timeline.append({
            'time': 0.0,
            'avg_jaccard': 0.0,
            'num_pairs': 0
        })
    
    def add_event(self, event_type: str, message: str, data: dict = None):
        """Add an event to the log."""
        self.events.append({
            'timestamp': time.time() - self.start_time,
            'type': event_type,
            'message': message,
            'data': data or {}
        })
    
    def update_metrics(
        self,
        cache_used: int,
        cache_total: int,
        hit_count: int,
        total_tokens: int,
        active_convs: int
    ):
        """Update simulation metrics."""
        elapsed = time.time() - self.start_time
        self.metrics['timestamps'].append(elapsed)
        
        # Calculate cache occupancy, ensuring valid values
        if cache_total > 0:
            occupancy = max(0, min(100, (cache_used / cache_total) * 100))
        else:
            occupancy = 0
        self.metrics['cache_occupancy'].append(occupancy)
        
        hit_rate = (hit_count / total_tokens * 100) if total_tokens > 0 else 0
        self.metrics['cache_hit_rate'].append(hit_rate)
        self.metrics['active_conversations'].append(active_convs)
        self.active_conv_history.append(active_convs)
        
        # Update analytics with current metrics
        self.update_analytics_point(hit_rate, active_convs)
        
        # Log cache state (every few seconds)
        current_time = time.time() - self.start_time
        if not self.cache_state_log or current_time - self.cache_state_log[-1]['timestamp'] >= 2.0:
            self.add_cache_state_log(cache_used, cache_total, hit_rate)
    
    def add_cache_state_log(self, blocks_used: int, blocks_total: int, hit_rate: float):
        """Add cache state log entry."""
        current_time = time.time() - self.start_time
        occupancy = (blocks_used / blocks_total * 100) if blocks_total > 0 else 0
        
        # Calculate instantaneous hit rate (if cache is empty, rate is 0)
        instant_hit_rate = hit_rate if blocks_used > 0 else 0.0
        
        entry = {
            'timestamp': current_time,
            'blocks_used': blocks_used,
            'blocks_total': blocks_total,
            'occupancy': occupancy,
            'hit_rate_cumulative': hit_rate,  # Overall average
            'hit_rate_instant': instant_hit_rate,  # Current state
            'last_hit': self.last_cache_hit.copy() if self.last_cache_hit else None,
            'recent_hits': self.recent_cache_hits.copy()  # Last 5 hits
        }
        
        self.cache_state_log.append(entry)
        # Keep last 100 entries
        if len(self.cache_state_log) > 100:
            self.cache_state_log.pop(0)
    
    def record_cache_hit(self, block_hash: str, tokens: List[int], conv_id: str):
        """Record a cache hit with block details."""
        # Decode tokens to text
        token_texts = decode_tokens(tokens[:10])
        
        hit = {
            'block_hash': block_hash,  # Full MD5 hash (32 chars)
            'token_count': len(tokens),
            'tokens_preview': tokens[:10],  # First 10 token IDs
            'tokens_text': token_texts,  # Decoded text
            'conv_id': conv_id,
            'timestamp': time.time() - self.start_time
        }
        
        self.last_cache_hit = hit
        
        # Add to recent hits (keep last 5)
        self.recent_cache_hits.append(hit)
        if len(self.recent_cache_hits) > 5:
            self.recent_cache_hits.pop(0)
    
    def update_analytics_point(self, hit_rate: float, active_convs: int):
        """Add a point to the heatmap data."""
        current_time = time.time() - self.start_time
        
        # Sample heatmap data every 0.5 seconds
        if not self.heatmap_data or current_time - self.heatmap_data[-1]['time'] >= 0.5:
            self.heatmap_data.append({
                'time': current_time,
                'hit_rate': hit_rate,
                'active_convs': active_convs
            })
            # Keep last 200 points
            if len(self.heatmap_data) > 200:
                self.heatmap_data.pop(0)
    
    def update_conversation_effectiveness(
        self,
        conv_id: str,
        cache_hits: int,
        total_tokens: int
    ):
        """Track per-conversation cache effectiveness."""
        if conv_id not in self.conv_effectiveness:
            self.conv_effectiveness[conv_id] = {
                'total_tokens': 0,
                'cached_tokens': 0,
                'requests': 0
            }
        
        self.conv_effectiveness[conv_id]['total_tokens'] += total_tokens
        self.conv_effectiveness[conv_id]['cached_tokens'] += cache_hits
        self.conv_effectiveness[conv_id]['requests'] += 1
        
        # Update aggregate prefix stats
        self.prefix_stats['total_cached_tokens'] += cache_hits
        self.prefix_stats['unique_tokens'] += (total_tokens - cache_hits)
        # Estimate common prefix (first 10% of cached tokens are likely common prefix)
        self.prefix_stats['common_prefix_tokens'] += int(cache_hits * 0.1)
    
    def get_analytics_summary(self) -> dict:
        """Get analytics data for visualization."""
        # Calculate per-conversation hit rates
        conv_hit_rates = []
        for conv_id, stats in self.conv_effectiveness.items():
            if stats['total_tokens'] > 0:
                hit_rate = (stats['cached_tokens'] / stats['total_tokens']) * 100
                conv_hit_rates.append({
                    'id': conv_id,
                    'hit_rate': hit_rate,
                    'requests': stats['requests'],
                    'total_tokens': stats['total_tokens']
                })
        
        # Sort by hit rate and take top 10
        conv_hit_rates.sort(key=lambda x: x['hit_rate'], reverse=True)
        top_convs = conv_hit_rates[:10]
        
        # Calculate conversation similarities
        similarities = self.calculate_conversation_similarities() if self.conversation_tokens else None
        
        return {
            'heatmap': self.heatmap_data[-50:],  # Last 50 points for display
            'top_conversations': top_convs,
            'prefix_overlap': self.prefix_stats.copy(),
            'conversation_similarities': similarities,
            'jaccard_timeline': self.jaccard_timeline.copy()
        }
    
    def add_conversation_snippet(
        self,
        conv_id: str,
        client_id: int,
        turns: List[Dict],
        cache_hits: int = 0,
        total_tokens: int = 0
    ):
        """Add or update a conversation snippet."""
        current_time = time.time()
        
        # Track conversation start time
        if conv_id not in self.conversation_start_times:
            self.conversation_start_times[conv_id] = current_time
        
        self.conversations[conv_id] = {
            'conv_id': conv_id,
            'client_id': client_id,
            'turns': turns,
            'num_turns': len(turns),
            'cache_hits': cache_hits,
            'total_tokens': total_tokens,
            'last_updated': current_time
        }
        
        # Track conversation effectiveness for analytics
        if total_tokens > 0:
            self.update_conversation_effectiveness(conv_id, cache_hits, total_tokens)
    
    def record_conversation_completion(self, conv_id: str):
        """Record when a conversation completes."""
        if conv_id in self.conversation_start_times:
            start_time = self.conversation_start_times[conv_id]
            duration = time.time() - start_time
            self.conversation_durations.append(duration)
            del self.conversation_start_times[conv_id]
    
    def record_conversation_tokens(self, conv_id: str, tokens: List[int]):
        """Record token sequence for a conversation."""
        if conv_id not in self.conversation_tokens:
            self.conversation_tokens[conv_id] = []
        self.conversation_tokens[conv_id] = tokens
    
    def update_jaccard_timeline(self):
        """Calculate and record average Jaccard similarity at current time."""
        if len(self.conversation_tokens) < 2:
            return  # Need at least 2 conversations to compare
        
        conv_ids = list(self.conversation_tokens.keys())
        jaccard_values = []
        
        # Calculate Jaccard for all pairs
        for i, conv_id_a in enumerate(conv_ids):
            for j, conv_id_b in enumerate(conv_ids):
                if i >= j:
                    continue  # Only upper triangle
                
                tokens_a = self.conversation_tokens[conv_id_a]
                tokens_b = self.conversation_tokens[conv_id_b]
                
                # Calculate Jaccard similarity
                set_a = set(tokens_a)
                set_b = set(tokens_b)
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = intersection / union if union > 0 else 0
                
                jaccard_values.append(jaccard)
        
        # Record average Jaccard
        if jaccard_values:
            avg_jaccard = sum(jaccard_values) / len(jaccard_values)
            self.jaccard_timeline.append({
                'time': time.time() - self.start_time,
                'avg_jaccard': avg_jaccard,
                'num_pairs': len(jaccard_values)
            })
    
    def calculate_conversation_similarities(self) -> Dict:
        """
        Calculate pairwise similarities between conversations.
        Returns metrics useful for cache analysis.
        """
        conv_ids = list(self.conversation_tokens.keys())[:20]  # Top 20 for visualization
        
        similarities = []
        for i, conv_id_a in enumerate(conv_ids):
            for j, conv_id_b in enumerate(conv_ids):
                if i >= j:
                    continue  # Only upper triangle
                
                tokens_a = self.conversation_tokens[conv_id_a]
                tokens_b = self.conversation_tokens[conv_id_b]
                
                # Calculate prefix overlap (most relevant for caching)
                prefix_len = 0
                for k in range(min(len(tokens_a), len(tokens_b))):
                    if tokens_a[k] == tokens_b[k]:
                        prefix_len += 1
                    else:
                        break
                
                # Calculate Jaccard similarity
                set_a = set(tokens_a)
                set_b = set(tokens_b)
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = intersection / union if union > 0 else 0
                
                similarities.append({
                    'conv_a': conv_id_a,
                    'conv_b': conv_id_b,
                    'prefix_overlap': prefix_len,
                    'jaccard_similarity': jaccard,
                    'total_tokens_a': len(tokens_a),
                    'total_tokens_b': len(tokens_b)
                })
        
        return {
            'conversation_ids': conv_ids,
            'similarities': similarities
        }
    
    def finalize(self, client_stats: Dict[int, ClientStats], 
                 total_conversations: int = 0, total_turns: int = 0):
        """Finalize simulation with client statistics."""
        import statistics
        
        self.status = "completed"
        self.total_conversations = total_conversations
        self.total_turns = total_turns
        
        # Calculate active conversation statistics
        if self.active_conv_history:
            self.active_conv_stats = {
                'min': min(self.active_conv_history),
                'max': max(self.active_conv_history),
                'avg': statistics.mean(self.active_conv_history),
                'std': statistics.stdev(self.active_conv_history) 
                       if len(self.active_conv_history) > 1 else 0
            }
        else:
            self.active_conv_stats = {
                'min': 0, 'max': 0, 'avg': 0, 'std': 0
            }
        
        # Calculate conversation duration statistics
        if self.conversation_durations:
            self.conv_duration_stats = {
                'min': min(self.conversation_durations),
                'max': max(self.conversation_durations),
                'avg': statistics.mean(self.conversation_durations),
                'std': statistics.stdev(self.conversation_durations)
                       if len(self.conversation_durations) > 1 else 0,
                'count': len(self.conversation_durations)
            }
        else:
            self.conv_duration_stats = {
                'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'count': 0
            }
        
        # Calculate Jaccard similarity statistics from timeline
        if self.jaccard_timeline and len(self.jaccard_timeline) > 1:
            # Skip t=0 initial point
            jaccard_values = [point['avg_jaccard'] 
                              for point in self.jaccard_timeline[1:]]
            if jaccard_values:
                self.jaccard_stats = {
                    'min': min(jaccard_values),
                    'max': max(jaccard_values),
                    'avg': statistics.mean(jaccard_values),
                    'std': statistics.stdev(jaccard_values)
                           if len(jaccard_values) > 1 else 0,
                    'count': len(jaccard_values)
                }
            else:
                self.jaccard_stats = {
                    'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'count': 0
                }
        else:
            self.jaccard_stats = {
                'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'count': 0
            }
        
        for client_id, stats in client_stats.items():
            self.client_stats[client_id] = {
                'requests_sent': stats.requests_sent,
                'total_input_tokens': stats.total_input_tokens,
                'total_output_tokens': stats.total_output_tokens,
                'cached_tokens': stats.cached_tokens,
                'cache_hit_rate': (stats.cached_tokens / 
                                  stats.total_input_tokens * 100)
                    if stats.total_input_tokens > 0 else 0
            }
    
    def get_summary(self) -> dict:
        """Get simulation summary."""
        return {
            'sim_id': self.sim_id,
            'status': self.status,
            'duration': time.time() - self.start_time,
            'events': len(self.events),
            'total_conversations': getattr(self, 'total_conversations', 0),
            'total_turns': getattr(self, 'total_turns', 0),
            'active_conv_stats': getattr(self, 'active_conv_stats', {}),
            'conv_duration_stats': getattr(self, 'conv_duration_stats', {}),
            'jaccard_stats': getattr(self, 'jaccard_stats', {}),
            'client_stats': self.client_stats,
            'metrics': self.metrics
        }


class WebVisualizer:
    """Adapter to collect data from simulator for web display."""
    
    def __init__(
        self,
        collector: WebSimulationCollector,
        max_total_turns: int = 0,
        time_limit: int = 60,
        total_blocks: int = 500
    ):
        self.collector = collector
        self.cache_used = 0
        self.cache_total = max(total_blocks, 1)  # Ensure at least 1
        self.total_hits = 0
        self.total_tokens = 0
        self.active_convs_per_client: Dict[int, int] = {}
        self.max_total_turns = max_total_turns
        self.time_limit = time_limit
        self.total_turns_processed = 0
        self.should_stop = False
        self.start_time = time.time()
        # Track conversation data for web display
        self.conversation_data: Dict[str, Dict] = {}
        # Track per-client stats in real-time
        self.client_stats_realtime: Dict[int, Dict] = {}
    
    def update_cache_state(self, used_blocks: int):
        """Update cache state."""
        # Ensure cache_used is never negative
        self.cache_used = max(0, used_blocks)
    
    def update_request(
        self,
        client_id: int,
        cache_hit_count: int,
        total_tokens: int
    ):
        """Update request metrics."""
        self.total_hits += cache_hit_count
        self.total_tokens += total_tokens
        
        # Track per-client stats in real-time
        if client_id not in self.client_stats_realtime:
            self.client_stats_realtime[client_id] = {
                'requests': 0,
                'tokens': 0,
                'cached_tokens': 0,
                'cache_hit_rate': 0.0
            }
        
        self.client_stats_realtime[client_id]['requests'] += 1
        self.client_stats_realtime[client_id]['tokens'] += total_tokens
        self.client_stats_realtime[client_id]['cached_tokens'] += cache_hit_count
        
        # Calculate hit rate for this client
        if self.client_stats_realtime[client_id]['tokens'] > 0:
            self.client_stats_realtime[client_id]['cache_hit_rate'] = (
                self.client_stats_realtime[client_id]['cached_tokens'] / 
                self.client_stats_realtime[client_id]['tokens'] * 100
            )
        
        # Update collector's client_stats for real-time display
        self.collector.client_stats[client_id] = {
            'requests_sent': self.client_stats_realtime[client_id]['requests'],
            'cache_hit_rate': self.client_stats_realtime[client_id]['cache_hit_rate']
        }
        
        # Track turns
        self.total_turns_processed += 1
        
        # Check turn limit
        if (self.max_total_turns > 0 and 
            self.total_turns_processed >= self.max_total_turns):
            self.should_stop = True
            self.collector.add_event(
                'info',
                f'Reached max total turns limit '
                f'({self.max_total_turns}), stopping...'
            )
        
        # Check time limit (only if not infinite)
        if self.time_limit >= 0:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                self.should_stop = True
                self.collector.add_event(
                    'info',
                    f'Reached time limit ({self.time_limit}s), stopping...'
                )
        
        # Update metrics
        active_convs = sum(self.active_convs_per_client.values())
        self.collector.update_metrics(
            self.cache_used,
            self.cache_total,
            self.total_hits,
            self.total_tokens,
            active_convs
        )
    
    def update_active_conversations(self, client_id: int, active_count: int):
        """Update active conversation count."""
        self.active_convs_per_client[client_id] = active_count
    
    def add_event(self, message: str):
        """Add event."""
        self.collector.add_event('info', message)
    
    def record_conversation_tokens(self, conv_id: str, tokens: List[int]):
        """Record full token sequence for a conversation."""
        self.collector.record_conversation_tokens(conv_id, tokens)
        # Update Jaccard timeline after recording tokens
        self.collector.update_jaccard_timeline()
    
    def add_conversation_snippet(
        self,
        client_id: int,
        conv_id: str,
        turn: int,
        text: str,
        is_user: bool = True,
        cache_hit_count: int = 0,
        token_count: int = 0
    ):
        """Add conversation snippet for web display."""
        # Initialize conversation if not exists
        if conv_id not in self.conversation_data:
            self.conversation_data[conv_id] = {
                'client_id': client_id,
                'conv_id': conv_id,
                'turns': [],
                'cache_hits': 0,
                'total_tokens': 0
            }
        
        conv = self.conversation_data[conv_id]
        
        # Update cache stats (only for user messages since that's when
        # cache hits are calculated)
        if is_user and token_count > 0:
            conv['cache_hits'] += cache_hit_count
            conv['total_tokens'] += token_count
        
        # Ensure we have enough turn slots
        while len(conv['turns']) <= turn:
            conv['turns'].append({'user': '', 'assistant': ''})
        
        # Add the text to the appropriate role
        role = 'user' if is_user else 'assistant'
        conv['turns'][turn][role] = text
        
        # Update the collector with formatted turn data
        formatted_turns = []
        for t in conv['turns']:
            if t['user']:
                formatted_turns.append({
                    'role': 'user',
                    'content': t['user']
                })
            if t['assistant']:
                formatted_turns.append({
                    'role': 'assistant',
                    'content': t['assistant']
                })
        
        self.collector.add_conversation_snippet(
            conv_id=conv_id,
            client_id=client_id,
            turns=formatted_turns,
            cache_hits=conv['cache_hits'],
            total_tokens=conv['total_tokens']
        )
    
    def record_conversation_completion(self, conv_id: str):
        """Record when a conversation completes."""
        self.collector.record_conversation_completion(conv_id)
    
    def record_cache_hit(self, block_hash: str, tokens: List[int], 
                         conv_id: str):
        """Record a cache hit for analytics."""
        self.collector.record_cache_hit(block_hash, tokens, conv_id)
    
    def start(self):
        """Start (no-op for web)."""
        pass
    
    def stop(self):
        """Stop (no-op for web)."""
        pass
    
    def keep_alive(self):
        """Keep alive (no-op for web)."""
        pass
    
    def close(self):
        """Close (no-op for web)."""
        pass


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="""
    <html>
        <head><title>vLLM Simulator</title></head>
        <body>
            <h1>vLLM Prefix Caching Simulator</h1>
            <p>Static files not found. Run from web/ directory.</p>
        </body>
    </html>
    """)


@app.get("/api/info")
async def get_info():
    """Get simulator information."""
    return {
        "name": "vLLM Prefill Model Simulator",
        "version": "1.0.0",
        "text_mode_available": TEXT_MODE_AVAILABLE,
        "features": [
            "Multi-user simulation",
            "Real-time metrics",
            "ShareGPT integration" if TEXT_MODE_AVAILABLE else None,
            "Configurable cache parameters"
        ]
    }


@app.get("/api/debug")
async def debug_info():
    """Debug endpoint to check active simulations."""
    debug_data = {}
    for sim_id, sim_data in active_simulations.items():
        collector = sim_data['collector']
        convs = list(collector.conversations.values())
        debug_data[sim_id] = {
            'status': sim_data['status'],
            'num_events': len(collector.events),
            'num_conversations': len(collector.conversations),
            'conversation_ids': list(collector.conversations.keys())[:5],
            'sample_conversation': convs[0] if convs else None,
            'all_conversations': convs[:3]  # First 3 conversations
        }
    return debug_data


@app.get("/api/test-conversations")
async def test_conversations():
    """Test endpoint to manually get conversations."""
    if not active_simulations:
        return {"error": "No active simulations"}
    
    sim_id = list(active_simulations.keys())[0]
    collector = active_simulations[sim_id]['collector']
    
    return {
        'sim_id': sim_id,
        'total_conversations': len(collector.conversations),
        'conversations': list(collector.conversations.values())[:5]
    }


@app.post("/api/simulate")
async def start_simulation(config: SimulationConfig):
    """Start a new simulation."""
    sim_id = f"sim_{int(time.time() * 1000)}"
    
    # Create collector
    collector = WebSimulationCollector(sim_id)
    collector.add_event('info', 'Simulation starting...')
    
    # Store simulation
    active_simulations[sim_id] = {
        'config': config.dict(),  # Pydantic v1 compatibility
        'collector': collector,
        'status': 'starting'
    }
    
    # Start simulation in background
    try:
        task = asyncio.create_task(run_simulation_task(sim_id, config, collector))
        print(f"[INFO] Background task created for {sim_id}: {task}")
    except Exception as e:
        print(f"[ERROR] Failed to create background task for {sim_id}: {e}")
        import traceback
        traceback.print_exc()
        active_simulations[sim_id]['status'] = 'error'
        collector.add_event('error', f'Failed to start simulation: {str(e)}')
        raise
    
    return {
        'sim_id': sim_id,
        'status': 'started',
        'message': 'Simulation started in background'
    }


async def run_simulation_task(
    sim_id: str,
    config: SimulationConfig,
    collector: WebSimulationCollector
):
    """Run simulation in background task."""
    try:
        active_simulations[sim_id]['status'] = 'running'
        collector.add_event('info', 'Generating conversations...')
        
        # Create distributions
        if config.num_turns:
            num_turns_dist = ConstantDist(config.num_turns)
        else:
            num_turns_dist = UniformDist(4, 8)
        
        prefix_tokens_dist = LognormalDist(config.prefix_tokens_avg)
        user_tokens_dist = UniformDist(
            config.user_tokens_avg // 2,
            config.user_tokens_avg + config.user_tokens_avg // 2
        )
        assistant_tokens_dist = UniformDist(
            config.assistant_tokens_avg - 20,
            config.assistant_tokens_avg + 20
        )
        
        # Parse conversation template first (needed for synthetic generation)
        template_map = {
            'quick_qa': ConversationTemplate.QUICK_QA,
            'standard': ConversationTemplate.STANDARD_CHAT,
            'deep_dive': ConversationTemplate.DEEP_DIVE,
            'debug': ConversationTemplate.DEBUG_SESSION,
            'mixed': ConversationTemplate.MIXED
        }
        template = template_map.get(
            config.conversation_template,
            ConversationTemplate.STANDARD_CHAT
        )
        
        collector.add_event(
            'info',
            f'Using template: {template.value}'
        )
        
        # Generate conversations
        if config.text_mode and TEXT_MODE_AVAILABLE and config.dataset_path:
            collector.add_event('info', 
                              f'Loading dataset: {config.dataset_path}')
            dataset = text_mode.TextModeDataset(
                dataset_path=config.dataset_path,
                tokenizer=TOKENIZER  # Use pre-loaded tokenizer
            )
            # Load large pool of conversations (controlled by time/turns limits)
            dataset.load_sharegpt_dataset(
                max_conversations=2000,
                min_turns=2,
                max_turns=20
            )
            conversations = dataset.get_text_conversations(
                num_conversations=1000,
                common_prefix=config.common_prefix_text
            )
        else:
            # Use text mode if tokenizer is available
            if TOKENIZER:
                collector.add_event('info', 'Generating text conversations with GPT-2 tokenization')
                from text_mode import TextModeDataset
                
                # Pass the pre-loaded tokenizer
                dataset = TextModeDataset(tokenizer=TOKENIZER)
                # Generate synthetic text conversations (no dataset path = synthetic)
                dataset.load_sharegpt_dataset(max_conversations=1000)
                conversations = dataset.get_text_conversations(
                    num_conversations=1000,
                    common_prefix="System: You are a helpful assistant."
                )
            else:
                collector.add_event('info', 'Generating synthetic conversations')
                
                # Generate large pool of conversations (controlled by time/turns limits)
                num_conversations = 1000  # Large pool
                conversations = []
                for i in range(num_conversations):
                    conv = generate_conversation_from_template(
                        conv_id=f"conv_{i:04d}",
                        template=template,
                        common_prefix_tokens=config.common_prefix_tokens,
                        client_id=0,
                        conv_counter=i
                    )
                    conversations.append(conv)
        
        collector.add_event('info', 
                          f'Generated {len(conversations)} conversations')
        
        # Create visualizer adapter
        visualizer = WebVisualizer(
            collector,
            max_total_turns=config.max_total_turns,
            time_limit=config.time_limit,
            total_blocks=config.total_blocks
        )
        
        # Log limits
        if config.max_total_turns > 0:
            collector.add_event(
                'info',
                f'Max total turns limit: {config.max_total_turns}'
            )
        if config.time_limit >= 0:
            collector.add_event(
                'info',
                f'Time limit: {config.time_limit} seconds'
            )
        elif config.time_limit == -1:
            collector.add_event(
                'info',
                'Time limit: infinite (will run until stopped)'
            )
        
        # Parse sampling strategy
        sampling = ConversationSampling.ROUND_ROBIN
        if config.sampling_strategy == "random":
            sampling = ConversationSampling.RANDOM
        
        collector.add_event('info', 'Starting simulation...')
        
        # Prepare conversation generator parameters for continuous mode
        # Using template-based generation
        generator_params = {
            'template': template,
            'common_prefix_tokens': config.common_prefix_tokens
        }
        
        # Run simulation
        client_stats = await run_multi_user_simulation(
            num_clients=config.num_clients,
            conversations=conversations,
            block_size=config.block_size,
            total_blocks=config.total_blocks,
            sampling_strategy=sampling,
            request_rate=config.request_rate,
            max_active_conversations=config.max_active_conversations,
            visualizer=visualizer,
            verbose=False,
            conversation_generator_params=generator_params
        )
        
        # Finalize - calculate totals
        total_conversations = len(collector.conv_effectiveness)  # Actual unique convs processed
        total_turns = sum(stats.requests_sent for stats in client_stats.values())
        
        collector.finalize(client_stats, total_conversations, total_turns)
        collector.add_event('success', 'Simulation completed successfully')
        active_simulations[sim_id]['status'] = 'completed'
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        collector.status = 'error'
        collector.add_event('error', f'Simulation failed: {str(e)}')
        collector.add_event('error', f'Traceback: {error_details}')
        active_simulations[sim_id]['status'] = 'error'
        print(f"ERROR in simulation {sim_id}:")
        print(error_details)


@app.get("/api/simulations")
async def list_simulations():
    """List all simulations."""
    return {
        'simulations': [
            {
                'sim_id': sim_id,
                'status': data['status'],
                'config': data['config']
            }
            for sim_id, data in active_simulations.items()
        ]
    }


@app.get("/api/simulation/{sim_id}")
async def get_simulation(sim_id: str):
    """Get simulation details."""
    if sim_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim_data = active_simulations[sim_id]
    collector = sim_data['collector']
    
    return {
        'sim_id': sim_id,
        'status': sim_data['status'],
        'config': sim_data['config'],
        'summary': collector.get_summary(),
        'num_conversations': len(collector.conversations),
        'conversation_ids': list(collector.conversations.keys())[:10]
    }


@app.websocket("/ws/{sim_id}")
async def websocket_endpoint(websocket: WebSocket, sim_id: str):
    """WebSocket for real-time simulation updates."""
    await websocket.accept()
    
    if sim_id not in active_simulations:
        await websocket.send_json({
            'type': 'error',
            'message': 'Simulation not found'
        })
        await websocket.close()
        return
    
    try:
        collector = active_simulations[sim_id]['collector']
        last_event_idx = 0
        sent_conversations = {}  # Track last sent timestamp for each conv
        
        while True:
            # Send new events
            if last_event_idx < len(collector.events):
                new_events = collector.events[last_event_idx:]
                await websocket.send_json({
                    'type': 'events',
                    'data': new_events
                })
                last_event_idx = len(collector.events)
            
            # Send new/updated conversations
            for conv_id, conv_data in collector.conversations.items():
                last_updated = conv_data['last_updated']
                last_sent = sent_conversations.get(conv_id, 0)
                
                # Send if: never sent before OR updated since last sent
                should_send = (conv_id not in sent_conversations or 
                               last_updated > last_sent)
                
                if should_send:
                    await websocket.send_json({
                        'type': 'conversation',
                        'data': conv_data
                    })
                    sent_conversations[conv_id] = time.time()
            
            # Send current metrics
            await websocket.send_json({
                'type': 'metrics',
                'data': collector.metrics
            })
            
            # Send analytics data (every update)
            await websocket.send_json({
                'type': 'analytics',
                'data': collector.get_analytics_summary()
            })
            
            # Send cache state log (last 20 entries)
            if collector.cache_state_log:
                await websocket.send_json({
                    'type': 'cache_log',
                    'data': collector.cache_state_log[-20:]
                })
            
            # Send status
            await websocket.send_json({
                'type': 'status',
                'data': {
                    'status': collector.status,
                    'client_stats': collector.client_stats
                }
            })
            
            # Check if completed
            if collector.status in ['completed', 'error']:
                await websocket.send_json({
                    'type': 'complete',
                    'data': collector.get_summary()
                })
                break
            
            await asyncio.sleep(0.5)  # Update every 500ms
            
    except WebSocketDisconnect:
        pass


@app.delete("/api/simulation/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation."""
    if sim_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del active_simulations[sim_id]
    return {'message': f'Simulation {sim_id} deleted'}


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), 
             name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting vLLM Simulator Web UI...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)

