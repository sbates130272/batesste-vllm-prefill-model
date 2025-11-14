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
    num_conversations: int = 100
    num_turns: Optional[int] = None
    block_size: int = 16
    total_blocks: int = 500
    common_prefix_tokens: int = 100
    prefix_tokens_avg: int = 200
    user_tokens_avg: int = 50
    assistant_tokens_avg: int = 100
    request_rate: float = 0.0
    max_active_conversations: int = 3
    max_total_turns: int = 0  # 0 = unlimited
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
    
    def add_conversation_snippet(
        self,
        conv_id: str,
        client_id: int,
        turns: List[Dict],
        cache_hits: int = 0,
        total_tokens: int = 0
    ):
        """Add or update a conversation snippet."""
        self.conversations[conv_id] = {
            'conv_id': conv_id,
            'client_id': client_id,
            'turns': turns,
            'num_turns': len(turns),
            'cache_hits': cache_hits,
            'total_tokens': total_tokens,
            'last_updated': time.time()
        }
    
    def finalize(self, client_stats: Dict[int, ClientStats]):
        """Finalize simulation with client statistics."""
        self.status = "completed"
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
    
    def add_conversation_snippet(
        self,
        client_id: int,
        conv_id: str,
        turn: int,
        text: str,
        is_user: bool = True
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
        'config': config.model_dump(),
        'collector': collector,
        'status': 'starting'
    }
    
    # Start simulation in background
    asyncio.create_task(run_simulation_task(sim_id, config, collector))
    
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
                tokenizer_name=config.tokenizer
            )
            dataset.load_sharegpt_dataset(
                max_conversations=config.num_conversations * 2,
                min_turns=2,
                max_turns=20
            )
            conversations = dataset.get_text_conversations(
                num_conversations=config.num_conversations,
                common_prefix=config.common_prefix_text
            )
        else:
            collector.add_event('info', 'Generating synthetic conversations')
            
            # Generate initial conversations using template
            conversations = []
            for i in range(config.num_conversations):
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
        
        # Finalize
        collector.finalize(client_stats)
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

