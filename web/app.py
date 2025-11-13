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
    ClientStats
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
    num_conversations: int = 20
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
    sampling_strategy: str = "round_robin"
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
        self.metrics['cache_occupancy'].append(cache_used / cache_total * 100)
        
        hit_rate = (hit_count / total_tokens * 100) if total_tokens > 0 else 0
        self.metrics['cache_hit_rate'].append(hit_rate)
        self.metrics['active_conversations'].append(active_convs)
    
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
        max_total_turns: int = 0
    ):
        self.collector = collector
        self.cache_used = 0
        self.cache_total = 1
        self.total_hits = 0
        self.total_tokens = 0
        self.active_convs_per_client: Dict[int, int] = {}
        self.max_total_turns = max_total_turns
        self.total_turns_processed = 0
        self.should_stop = False
    
    def update_cache_state(self, used_blocks: int):
        """Update cache state."""
        self.cache_used = used_blocks
    
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
        
        # Check if we should stop
        if (self.max_total_turns > 0 and 
            self.total_turns_processed >= self.max_total_turns):
            self.should_stop = True
            self.collector.add_event(
                'info',
                f'Reached max total turns limit '
                f'({self.max_total_turns}), stopping...'
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
    
    def add_conversation_snippet(self, *args, **kwargs):
        """Add conversation snippet (not needed for web)."""
        pass
    
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


@app.post("/api/simulate")
async def start_simulation(config: SimulationConfig):
    """Start a new simulation."""
    sim_id = f"sim_{int(time.time() * 1000)}"
    
    # Create collector
    collector = WebSimulationCollector(sim_id)
    collector.add_event('info', 'Simulation starting...')
    
    # Store simulation
    active_simulations[sim_id] = {
        'config': config.dict(),
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
            conversations = generate_synthetic_conversations(
                num_conversations=config.num_conversations,
                num_turns_dist=num_turns_dist,
                prefix_tokens_dist=prefix_tokens_dist,
                user_tokens_dist=user_tokens_dist,
                assistant_tokens_dist=assistant_tokens_dist,
                common_prefix_tokens=config.common_prefix_tokens
            )
        
        collector.add_event('info', 
                          f'Generated {len(conversations)} conversations')
        
        # Create visualizer adapter
        visualizer = WebVisualizer(
            collector,
            max_total_turns=config.max_total_turns
        )
        visualizer.cache_total = config.total_blocks
        
        if config.max_total_turns > 0:
            collector.add_event(
                'info',
                f'Max total turns limit: {config.max_total_turns}'
            )
        
        # Parse sampling strategy
        sampling = ConversationSampling.ROUND_ROBIN
        if config.sampling_strategy == "random":
            sampling = ConversationSampling.RANDOM
        
        collector.add_event('info', 'Starting simulation...')
        
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
            verbose=False
        )
        
        # Finalize
        collector.finalize(client_stats)
        collector.add_event('success', 'Simulation completed successfully')
        active_simulations[sim_id]['status'] = 'completed'
        
    except Exception as e:
        collector.status = 'error'
        collector.add_event('error', f'Simulation failed: {str(e)}')
        active_simulations[sim_id]['status'] = 'error'


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
        'summary': collector.get_summary()
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
        
        while True:
            # Send new events
            if last_event_idx < len(collector.events):
                new_events = collector.events[last_event_idx:]
                await websocket.send_json({
                    'type': 'events',
                    'data': new_events
                })
                last_event_idx = len(collector.events)
            
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

