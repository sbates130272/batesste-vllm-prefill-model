"""
Real-time visualization for multi-user vLLM simulator.

Uses matplotlib for live plots showing cache state, hit rates,
and client activity.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
from typing import Dict, List, Optional
import threading
import time


class SimulationVisualizer:
    """
    Real-time visualization dashboard for the multi-user simulator.
    """
    
    def __init__(
        self,
        num_clients: int,
        total_blocks: int,
        update_interval: int = 100
    ):
        """
        Initialize the visualizer.
        
        Args:
            num_clients: Number of clients in the simulation
            total_blocks: Total cache blocks available
            update_interval: Update interval in milliseconds
        """
        self.num_clients = num_clients
        self.total_blocks = total_blocks
        self.update_interval = update_interval
        
        # Data storage (thread-safe with lock)
        self.lock = threading.Lock()
        self.cache_timestamps = deque(maxlen=100)
        self.cache_used = deque(maxlen=100)
        self.hitrate_timestamps = deque(maxlen=100)
        self.cache_hit_rates = deque(maxlen=100)
        self.client_requests = {i: 0 for i in range(num_clients)}
        self.client_cache_hits = {i: 0 for i in range(num_clients)}
        self.client_input_tokens = {i: 0 for i in range(num_clients)}
        self.client_active_convs = {i: 0 for i in range(num_clients)}
        self.total_requests = 0
        self.total_cache_hits = 0
        self.total_input_tokens = 0
        self.start_time = time.time()
        
        # Event log for timeline
        self.events = deque(maxlen=50)
        
        # Setup matplotlib figure
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(
            'vLLM Multi-User Simulator - Real-Time Dashboard',
            fontsize=16,
            fontweight='bold'
        )
        
        # Create grid layout
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # Subplot 1: Cache Occupancy
        self.ax_cache = self.fig.add_subplot(gs[0, 0])
        self.ax_cache.set_title('Cache Occupancy', fontweight='bold')
        self.ax_cache.set_xlabel('Time (s)')
        self.ax_cache.set_ylabel('Blocks')
        self.ax_cache.set_ylim(0, total_blocks * 1.1)
        
        # Subplot 2: Cache Hit Rate
        self.ax_hitrate = self.fig.add_subplot(gs[0, 1])
        self.ax_hitrate.set_title('Cache Hit Rate', fontweight='bold')
        self.ax_hitrate.set_xlabel('Time (s)')
        self.ax_hitrate.set_ylabel('Hit Rate (%)')
        self.ax_hitrate.set_ylim(0, 100)
        
        # Subplot 3: Per-Client Requests
        self.ax_requests = self.fig.add_subplot(gs[1, 0])
        self.ax_requests.set_title('Requests per Client', fontweight='bold')
        self.ax_requests.set_xlabel('Client ID')
        self.ax_requests.set_ylabel('Total Requests')
        
        # Subplot 4: Active Conversations
        self.ax_convs = self.fig.add_subplot(gs[1, 1])
        self.ax_convs.set_title('Active Conversations', fontweight='bold')
        self.ax_convs.set_xlabel('Client ID')
        self.ax_convs.set_ylabel('Active Conversations')
        
        # Subplot 5: Event Timeline (bottom, spans both columns)
        self.ax_events = self.fig.add_subplot(gs[2, :])
        self.ax_events.set_title('Recent Events', fontweight='bold')
        self.ax_events.axis('off')
        
        # Animation
        self.ani = None
        self.running = True
        
    def update_cache_state(self, used_blocks: int):
        """Update cache occupancy data."""
        with self.lock:
            elapsed = time.time() - self.start_time
            self.cache_timestamps.append(elapsed)
            self.cache_used.append(used_blocks)
    
    def update_request(
        self,
        client_id: int,
        cached_tokens: int,
        total_tokens: int
    ):
        """
        Update request statistics.
        
        Args:
            client_id: ID of the client making the request
            cached_tokens: Number of tokens that were cached
            total_tokens: Total tokens in the request
        """
        with self.lock:
            self.client_requests[client_id] += 1
            self.total_requests += 1
            self.client_input_tokens[client_id] += total_tokens
            self.total_input_tokens += total_tokens
            
            if cached_tokens > 0:
                self.client_cache_hits[client_id] += cached_tokens
                self.total_cache_hits += cached_tokens
            
            # Update hit rate - calculate based on all processed tokens
            if self.total_input_tokens > 0:
                elapsed = time.time() - self.start_time
                hit_rate = (self.total_cache_hits / 
                           self.total_input_tokens) * 100
                hit_rate = min(100, max(0, hit_rate))  # Clamp to 0-100
                self.hitrate_timestamps.append(elapsed)
                self.cache_hit_rates.append(hit_rate)
    
    def update_active_conversations(
        self,
        client_id: int,
        active_count: int
    ):
        """Update active conversation count for a client."""
        with self.lock:
            self.client_active_convs[client_id] = active_count
    
    def add_event(self, event: str):
        """Add an event to the timeline."""
        with self.lock:
            elapsed = time.time() - self.start_time
            self.events.append(f"[{elapsed:6.2f}s] {event}")
    
    def _update_plot(self, frame):
        """Internal method to update all plots (called by animation)."""
        with self.lock:
            # Clear all axes
            self.ax_cache.clear()
            self.ax_hitrate.clear()
            self.ax_requests.clear()
            self.ax_convs.clear()
            self.ax_events.clear()
            
            # Plot 1: Cache Occupancy
            self.ax_cache.set_title('Cache Occupancy', fontweight='bold')
            self.ax_cache.set_xlabel('Time (s)')
            self.ax_cache.set_ylabel('Blocks')
            self.ax_cache.set_ylim(0, self.total_blocks * 1.1)
            if len(self.cache_timestamps) > 0 and len(self.cache_used) > 0:
                times = list(self.cache_timestamps)
                used = list(self.cache_used)
                self.ax_cache.plot(
                    times, used, 'b-', linewidth=2, label='Used'
                )
                self.ax_cache.axhline(
                    y=self.total_blocks,
                    color='r',
                    linestyle='--',
                    label='Capacity'
                )
                self.ax_cache.fill_between(
                    times, 0, used, alpha=0.3, color='blue'
                )
                self.ax_cache.legend(loc='upper left')
                self.ax_cache.grid(True, alpha=0.3)
            
            # Plot 2: Cache Hit Rate
            self.ax_hitrate.set_title('Cache Hit Rate', fontweight='bold')
            self.ax_hitrate.set_xlabel('Time (s)')
            self.ax_hitrate.set_ylabel('Hit Rate (%)')
            self.ax_hitrate.set_ylim(0, 100)
            if len(self.hitrate_timestamps) > 0 and \
               len(self.cache_hit_rates) > 0:
                times = list(self.hitrate_timestamps)
                rates = list(self.cache_hit_rates)
                self.ax_hitrate.plot(
                    times, rates, 'g-', linewidth=2, marker='o'
                )
                self.ax_hitrate.fill_between(
                    times, 0, rates, alpha=0.3, color='green'
                )
                self.ax_hitrate.grid(True, alpha=0.3)
                # Add current hit rate text
                if rates:
                    current_rate = rates[-1]
                    self.ax_hitrate.text(
                        0.02, 0.98,
                        f'Current: {current_rate:.1f}%',
                        transform=self.ax_hitrate.transAxes,
                        verticalalignment='top',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='wheat',
                            alpha=0.8
                        )
                    )
            
            # Plot 3: Per-Client Requests
            self.ax_requests.set_title('Requests per Client', 
                                      fontweight='bold')
            self.ax_requests.set_xlabel('Client ID')
            self.ax_requests.set_ylabel('Total Requests')
            client_ids = list(self.client_requests.keys())
            request_counts = [self.client_requests[i] 
                            for i in client_ids]
            if client_ids:
                bars = self.ax_requests.bar(
                    client_ids,
                    request_counts,
                    color='steelblue',
                    alpha=0.7
                )
                # Add value labels on bars
                for bar, count in zip(bars, request_counts):
                    height = bar.get_height()
                    if height > 0:
                        self.ax_requests.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height,
                            f'{int(count)}',
                            ha='center',
                            va='bottom'
                        )
                self.ax_requests.set_xticks(client_ids)
                self.ax_requests.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Active Conversations
            self.ax_convs.set_title('Active Conversations', 
                                   fontweight='bold')
            self.ax_convs.set_xlabel('Client ID')
            self.ax_convs.set_ylabel('Active Conversations')
            conv_counts = [self.client_active_convs[i] 
                          for i in client_ids]
            if client_ids:
                bars = self.ax_convs.bar(
                    client_ids,
                    conv_counts,
                    color='coral',
                    alpha=0.7
                )
                # Add value labels on bars
                for bar, count in zip(bars, conv_counts):
                    height = bar.get_height()
                    if height > 0:
                        self.ax_convs.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height,
                            f'{int(count)}',
                            ha='center',
                            va='bottom'
                        )
                self.ax_convs.set_xticks(client_ids)
                self.ax_convs.grid(True, alpha=0.3, axis='y')
            
            # Plot 5: Event Timeline
            self.ax_events.set_title('Recent Events', fontweight='bold')
            self.ax_events.axis('off')
            if self.events:
                # Show last 10 events
                recent_events = list(self.events)[-10:]
                event_text = '\n'.join(recent_events)
                self.ax_events.text(
                    0.02, 0.98,
                    event_text,
                    transform=self.ax_events.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=8,
                    bbox=dict(
                        boxstyle='round',
                        facecolor='lightgray',
                        alpha=0.5
                    )
                )
    
    def start(self):
        """Start the visualization animation."""
        # Make sure interactive mode is on
        plt.ion()
        
        # Show the figure first
        plt.show(block=False)
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,
            repeat=True
        )
        
        # Force initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Give it time to appear
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
    
    def keep_alive(self):
        """Keep the visualization window alive and process events."""
        try:
            # Process pending GUI events to allow updates
            self.fig.canvas.flush_events()
            # Also trigger a redraw if needed
            self.fig.canvas.draw_idle()
        except:
            pass
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)


class NullVisualizer:
    """
    Null object pattern for when visualization is disabled.
    All methods are no-ops.
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def update_cache_state(self, *args, **kwargs):
        pass
    
    def update_request(self, *args, **kwargs):
        pass
    
    def update_active_conversations(self, *args, **kwargs):
        pass
    
    def add_event(self, *args, **kwargs):
        pass
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def keep_alive(self):
        pass
    
    def close(self):
        pass


