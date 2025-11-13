# Real-Time Visualization Guide

The multi-user simulator includes a real-time visualization dashboard that
displays cache state, hit rates, and client activity as the simulation runs.

## Requirements

```bash
pip install matplotlib numpy
```

## Usage

Simply add the `--visualize` flag to any simulation command:

```bash
# Basic visualization
python3 src/multi_user_simulator.py --visualize

# With custom parameters
python3 src/multi_user_simulator.py --visualize \
  --num-clients 5 \
  --num-conversations 50 \
  --request-rate 1.0

# Slow down the simulation to see real-time updates
python3 src/multi_user_simulator.py --visualize \
  --num-clients 3 \
  --num-conversations 20 \
  --request-rate 2.0  # 2 requests/sec per client (slower)
```

## Dashboard Layout

The visualization shows 5 real-time plots:

```
┌─────────────────────────────────────────────────────────────┐
│  vLLM Multi-User Simulator - Real-Time Dashboard           │
├──────────────────────────────┬──────────────────────────────┤
│  1. Cache Occupancy          │  2. Cache Hit Rate           │
│  (Blocks used over time)     │  (% cached over time)        │
├──────────────────────────────┼──────────────────────────────┤
│  3. Requests per Client      │  4. Active Conversations     │
│  (Total requests bar chart)  │  (Per-client bar chart)      │
├──────────────────────────────┴──────────────────────────────┤
│  5. Recent Events Timeline                                  │
│  (Last 10 events with timestamps)                           │
└─────────────────────────────────────────────────────────────┘
```

## Plot Details

### 1. Cache Occupancy
- **X-axis**: Time (seconds)
- **Y-axis**: Number of KV cache blocks
- **Blue line**: Blocks currently in use
- **Red dashed line**: Total capacity
- **Blue fill**: Visual representation of utilization

**What to look for:**
- Spikes indicate many requests being processed
- High sustained usage may indicate need for more memory
- Oscillation shows natural request/free cycles

### 2. Cache Hit Rate
- **X-axis**: Time (seconds)
- **Y-axis**: Hit rate percentage (0-100%)
- **Green line**: Cache hit rate over time
- **Text box**: Current hit rate value

**What to look for:**
- Higher is better (more tokens reused from cache)
- Starts at 0% (cold cache), should increase as prefixes are reused
- Steady-state hit rate indicates caching effectiveness
- Low hit rate (<10%) may indicate insufficient shared prefixes

### 3. Requests per Client
- **X-axis**: Client ID
- **Y-axis**: Total requests completed
- **Blue bars**: Request count per client

**What to look for:**
- Balanced bars indicate even workload distribution
- Uneven bars expected if conversations differ in length
- Growing bars show active progress

### 4. Active Conversations
- **X-axis**: Client ID
- **Y-axis**: Currently active conversations
- **Coral bars**: Active conversation count per client

**What to look for:**
- Shows concurrency level per client
- Decreases as conversations complete
- Bounded by `--max-active-conversations` parameter

### 5. Recent Events Timeline
- Shows last 10 events with timestamps
- Format: `[time] Client X conv_ID Turn N: hit_rate% cached`

**Example:**
```
[  0.02s] Client 0 started with 10 convs
[  0.15s] C0 conv_0001 T0: 0% cached
[  0.18s] C0 conv_0002 T0: 35% cached
[  0.22s] C1 conv_0003 T0: 35% cached
```

## Interpreting the Visualization

### Healthy Cache Behavior

```
Cache Occupancy:     Cache Hit Rate:
    50 |                 100% |
       |    /\                |      ___________
    25 | __/  \__          50%|     /
       |          \            |    /
     0 |___________         0% |___/___________
       0s     5s    10s        0s     5s    10s
```

- Cache fills initially, then stabilizes
- Hit rate starts low, increases as cache warms up
- Sustained high hit rate (>30%) for multi-turn conversations

### Cache Pressure

```
Cache Occupancy (PRESSURE!):
   100 |████████████████████
       |█FULL CAPACITY!█████
    50 |████████████████████
       |████████████████████
     0 |
       0s     5s    10s
```

- Cache at or near capacity
- Consider increasing `--total-blocks`
- May see cache eviction in verbose output

### Poor Prefix Sharing

```
Cache Hit Rate (POOR!):
   100% |
        |
    50% |
        |
     0% |____________________
        0s     5s    10s
```

- Hit rate stays near 0%
- Indicates little to no prefix sharing
- Consider: more `--common-prefix-tokens` or fewer unique conversations

## WSL and Remote Systems

### Option 1: X11 Forwarding (Recommended)

```bash
# On Windows with VcXsrv or Xming running:
export DISPLAY=:0

# Run simulation with visualization
python3 src/multi_user_simulator.py --visualize
```

### Option 2: Save Plot to File

Modify the code to save plots periodically instead of displaying:

```python
# In visualizer.py, add a save method:
def save_plot(self, filename):
    self.fig.savefig(filename, dpi=150, bbox_inches='tight')

# In multi_user_simulator.py, periodically save:
if visualizer and iteration % 10 == 0:
    visualizer.save_plot(f'simulation_{iteration}.png')
```

### Option 3: View on Remote Machine

```bash
# Enable X11 forwarding when SSH'ing
ssh -X user@remote

# Run simulation
python3 src/multi_user_simulator.py --visualize
```

## Performance Notes

- **Update Interval**: Default is 100ms (10 Hz)
- **Data History**: Last 100 time points kept
- **CPU Impact**: Minimal (~1-2% on modern CPUs)
- **Memory**: ~50MB for matplotlib overhead

To reduce overhead, increase update interval in `main()`:

```python
visualizer = SimulationVisualizer(
    num_clients=args.num_clients,
    total_blocks=args.total_blocks,
    update_interval=500  # Update every 500ms instead of 100ms
)
```

## Troubleshooting

### "matplotlib not available"

```bash
pip install matplotlib numpy
```

### "Cannot connect to X server"

In WSL/remote environments:
```bash
export DISPLAY=:0  # Or appropriate display
```

Or use non-interactive backend:
```bash
export MPLBACKEND=Agg
```

### Visualization window not updating

- Check if simulation is running (verbose output)
- Try increasing `--request-rate` to slow down simulation
- Ensure matplotlib backend supports animation

### Simulation runs but no window appears

For WSL:
1. Install VcXsrv or Xming on Windows
2. Set `DISPLAY` environment variable
3. Allow X11 connections in firewall

## Examples

### Watch Cache Fill Up

```bash
python3 src/multi_user_simulator.py --visualize \
  --num-clients 5 \
  --num-conversations 20 \
  --total-blocks 100 \
  --request-rate 1.0  # Slow enough to watch
```

### Observe Cache Pressure

```bash
# Intentionally small cache
python3 src/multi_user_simulator.py --visualize \
  --num-clients 10 \
  --num-conversations 50 \
  --total-blocks 50  # Small cache!
  --max-active-conversations 5
```

### See High Cache Hit Rates

```bash
# Lots of shared prefixes
python3 src/multi_user_simulator.py --visualize \
  --num-clients 5 \
  --num-conversations 30 \
  --common-prefix-tokens 200  # Large shared prefix
  --prefix-tokens-avg 50  # Small unique prefixes
```

## Tips for Best Visualization

1. **Use request rate limiting** (`--request-rate 1.0` or higher)
   - Slows simulation so you can see updates
   - Without it, simulation may finish too quickly

2. **Start with fewer clients** (2-5)
   - Easier to follow individual client progress
   - Less visual clutter

3. **Enable verbose output** in separate terminal
   - See detailed cache operations
   - Correlate with visualization events

4. **Keep window open after completion**
   - Final state remains visible
   - Press Ctrl+C or close window to exit

## Integration with Your Workflow

### Screenshot for Reports

After simulation completes, window remains open:
- Take screenshot of final state
- Use for presentations or documentation
- Shows cache efficiency metrics

### Debugging Cache Issues

Run with visualization + verbose:
```bash
python3 src/multi_user_simulator.py --visualize --verbose \
  --num-clients 2 --num-conversations 4 | tee simulation.log
```

Watch visualization while reviewing verbose output to understand behavior.

### A/B Testing Configurations

Compare different cache sizes:

```bash
# Configuration A: Small cache
python3 src/multi_user_simulator.py --visualize --total-blocks 50

# Configuration B: Large cache
python3 src/multi_user_simulator.py --visualize --total-blocks 200
```

Observe hit rate differences.

## Advanced: Custom Visualization

The visualizer is modular and can be extended. See `src/visualizer.py`:

```python
class SimulationVisualizer:
    def add_custom_plot(self, ax, data):
        """Add your own custom plot"""
        ax.plot(data['x'], data['y'])
    
    def update_custom_metric(self, value):
        """Track custom metrics"""
        self.custom_data.append(value)
```

## Next Steps

- Try running simulations with different parameters
- Observe how cache hit rate changes with workload
- Use visualization to find optimal cache configuration
- Compare to vLLM's production behavior

