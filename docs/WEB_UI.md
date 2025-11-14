# Web UI Guide

The vLLM Prefix Caching Simulator includes a modern, interactive web 
interface for running simulations and visualizing results in real-time.

## Overview

The Web UI provides:
- **Interactive Configuration** - Adjust all simulation parameters via web form
- **Real-Time Visualization** - Live charts showing cache metrics
- **Event Logging** - Stream of simulation events as they occur
- **Multiple Simulations** - Run and track multiple simulations
- **No Installation** - Runs in your web browser

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r web/requirements.txt
```

Required packages:
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `websockets` - Real-time updates
- `pydantic` - Data validation

### 2. Start the Web Server

```bash
# From project root
python3 web/app.py

# Or from web directory
cd web
./app.py
```

Expected output:
```
Starting vLLM Simulator Web UI...
Open http://localhost:8000 in your browser
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

## Using the Web UI

### Configuration Panel

Configure your simulation with these parameters:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Number of Clients** | Concurrent users | 3 | 1-20 |
| **Conversations** | Initial conversations | 20 | 1-1000 |
| **Block Size** | Tokens per block | 16 | 1-128 |
| **Total Blocks** | Cache capacity | 500 | 10-100000 |
| **Request Rate** | Requests per second | 1.0 | 0-100 |
| **Max Active Convs** | Concurrent per client | 3 | 1-50 |
| **Max Total Turns** | Turn limit (0=unlimited) | 0 | 0-10000 |
| **Time Limit** | Duration in seconds (-1=infinite) | 60 | -1-3600 |

### Simulation Duration Controls

The simulator supports **continuous generation mode** where new 
conversations are automatically created as old ones complete. This allows 
for long-running tests and realistic workload simulation.

#### Max Total Turns
- **0** (default): No turn limit - simulation continues until time limit
- **>0**: Stop after processing this many turns across all conversations
- Useful for benchmarking with consistent workload size

#### Time Limit
- **-1**: Infinite - simulation runs continuously, generating new 
  conversations indefinitely (requires manual stop)
- **0 or more**: Stop after this many seconds
- Default: **60 seconds** for quick tests

**Examples:**
- **Quick test**: `Max Total Turns = 100`, `Time Limit = 60` (stops at 
  first limit reached)
- **Long run**: `Max Total Turns = 0`, `Time Limit = 300` (runs for 5 
  minutes)
- **Infinite**: `Max Total Turns = 0`, `Time Limit = -1` (runs until 
  manually stopped)

### Starting a Simulation

1. **Adjust parameters** in the configuration panel
2. **Set duration limits** (turn limit and/or time limit)
3. **Click "▶️ Start Simulation"**
4. **Watch real-time updates** in charts and event log
5. **Wait for completion** or click "⏹️ Stop"

### Real-Time Charts

The UI displays four live-updating charts:

#### 1. Cache Hit Rate (%)
- Shows percentage of tokens served from cache
- Green line indicates efficiency
- Higher is better (closer to 100%)
- Updates every 500ms

#### 2. Cache Occupancy (%)
- Shows how full the cache is
- Purple/blue line
- Helps identify if cache is undersized
- 100% means cache is full (may cause evictions)

#### 3. Active Conversations
- Number of concurrent active conversations
- Yellow/orange line
- Shows workload intensity over time
- Useful for understanding concurrency patterns

#### 4. Per-Client Statistics
- Bar chart showing cache hit rate per client
- Helps identify if clients have different patterns
- Useful for multi-tenant analysis

### Events Log

Real-time stream of simulation events:

```
[0.00s] info: Simulation starting...
[0.12s] info: Generating synthetic conversations
[0.25s] info: Generated 20 conversations
[0.27s] info: Starting simulation...
[2.45s] success: Simulation completed successfully
```

Event types:
- **info** - General information (blue)
- **success** - Successful completion (green)
- **error** - Errors or failures (red)

### Status Panel

Shows current simulation state:

**Running:**
```
▶️ Simulation running...
```

**Completed:**
```
✅ Simulation completed successfully

Total Requests: 120
Total Input Tokens: 15,840
Cached Tokens: 12,672
Overall Hit Rate: 80.0%
```

## Advanced Usage

### Custom Port

Run on a different port:

```bash
cd web
python3 -c "import uvicorn; from app import app; \
    uvicorn.run(app, host='0.0.0.0', port=8080)"
```

### Remote Access

To allow access from other machines:

```bash
# The server already binds to 0.0.0.0 by default
python3 web/app.py

# Access from other machine:
# http://YOUR_IP:8000
```

**Security Note**: In production, use HTTPS and authentication.

### Multiple Simulations

The Web UI supports running multiple simulations:

1. Each simulation gets a unique ID
2. WebSocket connects to specific simulation
3. Previous simulation data is retained
4. View past simulations via API

### API Endpoints

The Web UI exposes a REST API:

#### GET `/api/info`
Get simulator information:
```bash
curl http://localhost:8000/api/info
```

Response:
```json
{
  "name": "vLLM Prefill Model Simulator",
  "version": "1.0.0",
  "text_mode_available": true,
  "features": [
    "Multi-user simulation",
    "Real-time metrics",
    "ShareGPT integration",
    "Configurable cache parameters"
  ]
}
```

#### POST `/api/simulate`
Start a simulation:
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "num_clients": 5,
    "num_conversations": 50,
    "block_size": 16,
    "total_blocks": 1000
  }'
```

Response:
```json
{
  "sim_id": "sim_1699887654321",
  "status": "started",
  "message": "Simulation started in background"
}
```

#### GET `/api/simulations`
List all simulations:
```bash
curl http://localhost:8000/api/simulations
```

#### GET `/api/simulation/{sim_id}`
Get simulation details:
```bash
curl http://localhost:8000/api/simulation/sim_1699887654321
```

#### WebSocket `/ws/{sim_id}`
Real-time updates for a simulation.

### Programmatic Usage

Use the API programmatically:

```python
import requests
import json

# Start simulation
response = requests.post('http://localhost:8000/api/simulate',
    json={
        'num_clients': 5,
        'num_conversations': 100,
        'total_blocks': 2000
    })

sim_id = response.json()['sim_id']
print(f"Started simulation: {sim_id}")

# Wait and get results
import time
time.sleep(10)

response = requests.get(
    f'http://localhost:8000/api/simulation/{sim_id}'
)
results = response.json()
print(f"Hit rate: {results['summary']['client_stats']}")
```

## Architecture

### Backend (FastAPI)

- **app.py** - Main FastAPI application
- Runs simulations in background tasks
- Streams updates via WebSocket
- Stores active simulation data in memory

### Frontend (HTML/CSS/JS)

- **index.html** - Main page structure
- **style.css** - Modern, responsive styling
- **app.js** - WebSocket client, chart updates
- **Chart.js** - Real-time chart library (CDN)

### Data Flow

```
User Input → FastAPI → Background Task → Simulator
                                            ↓
WebSocket ← FastAPI ← Data Collector ← Simulator
    ↓
Frontend Charts
```

## Troubleshooting

### Port Already in Use

```
ERROR: [Errno 48] Address already in use
```

**Solution**: Kill existing process or use different port:
```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
python3 -c "import uvicorn; from web.app import app; \
    uvicorn.run(app, host='0.0.0.0', port=8001)"
```

### WebSocket Connection Failed

```
WebSocket error: Connection refused
```

**Solutions**:
1. Ensure server is running
2. Check firewall settings
3. Verify correct URL (ws:// not wss:// for localhost)
4. Check browser console for errors

### Charts Not Updating

**Solutions**:
1. Check browser console for JavaScript errors
2. Verify WebSocket connection is open
3. Ensure simulation is actually running
4. Try refreshing the page

### Slow Performance

For large simulations:
1. Reduce `request_rate` to 0 for max speed
2. Disable visualization in backend if not needed
3. Use fewer clients/conversations for testing
4. Check system resources (CPU/memory)

### CORS Errors

If accessing from different origin:

```python
# Add to app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Production Deployment

### With Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt web/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    -r web/requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "web/app.py"]
```

Build and run:
```bash
docker build -t vllm-simulator .
docker run -p 8000:8000 vllm-simulator
```

### With Nginx

Reverse proxy configuration:
```nginx
server {
    listen 80;
    server_name simulator.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/vllm-simulator.service`:
```ini
[Unit]
Description=vLLM Simulator Web UI
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 web/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable vllm-simulator
sudo systemctl start vllm-simulator
```

## Future Enhancements

Planned features:
- [ ] Save/load simulation configurations
- [ ] Export results to CSV/JSON
- [ ] Compare multiple simulation results
- [ ] ShareGPT dataset integration in web UI
- [ ] Authentication and user management
- [ ] Simulation history and replay
- [ ] Advanced filtering and search in logs
- [ ] Custom chart configurations
- [ ] Mobile-responsive improvements

## References

- **FastAPI**: https://fastapi.tiangolo.com/
- **Chart.js**: https://www.chartjs.org/
- **WebSocket API**: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
- **Main README**: [../README.md](../README.md)

---

For questions or issues, please file a GitHub issue or see the main 
documentation.

