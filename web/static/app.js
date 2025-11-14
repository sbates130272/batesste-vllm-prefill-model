// vLLM Simulator Web UI JavaScript

let currentSimId = null;
let ws = null;
let charts = {};

// Initialize charts
function initCharts() {
    const chartConfig = {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    };

    // Cache Hit Rate Chart
    charts.hitRate = new Chart(
        document.getElementById('hitRateChart'),
        {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [{
                    label: 'Hit Rate %',
                    data: [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...chartConfig.options,
                scales: {
                    ...chartConfig.options.scales,
                    y: {
                        ...chartConfig.options.scales.y,
                        max: 100
                    }
                }
            }
        }
    );

    // Cache Occupancy Chart
    charts.occupancy = new Chart(
        document.getElementById('occupancyChart'),
        {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [{
                    label: 'Occupancy %',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...chartConfig.options,
                scales: {
                    ...chartConfig.options.scales,
                    y: {
                        ...chartConfig.options.scales.y,
                        max: 100
                    }
                }
            }
        }
    );

    // Active Conversations Chart
    charts.activeConvs = new Chart(
        document.getElementById('activeConvsChart'),
        {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [{
                    label: 'Active Conversations',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            }
        }
    );

    // Client Stats Chart (Bar chart)
    charts.clientStats = new Chart(
        document.getElementById('clientStatsChart'),
        {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cache Hit Rate %',
                    data: [],
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Hit Rate %'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Client'
                        }
                    }
                }
            }
        }
    );
}

// Update charts with new data
function updateCharts(metrics) {
    const maxPoints = 100;
    
    if (metrics.timestamps && metrics.timestamps.length > 0) {
        const timestamps = metrics.timestamps.map(t => t.toFixed(1));
        
        // Update hit rate chart
        if (metrics.cache_hit_rate) {
            charts.hitRate.data.labels = timestamps.slice(-maxPoints);
            charts.hitRate.data.datasets[0].data = 
                metrics.cache_hit_rate.slice(-maxPoints);
            charts.hitRate.update('none');
        }
        
        // Update occupancy chart
        if (metrics.cache_occupancy) {
            charts.occupancy.data.labels = timestamps.slice(-maxPoints);
            charts.occupancy.data.datasets[0].data = 
                metrics.cache_occupancy.slice(-maxPoints);
            charts.occupancy.update('none');
        }
        
        // Update active conversations chart
        if (metrics.active_conversations) {
            charts.activeConvs.data.labels = timestamps.slice(-maxPoints);
            charts.activeConvs.data.datasets[0].data = 
                metrics.active_conversations.slice(-maxPoints);
            charts.activeConvs.update('none');
        }
    }
}

// Update client stats chart
function updateClientStats(clientStats) {
    const clientIds = Object.keys(clientStats).sort((a, b) => a - b);
    
    if (clientIds.length > 0) {
        charts.clientStats.data.labels = 
            clientIds.map(id => `Client ${id}`);
        charts.clientStats.data.datasets[0].data = 
            clientIds.map(id => clientStats[id].cache_hit_rate.toFixed(1));
        charts.clientStats.update('none');
    }
}

// Add event to log
function addEvent(event) {
    const eventsLog = document.getElementById('eventsLog');
    const entry = document.createElement('div');
    entry.className = `event-entry event-${event.type}`;
    
    entry.innerHTML = `
        <div class="event-timestamp">[${event.timestamp.toFixed(2)}s]</div>
        <div class="event-message">${event.message}</div>
    `;
    
    eventsLog.appendChild(entry);
    eventsLog.scrollTop = eventsLog.scrollHeight;
}

// Update status display
function updateStatus(status, clientStats = null) {
    const statusContent = document.getElementById('statusContent');
    
    let html = `<p class="status-${status}">${getStatusMessage(status)}</p>`;
    
    if (status === 'completed' && clientStats) {
        html += '<div class="status-details">';
        
        // Calculate totals
        let totalRequests = 0;
        let totalInputTokens = 0;
        let totalCachedTokens = 0;
        
        for (const stats of Object.values(clientStats)) {
            totalRequests += stats.requests_sent;
            totalInputTokens += stats.total_input_tokens;
            totalCachedTokens += stats.cached_tokens;
        }
        
        const overallHitRate = totalInputTokens > 0 ? 
            (totalCachedTokens / totalInputTokens * 100).toFixed(1) : 0;
        
        html += `
            <div><strong>Total Requests:</strong> 
                 ${totalRequests.toLocaleString()}</div>
            <div><strong>Total Input Tokens:</strong> 
                 ${totalInputTokens.toLocaleString()}</div>
            <div><strong>Cached Tokens:</strong> 
                 ${totalCachedTokens.toLocaleString()}</div>
            <div><strong>Overall Hit Rate:</strong> ${overallHitRate}%</div>
        `;
        html += '</div>';
    }
    
    statusContent.innerHTML = html;
}

function getStatusMessage(status) {
    const messages = {
        'idle': '⏸️ Ready to start simulation',
        'starting': '🔄 Starting simulation...',
        'running': '▶️ Simulation running...',
        'completed': '✅ Simulation completed successfully',
        'error': '❌ Simulation failed'
    };
    return messages[status] || status;
}

// Handle form submission
document.getElementById('configForm').addEventListener('submit', 
                                                      async (e) => {
    e.preventDefault();
    
    // Get form values
    const config = {
        num_clients: parseInt(document.getElementById('num_clients').value),
        num_conversations: parseInt(
            document.getElementById('num_conversations').value
        ),
        block_size: parseInt(document.getElementById('block_size').value),
        total_blocks: parseInt(document.getElementById('total_blocks').value),
        request_rate: parseFloat(
            document.getElementById('request_rate').value
        ),
        max_active_conversations: parseInt(
            document.getElementById('max_active_conversations').value
        ),
        max_total_turns: parseInt(
            document.getElementById('max_total_turns').value
        ),
        time_limit: parseInt(
            document.getElementById('time_limit').value
        ),
        conversation_template: document.getElementById('conversation_template')
            .value
    };
    
    // Clear previous data
    document.getElementById('eventsLog').innerHTML = '';
    Object.values(charts).forEach(chart => {
        chart.data.labels = [];
        chart.data.datasets.forEach(dataset => dataset.data = []);
        chart.update();
    });
    
    // Update UI
    document.getElementById('startBtn').disabled = true;
    document.getElementById('startBtn').classList.add('loading');
    updateStatus('starting');
    
    try {
        // Start simulation
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        currentSimId = result.sim_id;
        
        // Connect to WebSocket for real-time updates
        connectWebSocket(currentSimId);
        
        document.getElementById('stopBtn').disabled = false;
        
    } catch (error) {
        console.error('Error starting simulation:', error);
        updateStatus('error');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').classList.remove('loading');
    }
});

// Connect to WebSocket
function connectWebSocket(simId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${simId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('running');
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
            case 'events':
                message.data.forEach(addEvent);
                break;
            
            case 'metrics':
                updateCharts(message.data);
                break;
            
            case 'status':
                updateStatus(message.data.status, message.data.client_stats);
                if (message.data.client_stats && 
                    Object.keys(message.data.client_stats).length > 0) {
                    updateClientStats(message.data.client_stats);
                }
                break;
            
            case 'complete':
                console.log('Simulation complete', message.data);
                onSimulationComplete();
                break;
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('error');
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
    };
}

// Handle simulation completion
function onSimulationComplete() {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').classList.remove('loading');
    document.getElementById('stopBtn').disabled = true;
    
    if (ws) {
        ws.close();
        ws = null;
    }
}

// Stop button handler
document.getElementById('stopBtn').addEventListener('click', () => {
    if (ws) {
        ws.close();
    }
    onSimulationComplete();
    updateStatus('idle');
});

// Initialize on page load
window.addEventListener('load', () => {
    initCharts();
    console.log('vLLM Simulator Web UI initialized');
});

