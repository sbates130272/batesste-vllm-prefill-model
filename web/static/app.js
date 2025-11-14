// vLLM Simulator Web UI JavaScript

let currentSimId = null;
let ws = null;
let charts = {};
let conversationCount = 0;
let simulationStartTime = null;
let runtimeInterval = null;

// Model configurations for memory calculations
const MODEL_CONFIGS = {
    'llama2-7b': { hidden_dim: 4096, num_layers: 32, dtype_bytes: 2 },
    'llama2-13b': { hidden_dim: 5120, num_layers: 40, dtype_bytes: 2 },
    'llama2-70b': { hidden_dim: 8192, num_layers: 80, dtype_bytes: 2 },
    'llama3-8b': { hidden_dim: 4096, num_layers: 32, dtype_bytes: 2 },
    'llama3-70b': { hidden_dim: 8192, num_layers: 80, dtype_bytes: 2 },
    'mistral-7b': { hidden_dim: 4096, num_layers: 32, dtype_bytes: 2 },
    'mixtral-8x7b': { hidden_dim: 4096, num_layers: 32, dtype_bytes: 2 },
    'gpt-j-6b': { hidden_dim: 4096, num_layers: 28, dtype_bytes: 2 },
    'custom': { hidden_dim: 4096, num_layers: 32, dtype_bytes: 2 }
};

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
                    display: true,
                    position: 'top',
                    align: 'start',
                    labels: {
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
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
    if (metrics.timestamps && metrics.timestamps.length > 0) {
        const timestamps = metrics.timestamps.map(t => t.toFixed(1));
        
        // Update hit rate chart - show full timeline
        if (metrics.cache_hit_rate) {
            const currentValue = metrics.cache_hit_rate[metrics.cache_hit_rate.length - 1];
            charts.hitRate.data.labels = timestamps;
            charts.hitRate.data.datasets[0].data = metrics.cache_hit_rate;
            charts.hitRate.data.datasets[0].label = `Hit Rate % (${currentValue.toFixed(1)}%)`;
            charts.hitRate.update('none');
        }
        
        // Update occupancy chart - show full timeline
        if (metrics.cache_occupancy) {
            const currentValue = metrics.cache_occupancy[metrics.cache_occupancy.length - 1];
            charts.occupancy.data.labels = timestamps;
            charts.occupancy.data.datasets[0].data = metrics.cache_occupancy;
            charts.occupancy.data.datasets[0].label = `Occupancy % (${currentValue.toFixed(1)}%)`;
            charts.occupancy.update('none');
        }
        
        // Update active conversations chart - show full timeline
        if (metrics.active_conversations) {
            const currentValue = metrics.active_conversations[metrics.active_conversations.length - 1];
            charts.activeConvs.data.labels = timestamps;
            charts.activeConvs.data.datasets[0].data = metrics.active_conversations;
            charts.activeConvs.data.datasets[0].label = `Active Conversations (${currentValue})`;
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

// Update cache log display
function updateCacheLog(entries) {
    const cacheLog = document.getElementById('cacheLog');
    if (!cacheLog) return;
    
    // Keep only last 10 entries visible
    const lastEntries = entries.slice(-10);
    
    cacheLog.innerHTML = lastEntries.map(entry => {
        const time = entry.timestamp.toFixed(1);
        const occupancy = entry.occupancy.toFixed(1);
        const hitRateCumulative = entry.hit_rate_cumulative ? entry.hit_rate_cumulative.toFixed(1) : '0.0';
        const hitRateInstant = entry.hit_rate_instant ? entry.hit_rate_instant.toFixed(1) : '0.0';
        
        let hitInfo = '';
        if (entry.recent_hits && entry.recent_hits.length > 0) {
            const hitsHtml = entry.recent_hits.slice().reverse().slice(0, 3).map(hit => {
                // Display both token IDs and decoded text
                let tokensDisplay = '';
                if (hit.tokens_text && hit.tokens_text.length > 0) {
                    // Show decoded text with token IDs
                    tokensDisplay = hit.tokens_text.map((text, idx) => {
                        const tokenId = hit.tokens_preview[idx];
                        // Escape special chars and show text
                        const escapedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        return `<span title="Token ID: ${tokenId}">"${escapedText}"</span>`;
                    }).join(' ');
                } else {
                    // Fallback to just token IDs
                    tokensDisplay = hit.tokens_preview.map((tokenId, idx) => 
                        `[${idx}]=${tokenId}`
                    ).join(', ');
                }
                return `
                    <div style="margin-bottom: 4px; padding: 3px; background: rgba(255,255,255,0.5); border-radius: 2px;">
                        <div><strong>Hash:</strong> <code>${hit.block_hash}</code></div>
                        <div><strong>Conv:</strong> ${hit.conv_id}</div>
                        <div class="tokens-preview"><strong>Tokens (${hit.token_count}):</strong> ${tokensDisplay}${hit.token_count > 10 ? '...' : ''}</div>
                    </div>
                `;
            }).join('');
            
            hitInfo = `
                <div class="cache-hit-info">
                    <div class="hit-label">✓ Recent Cache Hits (${entry.recent_hits.length}):</div>
                    ${hitsHtml}
                </div>
            `;
        }
        
        return `
            <div class="cache-log-entry">
                <div class="timestamp">⏱ ${time}s</div>
                <div class="cache-stats">
                    <div class="stat-row">
                        <span class="stat-label">Blocks Used:</span>
                        <span class="stat-value">${entry.blocks_used} / ${entry.blocks_total}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Occupancy:</span>
                        <span class="stat-value">${occupancy}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Hit Rate (Now):</span>
                        <span class="stat-value">${hitRateInstant}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Hit Rate (Avg):</span>
                        <span class="stat-value">${hitRateCumulative}%</span>
                    </div>
                </div>
                ${hitInfo}
            </div>
        `;
    }).reverse().join('');
    
    // Auto-scroll to top (newest entry)
    cacheLog.scrollTop = 0;
}

// Display conversation snippet
function displayConversation(conv) {
    const conversationsBox = document.getElementById('conversationsBox');
    
    // Remove "no conversations" message if present
    const noConvMsg = conversationsBox.querySelector('.no-conversations');
    if (noConvMsg) {
        noConvMsg.remove();
    }
    
    // Create conversation item
    const convItem = document.createElement('div');
    convItem.className = 'conversation-item';
    convItem.id = `conv-${conv.conv_id}`;
    
    let turnsHtml = '';
    if (conv.turns && conv.turns.length > 0) {
        // Show last 2 turns for brevity
        const recentTurns = conv.turns.slice(-2);
        recentTurns.forEach(turn => {
            const isUser = turn.role === 'user';
            const userLabel = isUser ? `👤 User (Client ${conv.client_id})` : '🤖 Assistant';
            turnsHtml += `
                <div class="conversation-turn">
                    <div class="turn-label ${isUser ? 'turn-user' : 
                                                        'turn-assistant'}">
                        ${userLabel}:
                    </div>
                    <div class="turn-content">${escapeHtml(turn.content)}
                    </div>
                </div>
            `;
        });
    }
    
    convItem.innerHTML = `
        <div class="conversation-header">
            <div class="conversation-id">${conv.conv_id}</div>
            <div class="conversation-client">Client ${conv.client_id}</div>
        </div>
        ${turnsHtml}
        <div class="conversation-stats">
            <div class="stat-item">
                <span class="stat-label">Turns:</span>
                <span>${conv.num_turns || 0}</span>
            </div>
            ${conv.cache_hits !== undefined ? `
                <div class="stat-item cache-hit">
                    <span class="stat-label">Cache Hits:</span>
                    <span>${conv.cache_hits}</span>
                </div>
            ` : ''}
            ${conv.total_tokens !== undefined ? `
                <div class="stat-item">
                    <span class="stat-label">Tokens:</span>
                    <span>${conv.total_tokens}</span>
                </div>
            ` : ''}
        </div>
    `;
    
    // Check if conversation already exists and update it
    const existingConv = document.getElementById(`conv-${conv.conv_id}`);
    if (existingConv) {
        existingConv.replaceWith(convItem);
    } else {
        // Add to top of list
        conversationsBox.insertBefore(convItem, conversationsBox.firstChild);
        
        // Increment counter for new conversations
        conversationCount++;
        document.getElementById('convCount').textContent = 
            `(${conversationCount})`;
        
        // Limit to 10 conversations displayed
        const items = conversationsBox.querySelectorAll('.conversation-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update status display
function updateStatus(status, summaryData = null) {
    const statusContent = document.getElementById('statusContent');
    
    let html = `<p class="status-${status}">${getStatusMessage(status)}</p>`;
    
    // Add runtime for running simulations
    if (status === 'running' && simulationStartTime) {
        const elapsed = Math.floor((Date.now() - simulationStartTime) / 1000);
        html += `<div class="runtime-display" id="runtimeDisplay">
                   <strong>Runtime:</strong> ${formatTime(elapsed)}
                 </div>`;
    }
    
    if (status === 'completed' && summaryData) {
        html += '<div class="status-details">';
        
        const clientStats = summaryData.client_stats || summaryData;
        
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
        
        // Show final runtime
        if (simulationStartTime) {
            const totalTime = Math.floor((Date.now() - simulationStartTime) / 1000);
            html += `
                <div><strong>Total Runtime:</strong> ${formatTime(totalTime)}</div>
            `;
        }
        
        // Show total conversations and turns
        if (summaryData.total_conversations) {
            html += `
                <div><strong>Total Conversations:</strong> 
                     ${summaryData.total_conversations.toLocaleString()}</div>
            `;
        }
        
        if (summaryData.total_turns) {
            html += `
                <div><strong>Total Turns:</strong> 
                     ${summaryData.total_turns.toLocaleString()}</div>
            `;
        }
        
        // Show active conversation statistics
        if (summaryData.active_conv_stats) {
            const stats = summaryData.active_conv_stats;
            html += `
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                    <strong>Active Conversations:</strong>
                    Min: ${stats.min}, 
                    Max: ${stats.max}, 
                    Avg: ${stats.avg.toFixed(1)}, 
                    Std: ${stats.std.toFixed(2)}
                </div>
            `;
        }
        
        // Show conversation duration statistics
        if (summaryData.conv_duration_stats && summaryData.conv_duration_stats.count > 0) {
            const stats = summaryData.conv_duration_stats;
            html += `
                <div style="margin-top: 5px;">
                    <strong>Conversation Duration (sec):</strong>
                    Min: ${stats.min.toFixed(1)}, 
                    Max: ${stats.max.toFixed(1)}, 
                    Avg: ${stats.avg.toFixed(1)}, 
                    Std: ${stats.std.toFixed(2)}
                </div>
            `;
        }
        
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

// Format time as HH:MM:SS or MM:SS
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Update runtime display every second
function startRuntimeCounter() {
    // Clear any existing interval
    if (runtimeInterval) {
        clearInterval(runtimeInterval);
    }
    
    runtimeInterval = setInterval(() => {
        const runtimeDisplay = document.getElementById('runtimeDisplay');
        if (runtimeDisplay && simulationStartTime) {
            const elapsed = Math.floor((Date.now() - simulationStartTime) / 1000);
            runtimeDisplay.innerHTML = `<strong>Runtime:</strong> ${formatTime(elapsed)}`;
        }
    }, 1000);
}

// Stop runtime counter
function stopRuntimeCounter() {
    if (runtimeInterval) {
        clearInterval(runtimeInterval);
        runtimeInterval = null;
    }
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
        // num_conversations removed - controlled by turns/time instead
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
    document.getElementById('cacheLog').innerHTML = '';
    document.getElementById('conversationsBox').innerHTML = 
        '<p class="no-conversations">No active conversations yet</p>';
    conversationCount = 0;
    document.getElementById('convCount').textContent = '(0)';
    simulationStartTime = null;
    stopRuntimeCounter();
    Object.values(charts).forEach(chart => {
        chart.data.labels = [];
        chart.data.datasets.forEach(dataset => dataset.data = []);
        chart.update();
    });
    
    // Clear analytics charts
    clearAnalyticsCharts();
    
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
        const startBtn = document.getElementById('startBtn');
        startBtn.disabled = false;
        startBtn.classList.remove('loading');
        startBtn.textContent = '▶️ Start Simulation';
    }
});

// Connect to WebSocket
function connectWebSocket(simId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${simId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        simulationStartTime = Date.now();
        updateStatus('running');
        startRuntimeCounter();
        
        // Update button to show running state
        const startBtn = document.getElementById('startBtn');
        startBtn.textContent = '⏸️ Running Sim';
        startBtn.classList.remove('loading');
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
            
            case 'conversation':
                displayConversation(message.data);
                break;
            
            case 'analytics':
                console.log('Received analytics data:', message.data);
                updateAnalyticsCharts(message.data);
                break;
            
            case 'cache_log':
                updateCacheLog(message.data);
                break;
            
            case 'complete':
                console.log('Simulation complete', message.data);
                updateStatus('completed', message.data);
                onSimulationComplete();
                break;
            
            default:
                console.log('Unknown message type:', message.type);
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
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = false;
    startBtn.classList.remove('loading');
    startBtn.textContent = '▶️ Start Simulation';
    
    document.getElementById('stopBtn').disabled = true;
    
    stopRuntimeCounter();
    
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

// Calculate and display block size in memory
function updateBlockSizeMemory() {
    const modelPreset = document.getElementById('model_preset').value;
    const blockSize = parseInt(document.getElementById('block_size').value) || 16;
    const totalBlocks = parseInt(document.getElementById('total_blocks').value) || 500;
    
    const config = MODEL_CONFIGS[modelPreset];
    if (!config) return;
    
    // KV cache memory per token = 2 (K + V) × hidden_dim × num_layers × dtype_bytes
    const bytesPerToken = 2 * config.hidden_dim * config.num_layers * config.dtype_bytes;
    const bytesPerBlock = bytesPerToken * blockSize;
    const mibPerBlock = bytesPerBlock / (1024 * 1024);
    
    // Update block size display
    const memoryDisplay = document.getElementById('block_size_memory');
    memoryDisplay.textContent = `≈ ${mibPerBlock.toFixed(1)} MiB per block`;
    
    // Update total cache size display
    const totalCacheMiB = mibPerBlock * totalBlocks;
    const totalCacheGiB = totalCacheMiB / 1024;
    
    const totalMemoryDisplay = document.getElementById('total_cache_memory');
    if (totalCacheGiB >= 1) {
        totalMemoryDisplay.textContent = `Total cache: ${totalCacheGiB.toFixed(2)} GiB`;
    } else {
        totalMemoryDisplay.textContent = `Total cache: ${totalCacheMiB.toFixed(0)} MiB`;
    }
}

// Initialize on page load
window.addEventListener('load', () => {
    initCharts();
    initAnalyticsCharts();
    setupAnalyticsExportButtons();
    
    // Setup memory calculation listeners
    document.getElementById('model_preset').addEventListener('change', updateBlockSizeMemory);
    document.getElementById('block_size').addEventListener('input', updateBlockSizeMemory);
    document.getElementById('total_blocks').addEventListener('input', updateBlockSizeMemory);
    updateBlockSizeMemory();
    
    console.log('vLLM Simulator Web UI initialized');
});

