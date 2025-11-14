// Advanced Analytics for vLLM Simulator

let analyticsCharts = {};

// Initialize advanced analytics charts
function initAnalyticsCharts() {
    // Cache Hit Heatmap (scatter plot)
    analyticsCharts.heatmap = new Chart(
        document.getElementById('heatmapChart'),
        {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Cache Hit Rate',
                    data: [],
                    backgroundColor: function(context) {
                        const value = context.parsed ? context.parsed.y : 0;
                        return value > 75 ? 'rgba(40, 167, 69, 0.8)' :
                               value > 50 ? 'rgba(255, 193, 7, 0.8)' :
                               'rgba(220, 53, 69, 0.8)';
                    },
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (s)'
                        },
                        beginAtZero: true
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Cache Hit Rate (%)'
                        },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Hit Rate: ${context.parsed.y.toFixed(1)}% @ ${context.parsed.x.toFixed(1)}s`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        }
    );

    // Per-Conversation Effectiveness Chart (horizontal bar)
    analyticsCharts.convEffectiveness = new Chart(
        document.getElementById('convEffectivenessChart'),
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
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Cache Hit Rate (%)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Conversation'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const convData = context.chart.data.datasets[0].conversationData[context.dataIndex];
                                return [
                                    `Hit Rate: ${context.parsed.x.toFixed(1)}%`,
                                    `Requests: ${convData.requests}`,
                                    `Total Tokens: ${convData.total_tokens.toLocaleString()}`
                                ];
                            }
                        }
                    }
                }
            }
        }
    );

    // Prefix Overlap Analysis (doughnut chart)
    analyticsCharts.prefixOverlap = new Chart(
        document.getElementById('prefixOverlapChart'),
        {
            type: 'doughnut',
            data: {
                labels: ['Common Prefix', 'Unique Content', 'Cached Tokens'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(40, 167, 69, 0.8)'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value.toLocaleString()} tokens (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        }
    );
    
    // Conversation Similarity Matrix (bubble chart)
    analyticsCharts.similarity = new Chart(
        document.getElementById('similarityChart'),
        {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'Conversation Pairs',
                    data: [],
                    backgroundColor: function(context) {
                        if (!context.raw) return 'rgba(102, 126, 234, 0.6)';
                        // Color based on Jaccard similarity
                        const jaccard = context.raw.jaccard || 0;
                        if (jaccard > 0.7) return 'rgba(40, 167, 69, 0.7)';
                        if (jaccard > 0.4) return 'rgba(255, 193, 7, 0.7)';
                        return 'rgba(220, 53, 69, 0.7)';
                    },
                    borderColor: 'rgba(102, 126, 234, 0.8)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Conversation ID (numeric)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Conversation ID (numeric)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const data = context.raw;
                                return [
                                    `Pair: ${data.conv_a} ↔ ${data.conv_b}`,
                                    `Prefix Overlap: ${data.prefix_overlap} tokens`,
                                    `Jaccard Index: ${(data.jaccard * 100).toFixed(1)}%`,
                                    `Lengths: ${data.total_tokens_a} / ${data.total_tokens_b} tokens`
                                ];
                            }
                        }
                    }
                }
            }
        }
    );
    
    console.log('Analytics charts initialized');
}

// Update analytics charts with new data
function updateAnalyticsCharts(analyticsData) {
    console.log('updateAnalyticsCharts called with:', analyticsData);
    if (!analyticsData) {
        console.warn('No analytics data provided');
        return;
    }
    
    // Update heatmap
    if (analyticsData.heatmap && analyticsCharts.heatmap) {
        const heatmapPoints = analyticsData.heatmap.map(point => ({
            x: point.time,
            y: point.hit_rate
        }));
        
        analyticsCharts.heatmap.data.datasets[0].data = heatmapPoints;
        analyticsCharts.heatmap.update('none');
    }
    
    // Update conversation effectiveness
    if (analyticsData.top_conversations && analyticsCharts.convEffectiveness) {
        const topConvs = analyticsData.top_conversations.slice(0, 10);
        console.log('Top conversations:', topConvs);
        
        if (topConvs.length > 0) {
            analyticsCharts.convEffectiveness.data.labels = topConvs.map(c => c.id);
            analyticsCharts.convEffectiveness.data.datasets[0].data = topConvs.map(c => c.hit_rate);
            analyticsCharts.convEffectiveness.data.datasets[0].conversationData = topConvs;
            analyticsCharts.convEffectiveness.update('none');
        } else {
            console.warn('No conversation effectiveness data available');
        }
    } else {
        console.warn('Missing analytics data or chart:', {
            hasData: !!analyticsData.top_conversations,
            hasChart: !!analyticsCharts.convEffectiveness
        });
    }
    
    // Update prefix overlap
    if (analyticsData.prefix_overlap && analyticsCharts.prefixOverlap) {
        const overlap = analyticsData.prefix_overlap;
        analyticsCharts.prefixOverlap.data.datasets[0].data = [
            overlap.common_prefix_tokens || 0,
            overlap.unique_tokens || 0,
            overlap.total_cached_tokens || 0
        ];
        analyticsCharts.prefixOverlap.update('none');
    }
    
    // Update conversation similarity matrix
    if (analyticsData.conversation_similarities && analyticsCharts.similarity) {
        const simData = analyticsData.conversation_similarities;
        
        if (simData.conversation_ids && simData.similarities) {
            // Create a mapping of conv_id to numeric index
            const convIdToIndex = {};
            simData.conversation_ids.forEach((id, idx) => {
                convIdToIndex[id] = idx;
            });
            
            // Convert similarities to bubble chart data points
            const bubbleData = simData.similarities.map(sim => {
                const x = convIdToIndex[sim.conv_a] || 0;
                const y = convIdToIndex[sim.conv_b] || 0;
                // Bubble size based on prefix overlap (larger = more overlap)
                const r = Math.max(3, Math.min(15, sim.prefix_overlap / 10));
                
                return {
                    x: x,
                    y: y,
                    r: r,
                    conv_a: sim.conv_a,
                    conv_b: sim.conv_b,
                    prefix_overlap: sim.prefix_overlap,
                    jaccard: sim.jaccard_similarity,
                    total_tokens_a: sim.total_tokens_a,
                    total_tokens_b: sim.total_tokens_b
                };
            });
            
            analyticsCharts.similarity.data.datasets[0].data = bubbleData;
            analyticsCharts.similarity.update('none');
            
            console.log(`Updated similarity matrix with ${bubbleData.length} pairs`);
        }
    }
}

// Clear analytics charts
function clearAnalyticsCharts() {
    if (analyticsCharts.heatmap) {
        analyticsCharts.heatmap.data.datasets[0].data = [];
        analyticsCharts.heatmap.update();
    }
    
    if (analyticsCharts.convEffectiveness) {
        analyticsCharts.convEffectiveness.data.labels = [];
        analyticsCharts.convEffectiveness.data.datasets[0].data = [];
        analyticsCharts.convEffectiveness.data.datasets[0].conversationData = [];
        analyticsCharts.convEffectiveness.update();
    }
    
    if (analyticsCharts.prefixOverlap) {
        analyticsCharts.prefixOverlap.data.datasets[0].data = [0, 0, 0];
        analyticsCharts.prefixOverlap.update();
    }
    
    if (analyticsCharts.similarity) {
        analyticsCharts.similarity.data.datasets[0].data = [];
        analyticsCharts.similarity.update();
    }
}

// Export chart as PNG
function exportAnalyticsChart(chartKey, filename) {
    const chart = analyticsCharts[chartKey];
    if (!chart) {
        console.error('Chart not found:', chartKey);
        return;
    }
    
    const url = chart.toBase64Image();
    const link = document.createElement('a');
    link.download = filename || `${chartKey}-${Date.now()}.png`;
    link.href = url;
    link.click();
}

// Setup export button handlers
function setupAnalyticsExportButtons() {
    const exportHeatmap = document.getElementById('exportHeatmap');
    if (exportHeatmap) {
        exportHeatmap.addEventListener('click', () => {
            exportAnalyticsChart('heatmap', 'cache-heatmap.png');
        });
    }
    
    const exportConvChart = document.getElementById('exportConvChart');
    if (exportConvChart) {
        exportConvChart.addEventListener('click', () => {
            exportAnalyticsChart('convEffectiveness', 'conversation-effectiveness.png');
        });
    }
    
    const exportPrefixChart = document.getElementById('exportPrefixChart');
    if (exportPrefixChart) {
        exportPrefixChart.addEventListener('click', () => {
            exportAnalyticsChart('prefixOverlap', 'prefix-overlap.png');
        });
    }
    
    const exportSimilarityChart = document.getElementById('exportSimilarityChart');
    if (exportSimilarityChart) {
        exportSimilarityChart.addEventListener('click', () => {
            exportAnalyticsChart('similarity', 'conversation-similarity.png');
        });
    }
}

