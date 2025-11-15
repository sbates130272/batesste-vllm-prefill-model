// Advanced Analytics for vLLM Simulator

let analyticsCharts = {};

// Initialize advanced analytics charts
function initAnalyticsCharts() {
    // Cache Hit Rate Over Time (line chart)
    analyticsCharts.heatmap = new Chart(
        document.getElementById('heatmapChart'),
        {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cache Hit Rate %',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
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
                        display: true,
                        position: 'top',
                        align: 'start'
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
    
    // Conversation Similarity Heatmap
    analyticsCharts.similarity = new Chart(
        document.getElementById('similarityChart'),
        {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Jaccard Similarity',
                    data: [],
                    backgroundColor: [],
                    borderWidth: 1,
                    borderColor: '#fff'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        align: 'start',
                        labels: {
                            font: {
                                size: 11,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const jaccard = context.parsed.x;
                                return `Similarity: ${(jaccard * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Jaccard Similarity Index'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Conversation Pairs'
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
    
    // Update cache hit rate over time
    if (analyticsData.heatmap && analyticsCharts.heatmap) {
        const times = analyticsData.heatmap.map(point => point.time);
        const hitRates = analyticsData.heatmap.map(point => point.hit_rate);
        
        analyticsCharts.heatmap.data.labels = times;
        analyticsCharts.heatmap.data.datasets[0].data = hitRates;
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
    
    // Update conversation similarity heatmap
    if (analyticsData.conversation_similarities && analyticsCharts.similarity) {
        const simData = analyticsData.conversation_similarities;
        
        if (simData.conversation_ids && simData.similarities) {
            // Sort by Jaccard similarity (highest first)
            const allSorted = simData.similarities
                .sort((a, b) => b.jaccard_similarity - a.jaccard_similarity);
            
            // Get top 5, middle 5, and bottom 5
            let selectedSims = [];
            const totalPairs = allSorted.length;
            
            if (totalPairs >= 15) {
                // Top 5 (most similar)
                const top5 = allSorted.slice(0, 5);
                
                // Middle 5 (around median)
                const midStart = Math.floor((totalPairs - 5) / 2);
                const middle5 = allSorted.slice(midStart, midStart + 5);
                
                // Bottom 5 (least similar)
                const bottom5 = allSorted.slice(-5);
                
                // Combine: top, middle, bottom (maintain order for visual clarity)
                selectedSims = [...top5, ...middle5, ...bottom5];
            } else {
                // If we have fewer than 15 pairs, just show all of them
                selectedSims = allSorted;
            }
            
            // Create labels and data for horizontal bar chart
            const labels = selectedSims.map(sim => 
                `${sim.conv_a.slice(-4)} ↔ ${sim.conv_b.slice(-4)}`
            );
            
            const data = selectedSims.map(sim => sim.jaccard_similarity);
            
            // Calculate statistics across all pairs (not just displayed ones)
            const allJaccardValues = allSorted.map(s => s.jaccard_similarity);
            const avgJaccard = allJaccardValues.reduce((a, b) => a + b, 0) / allJaccardValues.length;
            const minJaccard = Math.min(...allJaccardValues);
            const maxJaccard = Math.max(...allJaccardValues);
            
            // Color gradient based on similarity (red = high similarity = bad diversity)
            const colors = selectedSims.map(sim => {
                const jaccard = sim.jaccard_similarity;
                if (jaccard > 0.7) return 'rgba(220, 53, 69, 0.8)';    // Red - very similar (bad)
                if (jaccard > 0.5) return 'rgba(255, 152, 0, 0.8)';    // Orange - moderately similar
                if (jaccard > 0.3) return 'rgba(255, 193, 7, 0.8)';    // Yellow - somewhat similar
                return 'rgba(40, 167, 69, 0.8)';                         // Green - different (good)
            });
            
            analyticsCharts.similarity.data.labels = labels;
            analyticsCharts.similarity.data.datasets[0].data = data;
            analyticsCharts.similarity.data.datasets[0].backgroundColor = colors;
            
            // Update chart title with statistics (across all pairs)
            analyticsCharts.similarity.data.datasets[0].label = 
                `Top/Mid/Low 5 (of ${totalPairs} pairs) | Avg: ${(avgJaccard * 100).toFixed(1)}% | Min: ${(minJaccard * 100).toFixed(1)}% | Max: ${(maxJaccard * 100).toFixed(1)}%`;
            
            analyticsCharts.similarity.update('none');
            
            console.log(`Updated similarity heatmap with ${selectedSims.length} pairs (Top/Mid/Bottom 5 of ${totalPairs})`);
        }
    }
}

// Clear analytics charts
function clearAnalyticsCharts() {
    if (analyticsCharts.heatmap) {
        analyticsCharts.heatmap.data.labels = [];
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

