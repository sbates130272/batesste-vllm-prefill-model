# ShareGPT Dataset Integration

This guide explains how to use real ShareGPT datasets with the vLLM 
prefix caching simulator for more realistic conversation modeling.

## Overview

ShareGPT is a collection of real conversations shared by ChatGPT users.
Using real conversation data provides:

- **Realistic token distributions** - Real user queries and responses
- **Natural conversation patterns** - Actual multi-turn dialogues
- **Production-like workloads** - Better cache behavior prediction
- **Diverse topics** - Wide range of conversation types

## Quick Start

### 1. Download a Dataset

```bash
# Download sample dataset (~1K conversations, ~10MB)
./scripts/download_sharegpt.py sg-sample

# Or download full dataset (~90K conversations, ~800MB)
./scripts/download_sharegpt.py sg90k
```

### 2. Analyze the Dataset

```bash
# Get statistics about the dataset
./scripts/analyze_dataset.py data/sg-sample.json

# View sample conversations
./scripts/analyze_dataset.py data/sg-sample.json --sample 5
```

### 3. Run Simulation with Real Data

```bash
# Run with ShareGPT dataset
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg-sample.json \
  --num-clients 5 \
  --num-conversations 100 \
  --visualize
```

## Available Datasets

### sg-sample (Recommended for Testing)
- **Size**: ~10MB
- **Conversations**: ~1,000
- **Best for**: Quick testing, development
- **Download**: `./scripts/download_sharegpt.py sg-sample`

### sg90k (Full Dataset)
- **Size**: ~800MB
- **Conversations**: ~90,000
- **Best for**: Production simulations, benchmarking
- **Download**: `./scripts/download_sharegpt.py sg90k`

## Dataset Format

The simulator supports multiple ShareGPT format variations:

### Standard Format
```json
[
  {
    "id": "conv_12345",
    "messages": [
      {"role": "user", "content": "What is machine learning?"},
      {"role": "assistant", "content": "Machine learning is..."}
    ]
  }
]
```

### Vicuna Format
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {"from": "human", "value": "What is machine learning?"},
      {"from": "gpt", "value": "Machine learning is..."}
    ]
  }
]
```

The simulator automatically detects and normalizes both formats.

## Advanced Usage

### Filtering Conversations

Control which conversations to use:

```bash
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg90k.json \
  --num-conversations 500 \
  --num-clients 10
```

The `--num-conversations` parameter:
- Randomly samples from the dataset
- Ensures diversity in simulation
- Controls memory usage

### Custom Tokenizer

Use different tokenizers for different models:

```bash
# Use GPT-2 tokenizer (default)
./src/multi_user_simulator.py --text-mode --tokenizer gpt2

# Use LLaMA tokenizer
./src/multi_user_simulator.py --text-mode \
  --tokenizer meta-llama/Llama-2-7b-hf

# Use Mistral tokenizer
./src/multi_user_simulator.py --text-mode \
  --tokenizer mistralai/Mistral-7B-v0.1
```

**Note**: Requires `transformers` library:
```bash
pip install transformers
```

### Adding System Prompts

Add a common prefix to all conversations (e.g., system prompt):

```bash
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg-sample.json \
  --common-prefix-text "You are a helpful AI assistant."
```

This simulates real production scenarios where all conversations 
share a system prompt.

### Creating Custom Samples

Create smaller samples from large datasets:

```bash
# Download full dataset and create 1000-conversation sample
./scripts/download_sharegpt.py sg90k --sample 1000
# Creates: data/sg90k_sample_1000.json

# Use the sample
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg90k_sample_1000.json
```

## Dataset Analysis

The `analyze_dataset.py` script provides detailed statistics:

```bash
./scripts/analyze_dataset.py data/sg-sample.json
```

Output includes:
- **Total conversations**
- **Turn distribution** (histogram)
- **Message length statistics** (user vs assistant)
- **Token estimates** (rough approximation)
- **Simulation recommendations** (block sizing, cache sizing)

Example output:
```
======================================================================
DATASET ANALYSIS
======================================================================

📊 Overview:
  Total Conversations: 1,000
  Format Type: standard

💬 Turns per Conversation:
  Min: 2
  Max: 24
  Mean: 6.4
  Median: 6

  Distribution (top 10):
      6 turns:   245 (24.5%) ████████████
      4 turns:   198 (19.8%) █████████
      8 turns:   156 (15.6%) ███████
...

📝 User Message Length (characters):
  Min: 10
  Max: 2,450
  Mean: 142
  Est. Tokens (÷4): ~36

🤖 Assistant Message Length (characters):
  Min: 15
  Max: 5,820
  Mean: 486
  Est. Tokens (÷4): ~122

💡 Simulation Recommendations:
  Block size: 16 or 32
  Blocks per conversation (avg): ~65 (bs=16) or ~32 (bs=32)
  Total blocks for 10 conversations: ~650 (bs=16)
  Total blocks for 50 conversations: ~3,250 (bs=16)
  Total blocks for 100 conversations: ~6,500 (bs=16)
```

## Cache Sizing Guidelines

Based on dataset analysis, configure cache size:

```bash
# For small-scale testing (10 concurrent conversations)
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg-sample.json \
  --num-clients 5 \
  --num-conversations 10 \
  --total-blocks 1000

# For production-like workloads (100 concurrent conversations)
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg90k.json \
  --num-clients 20 \
  --num-conversations 100 \
  --total-blocks 10000 \
  --block-size 16
```

### Rule of Thumb

1. **Analyze dataset** to get average tokens per conversation
2. **Calculate blocks per conversation**: 
   `avg_tokens / block_size`
3. **Multiply by concurrent conversations**:
   `blocks_per_conv * num_concurrent`
4. **Add 20-30% headroom** for cache efficiency

Example:
- Average conversation: 1,000 tokens
- Block size: 16
- Blocks per conversation: 1000 / 16 = ~63
- 50 concurrent conversations: 63 * 50 = 3,150 blocks
- With 25% headroom: 3,150 * 1.25 = ~4,000 blocks

```bash
--total-blocks 4000 --block-size 16
```

## Troubleshooting

### Dataset Not Found

```
Dataset file not found: data/sg-sample.json
Generating synthetic text conversations instead...
```

**Solution**: Download the dataset first:
```bash
./scripts/download_sharegpt.py sg-sample
```

### Tokenizer Issues

```
WARNING: transformers not installed
Falling back to mock tokenization...
```

**Solution**: Install transformers:
```bash
pip install transformers
```

Mock tokenization works but estimates tokens based on character count 
(~4 chars per token). Real tokenizers are more accurate.

### Memory Issues

Large datasets can consume significant memory.

**Solutions**:
1. Use `--num-conversations` to limit dataset size
2. Create a smaller sample: 
   `./scripts/download_sharegpt.py sg90k --sample 1000`
3. Use the sample dataset for testing

### Format Errors

```
Error loading dataset: Unknown dataset format
```

**Solution**: The dataset format isn't recognized. Check:
1. File is valid JSON: `python3 -m json.tool < dataset.json`
2. Has expected structure (see [Dataset Format](#dataset-format))
3. File isn't corrupted

## Integration with Visualization

ShareGPT datasets work seamlessly with the visualizer:

```bash
./src/multi_user_simulator.py \
  --text-mode \
  --dataset-path data/sg-sample.json \
  --num-clients 5 \
  --num-conversations 50 \
  --visualize
```

The visualization will show:
- Real conversation text snippets
- Actual token distributions
- Cache hit rates on realistic workloads

## Performance Considerations

### Dataset Loading

Large datasets take time to load and tokenize:

```
Loading ShareGPT dataset from: data/sg90k.json
Loaded 90,000 raw conversations
Tokenizing conversations...
✓ Using 1,000 conversations (2-20 turns)
```

**Tips**:
1. Pre-create samples for faster iteration
2. Use `--num-conversations` to limit processing
3. Consider caching tokenized datasets (future feature)

### Tokenization

Real tokenization is CPU-intensive:

- **GPT-2**: Fast (~10K tokens/sec)
- **LLaMA**: Slower (~5K tokens/sec)
- **Mock**: Very fast (no actual tokenization)

For quick testing, mock tokenization is sufficient. For accuracy, 
use a real tokenizer.

## Future Enhancements

Planned features for ShareGPT integration:

- [ ] Cached tokenized datasets (faster loading)
- [ ] Conversation filtering by topic/length
- [ ] Support for instruction-tuning datasets
- [ ] Dataset quality metrics
- [ ] Conversation replay with timing
- [ ] Multi-dataset mixing

## References

- **ShareGPT**: https://sharegpt.com/
- **Vicuna Dataset**: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers

---

For more information, see:
- [Main README](../README.md)
- [Visualization Guide](VISUALIZATION.md)
- [Multi-User Simulator](../README.md#multi-user-simulator)

