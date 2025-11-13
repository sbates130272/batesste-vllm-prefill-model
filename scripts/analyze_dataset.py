#!/usr/bin/env python3
"""
Analyze ShareGPT dataset for vLLM simulator.

Provides statistics and insights about a ShareGPT dataset to help
configure simulation parameters.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return (data.get('conversations') or 
                   data.get('data') or [])
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def analyze_dataset(conversations: List[Dict]) -> Dict[str, Any]:
    """
    Analyze dataset and return statistics.
    
    Returns dict with:
        - num_conversations
        - turn_counts (distribution)
        - message_lengths (stats)
        - total_tokens (estimate)
        - roles (distribution)
    """
    stats = {
        'num_conversations': len(conversations),
        'turn_counts': [],
        'user_msg_lengths': [],
        'assistant_msg_lengths': [],
        'role_counts': Counter(),
        'format_type': 'unknown'
    }
    
    for conv in conversations:
        # Detect format
        if 'messages' in conv:
            messages_key = 'messages'
            role_key = 'role'
            content_key = 'content'
            stats['format_type'] = 'standard'
        elif 'conversations' in conv:
            messages_key = 'conversations'
            role_key = 'from'
            content_key = 'value'
            stats['format_type'] = 'vicuna'
        else:
            continue
        
        messages = conv.get(messages_key, [])
        stats['turn_counts'].append(len(messages))
        
        for msg in messages:
            role = msg.get(role_key, '').lower()
            content = msg.get(content_key, '')
            
            # Normalize role
            if role in ['human', 'user']:
                role = 'user'
            elif role in ['gpt', 'assistant']:
                role = 'assistant'
            
            stats['role_counts'][role] += 1
            
            # Message length (chars)
            msg_len = len(content)
            if role == 'user':
                stats['user_msg_lengths'].append(msg_len)
            elif role == 'assistant':
                stats['assistant_msg_lengths'].append(msg_len)
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Overview:")
    print(f"  Total Conversations: {stats['num_conversations']:,}")
    print(f"  Format Type: {stats['format_type']}")
    
    # Turn distribution
    if stats['turn_counts']:
        turn_counts = stats['turn_counts']
        print(f"\n💬 Turns per Conversation:")
        print(f"  Min: {min(turn_counts)}")
        print(f"  Max: {max(turn_counts)}")
        print(f"  Mean: {sum(turn_counts)/len(turn_counts):.1f}")
        print(f"  Median: {sorted(turn_counts)[len(turn_counts)//2]}")
        
        # Distribution
        turn_dist = Counter(turn_counts)
        print(f"\n  Distribution (top 10):")
        for turns, count in turn_dist.most_common(10):
            pct = (count / len(turn_counts)) * 100
            bar = '█' * int(pct / 2)
            print(f"    {turns:3d} turns: {count:5d} "
                  f"({pct:5.1f}%) {bar}")
    
    # Message lengths
    if stats['user_msg_lengths']:
        user_lens = stats['user_msg_lengths']
        print(f"\n📝 User Message Length (characters):")
        print(f"  Min: {min(user_lens):,}")
        print(f"  Max: {max(user_lens):,}")
        print(f"  Mean: {sum(user_lens)/len(user_lens):.0f}")
        
        # Estimate tokens (rough: ~4 chars per token)
        avg_tokens = sum(user_lens)/len(user_lens) / 4
        print(f"  Est. Tokens (÷4): ~{avg_tokens:.0f}")
    
    if stats['assistant_msg_lengths']:
        asst_lens = stats['assistant_msg_lengths']
        print(f"\n🤖 Assistant Message Length (characters):")
        print(f"  Min: {min(asst_lens):,}")
        print(f"  Max: {max(asst_lens):,}")
        print(f"  Mean: {sum(asst_lens)/len(asst_lens):.0f}")
        
        # Estimate tokens
        avg_tokens = sum(asst_lens)/len(asst_lens) / 4
        print(f"  Est. Tokens (÷4): ~{avg_tokens:.0f}")
    
    # Role distribution
    if stats['role_counts']:
        print(f"\n👥 Role Distribution:")
        for role, count in stats['role_counts'].most_common():
            print(f"  {role}: {count:,}")
    
    # Cache sizing recommendations
    if (stats['turn_counts'] and 
        stats['user_msg_lengths'] and 
        stats['assistant_msg_lengths']):
        
        avg_turns = sum(stats['turn_counts']) / len(stats['turn_counts'])
        avg_user_tokens = (sum(stats['user_msg_lengths']) / 
                          len(stats['user_msg_lengths'])) / 4
        avg_asst_tokens = (sum(stats['assistant_msg_lengths']) / 
                          len(stats['assistant_msg_lengths'])) / 4
        
        # Estimate total tokens per conversation
        total_per_conv = (avg_user_tokens + avg_asst_tokens) * avg_turns
        
        print(f"\n💡 Simulation Recommendations:")
        print(f"  Block size: 16 or 32")
        print(f"  Blocks per conversation (avg): "
              f"~{total_per_conv/16:.0f} (bs=16) or "
              f"~{total_per_conv/32:.0f} (bs=32)")
        
        # For N concurrent conversations
        for n_conv in [10, 50, 100]:
            blocks_needed = (total_per_conv * n_conv) / 16
            print(f"  Total blocks for {n_conv} conversations: "
                  f"~{blocks_needed:.0f} (bs=16)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ShareGPT dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a dataset
  python3 analyze_dataset.py data/sg-sample.json

  # Analyze with sample output
  python3 analyze_dataset.py data/sg90k.json --sample 5
        """
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to ShareGPT JSON dataset'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Print N sample conversations'
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: File not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Loading dataset: {dataset_path}")
    conversations = load_dataset(dataset_path)
    
    if not conversations:
        print("Error: No conversations found in dataset")
        sys.exit(1)
    
    print(f"Loaded {len(conversations):,} conversations")
    print("Analyzing...")
    
    stats = analyze_dataset(conversations)
    print_statistics(stats)
    
    # Print sample conversations
    if args.sample > 0:
        print(f"\n" + "="*70)
        print(f"SAMPLE CONVERSATIONS (first {args.sample})")
        print("="*70)
        
        for i, conv in enumerate(conversations[:args.sample]):
            print(f"\n[Conversation {i+1}]")
            print(f"ID: {conv.get('id', 'N/A')}")
            
            # Get messages
            messages = (conv.get('messages') or 
                       conv.get('conversations') or [])
            
            for j, msg in enumerate(messages[:6]):  # Max 6 messages
                role = (msg.get('role') or 
                       msg.get('from') or 'unknown')
                content = (msg.get('content') or 
                          msg.get('value') or '')
                
                # Truncate long content
                if len(content) > 100:
                    content = content[:97] + "..."
                
                print(f"  [{role}]: {content}")
            
            if len(messages) > 6:
                print(f"  ... ({len(messages) - 6} more messages)")


if __name__ == "__main__":
    main()

