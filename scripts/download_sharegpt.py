#!/usr/bin/env python3
"""
Download ShareGPT dataset for vLLM simulator.

ShareGPT is a dataset of real conversations shared by ChatGPT users.
This script downloads a curated version suitable for testing.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


# Known ShareGPT dataset URLs
DATASETS = {
    "sg90k": {
        "name": "ShareGPT 90K (Vicuna)",
        "url": "https://huggingface.co/datasets/anon8231489123/"
               "ShareGPT_Vicuna_unfiltered/resolve/main/"
               "ShareGPT_V3_unfiltered_cleaned_split.json",
        "size": "~800MB",
        "conversations": "~90,000"
    },
    "sg-sample": {
        "name": "ShareGPT Sample (1K)",
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
               "playground/data/sample_data.json",
        "size": "~10MB",
        "conversations": "~1,000"
    }
}


def download_file(url: str, output_path: Path, show_progress: bool = True):
    """
    Download a file from URL to output_path.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        show_progress: Whether to show download progress
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        def report_progress(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% "
                      f"({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                      end='', flush=True)
        
        urllib.request.urlretrieve(url, output_path, report_progress)
        print("\n✓ Download complete!")
        
    except urllib.error.URLError as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


def validate_dataset(path: Path) -> bool:
    """
    Validate that the downloaded file is valid ShareGPT JSON.
    
    Args:
        path: Path to the dataset file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        print(f"\nValidating dataset...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            num_convs = len(data)
            if num_convs > 0:
                # Check first conversation has expected format
                conv = data[0]
                if 'conversations' in conv or 'messages' in conv:
                    print(f"✓ Valid ShareGPT format")
                    print(f"✓ Found {num_convs:,} conversations")
                    return True
        
        print("✗ Invalid format: Expected list of conversations")
        return False
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def create_sample(input_path: Path, output_path: Path,
                  sample_size: int = 100):
    """
    Create a smaller sample from a large dataset.
    
    Args:
        input_path: Path to full dataset
        output_path: Path to save sample
        sample_size: Number of conversations to include
    """
    import random
    
    print(f"\nCreating sample with {sample_size} conversations...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            sample = random.sample(data, min(sample_size, len(data)))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Sample saved to: {output_path}")
            print(f"✓ Contains {len(sample)} conversations")
        
    except Exception as e:
        print(f"✗ Error creating sample: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download ShareGPT dataset for vLLM simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  sg-sample : ShareGPT sample (~1K conversations, ~10MB)
  sg90k     : ShareGPT 90K full dataset (~90K conversations, ~800MB)

Examples:
  # Download sample dataset (recommended for testing)
  python3 download_sharegpt.py sg-sample

  # Download full dataset
  python3 download_sharegpt.py sg90k

  # Create a smaller sample from full dataset
  python3 download_sharegpt.py sg90k --sample 1000

  # Custom output location
  python3 download_sharegpt.py sg-sample --output my_dataset.json
        """
    )
    
    parser.add_argument(
        'dataset',
        choices=list(DATASETS.keys()),
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: data/<dataset_name>.json)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Create a sample with N conversations after downloading'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force download even if file exists'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to data/ directory
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / f"{args.dataset}.json"
    
    # Check if file exists
    if output_path.exists() and not args.force:
        print(f"File already exists: {output_path}")
        print("Use --force to download again")
        sys.exit(0)
    
    # Show dataset info
    dataset_info = DATASETS[args.dataset]
    print("="*70)
    print(f"Dataset: {dataset_info['name']}")
    print(f"Size: {dataset_info['size']}")
    print(f"Conversations: {dataset_info['conversations']}")
    print("="*70)
    print()
    
    # Download
    download_file(dataset_info['url'], output_path)
    
    # Validate
    if not validate_dataset(output_path):
        print("\n✗ Dataset validation failed!")
        sys.exit(1)
    
    # Create sample if requested
    if args.sample:
        sample_path = output_path.with_name(
            f"{output_path.stem}_sample_{args.sample}.json"
        )
        create_sample(output_path, sample_path, args.sample)
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nDataset ready at: {output_path}")
    print(f"\nTo use in simulator:")
    print(f"  ./src/multi_user_simulator.py --text-mode \\")
    print(f"    --dataset-path {output_path}")
    print()


if __name__ == "__main__":
    main()

