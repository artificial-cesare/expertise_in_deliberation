#!/usr/bin/env python3
"""
Expertise Scoring Tool for Citizens' Assembly Transcripts
Evaluates expertise level of speech acts using LLM analysis.
"""

import os
import csv
import sys
import argparse
from time import sleep
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'handcrafted_features'))
from filter_function import is_substantive_speech_act
from llm_utils import extract_topic, score_expertise_with_llm

def load_csv_data(csv_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load CSV file and return headers and rows as dictionaries.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (headers list, rows as list of dicts)
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames)  # Convert to list
            rows = list(reader)
        return headers, rows
    except Exception as e:
        print(f"Error loading {csv_path}: {str(e)}")
        return None, None

def get_text_from_row(row: Dict) -> str:
    """
    Extract text from row - tries 'text' column first, then 'comment'.
    
    Args:
        row: Dictionary representing a CSV row
    
    Returns:
        Text content or empty string
    """
    # Check both 'text' and 'comment' columns
    text = row.get('text', '').strip()
    if not text:
        text = row.get('comment', '').strip()
    return text

def get_context_window(rows: List[Dict], current_idx: int, window_size: int = 20) -> str:
    """
    Get previous N speech acts as context.
    
    Args:
        rows: List of all rows
        current_idx: Index of current row
        window_size: Number of previous rows to include
    
    Returns:
        Formatted context string
    """
    start_idx = max(0, current_idx - window_size)
    context_rows = rows[start_idx:current_idx]
    
    context_parts = []
    for i, row in enumerate(context_rows, start=start_idx + 1):
        # Try different possible speaker column names
        speaker = row.get('speaker', row.get('name', row.get('Speaker', 'Unknown')))
        text = get_text_from_row(row)
        if text:
            context_parts.append(f"[{i}] {speaker}: {text}")
    
    return "\n".join(context_parts) if context_parts else "No previous context available."

def process_csv_file(
    csv_path: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    rate_limit_delay: float = 0.5,
    force_rescore: bool = False,
    content_word_threshold: int = 5
) -> bool:
    """
    Process a single CSV file and add expertise scores.
    Only scores substantive speech acts based on content word count.
    
    Args:
        csv_path: Path to CSV file
        api_key: OpenAI API key
        model: Model to use
        rate_limit_delay: Delay between API calls in seconds
        force_rescore: If True, rescore even if already scored
        content_word_threshold: Minimum content words required for scoring
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Processing: {csv_path}")
    
    # Load CSV
    headers, rows = load_csv_data(csv_path)
    if headers is None or rows is None:
        return False
    
    # Check if num_non_function_words column exists
    if 'num_non_function_words' not in headers:
        print(f"  ✗ Error: 'num_non_function_words' column not found. Run extract_expert_features.py first.\n")
        return False
    
    # Add new columns if not present
    if 'expertise_llm_score' not in headers:
        headers.append('expertise_llm_score')
    if 'expertise_justification' not in headers:
        headers.append('expertise_justification')
    
    # Extract topic
    topic = extract_topic(csv_path)
    print(f"  Topic: {topic}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Content word threshold: {content_word_threshold}")
    
    # Process each row
    scored_count = 0
    skipped_count = 0
    filtered_out_count = 0
    
    for idx, row in enumerate(rows):
        # Skip if already scored (unless force_rescore is True)
        if not force_rescore and row.get('expertise_llm_score'):
            skipped_count += 1
            continue
        
        text = get_text_from_row(row)
        
        # Skip empty contributions
        if not text:
            row['expertise_llm_score'] = ''
            row['expertise_justification'] = ''
            continue
        
        # Check if substantive based on content word count
        try:
            num_content_words = int(row.get('num_non_function_words', 0))
        except (ValueError, TypeError):
            num_content_words = 0
        
        if not is_substantive_speech_act(num_content_words, content_word_threshold):
            # Mark as filtered out (NA won't affect statistics)
            row['expertise_llm_score'] = 'NA'
            row['expertise_justification'] = f'Filtered: only {num_content_words} content words'
            filtered_out_count += 1
            continue
        
        # Get context
        context = get_context_window(rows, idx)
        
        # Score with LLM
        print(f"  Scoring row {idx + 1}/{len(rows)}...", end='\r')
        rating, justification = score_expertise_with_llm(
            text, context, topic, api_key, model
        )
        
        if rating is not None:
            row['expertise_llm_score'] = str(rating)
            row['expertise_justification'] = justification
            scored_count += 1
        else:
            row['expertise_llm_score'] = ''
            row['expertise_justification'] = f'Error: {justification}'
        
        # Rate limiting
        sleep(rate_limit_delay)
    
    print()  # New line after progress
    
    if skipped_count > 0:
        print(f"  ℹ Skipped {skipped_count} already-scored rows")
    if filtered_out_count > 0:
        print(f"  ℹ Filtered out {filtered_out_count} rows (< {content_word_threshold} content words)")
    
    # Write updated CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  ✓ Successfully scored {scored_count} new rows")
        print(f"  Updated file: {csv_path}\n")
        return True
    
    except Exception as e:
        print(f"  ✗ Error writing file: {str(e)}\n")
        return False

def main():
    """Main function to process CSV files."""
    parser = argparse.ArgumentParser(
        description='Score expertise in Citizens Assembly CSV transcripts using LLM'
    )
    parser.add_argument(
        '--base-dir',
        default='data/assemblies/processed_files/',
        help='Base directory containing CSV folders'
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        default=['8th_amendment', 'gendereq', 'scotland'],
        help='Specific folders to process (default: 8th_amendment gendereq scotland_csv)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--force-rescore',
        action='store_true',
        help='Re-score all rows even if already scored'
    )
    parser.add_argument(
        '--content-threshold',
        type=int,
        default=5,
        help='Minimum content words required for scoring (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"{'='*80}")
    print(f"Expertise Scoring for Citizens' Assembly Transcripts")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Base directory: {args.base_dir}")
    print(f"Folders to process: {', '.join(args.folders)}")
    print(f"Rate limit delay: {args.delay}s")
    print(f"Force re-score: {args.force_rescore}")
    print(f"Content word threshold: {args.content_threshold}")
    print(f"{'='*80}\n")
    
    total_files = 0
    successful_files = 0
    
    for folder in args.folders:
        folder_path = os.path.join(args.base_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"⚠ Folder not found: {folder_path}\n")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing folder: {folder}")
        print(f"{'='*80}\n")
        
        # Get all CSV files
        try:
            csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        except Exception as e:
            print(f"Error reading folder: {str(e)}\n")
            continue
        
        if not csv_files:
            print(f"No CSV files found in {folder}\n")
            continue
        
        print(f"Found {len(csv_files)} CSV file(s)\n")
        
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            total_files += 1
            
            try:
                if process_csv_file(
                    csv_path, 
                    api_key, 
                    args.model, 
                    args.delay, 
                    args.force_rescore,
                    args.content_threshold
                ):
                    successful_files += 1
            except Exception as e:
                print(f"  ✗ Unexpected error: {str(e)}\n")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {total_files}")
    print(f"Successfully updated: {successful_files}")
    print(f"Failed: {total_files - successful_files}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
