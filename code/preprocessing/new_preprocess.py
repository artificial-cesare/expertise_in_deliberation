#!/usr/bin/env python3
"""
Optimized Speech Extraction Tool for Citizens' Assembly Transcripts
Single-pass LLM architecture: Raw -> LLM Clean -> Regex Extract -> CSV.
"""

import os
import re
import csv
import sys
import argparse
import json
from time import sleep
from pathlib import Path
from openai import OpenAI
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional dependencies
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# ============================================================================
# Text Processing & Cleaning
# ============================================================================

def safe_fix_spacing(text):
    """
    Basic spacing fixes that won't break names like 'McDonald'.
    """
    # Fix spacing between letters and numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # Collapse multiple spaces/tabs
    return re.sub(r'[ \t]+', ' ', text).strip()

def split_text_with_overlap(text, chunk_size=8000, overlap=500):
    """
    Splits text into chunks with overlap to preserve context across boundaries.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # If we are not at the end, try to break at a paragraph or newline
        if end < len(text):
            # Look for last newline in the window
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1 and last_newline > start + (chunk_size // 2):
                end = last_newline + 1
        
        chunks.append(text[start:end])
        start = end - overlap # Move pointer back by overlap amount
        
    return chunks

# ============================================================================
# Core Logic: LLM Cleaning + Regex Extraction
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def clean_transcript_segment(client, text, model="gpt-4o-mini"):
    """
    Uses LLM to standardize text format.
    Enforces '### SPEAKER NAME:' format for easy regex parsing later.
    """
    prompt = """You are a transcript cleaner. 
Input is raw text from PDF/OCR. Output CLEAN, STRUCTURED text.

RULES:
1. Format ALL speaker changes exactly as: ### SPEAKER NAME:
2. Start every speech on a new line after the speaker marker.
3. Fix OCR errors (e.g., "Facilit ator" -> "Facilitator").
4. Remove headers, footers, page numbers, and non-speech text.
5. DO NOT summarize. Keep exact speech content.
6. If a speech is cut off at the start/end, just transcribe what is there.
7. Extract the date of the meeting if mentioned at start, do NOT add otherwise, return it at the beginning as: DATE: DD/MM/YYYY

EXAMPLE:
Input: "
commitee meeting fifteenth November 2020 
CHAIR PERSON \n
Welcome. \n
MR. SMITH: Hi."

Output:
DATE: 15/11/2020
### CHAIRPERSON:
Welcome.
### MR. SMITH:
Hi.

TEXT TO CLEAN:
""" + text

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Deterministic
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ⚠️ API Error: {e}")
        return text # Fail gracefully by returning original

def parse_structured_text(cleaned_text):
    """
    Extracts speeches using Regex from the LLM-standardized text.
    Replaces the second LLM pass.
    """
    speeches = []
    
    # Regex finds: ### SPEAKER NAME: (Group 1), followed by text (Group 2)
    # Flags: Dotall so . matches newlines
    pattern = re.compile(r'###\s*(.*?):\s*(.*?)(?=(?:###|$))', re.DOTALL)
    
    matches = pattern.findall(cleaned_text)
    
    for speaker, content in matches:
        speaker = speaker.strip().upper()
        # Remove internal newlines in speech for CSV cleanliness, or keep them
        content = re.sub(r'\n+', ' ', content).strip()
        
        if speaker and content:
            speeches.append(("", speaker, content)) # Date is added later
            
    return speeches

# ============================================================================
# Orchestration
# ============================================================================

def process_file(file_path, client, model="gpt-4o-mini"):
    """
    Reads file -> Chunks -> LLM Clean -> Regex Parse -> Merge.
    """
    ext = file_path.suffix.lower()
    
    # 1. Read Text
    try:
        if ext == '.pdf':
            reader = PyPDF2.PdfReader(file_path)
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == '.rtf':
            if not rtf_to_text: raise ImportError("pip install striprtf")
            with open(file_path, 'r', errors='replace') as f:
                full_text = rtf_to_text(f.read())
        elif ext in ['.html', '.htm']:
            if not BeautifulSoup: raise ImportError("pip install beautifulsoup4")
            with open(file_path, 'r', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                full_text = soup.get_text(separator='\n')
        else:
            print(f"  ⚠️ Unsupported file type: {ext}")
            return []
            
        full_text = safe_fix_spacing(full_text)
    except Exception as e:
        print(f"  ❌ Read Error: {e}")
        return []

    if not full_text.strip(): return []

    # 2. Chunking (with overlap)
    chunks = split_text_with_overlap(full_text)
    print(f"  Processing {len(chunks)} chunks...", end='', flush=True)

    all_speeches = []
    
    # 3. Process Chunks
    for i, chunk in enumerate(chunks):
        cleaned = clean_transcript_segment(client, chunk, model)
        chunk_speeches = parse_structured_text(cleaned)
        all_speeches.extend(chunk_speeches)
        print(".", end='', flush=True)
        sleep(0.5) # Rate limit safety
    
    print(" Done.")

    # 4. Merge Duplicates & Fix Boundaries
    # Because of overlap, we might have duplicate speeches or split speeches.
    # Simple logic: If speaker is same as previous, append text.
    if not all_speeches:
        return []
    
    merged = []
    curr_speaker, curr_text = all_speeches[0][1], all_speeches[0][2]
    
    for _, speaker, text in all_speeches[1:]:
        if speaker == curr_speaker:
            # Skip if text is substring of current (exact duplicate)
            if text in curr_text:
                continue
            # Skip if current text ends with start of new text (overlap)
            if curr_text in text:
                curr_text = text  # New text contains old, use new
                continue
            # Check for partial overlap at boundary
            overlap_found = False
            for overlap_len in range(min(len(curr_text), len(text), 200), 10, -1):
                if curr_text.endswith(text[:overlap_len]):
                    curr_text += text[overlap_len:]  # Append only non-overlapping part
                    overlap_found = True
                    break
            if not overlap_found:
                curr_text += " " + text
        else:
            merged.append(("", curr_speaker, curr_text))
            curr_speaker, curr_text = speaker, text
    
    merged.append(("", curr_speaker, curr_text))
    return merged

# ============================================================================
# Main
# ============================================================================

# Replace the argument parser section (around line 207):

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='data/assemblies/raw')
    parser.add_argument('--output-dir', default='data/assemblies/new_processed_files')
    parser.add_argument('--folders', nargs='+', default=['8th_amendment', 'gendereq', 'scotland', 'experts'])
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--limit', type=int, default=None, help='Process only N files per folder (for testing)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without running')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already have output CSVs')
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("❌ Missing OPENAI_API_KEY")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be processed\n")
        client = None
    else:
        client = OpenAI(api_key=api_key)
    
    total_to_process = 0
    total_skipped = 0
    
    for folder in args.folders:
        input_path = Path(args.base_dir) / folder
        output_path = Path(args.output_dir) / folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            print(f"⚠️  {folder}: Directory not found")
            continue
        
        files = list(input_path.glob('**/*'))
        supported = ['.pdf', '.rtf', '.html', '.htm']
        files = [f for f in files if f.suffix.lower() in supported]
        
        # Apply limit for testing
        if args.limit:
            files = files[:args.limit]

        print(f"\n📂 {folder}: Found {len(files)} files")

        for file_path in files:
            csv_path = output_path / f"new_{file_path.stem}_speeches.csv"
            
            # Skip existing files if flag set
            if args.skip_existing and csv_path.exists():
                print(f"  ⏭️  {file_path.name} (already exists)")
                total_skipped += 1
                continue
            
            if args.dry_run:
                print(f"  📄 Would process: {file_path.name} -> {csv_path.name}")
                total_to_process += 1
                continue
            
            print(f"📄 {file_path.name}")
            
            speeches = process_file(file_path, client, model=args.model)
            
            if speeches:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['date', 'speaker', 'text'])
                    writer.writerows(speeches)
                print(f"  ✅ Saved {len(speeches)} speeches to CSV")
                total_to_process += 1
            else:
                print("  ⚠️ No speeches extracted")
    
    if args.dry_run:
        print(f"\n{'='*50}")
        print(f"DRY RUN SUMMARY: {total_to_process} files would be processed")
        if args.skip_existing:
            print(f"                 {total_skipped} files would be skipped (existing)")

if __name__ == "__main__":
    main()