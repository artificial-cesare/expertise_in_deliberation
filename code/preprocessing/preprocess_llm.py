#!/usr/bin/env python3
"""
Multi-Format Speech Extraction Tool for Citizens' Assembly Transcripts
Extracts speaker speeches from PDF, RTF, and HTML files using two-pass LLM processing.
"""

import os
import re
import csv
import sys
import PyPDF2
from pathlib import Path
from time import sleep
import argparse
import json

# ============================================================================
# Text Chunking Functions
# ============================================================================

def split_text_into_chunks(text, max_chars=10000):
    """
    Split text into chunks that fit within token limits.
    Tries to split at paragraph boundaries.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (~2500 tokens at 4:1 ratio)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')  # Split on double newlines for better paragraph detection
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds limit, save current chunk and start new one
        if len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If any chunk is still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split very large chunks by sentences
            sentences = re.split(r'([.!?]\s+)', chunk)
            sub_chunk = ""
            for i in range(0, len(sentences), 2):
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                if len(sub_chunk) + len(sentence) > max_chars and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sentence
                else:
                    sub_chunk += sentence
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
    
    return final_chunks

def split_by_speech_acts(text, max_chars=12000):
    """
    Split cleaned text into chunks by complete speech acts.
    Ensures no speech is split across chunks.
    
    Args:
        text: Cleaned text with speaker labels (SPEAKER:)
        max_chars: Maximum characters per chunk (~3000 tokens at 4:1 ratio)
    
    Returns:
        List of text chunks, each containing complete speeches
    """
    # Split by speaker pattern (SPEAKER_NAME:)
    # Pattern: uppercase words, numbers, spaces, hyphens, periods, followed by colon
    speaker_pattern = r'\n([A-Z][A-Z0-9\s\.\-]+:)'
    
    # Find all speaker positions
    matches = list(re.finditer(speaker_pattern, text))
    
    if not matches:
        # No speakers found, fall back to regular chunking
        return split_text_into_chunks(text, max_chars)
    
    chunks = []
    current_chunk = ""
    
    # Add text before first speaker if any
    first_match_start = matches[0].start()
    if first_match_start > 0:
        current_chunk = text[:first_match_start].strip()
    
    # Process each speech act
    for i, match in enumerate(matches):
        # Get speech text (from this speaker to next speaker or end)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        speech = text[start:end].strip()
        
        # If adding this speech exceeds limit
        if len(current_chunk) + len(speech) + 1 > max_chars and current_chunk:
            # Save current chunk and start new one with this speech
            chunks.append(current_chunk.strip())
            current_chunk = speech
        else:
            # Add speech to current chunk
            current_chunk += "\n" + speech if current_chunk else speech
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ============================================================================
# LLM Processing Functions
# ============================================================================

def clean_and_structure_text_with_llm(text, api_key, model="gpt-4o-mini", delay=0.5):
    """
    Clean extracted text and structure it for speech extraction using LLM.
    Processes text in chunks to handle long documents.
    
    Args:
        text: Raw text to clean and structure
        api_key: API key for the LLM service
        model: Model to use (default: gpt-4o-mini for cost efficiency)
        delay: Delay between API calls in seconds
    
    Returns:
        Cleaned and structured text with properly formatted speaker labels
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Split text into manageable chunks
        chunks = split_text_into_chunks(text, max_chars=10000)
        print(f"  Pass 1: Cleaning {len(chunks)} chunk(s)...")
        
        cleaned_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  Cleaning chunk {i}/{len(chunks)}...", end='\r')
            
            prompt = f"""Clean and structure this Citizens' Assembly transcript excerpt.

TASKS:
1. Fix spacing errors: "Facilit ator" → "Facilitator"
2. Fix OCR errors and typos
3. Standardize ALL speaker labels to: "SPEAKER_NAME:" (all caps + colon)
4. Put each speaker turn on a NEW LINE, do not break speeches
5. Remove page numbers, headers, footers
6. DO NOT modify speech content

SPEAKER LABEL EXAMPLES:
"MS. FINEGAN:", "CHAIRPERSON:", "TABLE 1 - FACILITATOR:", "Facilitator 1:", "Chair" (alone), "Catherine Day -" → "CATHERINE DAY:"
"A. DR DUNN:" → "DR DUNN:"
"Q. TABLE 4 - MICHELLE:" → "TABLE 4 - MICHELLE:"
"B 1 -" (not a speaker) → leave as-is
"House -" (not a speaker) → leave as-is

{f"Part {i}/{len(chunks)} - Continue from previous part." if i > 1 else ""}

Text:
{chunk}"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You clean transcripts: fix OCR errors, standardize speaker labels, preserve speech content exactly."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=12000
            )
            
            cleaned_chunk = response.choices[0].message.content.strip()
            cleaned_chunks.append(cleaned_chunk)
            
            # Delay between chunks
            if i < len(chunks):
                sleep(delay)
        
        print()  # New line after progress updates
        
        # Combine cleaned chunks
        cleaned_text = "\n\n".join(cleaned_chunks)
        return cleaned_text
    
    except Exception as e:
        print(f"\n  ⚠️  Cleaning API call failed: {str(e)}")
        return text  # Return original text if API fails

def extract_speeches_with_llm(cleaned_text, api_key, model="gpt-4o-mini", delay=0.5):
    """
    Extract structured speech data from cleaned text using speech-act based chunking.
    
    Args:
        cleaned_text: Pre-cleaned text with standardized speaker labels
        api_key: API key for the LLM service
        model: Model to use
        delay: Delay between API calls in seconds
    
    Returns:
        List of [date, speaker, text] entries
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Split by complete speech acts
        chunks = split_by_speech_acts(cleaned_text, max_chars=12000)
        print(f"  Pass 2: Extracting from {len(chunks)} speech-act chunk(s)...")
        
        all_speeches = []
        date_found = ""
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  Extracting chunk {i}/{len(chunks)}...", end='\r')
            
            prompt = f"""Extract all speeches from this transcript excerpt.

INSTRUCTIONS:
1. {"Find the meeting date (format: 'DD Month YYYY'). Return empty string if not found." if i == 1 else f"Use date: '{date_found}'"}
2. For EACH speaker turn (format: SPEAKER_NAME:), extract:
   - Speaker name (without colon)
   - Complete speech text (everything until next speaker)
   - same speakers may appear multiple times, keep the speeches separate if not consecutive
3. Return JSON:
   {{
     "date": "DD Month YYYY or empty",
     "speeches": [
       {{"speaker": "CHAIRPERSON", "text": "Welcome..."}},
       {{"speaker": "MS. FINEGAN", "text": "Thank you..."}}
     ]
   }}

RULES:
- Extract EVERY speaker turn in order
- Keep speaker names exactly as written (all caps, no colon)
- Preserve exact wording of speeches
- Skip non-speech content (headers, page numbers)
- Combine multi-line speeches into single text field
- Return ONLY valid JSON

{f"Part {i}/{len(chunks)}" if len(chunks) > 1 else ""}

Transcript:
{chunk}"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract speech data from transcripts and return valid JSON. You preserve exact wording."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=14000,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(result)
                
                # Extract date from first chunk only
                if i == 1 and "date" in data and data["date"] and not date_found:
                    date_found = data["date"]
                
                # Extract speeches
                speeches = data.get("speeches", [])
                if not speeches and isinstance(data, list):
                    speeches = data
                
                # Convert to list format
                for entry in speeches:
                    speaker = entry.get("speaker", "UNKNOWN").strip()
                    text = entry.get("text", "").strip()
                    
                    if speaker and text:
                        # Add date only to very first speech
                        speech_date = date_found if len(all_speeches) == 0 else ""
                        all_speeches.append([speech_date, speaker, text])
            
            except json.JSONDecodeError as e:
                print(f"\n  ⚠️  JSON parse error in chunk {i}: {e}")
                print(f"  Response preview: {result[:300]}...")
            
            # Delay between chunks
            if i < len(chunks):
                sleep(delay)
        
        print()  # New line after progress
        return all_speeches
    
    except Exception as e:
        print(f"\n  ⚠️  Extraction API call failed: {str(e)}")
        return []

# ============================================================================
# Post-Processing Functions
# ============================================================================

def merge_consecutive_speakers(speeches):
    """
    Merge consecutive speeches from the same speaker.
    Handles cases where speech acts were incorrectly split.
    
    Args:
        speeches: List of [date, speaker, text]
    
    Returns:
        Merged list of [date, speaker, text]
    """
    if not speeches:
        return []
    
    merged = []
    current_date, current_speaker, current_text = speeches[0]
    
    for i in range(1, len(speeches)):
        date, speaker, text = speeches[i]
        
        # Same speaker? Merge the text
        if speaker == current_speaker:
            current_text += " " + text
        else:
            # Different speaker: save previous, start new
            merged.append([current_date, current_speaker, current_text])
            current_date = date if date else ""
            current_speaker = speaker
            current_text = text
    
    # Don't forget the last one
    merged.append([current_date, current_speaker, current_text])
    
    return merged

# ============================================================================
# Text Extraction Functions
# ============================================================================

def fix_spacing(text):
    """Basic spacing fixes for extracted text."""
    # Add space between lowercase and uppercase letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add space between letters and numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # Collapse multiple spaces but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    full_text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = fix_spacing(page_text)
                full_text += page_text + "\n"
    return full_text, len(pdf_reader.pages)

def extract_text_from_rtf(file_path):
    """Extract text from RTF file."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise ImportError("striprtf library required. Install: pip install striprtf")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        rtf_content = file.read()
        full_text = rtf_to_text(rtf_content)
        full_text = fix_spacing(full_text)
    return full_text

def extract_text_from_html(file_path):
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 library required. Install: pip install beautifulsoup4")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, nav, header, footer
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Get text
        full_text = soup.get_text(separator='\n')
        # Clean up whitespace
        lines = (line.strip() for line in full_text.splitlines())
        full_text = '\n'.join(line for line in lines if line)
        full_text = fix_spacing(full_text)
    return full_text

# ============================================================================
# Main Processing Pipeline
# ============================================================================

def extract_speeches_from_file(file_path, output_dir, api_key, model="gpt-4o-mini", delay=0.5):
    """
    Extract speeches from a single file using two-pass LLM processing.
    
    Args:
        file_path: Path to input file
        output_dir: Directory for output CSV and debug files
        api_key: API key for LLM service
        model: Model to use
        delay: Delay between API calls
    
    Returns:
        List of speeches [date, speaker, text]
    """
    speeches = []
    
    try:
        file_ext = file_path.lower()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Step 1: Extract text based on file type
        print(f"  [1/4] Extracting text from {file_ext.split('.')[-1].upper()}...", end=' ')
        
        if file_ext.endswith('.pdf'):
            full_text, num_pages = extract_text_from_pdf(file_path)
            print(f"✓ ({num_pages} pages, {len(full_text):,} chars)")
        elif file_ext.endswith('.rtf'):
            full_text = extract_text_from_rtf(file_path)
            print(f"✓ ({len(full_text):,} chars)")
        elif file_ext.endswith('.html') or file_ext.endswith('.htm'):
            full_text = extract_text_from_html(file_path)
            print(f"✓ ({len(full_text):,} chars)")
        else:
            print(f"\n  ❌ Unsupported file type")
            return speeches
        
        if not full_text.strip():
            print(f"  ❌ No text extracted")
            return speeches
        
        # Save raw text debug file
        debug_raw_file = os.path.join(output_dir, f"{base_name}_debug_raw.txt")
        with open(debug_raw_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # Step 2: Clean and structure text with LLM
        print(f"  [2/4] Cleaning text with LLM...")
        cleaned_text = clean_and_structure_text_with_llm(full_text, api_key, model, delay)
        
        # Save cleaned text debug file
        debug_cleaned_file = os.path.join(output_dir, f"{base_name}_debug_cleaned.txt")
        with open(debug_cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"        Cleaned: {len(cleaned_text):,} chars")
        
        # Step 3: Extract speeches from cleaned text
        print(f"  [3/4] Extracting speeches with LLM...")
        speeches = extract_speeches_with_llm(cleaned_text, api_key, model, delay)
        
        if not speeches:
            print(f"  ⚠️  No speeches extracted")
            return speeches
        
        print(f"        Extracted: {len(speeches)} speeches")
        
        # Step 4: Post-process - merge consecutive same speakers
        print(f"  [4/4] Post-processing...")
        merged_speeches = merge_consecutive_speakers(speeches)
        
        if len(merged_speeches) < len(speeches):
            print(f"        Merged: {len(speeches)} → {len(merged_speeches)} speeches")
        
        # Save debug JSON
        debug_json = os.path.join(output_dir, f"{base_name}_debug_speeches.json")
        with open(debug_json, 'w', encoding='utf-8') as f:
            date_from_speeches = merged_speeches[0][0] if merged_speeches else ""
            json.dump({
                "file": os.path.basename(file_path),
                "date": date_from_speeches,
                "total_speeches": len(merged_speeches),
                "speeches": [
                    {"speaker": s[1], "text": s[2]}
                    for s in merged_speeches
                ]
            }, f, indent=2, ensure_ascii=False)
        
        return merged_speeches
    
    except Exception as e:
        print(f"\n  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return speeches

def parse_to_csv(file_path, output_csv, output_dir, api_key, model="gpt-4o-mini", delay=0.5):
    """
    Parse a file and save speeches to CSV.
    
    Args:
        file_path: Path to input file
        output_csv: Output CSV path
        output_dir: Directory for debug files
        api_key: API key for LLM service
        model: Model to use
        delay: Delay between API calls
    
    Returns:
        Number of speeches extracted
    """
    speeches = extract_speeches_from_file(file_path, output_dir, api_key, model, delay)
    
    if not speeches:
        print(f"  ⚠️  No speeches to save")
        return 0
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'speaker', 'text'])
        writer.writerows(speeches)
    
    # Preview first 3 speeches
    print(f"\n  📊 Preview (first 3 speeches):")
    print(f"  {'─' * 70}")
    for i, (date, speaker, text) in enumerate(speeches[:3], 1):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  {i}. [{date or 'no date'}] {speaker}")
        print(f"     {preview}")
    print(f"  {'─' * 70}")
    
    return len(speeches)

# ============================================================================
# Batch Processing
# ============================================================================

def find_all_files(base_dir, extensions):
    """Recursively find all files with given extensions."""
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_dir)
                all_files.append((full_path, rel_path, file))
    return all_files

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract speeches from Citizens' Assembly transcripts (PDF/RTF/HTML)"
    )
    parser.add_argument(
        '--base-dir',
        default='data/assemblies/raw',
        help='Base directory with files (default: data/assemblies/raw)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/assemblies/processed_files',
        help='Output directory for CSVs (default: data/assemblies/processed_files)'
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        default=['8th_amendment', 'gendereq', 'expert_htmls'],
        help='Folders to process (default: 8th_amendment gendereq expert_htmls)'
    )
    parser.add_argument(
        '--file',
        help='Process only this specific file'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='OpenAI model (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check API key
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Print header
    print("=" * 80)
    print("MULTI-FORMAT SPEECH EXTRACTION TOOL")
    print("=" * 80)
    print(f"Base directory:   {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model:            {args.model}")
    print(f"Folders:          {', '.join(args.folders)}")
    if args.file:
        print(f"Single file:      {args.file}")
    print(f"API delay:        {args.delay}s")
    print("=" * 80)
    
    # Stats
    total_files = 0
    total_speeches = 0
    failed_files = []
    
    SUPPORTED_EXTENSIONS = ('.pdf', '.rtf', '.html', '.htm')
    
    # Process each folder
    for folder_name in args.folders:
        folder_path = os.path.join(args.base_dir, folder_name)
        
        print(f"\n{'═' * 80}")
        print(f"📁 FOLDER: {folder_name}")
        print(f"{'═' * 80}")
        
        if not os.path.isdir(folder_path):
            print(f"⚠️  Directory not found: {folder_path}")
            continue
        
        # Find all files
        all_files = find_all_files(folder_path, SUPPORTED_EXTENSIONS)
        
        # Filter to single file if specified
        if args.file:
            all_files = [(p, r, n) for p, r, n in all_files if n == args.file]
            if not all_files:
                print(f"⚠️  File '{args.file}' not found")
                continue
        
        if not all_files:
            print(f"⚠️  No supported files found")
            continue
        
        print(f"Found {len(all_files)} file(s)\n")
        
        # Process files
        for file_path, rel_path, file_name in all_files:
            # Create output directory (preserve structure)
            # rel_path is relative to folder_path, so we need to prepend folder_name
            if rel_path == '.':
                # File is directly in folder_name
                full_rel_path = folder_name
            else:
                # File is in a subfolder
                full_rel_path = os.path.join(folder_name, rel_path)
            
            output_subdir = os.path.join(args.output_dir, full_rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Output paths
            base_name = os.path.splitext(file_name)[0]
            output_csv = os.path.join(output_subdir, f"{base_name}_speeches.csv")
            
            print(f"\n📄 {file_name}")
            print(f"   {'─' * 70}")
            
            try:
                num_speeches = parse_to_csv(
                    file_path,
                    output_csv,
                    output_subdir,
                    API_KEY,
                    args.model,
                    args.delay
                )
                
                if num_speeches > 0:
                    total_files += 1
                    total_speeches += num_speeches
                    print(f"\n   ✅ Saved: {output_csv}")
                else:
                    failed_files.append(file_name)
                    print(f"\n   ⚠️  No output generated")
            
            except Exception as e:
                print(f"\n   ❌ Failed: {e}")
                failed_files.append(file_name)
            
            print()
    
    # Summary
    print(f"{'═' * 80}")
    print("SUMMARY")
    print(f"{'═' * 80}")
    print(f"✅ Processed: {total_files} files")
    print(f"📊 Speeches:  {total_speeches} total")
    if failed_files:
        print(f"❌ Failed:    {len(failed_files)} files")
        for f in failed_files[:5]:
            print(f"   - {f}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    print(f"\n💾 Output: {args.output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()