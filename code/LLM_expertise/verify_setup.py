#!/usr/bin/env python3
"""
Verify Setup for Context Size Calibration
Checks dependencies, data availability, and configuration.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required Python packages are installed."""
    print("\n[1/5] Checking Python dependencies...")

    required = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'tiktoken',
        'openai',
        'pingouin'
    ]

    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)

    if missing:
        print(f"\n  ERROR: Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False

    print("  All dependencies installed!")
    return True


def check_api_key():
    """Check if OpenAI API key is set."""
    print("\n[2/5] Checking OpenAI API key...")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("  ✗ OPENAI_API_KEY not set")
        print("  Set with: export OPENAI_API_KEY='your-key-here'")
        return False

    print(f"  ✓ API key found (length: {len(api_key)} characters)")
    return True


def check_data_directory(base_dir: str):
    """Check if data directory and CSV files exist."""
    print(f"\n[3/5] Checking data directory: {base_dir}...")

    if not os.path.exists(base_dir):
        print(f"  ✗ Directory not found: {base_dir}")
        return False

    assemblies = ["8th_amendment", "gendereq", "scotland"]
    found_files = {}

    for assembly in assemblies:
        assembly_path = os.path.join(base_dir, assembly)

        if not os.path.exists(assembly_path):
            print(f"  ✗ Assembly folder not found: {assembly}")
            found_files[assembly] = 0
            continue

        csv_files = [f for f in os.listdir(assembly_path) if f.endswith('.csv')]
        found_files[assembly] = len(csv_files)

        print(f"  ✓ {assembly}: {len(csv_files)} CSV files")

    total_files = sum(found_files.values())

    if total_files == 0:
        print("  ✗ No CSV files found!")
        return False

    print(f"  Total CSV files: {total_files}")
    return True


def check_csv_structure(base_dir: str):
    """Check if CSV files have required columns."""
    print(f"\n[4/5] Checking CSV structure...")

    import csv

    required_cols = ['speaker', 'text']
    recommended_cols = ['num_non_function_words', 'date']

    assemblies = ["8th_amendment", "gendereq", "scotland"]

    for assembly in assemblies:
        assembly_path = os.path.join(base_dir, assembly)

        if not os.path.exists(assembly_path):
            continue

        csv_files = [f for f in os.listdir(assembly_path) if f.endswith('.csv')]

        if not csv_files:
            continue

        # Check first file
        sample_file = os.path.join(assembly_path, csv_files[0])

        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = list(reader.fieldnames)

            missing_required = [col for col in required_cols if col not in headers]
            missing_recommended = [col for col in recommended_cols if col not in headers]

            if missing_required:
                print(f"  ✗ {assembly}: Missing required columns: {', '.join(missing_required)}")
                return False

            if missing_recommended:
                print(f"  ⚠ {assembly}: Missing recommended columns: {', '.join(missing_recommended)}")
            else:
                print(f"  ✓ {assembly}: All columns present")

        except Exception as e:
            print(f"  ✗ {assembly}: Error reading CSV: {e}")
            return False

    print("  CSV structure verified!")
    return True


def test_tokenizer():
    """Test tiktoken tokenizer."""
    print("\n[5/5] Testing tokenizer...")

    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        test_text = "This is a test sentence to verify tokenization."
        tokens = encoding.encode(test_text)

        print(f"  ✓ Tokenizer working")
        print(f"  Test: \"{test_text}\"")
        print(f"  Tokens: {len(tokens)}")

        return True

    except Exception as e:
        print(f"  ✗ Tokenizer error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("="*80)
    print("CONTEXT SIZE CALIBRATION: SETUP VERIFICATION")
    print("="*80)

    # Default data directory (relative to this script)
    script_dir = Path(__file__).parent
    default_base_dir = script_dir / "../../data/assemblies/processed_files"
    default_base_dir = default_base_dir.resolve()

    checks = [
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Data Directory", lambda: check_data_directory(str(default_base_dir))),
        ("CSV Structure", lambda: check_csv_structure(str(default_base_dir))),
        ("Tokenizer", test_tokenizer),
    ]

    results = {}

    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    all_passed = True

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✓ All checks passed! You're ready to run the calibration experiment.")
        print("\nNext steps:")
        print("  1. Review configuration in context_size_calibration.py")
        print("  2. Run: ./run_calibration.sh")
        print("     OR")
        print("     python context_size_calibration.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above before running.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements_calibration.txt")
        print("  - Set API key: export OPENAI_API_KEY='your-key-here'")
        print("  - Check data path: adjust base_dir in script if needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
