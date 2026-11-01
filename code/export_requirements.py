"""
Export current conda/pip environment to requirements.txt
Filters out any lines that might contain API keys or secrets.
"""

import subprocess
import re
import sys

def export_requirements(output_file="qwen3_requirements.txt"):
    # Patterns that might indicate sensitive information
    sensitive_patterns = [
        r'api[-_]?key',
        r'secret',
        r'token',
        r'credential',
        r'password',
        r'auth',
    ]
    
    # Get pip freeze output from the interpreter running this script
    # (a bare 'pip' can resolve to a different environment, or be missing)
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'freeze'],
        capture_output=True,
        text=True
    )
    
    lines = result.stdout.strip().split('\n')
    
    # Filter out sensitive lines and editable installs with local paths
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines matching sensitive patterns
        is_sensitive = any(
            re.search(pattern, line, re.IGNORECASE) 
            for pattern in sensitive_patterns
        )
        
        # Skip editable installs that might expose local paths
        if line.startswith('-e') and '/home/' in line:
            continue
        if line.startswith('-e') and '/Users/' in line:
            continue
            
        if not is_sensitive:
            filtered_lines.append(line)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(filtered_lines))
    
    print(f"✅ Saved {len(filtered_lines)} packages to {output_file}")
    print(f"   (Filtered out {len(lines) - len(filtered_lines)} potentially sensitive entries)")

if __name__ == "__main__":
    export_requirements()