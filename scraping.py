#!/usr/bin/env python3
"""
Git patch extractor that generates agentic tool calls for find-replace operations.
Designed specifically for AI coding agents.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error

def run_cmd(cmd, cwd=None, timeout=60):
    """Run a git command and return the output"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"⚠️ Command failed: {' '.join(cmd)}")
            print(f"   Error: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⚠️ Command timed out: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"⚠️ Command error: {' '.join(cmd)} - {e}")
        return None

def clone_repository(repo_url, target_dir):
    """Clone a git repository"""
    print(f"🔄 Cloning repository: {repo_url}")
    
    if target_dir.exists():
        print(f"📁 Repository already exists at {target_dir}")
        return target_dir
    
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["git", "clone", repo_url, str(target_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"❌ Failed to clone: {result.stderr}")
        sys.exit(1)
    
    print(f"✅ Cloned to: {target_dir}")
    return target_dir

def get_commits(repo_path, max_commits=500):
    """Get commit information from the repository"""
    print(f"📝 Getting commits from {repo_path}")
    
    cmd = [
        "git", "log",
        f"--max-count={max_commits}",
        "--pretty=format:%H|%an|%ae|%ad|%s",
        "--date=iso"
    ]
    
    output = run_cmd(cmd, cwd=repo_path)
    if not output:
        print("❌ No commits found")
        return []
    
    commits = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        
        parts = line.split('|', 4)
        if len(parts) >= 5:
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "email": parts[2], 
                "date": parts[3],
                "message": parts[4]
            })
    
    print(f"✅ Found {len(commits)} commits")
    return commits

def get_patch_for_commit(repo_path, commit_hash):
    """Get the patch for a specific commit"""
    cmd = ["git", "show", "--no-merges", commit_hash]
    return run_cmd(cmd, cwd=repo_path)

def parse_patch_to_tool_calls(patch_content, commit_hash):
    """Parse a git patch into find-replace tool calls"""
    if not patch_content:
        return []
    
    tool_calls = []
    lines = patch_content.split('\n')
    current_file = None
    old_lines = []
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('diff --git'):
            # Save previous file changes
            if current_file and (old_lines or new_lines):
                if old_lines or new_lines:  # Only add if there are actual changes
                    tool_calls.append({
                        "file": current_file,
                        "old_str": '\n'.join(old_lines),
                        "new_str": '\n'.join(new_lines),
                        "commit": commit_hash
                    })
            
            # Parse new file path
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:]  # Remove 'b/' prefix
            old_lines = []
            new_lines = []
            
        elif line.startswith('+++') or line.startswith('---'):
            # Skip file headers
            pass
            
        elif line.startswith('@@'):
            # End of previous hunk, start of new hunk
            if old_lines or new_lines:
                tool_calls.append({
                    "file": current_file or "unknown",
                    "old_str": '\n'.join(old_lines),
                    "new_str": '\n'.join(new_lines), 
                    "commit": commit_hash
                })
                old_lines = []
                new_lines = []
                
        elif current_file and (line.startswith(' ') or line.startswith('-') or line.startswith('+')):
            content = line[1:] if len(line) > 0 else ""
            
            if line.startswith(' '):
                # Context line - appears in both old and new
                old_lines.append(content)
                new_lines.append(content)
            elif line.startswith('-'):
                # Removed line - only in old
                old_lines.append(content)
            elif line.startswith('+'):
                # Added line - only in new  
                new_lines.append(content)
        
        i += 1
    
    # Handle final file
    if current_file and (old_lines or new_lines):
        tool_calls.append({
            "file": current_file,
            "old_str": '\n'.join(old_lines),
            "new_str": '\n'.join(new_lines),
            "commit": commit_hash
        })
    
    # Filter out empty changes
    filtered_calls = []
    for tc in tool_calls:
        if tc["old_str"].strip() or tc["new_str"].strip():
            # Only include if there's actual content change
            if tc["old_str"] != tc["new_str"]:
                filtered_calls.append(tc)
    
    return filtered_calls

def create_agentic_file(commit, tool_calls, output_dir):
    """Create an agentic coding file with tool calls"""
    commit_hash = commit["hash"]
    
    # Create subdirectory based on first 2 chars of hash
    subdir = output_dir / commit_hash[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    
    # Build the agentic content
    content_lines = []
    content_lines.append(f"COMMIT: {commit_hash}")
    content_lines.append(f"AUTHOR: {commit['author']} <{commit['email']}>")
    content_lines.append(f"DATE: {commit['date']}")
    content_lines.append(f"MESSAGE: {commit['message']}")
    content_lines.append("")
    
    if tool_calls:
        content_lines.append("=== AGENTIC_TOOL_CALLS ===")
        for i, tc in enumerate(tool_calls, 1):
            content_lines.append(f"TOOL_CALL_{i}:")
            content_lines.append(f"FILE: {tc['file']}")
            content_lines.append("ACTION: find_and_replace")
            content_lines.append("OLD_STR:")
            content_lines.append(tc['old_str'])
            content_lines.append("NEW_STR:")
            content_lines.append(tc['new_str'])
            content_lines.append("")
    else:
        content_lines.append("=== NO_TOOL_CALLS ===")
        content_lines.append("This commit contains no actionable find-replace operations")
        content_lines.append("")
    
    content_lines.append("----- END_COMMIT -----")
    
    # Save text file
    text_file = subdir / f"{commit_hash}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))
    
    # Save JSON file for machine processing
    json_file = subdir / f"{commit_hash}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "commit": commit,
            "tool_calls": tool_calls,
            "file_count": len(set(tc["file"] for tc in tool_calls)),
            "change_count": len(tool_calls)
        }, f, indent=2, ensure_ascii=False)
    
    return len(tool_calls)

def process_repository(repo_path, output_dir, max_commits=500, limit=200):
    """Process repository and generate agentic tool calls"""
    print(f"🤖 Processing repository for agentic tool calls...")
    
    # Create agentic output directory
    agentic_dir = output_dir / "agentic_tool_calls"
    agentic_dir.mkdir(parents=True, exist_ok=True)
    
    # Get commits
    commits = get_commits(repo_path, max_commits)
    if not commits:
        return 0, 0
    
    # Process commits and generate tool calls
    processed_commits = 0
    total_tool_calls = 0
    
    for commit in commits[:limit]:
        try:
            commit_hash = commit["hash"]
            print(f"   📝 Processing {commit_hash[:8]}...")
            
            # Get patch for this commit
            patch = get_patch_for_commit(repo_path, commit_hash)
            if not patch:
                print(f"      ⚠️ No patch found")
                continue
            
            # Parse patch into tool calls
            tool_calls = parse_patch_to_tool_calls(patch, commit_hash)
            if not tool_calls:
                print(f"      ⚠️ No tool calls generated")
                continue
            
            print(f"      ✅ Generated {len(tool_calls)} tool calls")
            
            # Create agentic file
            tc_count = create_agentic_file(commit, tool_calls, agentic_dir)
            
            processed_commits += 1
            total_tool_calls += tc_count
            
            if processed_commits % 10 == 0:
                print(f"   📊 Processed {processed_commits} commits, {total_tool_calls} tool calls so far...")
            
        except Exception as e:
            print(f"      ❌ Error processing {commit.get('hash', 'unknown')[:8]}: {e}")
            continue
    
    return processed_commits, total_tool_calls

def create_usage_guide(output_dir, processed_commits, total_tool_calls):
    """Create a usage guide for the generated tool calls"""
    guide_file = output_dir / "AGENTIC_USAGE_GUIDE.md"
    
    guide_content = f"""# Agentic Tool Calls Usage Guide

## Summary
- **Processed commits**: {processed_commits}
- **Total tool calls**: {total_tool_calls}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure
```
agentic_tool_calls/
├── ab/
│   ├── abc123def.txt    # Human-readable format
│   ├── abc123def.json   # Machine-readable format
│   └── ...
├── cd/
│   └── ...
└── AGENTIC_USAGE_GUIDE.md (this file)
```

## Tool Call Format

Each tool call represents a find-and-replace operation:

### Text Format (.txt files):
```
TOOL_CALL_1:
FILE: src/main.py
ACTION: find_and_replace
OLD_STR:
def old_function():
    return "old"
NEW_STR:
def new_function():
    return "new"
```

### JSON Format (.json files):
```json
{{
  "commit": {{
    "hash": "abc123...",
    "author": "Developer Name",
    "email": "dev@example.com",
    "date": "2024-01-01 12:00:00 +0000",
    "message": "Commit message"
  }},
  "tool_calls": [
    {{
      "file": "src/main.py",
      "old_str": "def old_function():\\n    return \\"old\\"",
      "new_str": "def new_function():\\n    return \\"new\\"",
      "commit": "abc123..."
    }}
  ]
}}
```

## Usage in AI Coding Systems

1. **Load tool calls**: Parse JSON files to extract structured operations
2. **Apply changes**: Use find-and-replace operations sequentially
3. **Context awareness**: Use commit info for understanding changes
4. **Validation**: Verify changes before applying

## Example Python Usage

```python
import json
from pathlib import Path

def apply_tool_calls(json_file_path, target_codebase_path):
    with open(json_file_path) as f:
        data = json.load(f)
    
    for tool_call in data['tool_calls']:
        file_path = target_codebase_path / tool_call['file']
        
        if file_path.exists():
            content = file_path.read_text()
            new_content = content.replace(
                tool_call['old_str'], 
                tool_call['new_str']
            )
            file_path.write_text(new_content)
            print(f"Applied change to {{tool_call['file']}}")
```

This format enables AI agents to systematically understand and replay code changes!
"""
    
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 Usage guide created: {guide_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract git patches as agentic tool calls for AI coding systems"
    )
    parser.add_argument("--repo", required=True, 
                       help="Git repository URL or local path")
    parser.add_argument("--output", required=True,
                       help="Output directory for generated files")
    parser.add_argument("--max-commits", type=int, default=500,
                       help="Maximum commits to scan (default: 500)")
    parser.add_argument("--limit", type=int, default=200,
                       help="Maximum commits to process (default: 200)")
    
    args = parser.parse_args()
    
    # Prepare output directory
    output_dir = Path(args.output).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle repository (clone if URL, validate if local path)
    if args.repo.startswith(('http://', 'https://', 'git@')):
        # It's a URL - clone it
        repo_name = args.repo.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        repo_path = output_dir / "repos" / repo_name
        clone_repository(args.repo, repo_path)
    else:
        # It's a local path
        repo_path = Path(args.repo).absolute()
        if not repo_path.exists():
            print(f"❌ Repository path does not exist: {repo_path}")
            sys.exit(1)
        if not (repo_path / ".git").exists():
            print(f"❌ Not a git repository: {repo_path}")
            sys.exit(1)
    
    print(f"🎯 Repository: {repo_path}")
    print(f"📁 Output: {output_dir}")
    
    # Process repository
    processed_commits, total_tool_calls = process_repository(
        repo_path, output_dir, args.max_commits, args.limit
    )
    
    # Create usage guide
    create_usage_guide(output_dir, processed_commits, total_tool_calls)
    
    print(f"\n🎉 Agentic Tool Call Extraction Complete!")
    print(f"   📊 Processed: {processed_commits} commits")
    print(f"   🔧 Generated: {total_tool_calls} tool calls")
    print(f"   📁 Output: {output_dir}/agentic_tool_calls/")
    print(f"   📖 Guide: {output_dir}/AGENTIC_USAGE_GUIDE.md")

if __name__ == "__main__":
    main()