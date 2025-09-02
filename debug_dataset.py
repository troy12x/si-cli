#!/usr/bin/env python3
"""
Debug script to check dataset.json structure
"""

import json
from pathlib import Path

def debug_dataset(file_path="dataset.json"):
    """Print dataset structure for debugging."""
    
    if not Path(file_path).exists():
        print(f"❌ File {file_path} not found")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 File: {file_path}")
        print(f"📊 Type: {type(data)}")
        print(f"📏 Length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
        print()
        
        if isinstance(data, list):
            print("🔍 First few entries:")
            for i, entry in enumerate(data[:3]):  # Show first 3 entries
                print(f"\n--- Entry {i} ---")
                print(f"Type: {type(entry)}")
                if isinstance(entry, dict):
                    print(f"Keys: {list(entry.keys())}")
                    for key, value in entry.items():
                        if isinstance(value, list):
                            print(f"  {key}: [list with {len(value)} items]")
                            for j, item in enumerate(value[:2]):  # Show first 2 items
                                print(f"    [{j}]: {type(item)} - {item}")
                        else:
                            print(f"  {key}: {type(value)} - {value}")
                else:
                    print(f"Content: {entry}")
        else:
            print(f"🔍 Content preview: {str(data)[:200]}...")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_dataset()
