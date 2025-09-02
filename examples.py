#!/usr/bin/env python3
"""
Example usage scripts for the Synthetic Dataset Generator CLI
"""

import subprocess
import sys
from pathlib import Path

def run_example(name: str, command: list):
    """Run an example command and display results."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.stdout:
            print("OUTPUT:")
            print(result.stdout)
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running example: {e}")

def main():
    """Run example commands to demonstrate the CLI."""
    
    examples = [
        {
            "name": "Python Programming Tutorial Dataset",
            "command": [
                sys.executable, "main.py", "generate",
                "--topic", "Python programming tutorials",
                "--input-template", "How do I {specific_task} in Python?",
                "--output-template", "Here's how to {specific_task} in Python: {detailed_explanation}",
                "--num-entries", "2",
                "--output-file", "python_tutorial_dataset.json",
                "--verbose"
            ]
        },
        {
            "name": "JavaScript Web Development Q&A",
            "command": [
                sys.executable, "main.py", "generate",
                "--topic", "JavaScript web development",
                "--input-template", "What is {concept} in JavaScript and how do I use it?",
                "--output-template", "{concept} in JavaScript is {definition}. Here's how to use it: {usage_example}",
                "--num-entries", "1",
                "--output-file", "javascript_qa_dataset.json"
            ]
        },
        {
            "name": "Machine Learning Concepts",
            "command": [
                sys.executable, "main.py", "generate",
                "--topic", "Machine learning fundamentals",
                "--input-template", "Explain {ml_algorithm} and when I should use it",
                "--output-template", "{ml_algorithm} is {explanation}. You should use it when {use_cases}",
                "--num-entries", "1",
                "--output-file", "ml_concepts_dataset.json"
            ]
        },
        {
            "name": "Validate Generated Dataset",
            "command": [
                sys.executable, "main.py", "validate",
                "--file", "python_tutorial_dataset.json"
            ]
        }
    ]
    
    print("Synthetic Dataset Generator - Example Usage")
    print("=" * 60)
    print("This script demonstrates various ways to use the CLI tool.")
    print("Make sure you have installed the requirements first:")
    print("pip install -r requirements.txt")
    
    for example in examples:
        run_example(example["name"], example["command"])
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("Check the generated JSON files to see the results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
