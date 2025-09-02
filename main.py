#!/usr/bin/env python3
"""
Synthetic Dataset Generator CLI
A tool for generating synthetic datasets using open-source models like Qwen-3-4B-Instruct-2507
"""

import click
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dataset_generator import DatasetGenerator

console = Console()

@click.command()
@click.option('--topic', '-t', required=True, help='Overall topic for the dataset generation')
@click.option('--input-template', '-i', required=True, help='Template for user input messages')
@click.option('--output-template', '-o', required=True, help='Template for assistant output messages')
@click.option('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct', help='Model to use for generation')
@click.option('--output-file', '-f', default='dataset.json', help='Output file for the generated dataset')
@click.option('--num-entries', '-n', default=1, help='Number of dataset entries to generate')
@click.option('--existing-file', '-e', help='Path to existing dataset file to avoid duplicates')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def generate(topic: str, input_template: str, output_template: str, model: str, 
            output_file: str, num_entries: int, existing_file: Optional[str], verbose: bool):
    """Generate synthetic dataset entries using an open-source language model."""
    
    console.print(f"[bold blue]Synthetic Dataset Generator[/bold blue]")
    console.print(f"Topic: [green]{topic}[/green]")
    console.print(f"Model: [yellow]{model}[/yellow]")
    console.print(f"Entries to generate: [cyan]{num_entries}[/cyan]")
    console.print()
    
    # Load existing dataset for uniqueness checking
    existing_entries = []
    if existing_file and Path(existing_file).exists():
        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_entries = existing_data
                console.print(f"[yellow]Loaded {len(existing_entries)} existing entries for uniqueness checking[/yellow]")
        except Exception as e:
            console.print(f"[red]Warning: Could not load existing file: {e}[/red]")
    
    # Initialize generator
    generator = DatasetGenerator(
        model_name=model,
        topic=topic,
        input_template=input_template,
        output_template=output_template,
        existing_entries=existing_entries,
        verbose=verbose
    )
    
    generated_entries = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating dataset entries...", total=num_entries)
        
        for i in range(num_entries):
            try:
                entry = generator.generate_entry()
                if entry:
                    generated_entries.extend(entry)
                    progress.update(task, advance=1, description=f"Generated entry {i+1}/{num_entries}")
                else:
                    console.print(f"[red]Failed to generate entry {i+1}[/red]")
            except Exception as e:
                console.print(f"[red]Error generating entry {i+1}: {e}[/red]")
                if verbose:
                    console.print_exception()
    
    # Save generated entries
    if generated_entries:
        output_path = Path(output_file)
        
        # If appending to existing file
        if existing_file and Path(existing_file).exists() and str(output_path) == existing_file:
            all_entries = existing_entries + generated_entries
        else:
            all_entries = generated_entries
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_entries, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✓ Successfully generated {len(generated_entries)//2} dataset entries[/green]")
            console.print(f"[green]✓ Saved to: {output_path.absolute()}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving dataset: {e}[/red]")
    else:
        console.print("[red]No entries were generated[/red]")

@click.command()
@click.option('--file', '-f', required=True, help='Dataset file to validate')
def validate(file: str):
    """Validate a dataset file format."""
    
    console.print(f"[bold blue]Validating dataset file: {file}[/bold blue]")
    
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            console.print("[red]✗ Dataset must be a JSON array[/red]")
            return
        
        if len(data) % 2 != 0:
            console.print("[red]✗ Dataset must have even number of entries (user/assistant pairs)[/red]")
            return
        
        valid_entries = 0
        for i in range(0, len(data), 2):
            if i + 1 >= len(data):
                break
                
            user_entry = data[i]
            assistant_entry = data[i + 1]
            
            if (user_entry.get('role') == 'user' and 
                assistant_entry.get('role') == 'assistant' and
                'content' in user_entry and 'content' in assistant_entry):
                valid_entries += 1
            else:
                console.print(f"[yellow]Warning: Invalid entry pair at index {i}[/yellow]")
        
        console.print(f"[green]✓ Valid dataset with {valid_entries} entry pairs[/green]")
        console.print(f"[green]✓ Total messages: {len(data)}[/green]")
        
    except FileNotFoundError:
        console.print(f"[red]✗ File not found: {file}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error validating file: {e}[/red]")

@click.group()
def cli():
    """Synthetic Dataset Generator CLI - Generate training datasets using open-source models."""
    pass

cli.add_command(generate)
cli.add_command(validate)

if __name__ == '__main__':
    cli()
