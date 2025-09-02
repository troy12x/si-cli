#!/usr/bin/env python3
"""
Synthetic Dataset Generator CLI
A tool for generating synthetic datasets using open-source models like Qwen-3-4B-Instruct-2507
"""

import click
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt, Confirm
from dataset_generator import DatasetGenerator
import yaml

console = Console()

def save_config(config_data: Dict, config_file: str = "dataset_config.yaml"):
    """Save configuration to a YAML file."""
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        console.print(f"[green]✓ Configuration saved to {config_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")

def load_config(config_file: str = "dataset_config.yaml") -> Optional[Dict]:
    """Load configuration from a YAML file."""
    try:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
    return None

def interactive_generate(config_file: Optional[str] = None):
    """Run the generator in interactive mode with prompts."""
    console.print("[bold blue]🤖 Synthetic Dataset Generator - Interactive Mode[/bold blue]")
    
    # Check for existing config files
    config = None
    if config_file and Path(config_file).exists():
        config = load_config(config_file)
        if config:
            console.print(f"[green]📁 Loaded configuration from {config_file}[/green]")
    else:
        # Look for existing dataset_config.yaml files
        existing_configs = []
        for config_path in Path('.').glob('*config*.yaml'):
            if config_path.name != 'config.yaml' and config_path.name != 'hf_config.yaml':
                existing_configs.append(str(config_path))
        
        if existing_configs:
            console.print("\n[yellow]📋 Found existing configuration files:[/yellow]")
            for i, cfg in enumerate(existing_configs, 1):
                console.print(f"  {i}. {cfg}")
            
            load_choice = Confirm.ask("\n[cyan]🔄 Would you like to load an existing configuration?[/cyan]", default=False)
            
            if load_choice:
                if len(existing_configs) == 1:
                    config_file = existing_configs[0]
                else:
                    choice = IntPrompt.ask(f"[cyan]Which config file? (1-{len(existing_configs)})[/cyan]", 
                                         default=1)
                    if 1 <= choice <= len(existing_configs):
                        config_file = existing_configs[choice - 1]
                
                if config_file:
                    config = load_config(config_file)
                    if config:
                        console.print(f"[green]📁 Loaded configuration from {config_file}[/green]")
            else:
                console.print("[blue]✨ Starting with fresh configuration[/blue]")
    
    # If config is loaded and complete, use it directly without prompting
    if config and all(key in config for key in ['topic', 'input_template', 'output_template', 'num_entries']):
        console.print("[green]✅ Using saved configuration - no prompts needed![/green]")
        topic = config['topic']
        input_template = config['input_template']
        output_template = config['output_template']
        variable_descriptions = config.get('variable_descriptions', {})
        num_entries = config['num_entries']
        max_tokens = config.get('max_tokens', 512)
        model = config.get('model', 'Qwen/Qwen3-4B-Instruct-2507')
        batch_size = config.get('batch_size', 2)
        output_file = config.get('output_file', 'dataset.json')
        
        # Show loaded configuration
        console.print(f"\n[dim]📋 Loaded Configuration:[/dim]")
        console.print(f"Topic: [cyan]{topic}[/cyan]")
        console.print(f"Input template: [yellow]{input_template}[/yellow]")
        console.print(f"Output template: [yellow]{output_template}[/yellow]")
        console.print(f"Entries: [cyan]{num_entries}[/cyan]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(f"Output file: [magenta]{output_file}[/magenta]")
        
        if not Confirm.ask("\n[bold yellow]🚀 Start generation with these settings?[/bold yellow]", default=True):
            console.print("[red]❌ Generation cancelled.[/red]")
            return
        
        verbose = config.get('verbose', False)
    else:
        console.print("[dim]Answer the following questions to generate your dataset:[/dim]\n")
        
        # Get topic
        topic = Prompt.ask("[cyan]📝 What topic should the dataset focus on?[/cyan]", 
                          default=config.get('topic', 'Python programming') if config else "Python programming")
    
        # Get input template with examples
        console.print("\n[yellow]💡 Input Template Examples:[/yellow]")
        console.print("  • How do I {task} in Python?")
        console.print("  • What's the best way to {action} using {technology}?")
        console.print("  • I'm getting {error} when {action}. How to fix?")
        
        input_template = Prompt.ask("\n[cyan]📥 Enter your input template (use {variable} for placeholders)[/cyan]",
                                   default=config.get('input_template', 'How do I {task} in Python?') if config else "How do I {task} in Python?")
        
        # Get output template with examples
        console.print("\n[yellow]💡 Output Template Examples:[/yellow]")
        console.print("  • Here's how to {task}: {explanation}")
        console.print("  • To {action} with {technology}, you can: {steps}")
        console.print("  • This {error} occurs because {cause}. Solution: {fix}")
        
        output_template = Prompt.ask("\n[cyan]📤 Enter your output template (use {variable} for placeholders)[/cyan]",
                                    default=config.get('output_template', "Here's how to {task} in Python: {explanation}") if config else "Here's how to {task} in Python: {explanation}")
    
        # Extract variables from both templates and ask for descriptions
        import re
        input_vars = set(re.findall(r'\{(\w+)\}', input_template))
        output_vars = set(re.findall(r'\{(\w+)\}', output_template))
        all_vars = sorted(input_vars | output_vars)
        
        variable_descriptions = {}
        if all_vars:
            console.print("[dim]Please describe what each variable should represent:[/dim]\n")
            
            for var in all_vars:
                saved_desc = config.get('variable_descriptions', {}).get(var) if config else None
                default_desc = saved_desc or f"specific {var} related to {topic}"
                description = Prompt.ask(f"[cyan]What should {{[bold]{var}[/bold]}} represent?[/cyan]",
                                       default=default_desc)
                variable_descriptions[var] = description
        
        # Get number of entries
        num_entries = IntPrompt.ask("\n[cyan]🔢 How many dataset entries do you want to generate?[/cyan]", 
                                   default=config.get('num_entries', 5) if config else 5)
        
        # Get max tokens
        max_tokens = IntPrompt.ask("[cyan]⚡ Maximum new tokens per entry?[/cyan]", 
                                  default=config.get('max_tokens', 512) if config else 512)
    
        # Get model choice
        console.print("\n[yellow]🤖 Available Models:[/yellow]")
        console.print("  1. Qwen/Qwen2.5-3B-Instruct (Recommended)")
        console.print("  2. Qwen/Qwen2.5-7B-Instruct")
        console.print("  3. microsoft/DialoGPT-medium")
        console.print("  4. facebook/blenderbot-400M-distill")
        console.print("  5. Custom model (enter your own)")
        
        model_choice = IntPrompt.ask("[cyan]🔧 Choose model (1-5)?[/cyan]", default=1)
        models = [
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct", 
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill"
        ]
        
        if model_choice == 5:
            model = Prompt.ask("[cyan]🔧 Enter custom model name (e.g., microsoft/DialoGPT-large)[/cyan]")
        else:
            model = models[model_choice - 1] if 1 <= model_choice <= len(models) else models[0]
        
        # Get batch size for multi-GPU
        batch_size = IntPrompt.ask("[cyan]🚀 Batch size for parallel generation?[/cyan]", 
                                  default=config.get('batch_size', 2) if config else 2)
        
        # Get output file
        output_file = Prompt.ask("[cyan]💾 Output filename?[/cyan]", 
                                default=config.get('output_file', 'dataset.json') if config else "dataset.json")
        
        # Ask about verbose mode
        verbose = Confirm.ask("[cyan]🔍 Enable verbose output?[/cyan]", default=False)
    
        # Show configuration summary
        console.print(f"\n[bold green]📋 Configuration Summary:[/bold green]")
        console.print(f"Topic: [cyan]{topic}[/cyan]")
        console.print(f"Input template: [yellow]{input_template}[/yellow]")
        console.print(f"Output template: [yellow]{output_template}[/yellow]")
        console.print(f"Variables: [magenta]{list(variable_descriptions.keys())}[/magenta]")
        console.print(f"Entries: [cyan]{num_entries}[/cyan]")
        console.print(f"Max tokens: [cyan]{max_tokens}[/cyan]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
        console.print(f"Output file: [magenta]{output_file}[/magenta]")
        
        # Ask to save configuration
        save_config_choice = Confirm.ask("\n[cyan]💾 Save this configuration for future use?[/cyan]", default=True)
        if save_config_choice:
            config_filename = Prompt.ask("[cyan]Config filename?[/cyan]", default="dataset_config.yaml")
            config_data = {
                'topic': topic,
                'input_template': input_template,
                'output_template': output_template,
                'variable_descriptions': variable_descriptions,
                'num_entries': num_entries,
                'max_tokens': max_tokens,
                'batch_size': batch_size,
                'output_file': output_file,
                'model': model
            }
            save_config(config_data, config_filename)
        
        if not Confirm.ask("\n[bold yellow]🚀 Start generation?[/bold yellow]", default=True):
            console.print("[red]❌ Generation cancelled.[/red]")
            return
    
    # Call the generation function
    generate_dataset(
        topic=topic,
        input_template=input_template,
        output_template=output_template,
        model=model,
        output_file=output_file,
        num_entries=num_entries,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_multi_gpu=True,
        max_gpus=None,
        existing_file=None,
        verbose=verbose,
        variable_descriptions=variable_descriptions
    )

def generate_dataset(topic: str, input_template: str, output_template: str, model: str, 
                    output_file: str, num_entries: int, max_tokens: int, batch_size: int, 
                    use_multi_gpu: bool, max_gpus: Optional[int], existing_file: Optional[str], 
                    verbose: bool, variable_descriptions: Optional[Dict[str, str]] = None):
    """Core dataset generation function."""
    console.print(f"\n[bold blue]🚀 Starting Dataset Generation[/bold blue]")
    console.print(f"Topic: [green]{topic}[/green]")
    console.print(f"Model: [yellow]{model}[/yellow]")
    console.print(f"Entries to generate: [cyan]{num_entries}[/cyan]")
    console.print(f"Max tokens per entry: [cyan]{max_tokens}[/cyan]")
    console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"Multi-GPU: [cyan]{use_multi_gpu}[/cyan]")
    if max_gpus:
        console.print(f"Max GPUs: [cyan]{max_gpus}[/cyan]")
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
        verbose=verbose,
        use_multi_gpu=use_multi_gpu,
        max_gpus=max_gpus,
        max_tokens=max_tokens,
        variable_descriptions=variable_descriptions or {}
    )
    
    generated_entries = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Calculate number of batches needed
        num_batches = (num_entries + batch_size - 1) // batch_size
        task = progress.add_task("Generating dataset entries...", total=num_batches)
        
        entries_generated = 0
        for batch_idx in range(num_batches):
            try:
                # Calculate entries for this batch
                remaining_entries = num_entries - entries_generated
                current_batch_size = min(batch_size, remaining_entries)
                
                # Generate batch
                batch_entries = generator.generate_batch(current_batch_size)
                
                # Add successful entries
                for entry in batch_entries:
                    if entry:
                        generated_entries.append(entry)
                        entries_generated += 1
                
                progress.update(task, advance=1, 
                              description=f"Generated {entries_generated}/{num_entries} entries (batch {batch_idx+1}/{num_batches})")
                
                if len(batch_entries) < current_batch_size:
                    console.print(f"[yellow]Warning: Only generated {len(batch_entries)}/{current_batch_size} entries in batch {batch_idx+1}[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Error generating batch {batch_idx+1}: {e}[/red]")
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
            if use_multi_gpu and generator.device_info['use_multi_gpu']:
                console.print(f"[green]✓ Used {generator.device_info['available_gpus']} GPUs for generation[/green]")
            console.print(f"[green]✓ Saved to: {output_path.absolute()}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving dataset: {e}[/red]")
    else:
        console.print("[red]No entries were generated[/red]")

@click.command()
@click.option('--topic', '-t', help='Overall topic for the dataset generation')
@click.option('--input-template', '-i', help='Template for user input messages')
@click.option('--output-template', '-o', help='Template for assistant output messages')
@click.option('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct', help='Model to use for generation')
@click.option('--output-file', '-f', default='dataset.json', help='Output file for the generated dataset')
@click.option('--num-entries', '-n', default=1, help='Number of dataset entries to generate')
@click.option('--max-tokens', default=512, help='Maximum new tokens to generate per entry')
@click.option('--batch-size', '-b', default=1, help='Batch size for parallel generation (multi-GPU)')
@click.option('--use-multi-gpu/--no-multi-gpu', default=True, help='Enable/disable multi-GPU support')
@click.option('--max-gpus', type=int, help='Maximum number of GPUs to use')
@click.option('--existing-file', '-e', help='Path to existing dataset file to avoid duplicates')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def generate(topic: Optional[str], input_template: Optional[str], output_template: Optional[str], model: str, 
            output_file: str, num_entries: int, max_tokens: int, batch_size: int, use_multi_gpu: bool, 
            max_gpus: Optional[int], existing_file: Optional[str], verbose: bool):
    """Generate synthetic dataset entries using an open-source language model with multi-GPU support."""
    
    # If no topic provided, run in interactive mode
    if not topic or not input_template or not output_template:
        interactive_generate()
        return
    
    # Call the core generation function
    generate_dataset(
        topic=topic,
        input_template=input_template,
        output_template=output_template,
        model=model,
        output_file=output_file,
        num_entries=num_entries,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_multi_gpu=use_multi_gpu,
        max_gpus=max_gpus,
        existing_file=existing_file,
        verbose=verbose,
        variable_descriptions={}
    )

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
    # If no arguments provided, run in interactive mode
    if len(sys.argv) == 1:
        interactive_generate()
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        # Load specific config file if provided
        interactive_generate(sys.argv[1])
    else:
        cli()
