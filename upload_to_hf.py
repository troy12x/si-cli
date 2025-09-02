#!/usr/bin/env python3
"""
Upload Dataset to Hugging Face Repository
Uploads generated dataset.json files to a Hugging Face dataset repository
"""

import json
import os
from pathlib import Path
from typing import Optional
import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from huggingface_hub import HfApi, upload_file, login
from huggingface_hub.utils import HfHubHTTPError

console = Console()

def load_hf_config(config_file: str = "hf_config.yaml") -> Optional[dict]:
    """Load Hugging Face configuration from YAML file."""
    try:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load {config_file}: {e}[/yellow]")
    return None

def authenticate_hf(token: str) -> bool:
    """Authenticate with Hugging Face using the provided token."""
    try:
        login(token=token)
        console.print("[green]✓ Successfully authenticated with Hugging Face[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Authentication failed: {e}[/red]")
        return False

def validate_dataset(file_path: str) -> bool:
    """Validate the dataset format."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            console.print("[red]✗ Dataset must be a JSON array[/red]")
            return False
        
        if len(data) == 0:
            console.print("[red]✗ Dataset is empty[/red]")
            return False
        
        # Check each entry
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                console.print(f"[red]✗ Entry {i} must be an object[/red]")
                return False
            
            if 'messages' not in entry:
                console.print(f"[red]✗ Entry {i} must have 'messages' field[/red]")
                return False
            
            messages = entry['messages']
            if not isinstance(messages, list) or len(messages) != 2:
                console.print(f"[red]✗ Entry {i} messages must be an array with exactly 2 messages[/red]")
                return False
            
            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    console.print(f"[red]✗ Entry {i}, message {j} must be an object[/red]")
                    return False
                
                if 'content' not in message or 'role' not in message:
                    console.print(f"[red]✗ Entry {i}, message {j} must have 'content' and 'role' fields[/red]")
                    return False
                
                if message['role'] not in ['user', 'assistant']:
                    console.print(f"[red]✗ Entry {i}, message {j} role must be 'user' or 'assistant'[/red]")
                    return False
        
        console.print(f"[green]✓ Dataset validation passed ({len(data)} entries)[/green]")
        return True
        
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON format: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Validation error: {e}[/red]")
        return False

def upload_dataset(file_path: str, repo_id: str, token: str, commit_message: Optional[str] = None) -> bool:
    """Upload dataset file to Hugging Face repository."""
    try:
        # Authenticate
        if not authenticate_hf(token):
            return False
        
        # Validate dataset
        if not validate_dataset(file_path):
            return False
        
        # Prepare commit message
        if not commit_message:
            file_size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            num_entries = len(data) // 2
            commit_message = f"Add dataset with {num_entries} conversation pairs ({file_size} bytes)"
        
        # Upload file
        console.print(f"[yellow]📤 Uploading {file_path} to {repo_id}...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Uploading to Hugging Face...", total=None)
            
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=Path(file_path).name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                token=token
            )
            
            progress.update(task, description="Upload completed!")
        
        console.print(f"[green]✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}[/green]")
        return True
        
    except HfHubHTTPError as e:
        if "401" in str(e):
            console.print("[red]✗ Authentication failed. Please check your token.[/red]")
        elif "404" in str(e):
            console.print(f"[red]✗ Repository {repo_id} not found or not accessible.[/red]")
        else:
            console.print(f"[red]✗ HTTP Error: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Upload failed: {e}[/red]")
        return False

@click.command()
@click.option('--file', '-f', default='dataset.json', help='Dataset file to upload')
@click.option('--repo', '-r', help='Hugging Face repository ID (default from config)')
@click.option('--token', '-t', help='Hugging Face token (or set HF_TOKEN env var)')
@click.option('--message', '-m', help='Commit message')
@click.option('--validate-only', is_flag=True, help='Only validate the dataset without uploading')
def upload(file: str, repo: str, token: Optional[str], message: Optional[str], validate_only: bool):
    """Upload dataset to Hugging Face repository."""
    
    console.print("[bold blue]🤗 Hugging Face Dataset Uploader[/bold blue]")
    console.print(f"File: [cyan]{file}[/cyan]")
    # Load config for defaults
    config = load_hf_config()
    
    # Use repo from config if not provided
    if not repo and config:
        repo = config.get('repository', {}).get('id', 'eyad-silx/si-cli')
    elif not repo:
        repo = 'eyad-silx/si-cli'
    
    console.print(f"Repository: [yellow]{repo}[/yellow]")
    
    # Use default file from config if not specified
    if file == 'dataset.json' and config:
        file = config.get('defaults', {}).get('dataset_file', 'dataset.json')
    
    # Check if file exists
    if not Path(file).exists():
        console.print(f"[red]✗ File {file} not found[/red]")
        return
    
    # Validate only mode
    if validate_only:
        console.print("\n[yellow]🔍 Validation mode - not uploading[/yellow]")
        validate_dataset(file)
        return
    
    # Get token from parameter, config file, or environment
    if not token:
        # Try to load from config file
        config = load_hf_config()
        if config and config.get('token'):
            token = config['token']
            console.print("[green]📁 Using token from hf_config.yaml[/green]")
    
    if not token:
        token = os.getenv('HF_TOKEN')
        if token:
            console.print("[green]🔑 Using token from HF_TOKEN environment variable[/green]")
    
    if not token:
        console.print("[red]✗ No Hugging Face token found. Options:[/red]")
        console.print("  1. Use --token parameter")
        console.print("  2. Set HF_TOKEN environment variable")
        console.print("  3. Add token to hf_config.yaml")
        return
    
    # Upload dataset
    console.print(f"\n[yellow]🚀 Starting upload process...[/yellow]")
    success = upload_dataset(file, repo, token, message)
    
    if success:
        console.print(f"\n[bold green]🎉 Upload completed successfully![/bold green]")
        console.print(f"[dim]View your dataset at: https://huggingface.co/datasets/{repo}[/dim]")
    else:
        console.print(f"\n[bold red]❌ Upload failed[/bold red]")

if __name__ == '__main__':
    upload()
