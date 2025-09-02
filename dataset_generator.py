"""
Dataset Generator Core Module
Handles the generation of synthetic dataset entries using language models
"""

import json
import hashlib
import re
import os
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from rich.console import Console

console = Console()

class DatasetGenerator:
    """Core class for generating synthetic dataset entries."""
    
    def __init__(self, model_name: str, topic: str, input_template: str, 
                 output_template: str, existing_entries: List[Dict] = None, verbose: bool = False,
                 use_multi_gpu: bool = True, max_gpus: Optional[int] = None, max_tokens: int = 512):
        """
        Initialize the dataset generator.
        
        Args:
            model_name: Name of the model to use for generation
            topic: Overall topic for dataset generation
            input_template: Template for user input messages
            output_template: Template for assistant output messages
            existing_entries: List of existing entries to avoid duplicates
            verbose: Enable verbose logging
            use_multi_gpu: Enable multi-GPU support
            max_gpus: Maximum number of GPUs to use
            max_tokens: Maximum new tokens to generate per entry
        """
        self.model_name = model_name
        self.topic = topic
        self.input_template = input_template
        self.output_template = output_template
        self.existing_entries = existing_entries or []
        self.verbose = verbose
        self.use_multi_gpu = use_multi_gpu
        self.max_gpus = max_gpus
        self.max_tokens = max_tokens
        self.existing_hashes = set()
        self.device_info = self._detect_gpu_setup()
        
        # Create hashes of existing entries for uniqueness checking
        self._build_existing_hashes()
        
        # Initialize model and tokenizer
        self._initialize_model()
    
    def _build_existing_hashes(self):
        """Build hash set of existing entries for uniqueness checking."""
        for i in range(0, len(self.existing_entries), 2):
            if i + 1 < len(self.existing_entries):
                user_content = self.existing_entries[i].get('content', '')
                assistant_content = self.existing_entries[i + 1].get('content', '')
                entry_hash = self._hash_entry(user_content, assistant_content)
                self.existing_hashes.add(entry_hash)
    
    def _hash_entry(self, user_content: str, assistant_content: str) -> str:
        """Create a hash for an entry pair to check uniqueness."""
        combined = f"{user_content.strip()}{assistant_content.strip()}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _detect_gpu_setup(self) -> Dict:
        """Detect available GPUs and setup configuration."""
        gpu_info = {
            'available_gpus': 0,
            'gpu_names': [],
            'total_memory': 0,
            'use_multi_gpu': False,
            'device_map': None
        }
        
        if torch.cuda.is_available():
            gpu_info['available_gpus'] = torch.cuda.device_count()
            
            for i in range(gpu_info['available_gpus']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                gpu_info['gpu_names'].append(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                gpu_info['total_memory'] += gpu_memory
            
            # Determine if we should use multi-GPU
            if (self.use_multi_gpu and 
                gpu_info['available_gpus'] > 1 and 
                (self.max_gpus is None or gpu_info['available_gpus'] <= self.max_gpus)):
                gpu_info['use_multi_gpu'] = True
                
                # Create device map for model parallelism
                if gpu_info['available_gpus'] >= 2:
                    gpu_info['device_map'] = 'auto'  # Let transformers handle distribution
        
        if self.verbose:
            console.print(f"[cyan]GPU Setup:[/cyan]")
            console.print(f"  Available GPUs: {gpu_info['available_gpus']}")
            for gpu_name in gpu_info['gpu_names']:
                console.print(f"  {gpu_name}")
            console.print(f"  Multi-GPU enabled: {gpu_info['use_multi_gpu']}")
            console.print(f"  Total GPU memory: {gpu_info['total_memory']:.1f}GB")
        
        return gpu_info
    
    def _initialize_model(self):
        """Initialize the language model and tokenizer with multi-GPU support."""
        try:
            if self.verbose:
                console.print(f"[yellow]Loading model: {self.model_name}[/yellow]")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device and dtype
            if self.device_info['available_gpus'] > 0:
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32
            
            # Load model with appropriate device mapping
            model_kwargs = {
                'torch_dtype': dtype,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True
            }
            
            if self.device_info['use_multi_gpu']:
                # Multi-GPU setup
                model_kwargs['device_map'] = 'auto'
                if self.verbose:
                    console.print(f"[green]Using multi-GPU setup with {self.device_info['available_gpus']} GPUs[/green]")
            elif device == "cuda":
                # Single GPU setup
                model_kwargs['device_map'] = {'': 0}
                if self.verbose:
                    console.print(f"[green]Using single GPU setup[/green]")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            pipeline_kwargs = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'torch_dtype': dtype,
                'return_full_text': False
            }
            
            if not self.device_info['use_multi_gpu'] and device == "cuda":
                pipeline_kwargs['device'] = 0
            
            self.generator = pipeline("text-generation", **pipeline_kwargs)
            
            if self.verbose:
                console.print(f"[green]✓ Model loaded successfully[/green]")
                if self.device_info['use_multi_gpu']:
                    console.print(f"[green]✓ Model distributed across {self.device_info['available_gpus']} GPUs[/green]")
                
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the model."""
        return f"""You are a synthetic dataset generator. Your task is to generate training data for a conversational AI system.

Topic: {self.topic}

Instructions:
1. Generate content that is relevant to the topic: "{self.topic}"
2. Follow the input template exactly for the user's request
3. Follow the output template exactly for the assistant's response
4. Generate text only, no extra formatting
5. Make the content unique and varied
6. Ensure high quality and realistic conversations

Input Template: {self.input_template}
Output Template: {self.output_template}

Generate one complete conversation pair (user request + assistant response) that follows these templates and relates to the topic."""
    
    def _extract_conversation_pair(self, generated_text: str) -> Optional[Tuple[str, str]]:
        """Extract user and assistant content from generated text."""
        try:
            # Look for patterns that indicate user and assistant messages
            lines = generated_text.strip().split('\n')
            user_content = ""
            assistant_content = ""
            
            current_role = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for role indicators
                if line.lower().startswith(('user:', 'human:', 'question:')):
                    if current_role == 'assistant' and current_content:
                        assistant_content = '\n'.join(current_content).strip()
                    current_role = 'user'
                    current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                elif line.lower().startswith(('assistant:', 'ai:', 'answer:', 'response:')):
                    if current_role == 'user' and current_content:
                        user_content = '\n'.join(current_content).strip()
                    current_role = 'assistant'
                    current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                else:
                    if current_role:
                        current_content.append(line)
            
            # Handle the last role
            if current_role == 'assistant' and current_content:
                assistant_content = '\n'.join(current_content).strip()
            elif current_role == 'user' and current_content:
                user_content = '\n'.join(current_content).strip()
            
            # If we couldn't parse roles, try to split the text in half
            if not user_content or not assistant_content:
                text_parts = generated_text.strip().split('\n\n')
                if len(text_parts) >= 2:
                    user_content = text_parts[0].strip()
                    assistant_content = '\n\n'.join(text_parts[1:]).strip()
            
            # Clean up the content
            user_content = self._clean_content(user_content)
            assistant_content = self._clean_content(assistant_content)
            
            if user_content and assistant_content:
                return user_content, assistant_content
            
            return None
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error extracting conversation pair: {e}[/red]")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content."""
        # Remove common prefixes
        prefixes = ['user:', 'assistant:', 'human:', 'ai:', 'question:', 'answer:', 'response:']
        content = content.strip()
        
        for prefix in prefixes:
            if content.lower().startswith(prefix):
                content = content[len(prefix):].strip()
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _is_unique(self, user_content: str, assistant_content: str) -> bool:
        """Check if the generated entry is unique."""
        entry_hash = self._hash_entry(user_content, assistant_content)
        return entry_hash not in self.existing_hashes
    
    def generate_batch(self, batch_size: int = 1, max_retries: int = 3) -> List[List[Dict]]:
        """
        Generate multiple dataset entries in batch for better GPU utilization.
        
        Args:
            batch_size: Number of entries to generate in parallel
            max_retries: Maximum number of retries if generation fails
            
        Returns:
            List of entry pairs (each containing user and assistant messages)
        """
        if batch_size == 1:
            entry = self.generate_entry(max_retries)
            return [entry] if entry else []
        
        # Create multiple system prompts for batch generation
        system_prompts = [self._create_system_prompt() for _ in range(batch_size)]
        generated_entries = []
        
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    console.print(f"[yellow]Batch generation attempt {attempt + 1}/{max_retries} (size: {batch_size})[/yellow]")
                
                # Generate multiple responses in batch
                responses = self.generator(
                    system_prompts,
                    max_new_tokens=self.max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    batch_size=min(batch_size, 4)  # Limit batch size to prevent OOM
                )
                
                for i, response in enumerate(responses):
                    generated_text = response['generated_text']
                    
                    # Remove the system prompt from the generated text
                    if system_prompts[i] in generated_text:
                        generated_text = generated_text.replace(system_prompts[i], "").strip()
                    
                    # Extract conversation pair
                    pair = self._extract_conversation_pair(generated_text)
                    
                    if pair:
                        user_content, assistant_content = pair
                        
                        # Check uniqueness
                        if self._is_unique(user_content, assistant_content):
                            # Add to existing hashes to prevent future duplicates
                            entry_hash = self._hash_entry(user_content, assistant_content)
                            self.existing_hashes.add(entry_hash)
                            
                            # Create the dataset entry
                            entry = [
                                {"content": user_content, "role": "user"},
                                {"content": assistant_content, "role": "assistant"}
                            ]
                            generated_entries.append(entry)
                
                if generated_entries:
                    if self.verbose:
                        console.print(f"[green]✓ Generated {len(generated_entries)} unique entries in batch[/green]")
                    return generated_entries
                    
            except Exception as e:
                if self.verbose:
                    console.print(f"[red]Batch generation error (attempt {attempt + 1}): {e}[/red]")
        
        return generated_entries
    
    def generate_entry(self, max_retries: int = 3) -> Optional[List[Dict]]:
        """
        Generate a single dataset entry (user + assistant pair).
        
        Args:
            max_retries: Maximum number of retries if generation fails
            
        Returns:
            List containing user and assistant message dictionaries, or None if failed
        """
        system_prompt = self._create_system_prompt()
        
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    console.print(f"[yellow]Generation attempt {attempt + 1}/{max_retries}[/yellow]")
                
                # Generate text using the model
                response = self.generator(
                    system_prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = response[0]['generated_text']
                
                # Remove the system prompt from the generated text
                if system_prompt in generated_text:
                    generated_text = generated_text.replace(system_prompt, "").strip()
                
                if self.verbose:
                    console.print(f"[cyan]Generated text: {generated_text[:200]}...[/cyan]")
                
                # Extract conversation pair
                pair = self._extract_conversation_pair(generated_text)
                
                if pair:
                    user_content, assistant_content = pair
                    
                    # Check uniqueness
                    if self._is_unique(user_content, assistant_content):
                        # Add to existing hashes to prevent future duplicates
                        entry_hash = self._hash_entry(user_content, assistant_content)
                        self.existing_hashes.add(entry_hash)
                        
                        # Create the dataset entry
                        entry = [
                            {
                                "content": user_content,
                                "role": "user"
                            },
                            {
                                "content": assistant_content,
                                "role": "assistant"
                            }
                        ]
                        
                        if self.verbose:
                            console.print(f"[green]✓ Generated unique entry[/green]")
                        
                        return entry
                    else:
                        if self.verbose:
                            console.print(f"[yellow]Entry not unique, retrying...[/yellow]")
                
            except Exception as e:
                if self.verbose:
                    console.print(f"[red]Generation error (attempt {attempt + 1}): {e}[/red]")
                
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed to generate entry after {max_retries} attempts[/red]")
        
        return None
