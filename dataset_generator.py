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
                 use_multi_gpu: bool = True, max_gpus: Optional[int] = None, max_tokens: int = 512,
                 variable_descriptions: Dict[str, str] = None):
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
            variable_descriptions: Dictionary mapping variable names to their descriptions
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
        self.variable_descriptions = variable_descriptions or {}
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
                'dtype': dtype,  # Fixed: torch_dtype is deprecated
                'trust_remote_code': True,
                'low_cpu_mem_usage': True
            }
            
            if self.device_info['use_multi_gpu'] and self.device_info['available_gpus'] > 1:
                # Multi-GPU setup - force model parallelism across all GPUs
                num_gpus = self.device_info['available_gpus']
                device_map = {}
                
                # Distribute model layers across all available GPUs
                for i in range(num_gpus):
                    device_map[f'cuda:{i}'] = f'cuda:{i}'
                
                model_kwargs['device_map'] = 'auto'  # Let transformers auto-distribute
                if self.verbose:
                    console.print(f"[green]Using multi-GPU setup with {num_gpus} GPUs - model parallelism enabled[/green]")
            elif device == "cuda":
                # Single GPU setup
                model_kwargs['device_map'] = {'': 0}
                if self.verbose:
                    console.print(f"[green]Using single GPU setup[/green]")
            else:
                # CPU-only setup - don't use device_map
                if self.verbose:
                    console.print(f"[yellow]Using CPU-only setup[/yellow]")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            pipeline_kwargs = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'return_full_text': False
            }
            
            # Only set device for single GPU, let multi-GPU handle itself
            if not self.device_info['use_multi_gpu'] and device == "cuda":
                pipeline_kwargs['device'] = 0
            elif device == "cpu":
                pipeline_kwargs['device'] = -1  # CPU device
            
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
        # Extract variables from templates
        import re
        input_vars = re.findall(r'\{([^}]+)\}', self.input_template)
        output_vars = re.findall(r'\{([^}]+)\}', self.output_template)
        all_vars = list(set(input_vars + output_vars))
        
        var_instructions = ""
        if all_vars:
            var_instructions = f"""

Template Variables:
Your templates contain these variables: {', '.join(all_vars)}
For each variable like {{variable_name}}, you must:
1. Replace it with appropriate, specific content based on the descriptions below
2. Make sure the content fits naturally in the sentence
3. Vary the content for each generation to create diversity
4. Keep content relevant to the topic "{self.topic}"

Variable Descriptions:"""
            
            # Add user-provided descriptions or defaults
            for var in all_vars:
                if var in self.variable_descriptions:
                    description = self.variable_descriptions[var]
                else:
                    # Provide smart defaults based on common variable names
                    if var.lower() in ['task', 'action']:
                        description = "specific programming task or action"
                    elif var.lower() in ['problem', 'issue', 'error']:
                        description = "specific problem, issue, or error type"
                    elif var.lower() in ['concept', 'topic']:
                        description = "programming concept or topic"
                    elif var.lower() in ['fix', 'solution']:
                        description = "detailed solution or fix"
                    elif var.lower() in ['explanation', 'details']:
                        description = "detailed explanation or implementation details"
                    else:
                        description = f"specific {var} related to {self.topic}"
                
                var_instructions += f"\n- {{{var}}}: {description}"
            
            var_instructions += f"\n\nAlways replace ALL variables with concrete, specific content. Make each generation unique and varied."
        
        return f"""You are a synthetic dataset generator. Your task is to generate training data for a conversational AI system.

Topic: {self.topic}

Instructions:
1. Generate content that is relevant to the topic: "{self.topic}"
2. Use the input template as a guide for the user's request
3. Use the output template as a guide for the assistant's response
4. Replace ALL template variables {{variable}} with specific, concrete content
5. Generate text only, no extra formatting or role labels
6. Make the content unique, varied, and realistic
7. Ensure high quality conversations that would be useful for training{var_instructions}

Input Template: {self.input_template}
Output Template: {self.output_template}

Generate one complete conversation pair following this format:

User: [Generated user message based on input template with variables filled]
Assistant: [Generated assistant response based on output template with variables filled]

Make sure to replace all {{variables}} with actual content according to their descriptions above."""
    
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
                
                # Check for role indicators (more flexible matching)
                line_lower = line.lower()
                if any(line_lower.startswith(prefix) for prefix in ['user:', 'human:', 'question:', 'q:']):
                    if current_role == 'assistant' and current_content:
                        assistant_content = '\n'.join(current_content).strip()
                    current_role = 'user'
                    # Extract content after the colon
                    content = line.split(':', 1)[1].strip() if ':' in line else line
                    current_content = [content] if content else []
                elif any(line_lower.startswith(prefix) for prefix in ['assistant:', 'ai:', 'answer:', 'response:', 'a:']):
                    if current_role == 'user' and current_content:
                        user_content = '\n'.join(current_content).strip()
                    current_role = 'assistant'
                    # Extract content after the colon
                    content = line.split(':', 1)[1].strip() if ':' in line else line
                    current_content = [content] if content else []
                else:
                    if current_role:
                        current_content.append(line)
            
            # Handle the last role
            if current_role == 'assistant' and current_content:
                assistant_content = '\n'.join(current_content).strip()
            elif current_role == 'user' and current_content:
                user_content = '\n'.join(current_content).strip()
            
            # If we couldn't parse roles, try different splitting strategies
            if not user_content or not assistant_content:
                # Try splitting by double newlines first
                text_parts = generated_text.strip().split('\n\n')
                if len(text_parts) >= 2:
                    user_content = text_parts[0].strip()
                    assistant_content = '\n\n'.join(text_parts[1:]).strip()
                else:
                    # Try splitting by single newlines and take first two non-empty parts
                    parts = [part.strip() for part in generated_text.strip().split('\n') if part.strip()]
                    if len(parts) >= 2:
                        user_content = parts[0]
                        assistant_content = '\n'.join(parts[1:])
            
            # Clean up the content
            user_content = self._clean_content(user_content)
            assistant_content = self._clean_content(assistant_content)
            
            # Validate that we have meaningful content
            if user_content and assistant_content and len(user_content) > 5 and len(assistant_content) > 5:
                return user_content, assistant_content
            
            return None
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error extracting conversation pair: {e}[/red]")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content."""
        # Remove common prefixes (case insensitive)
        prefixes = ['user:', 'assistant:', 'human:', 'ai:', 'question:', 'answer:', 'response:', 'q:', 'a:']
        content = content.strip()
        
        for prefix in prefixes:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
        
        # Remove quotes if they wrap the entire content
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1].strip()
        elif content.startswith("'") and content.endswith("'"):
            content = content[1:-1].strip()
        
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove any remaining template variables that weren't filled
        content = re.sub(r'\{[^}]*\}', '', content)
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
                
                # Handle both single response and batch responses
                if isinstance(responses, list):
                    response_list = responses
                else:
                    response_list = [responses]
                
                for i, response in enumerate(response_list):
                    if isinstance(response, dict) and 'generated_text' in response:
                        generated_text = response['generated_text']
                    elif isinstance(response, list) and len(response) > 0:
                        generated_text = response[0]['generated_text']
                    else:
                        continue
                    
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
                            
                            # Create the dataset entry in messages format
                            entry = {
                                "messages": [
                                    {"content": user_content, "role": "user"},
                                    {"content": assistant_content, "role": "assistant"}
                                ]
                            }
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
                        
                        # Create the dataset entry in messages format
                        entry = {
                            "messages": [
                                {
                                    "content": user_content,
                                    "role": "user"
                                },
                                {
                                    "content": assistant_content,
                                    "role": "assistant"
                                }
                            ]
                        }
                        
                        if self.verbose:
                            console.print(f"[green]✓ Generated unique entry[/green]")
                        
                        return entry  # Return single entry
                    else:
                        if self.verbose:
                            console.print(f"[yellow]Entry not unique, retrying...[/yellow]")
                
            except Exception as e:
                if self.verbose:
                    console.print(f"[red]Generation error (attempt {attempt + 1}): {e}[/red]")
                
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed to generate entry after {max_retries} attempts[/red]")
        
        return None
