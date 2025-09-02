"""
Dataset Generator Core Module
Handles the generation of synthetic dataset entries using language models
"""

import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from rich.console import Console

console = Console()

class DatasetGenerator:
    """Core class for generating synthetic dataset entries."""
    
    def __init__(self, model_name: str, topic: str, input_template: str, 
                 output_template: str, existing_entries: List[Dict] = None, verbose: bool = False):
        """
        Initialize the dataset generator.
        
        Args:
            model_name: Name of the model to use for generation
            topic: Overall topic for dataset generation
            input_template: Template for user input messages
            output_template: Template for assistant output messages
            existing_entries: List of existing entries to avoid duplicates
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.topic = topic
        self.input_template = input_template
        self.output_template = output_template
        self.existing_entries = existing_entries or []
        self.verbose = verbose
        self.existing_hashes = set()
        
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
    
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            if self.verbose:
                console.print(f"[yellow]Loading model: {self.model_name}[/yellow]")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            if self.verbose:
                console.print(f"[green]✓ Model loaded successfully on {device}[/green]")
                
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
                    max_new_tokens=512,
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
