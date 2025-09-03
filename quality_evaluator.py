#!/usr/bin/env python3
"""
Data Quality Evaluator
Uses the same model to judge and score generated dataset entries
"""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

console = Console()

class DataQualityEvaluator:
    """Evaluates dataset quality using the same model that generated it."""
    
    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.tokenizer = None
        self.model = None
        self.evaluator = None
        self.supports_chat_template = False
        
    def _initialize_model(self):
        """Initialize the model for evaluation."""
        if self.verbose:
            console.print(f"[yellow]Loading evaluator model: {self.model_name}[/yellow]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check chat template support
        self.supports_chat_template = hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create text generation pipeline
        self.evaluator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        if self.verbose:
            console.print("[green]✓ Evaluator model loaded successfully[/green]")
    
    def _create_evaluation_prompt(self, user_msg: str, assistant_msg: str, topic: str) -> str:
        """Create evaluation prompt for the model to judge quality."""
        
        evaluation_criteria = f"""
Evaluate this conversation about {topic} on a scale of 1-10:

User: {user_msg}
Assistant: {assistant_msg}

Rate each aspect (1-10):
1. RELEVANCE: Does the assistant directly address the user's question?
2. ACCURACY: Is the technical information correct and reliable?
3. COMPLETENESS: Does it provide a complete solution/answer?
4. CLARITY: Is the response clear and easy to understand?
5. SPECIFICITY: Are examples concrete and specific (not generic)?

Format your response as:
RELEVANCE: [score]
ACCURACY: [score] 
COMPLETENESS: [score]
CLARITY: [score]
SPECIFICITY: [score]
OVERALL: [average score]
REASON: [brief explanation]"""

        if self.supports_chat_template:
            messages = [
                {"role": "system", "content": "You are an expert evaluator of technical conversations. Provide honest, accurate scores."},
                {"role": "user", "content": evaluation_criteria}
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            return evaluation_criteria
    
    def evaluate_entry(self, entry: Dict) -> Dict:
        """Evaluate a single dataset entry and return scores."""
        if not self.evaluator:
            self._initialize_model()
        
        messages = entry.get('messages', [])
        if len(messages) != 2:
            return {"error": "Invalid entry format"}
        
        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(user_msg, assistant_msg, "React JS")
        
        try:
            # Generate evaluation
            response = self.evaluator(
                eval_prompt,
                max_new_tokens=300,
                temperature=0.3,  # Lower temperature for consistent scoring
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            evaluation_text = response[0]['generated_text']
            scores = self._parse_scores(evaluation_text)
            
            return {
                "user_message": user_msg,
                "assistant_message": assistant_msg,
                "scores": scores,
                "evaluation_text": evaluation_text
            }
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error evaluating entry: {e}[/red]")
            return {"error": str(e)}
    
    def _parse_scores(self, evaluation_text: str) -> Dict:
        """Parse scores from evaluation text."""
        scores = {
            "relevance": 0,
            "accuracy": 0, 
            "completeness": 0,
            "clarity": 0,
            "specificity": 0,
            "overall": 0,
            "reason": ""
        }
        
        lines = evaluation_text.upper().split('\n')
        for line in lines:
            line = line.strip()
            if 'RELEVANCE:' in line:
                scores["relevance"] = self._extract_score(line)
            elif 'ACCURACY:' in line:
                scores["accuracy"] = self._extract_score(line)
            elif 'COMPLETENESS:' in line:
                scores["completeness"] = self._extract_score(line)
            elif 'CLARITY:' in line:
                scores["clarity"] = self._extract_score(line)
            elif 'SPECIFICITY:' in line:
                scores["specificity"] = self._extract_score(line)
            elif 'OVERALL:' in line:
                scores["overall"] = self._extract_score(line)
            elif 'REASON:' in line:
                scores["reason"] = line.replace('REASON:', '').strip()
        
        # Calculate overall if not provided
        if scores["overall"] == 0:
            individual_scores = [scores["relevance"], scores["accuracy"], 
                               scores["completeness"], scores["clarity"], scores["specificity"]]
            valid_scores = [s for s in individual_scores if s > 0]
            if valid_scores:
                scores["overall"] = round(sum(valid_scores) / len(valid_scores), 1)
        
        return scores
    
    def _extract_score(self, line: str) -> float:
        """Extract numeric score from evaluation line."""
        import re
        # Look for numbers (including decimals)
        numbers = re.findall(r'\d+\.?\d*', line)
        if numbers:
            try:
                score = float(numbers[0])
                return min(10.0, max(1.0, score))  # Clamp between 1-10
            except ValueError:
                pass
        return 0.0
    
    def evaluate_dataset(self, dataset_file: str, min_score: float = 6.0, 
                        output_file: str = None) -> Dict:
        """Evaluate entire dataset and filter by quality."""
        
        if not Path(dataset_file).exists():
            console.print(f"[red]✗ Dataset file {dataset_file} not found[/red]")
            return {}
        
        # Load dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            console.print("[red]✗ Dataset must be a JSON array[/red]")
            return {}
        
        console.print(f"[blue]🔍 Evaluating {len(dataset)} entries...[/blue]")
        
        evaluated_entries = []
        high_quality_entries = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating entries...", total=len(dataset))
            
            for i, entry in enumerate(dataset):
                evaluation = self.evaluate_entry(entry)
                
                if "error" not in evaluation:
                    evaluated_entries.append(evaluation)
                    overall_score = evaluation["scores"]["overall"]
                    
                    if overall_score >= min_score:
                        high_quality_entries.append(entry)
                        if self.verbose:
                            console.print(f"[green]✓ Entry {i+1}: Score {overall_score}/10 - KEPT[/green]")
                    else:
                        if self.verbose:
                            console.print(f"[red]✗ Entry {i+1}: Score {overall_score}/10 - FILTERED[/red]")
                
                progress.update(task, advance=1, 
                              description=f"Evaluated {i+1}/{len(dataset)} entries")
        
        # Save results
        results = {
            "original_count": len(dataset),
            "evaluated_count": len(evaluated_entries),
            "high_quality_count": len(high_quality_entries),
            "filter_threshold": min_score,
            "evaluations": evaluated_entries
        }
        
        if output_file:
            # Save filtered dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(high_quality_entries, f, indent=2, ensure_ascii=False)
            
            # Save evaluation report
            report_file = output_file.replace('.json', '_evaluation_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✓ Filtered dataset saved to: {output_file}[/green]")
            console.print(f"[green]✓ Evaluation report saved to: {report_file}[/green]")
        
        # Print summary
        kept_percentage = (len(high_quality_entries) / len(dataset)) * 100
        console.print(f"\n[bold blue]📊 Quality Evaluation Summary:[/bold blue]")
        console.print(f"Original entries: [cyan]{len(dataset)}[/cyan]")
        console.print(f"High quality entries: [green]{len(high_quality_entries)}[/green]")
        console.print(f"Filtered out: [red]{len(dataset) - len(high_quality_entries)}[/red]")
        console.print(f"Quality rate: [yellow]{kept_percentage:.1f}%[/yellow]")
        
        if evaluated_entries:
            avg_scores = {}
            for metric in ["relevance", "accuracy", "completeness", "clarity", "specificity", "overall"]:
                scores = [e["scores"][metric] for e in evaluated_entries if e["scores"][metric] > 0]
                if scores:
                    avg_scores[metric] = sum(scores) / len(scores)
            
            console.print(f"\n[bold blue]📈 Average Scores:[/bold blue]")
            for metric, score in avg_scores.items():
                console.print(f"{metric.title()}: [cyan]{score:.1f}/10[/cyan]")
        
        return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        console.print("[red]Usage: python quality_evaluator.py <dataset.json> [min_score] [output_file][/red]")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    output_file = sys.argv[3] if len(sys.argv) > 3 else dataset_file.replace('.json', '_filtered.json')
    
    evaluator = DataQualityEvaluator("Qwen/Qwen3-4B-Instruct-2507", verbose=True)
    evaluator.evaluate_dataset(dataset_file, min_score, output_file)
