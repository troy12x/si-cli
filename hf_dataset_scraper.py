#!/usr/bin/env python3
"""
HuggingFace Dataset Domain Scraper
Scrapes domain URLs from HuggingFace datasets for math/science content sources
"""

import requests
import json
import time
from typing import List, Set
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import re

console = Console()

class HuggingFaceDatasetScraper:
    """Scraper for HuggingFace dataset domains."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.domains = set()
    
    def scrape_megamath_dataset(self) -> Set[str]:
        """Scrape domains from MegaMath-Web-Pro-Max dataset."""
        console.print("[blue]🔍 Scraping MegaMath-Web-Pro-Max dataset...[/blue]")
        
        # HuggingFace dataset API endpoint
        base_url = "https://datasets-server.huggingface.co/rows"
        dataset = "OctoThinker/MegaMath-Web-Pro-Max"
        config = "default"
        split = "train"
        
        domains = set()
        offset = 0
        limit = 100
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching dataset rows...", total=None)
            
            while True:
                try:
                    # API request to get dataset rows
                    url = f"{base_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={limit}"
                    
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    rows = data.get('rows', [])
                    
                    if not rows:
                        break
                    
                    # Extract domains from each row
                    for row in rows:
                        row_data = row.get('row', {})
                        domain = row_data.get('domain')
                        
                        if domain and isinstance(domain, str):
                            # Clean domain (remove www, protocols, etc.)
                            clean_domain = self.clean_domain(domain)
                            if clean_domain and clean_domain not in domains:
                                domains.add(clean_domain)
                                # Save progressively if it's a math/science domain
                                if self.is_math_science_domain(clean_domain):
                                    self.save_domain_progressively(clean_domain)
                    
                    console.print(f"[cyan]📊 Processed {offset + len(rows)} rows, found {len(domains)} unique domains[/cyan]")
                    
                    offset += limit
                    time.sleep(self.delay)
                    
                    # Safety check to avoid infinite loops
                    if offset > 50000:  # Adjust based on dataset size
                        console.print("[yellow]⚠️  Reached safety limit, stopping...[/yellow]")
                        break
                        
                except requests.exceptions.RequestException as e:
                    console.print(f"[red]❌ Error fetching data: {e}[/red]")
                    break
                except Exception as e:
                    console.print(f"[red]❌ Unexpected error: {e}[/red]")
                    break
        
        console.print(f"[green]✅ Found {len(domains)} unique domains[/green]")
        return domains
    
    def clean_domain(self, domain: str) -> str:
        """Clean and normalize domain names."""
        if not domain:
            return ""
        
        # Remove protocols
        domain = re.sub(r'^https?://', '', domain)
        domain = re.sub(r'^www\.', '', domain)
        
        # Remove paths and parameters
        domain = domain.split('/')[0]
        domain = domain.split('?')[0]
        domain = domain.split('#')[0]
        
        # Remove port numbers
        domain = domain.split(':')[0]
        
        # Convert to lowercase
        domain = domain.lower().strip()
        
        # Validate domain format
        if not re.match(r'^[a-z0-9.-]+\.[a-z]{2,}$', domain):
            return ""
        
        return domain
    
    def is_math_science_domain(self, domain: str) -> bool:
        """Check if domain is likely to contain math/science content."""
        math_science_keywords = [
            'edu', 'university', 'college', 'academic', 'research',
            'math', 'mathematics', 'science', 'physics', 'chemistry',
            'biology', 'engineering', 'mit', 'stanford', 'harvard',
            'arxiv', 'scholar', 'journal', 'publication', 'institute',
            'lab', 'department', 'faculty', 'course', 'lecture'
        ]
        
        # Check if domain contains math/science keywords
        if any(keyword in domain for keyword in math_science_keywords):
            return True
        # Include .edu domains by default
        elif domain.endswith('.edu'):
            return True
        
        return False
    
    def filter_math_science_domains(self, domains: Set[str]) -> Set[str]:
        """Filter domains that are likely to contain math/science content."""
        math_science_keywords = [
            'edu', 'university', 'college', 'academic', 'research',
            'math', 'mathematics', 'science', 'physics', 'chemistry',
            'biology', 'engineering', 'mit', 'stanford', 'harvard',
            'arxiv', 'scholar', 'journal', 'publication', 'institute',
            'lab', 'department', 'faculty', 'course', 'lecture'
        ]
        
        filtered_domains = set()
        
        for domain in domains:
            # Check if domain contains math/science keywords
            if any(keyword in domain for keyword in math_science_keywords):
                filtered_domains.add(domain)
            # Include .edu domains by default
            elif domain.endswith('.edu'):
                filtered_domains.add(domain)
        
        console.print(f"[blue]🔬 Filtered to {len(filtered_domains)} math/science domains[/blue]")
        return filtered_domains
    
    def save_domain_progressively(self, domain: str, filename: str = 'sources.txt'):
        """Save a single domain progressively to sources.txt."""
        import os
        
        try:
            # Check if file exists
            file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
            
            if not file_exists:
                # Create file with header
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# Math and Science Domain Sources\n")
                    f.write("# Extracted from HuggingFace MegaMath-Web-Pro-Max dataset\n\n")
            
            # Append domain as URL
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"https://{domain}\n")
            
            console.print(f"[cyan]💾 Saved: {domain}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]❌ Error saving domain: {e}[/red]")

    def save_domains_to_file(self, domains: Set[str], filename: str = 'sources.txt'):
        """Save domains to sources.txt file."""
        try:
            # Convert domains to URLs
            urls = [f"https://{domain}" for domain in sorted(domains)]
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Math and Science Domain Sources\n")
                f.write("# Extracted from HuggingFace MegaMath-Web-Pro-Max dataset\n\n")
                
                for url in urls:
                    f.write(f"{url}\n")
            
            console.print(f"[green]💾 Saved {len(urls)} domains to {filename}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error saving domains: {e}[/red]")

def main():
    """Main function to scrape HuggingFace dataset domains."""
    console.print("[bold blue]HuggingFace Dataset Domain Scraper[/bold blue]")
    
    scraper = HuggingFaceDatasetScraper(delay=0.5)
    
    # Scrape domains from MegaMath dataset
    all_domains = scraper.scrape_megamath_dataset()
    
    if not all_domains:
        console.print("[red]❌ No domains found[/red]")
        return
    
    # Filter for math/science domains
    filtered_domains = scraper.filter_math_science_domains(all_domains)
    
    # Save to sources.txt
    scraper.save_domains_to_file(filtered_domains)
    
    console.print("[green]🎉 Domain scraping completed![/green]")

if __name__ == "__main__":
    main()
