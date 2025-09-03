#!/usr/bin/env python3
"""
High-Quality Retrieval Dataset Scraper
Scrapes coherent, high-quality resources for retrieval training datasets
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import base64
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

console = Console()

@dataclass
class ScrapedContent:
    """Represents scraped content with metadata."""
    url: str
    title: str
    content: str
    source_type: str
    quality_score: float
    word_count: int
    coherence_score: float
    timestamp: str

class QualityFilter:
    """Filters content based on quality metrics."""
    
    def __init__(self, min_words: int = 100, max_words: int = 5000):
        self.min_words = min_words
        self.max_words = max_words
    
    def calculate_quality_score(self, content: str, title: str) -> float:
        """Calculate content quality score (0-10)."""
        score = 5.0  # Base score
        
        # Word count scoring
        word_count = len(content.split())
        if self.min_words <= word_count <= self.max_words:
            score += 1.0
        elif word_count < self.min_words:
            score -= 2.0
        
        # Structure scoring
        if title and len(title) > 10:
            score += 0.5
        
        # Code presence (good for technical content)
        if '```' in content or '<code>' in content or 'function' in content.lower():
            score += 1.0
        
        # Technical keywords
        tech_keywords = ['react', 'javascript', 'python', 'api', 'component', 'function', 'method']
        keyword_count = sum(1 for keyword in tech_keywords if keyword in content.lower())
        score += min(keyword_count * 0.3, 1.5)
        
        # Sentence structure
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:  # Good sentence length
            score += 0.5
        
        # Avoid spam indicators
        if content.count('!') > word_count * 0.05:  # Too many exclamations
            score -= 1.0
        
        return min(10.0, max(0.0, score))
    
    def calculate_coherence_score(self, content: str) -> float:
        """Calculate content coherence score (0-10)."""
        score = 5.0
        
        # Paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 1.0
        
        # Consistent terminology
        sentences = content.split('.')
        if len(sentences) > 3:
            # Check for repeated key terms (indicates focus)
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Only count meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            repeated_terms = sum(1 for freq in word_freq.values() if freq >= 3)
            score += min(repeated_terms * 0.2, 1.0)
        
        return min(10.0, max(0.0, score))

class WebScraper:
    """High-quality web scraper for technical content."""
    
    def __init__(self, delay: float = 1.0, max_workers: int = 3):
        self.delay = delay
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.quality_filter = QualityFilter()
        self.scraped_urls = set()
        self.lock = threading.Lock()
    
    def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape a single URL and return structured content."""
        try:
            with self.lock:
                if url in self.scraped_urls:
                    return None
                self.scraped_urls.add(url)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('title') or soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            if not content or len(content.split()) < 50:
                return None
            
            # Calculate quality scores
            quality_score = self.quality_filter.calculate_quality_score(content, title)
            coherence_score = self.quality_filter.calculate_coherence_score(content)
            
            # Determine source type
            source_type = self._determine_source_type(url, soup)
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                source_type=source_type,
                quality_score=quality_score,
                word_count=len(content.split()),
                coherence_score=coherence_score,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            console.print(f"[red]Error scraping {url}: {e}[/red]")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML with proper cleaning."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '.post-content', 
            '.entry-content', '.article-body', '#content'
        ]
        
        content = ""
        for selector in main_selectors:
            main_elem = soup.select_one(selector)
            if main_elem:
                content = self._clean_extracted_text(main_elem)
                break
        
        if not content:
            # Fallback to body
            body = soup.find('body')
            if body:
                content = self._clean_extracted_text(body)
            else:
                content = self._clean_extracted_text(soup)
        
        return content
    
    def _clean_extracted_text(self, element) -> str:
        """Clean extracted text with proper formatting."""
        # Handle code blocks specially
        code_blocks = []
        for i, code_elem in enumerate(element.find_all(['code', 'pre'])):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_text = code_elem.get_text()
            # Clean code but preserve structure
            cleaned_code = re.sub(r'\n\s*\n', '\n', code_text.strip())
            code_blocks.append(cleaned_code)
            code_elem.replace_with(placeholder)
        
        # Extract text with proper spacing
        text = element.get_text(separator=' ', strip=True)
        
        # Clean up the text
        text = self._clean_text_formatting(text)
        
        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            # Format code block nicely
            formatted_code = f"\n\n```\n{code_block}\n```\n\n"
            text = text.replace(placeholder, formatted_code)
        
        return text.strip()
    
    def _clean_text_formatting(self, text: str) -> str:
        """Clean text formatting issues from scraping."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common scraping artifacts
        text = re.sub(r'\s*\n\s*', '\n', text)  # Clean newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Fix common HTML artifacts
        text = re.sub(r'\s*\|\s*', ' | ', text)  # Navigation separators
        text = re.sub(r'\s*>\s*', ' > ', text)   # Breadcrumb separators
        
        # Remove standalone single characters (common scraping artifacts)
        text = re.sub(r'\n\s*[a-zA-Z]\s*\n', '\n', text)
        
        # Clean up repeated words (scraping duplicates)
        words = text.split()
        cleaned_words = []
        prev_word = ""
        for word in words:
            if word.lower() != prev_word.lower() or len(word) <= 3:
                cleaned_words.append(word)
                prev_word = word
        
        text = ' '.join(cleaned_words)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean paragraph breaks
        
        return text.strip()
    
    def _determine_source_type(self, url: str, soup: BeautifulSoup) -> str:
        """Determine the type of source."""
        domain = urlparse(url).netloc.lower()
        
        if 'stackoverflow.com' in domain:
            return 'stackoverflow'
        elif 'github.com' in domain:
            return 'github'
        elif 'medium.com' in domain or 'dev.to' in domain:
            return 'blog'
        elif 'docs.' in domain or 'documentation' in url.lower():
            return 'documentation'
        elif 'tutorial' in url.lower() or 'guide' in url.lower():
            return 'tutorial'
        else:
            return 'general'

class GitHubFileContentScraper:
    """Scraper for individual GitHub repository files with quality metrics."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.session = requests.Session()
        # No need for API headers since we're using raw URLs
        self.quality_filter = QualityFilter()
    
    def load_path_analysis(self, analysis_file: str = 'react_github_paths_analysis.json') -> List[Dict]:
        """Load GitHub path analysis data."""
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Analysis file {analysis_file} not found[/red]")
            return []
    
    def get_file_content(self, owner: str, repo: str, file_path: str, branch: str = 'main') -> Optional[str]:
        """Get content of a specific file from GitHub repository using raw URL."""
        # Use raw.githubusercontent.com to get file content directly
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        
        try:
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            return response.text
                
        except requests.exceptions.RequestException as e:
            # Try with 'master' branch as fallback
            if branch == 'main':
                return self.get_file_content(owner, repo, file_path, 'master')
            console.print(f"[yellow]Failed to fetch {file_path} from {owner}/{repo}: {e}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[yellow]Error fetching {file_path}: {e}[/yellow]")
            return None
    
    def analyze_file_quality(self, content: str, file_info: Dict) -> Dict[str, Any]:
        """Analyze quality metrics for a file."""
        if not content:
            return {'quality_score': 0, 'coherence_score': 0, 'word_count': 0}
        
        word_count = len(content.split())
        
        # Base quality score
        quality_score = self.quality_filter.calculate_quality_score(content, file_info.get('path', ''))
        
        # File-specific scoring
        extension = file_info.get('extension', '').lower()
        
        # Bonus for source files
        if file_info.get('is_source', False):
            quality_score += 1.0
        
        # Bonus for well-structured files
        if extension in ['.js', '.jsx', '.ts', '.tsx']:
            # Check for React patterns
            react_patterns = ['import React', 'useState', 'useEffect', 'component', 'props']
            pattern_count = sum(1 for pattern in react_patterns if pattern.lower() in content.lower())
            quality_score += pattern_count * 0.2
            
            # Check for good practices
            if 'export default' in content:
                quality_score += 0.3
            if 'PropTypes' in content or 'interface' in content:
                quality_score += 0.5
        
        # Coherence score based on structure
        coherence_score = self.quality_filter.calculate_coherence_score(content)
        
        return {
            'quality_score': min(10.0, quality_score),
            'coherence_score': coherence_score,
            'word_count': word_count,
            'has_comments': '//' in content or '/*' in content,
            'has_documentation': '/**' in content or '@param' in content,
            'complexity_score': self._calculate_complexity_score(content, extension)
        }
    
    def _calculate_complexity_score(self, content: str, extension: str) -> float:
        """Calculate code complexity score."""
        if extension not in ['.js', '.jsx', '.ts', '.tsx', '.py']:
            return 0.0
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # Count functions, classes, and control structures
        function_count = len(re.findall(r'function\s+\w+|const\s+\w+\s*=.*=>', content))
        class_count = len(re.findall(r'class\s+\w+', content))
        control_structures = len(re.findall(r'\b(if|for|while|switch|try)\b', content))
        
        # Complexity based on structure
        complexity = (function_count * 0.5 + class_count * 1.0 + control_structures * 0.3) / len(non_empty_lines) * 100
        
        return min(10.0, complexity)
    
    def scrape_repository_files(self, repo_data: Dict, max_files_per_repo: int = 50, output_file: str = None) -> List[Dict]:
        """Scrape individual files from a repository."""
        if 'error' in repo_data:
            return []
        
        owner = repo_data['owner']
        repo = repo_data['repo']
        file_paths = repo_data.get('file_paths', [])
        
        # Filter for source files and important files
        priority_files = []
        for file_info in file_paths:
            if file_info['type'] == 'blob' and (
                file_info.get('is_source', False) or 
                file_info.get('is_config', False) or
                file_info.get('extension') in ['.md', '.json', '.yml', '.yaml']
            ):
                priority_files.append(file_info)
        
        # Sort by importance and limit
        priority_files.sort(key=lambda x: (
            x.get('is_source', False),
            x.get('size', 0) > 100,  # Prefer non-empty files
            -x.get('depth', 0)  # Prefer files closer to root
        ), reverse=True)
        
        priority_files = priority_files[:max_files_per_repo]
        
        scraped_files = []
        
        console.print(f"[blue]Scraping {len(priority_files)} files from {owner}/{repo}[/blue]")
        
        for file_info in priority_files:
            file_path = file_info['path']
            content = self.get_file_content(owner, repo, file_path)
            
            if content:
                quality_metrics = self.analyze_file_quality(content, file_info)
                
                # Only include high-quality files with sufficient content
                if quality_metrics['quality_score'] >= 5.0 and quality_metrics['word_count'] >= 90:
                    file_id = hashlib.md5(f"{owner}/{repo}/{file_path}".encode()).hexdigest()[:12]
                    
                    file_data = {
                        'id': file_id,
                        'url': f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}",
                        'title': f"{repo} - {file_path}",
                        'content': content,
                        'metadata': {
                            'source_type': 'github_file',
                            'repository': f"{owner}/{repo}",
                            'file_path': file_path,
                            'file_extension': file_info.get('extension'),
                            'file_size': file_info.get('size', 0),
                            'is_source': file_info.get('is_source', False),
                            'is_config': file_info.get('is_config', False),
                            'depth': file_info.get('depth', 0),
                            'quality_score': quality_metrics['quality_score'],
                            'coherence_score': quality_metrics['coherence_score'],
                            'word_count': quality_metrics['word_count'],
                            'complexity_score': quality_metrics['complexity_score'],
                            'has_comments': quality_metrics['has_comments'],
                            'has_documentation': quality_metrics['has_documentation'],
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    scraped_files.append(file_data)
                    
                    # Save incrementally if output file is provided
                    if output_file:
                        self.save_file_incrementally(file_data, output_file)
                else:
                    console.print(f"[yellow]⏭️  Skipped {file_path} (words: {quality_metrics['word_count']}, quality: {quality_metrics['quality_score']:.1f})[/yellow]")
            
            # Rate limiting
            time.sleep(0.1)
        
        console.print(f"[green]Successfully scraped {len(scraped_files)} high-quality files from {owner}/{repo}[/green]")
        return scraped_files
    
    def save_file_incrementally(self, file_data: Dict, output_file: str):
        """Save a single file to the dataset incrementally."""
        import os
        
        # Check if file exists and has content
        file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        
        try:
            if file_exists:
                # Read existing data
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Append new file
                existing_data.append(file_data)
                
                # Write back
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            else:
                # Create new file with first entry
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([file_data], f, indent=2, ensure_ascii=False)
                    
            console.print(f"[cyan]💾 Saved: {file_data['title']}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error saving file incrementally: {e}[/red]")

class HighQualitySourceAgent:
    """Agent that targets high-quality sources for specific topics."""
    
    def __init__(self, topic: str = "React JavaScript"):
        self.topic = topic
        self.scraper = WebScraper()
        self.github_scraper = GitHubFileContentScraper()

    def search_stackoverflow(self, query: str, max_results: int = 20) -> List[str]:
        """Search StackOverflow for high-quality Q&A."""
        try:
            # StackOverflow search API
            api_url = f"https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'q': query,
                'accepted': 'True',
                'site': 'stackoverflow',
                'pagesize': max_results,
                'filter': 'withbody'
            }
            
            response = requests.get(api_url, params=params)
            data = response.json()
            
            urls = []
            for item in data.get('items', []):
                if item.get('score', 0) >= 5:  # Only high-scored questions
                    urls.append(item['link'])
            
            return urls
            
        except Exception as e:
            console.print(f"[red]Error searching StackOverflow: {e}[/red]")
            return []
    
    def scrape_high_quality_content(self, max_pages: int = 50, min_quality: float = 6.0) -> List[ScrapedContent]:
        """Scrape high-quality content from multiple sources."""
        
        console.print(f"[blue]🔍 Scraping high-quality content for: {self.topic}[/blue]")
        
        # Get source URLs
        source_urls = self.get_high_quality_sources()
        
        # Add StackOverflow results
        so_query = self.topic.replace(' ', '+')
        so_urls = self.search_stackoverflow(so_query, 30)
        source_urls.extend(so_urls)
        
        # Limit total URLs
        source_urls = source_urls[:max_pages]
        
        console.print(f"[cyan]Found {len(source_urls)} sources to scrape[/cyan]")
        
        scraped_content = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scraping content...", total=len(source_urls))
            
            with ThreadPoolExecutor(max_workers=self.scraper.max_workers) as executor:
                future_to_url = {executor.submit(self.scraper.scrape_url, url): url for url in source_urls}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content and content.quality_score >= min_quality:
                            scraped_content.append(content)
                            console.print(f"[green]✓ {url} - Quality: {content.quality_score:.1f}[/green]")
                        else:
                            console.print(f"[yellow]⚠ {url} - Low quality, skipped[/yellow]")
                    except Exception as e:
                        console.print(f"[red]✗ {url} - Error: {e}[/red]")
                    
                    progress.update(task, advance=1)
                    time.sleep(self.scraper.delay)
        
        # Sort by quality score
        scraped_content.sort(key=lambda x: x.quality_score, reverse=True)
        
        console.print(f"[green]✓ Scraped {len(scraped_content)} high-quality documents[/green]")
        return scraped_content

def create_retrieval_dataset(scraped_content: List[ScrapedContent], output_file: str = "retrieval_dataset.json"):
    """Convert scraped content to retrieval dataset format."""
    
    dataset = []
    
    for content in scraped_content:
        # Create retrieval entry
        entry = {
            "id": hashlib.md5(content.url.encode()).hexdigest()[:12],
            "url": content.url,
            "title": content.title,
            "content": content.content,
            "metadata": {
                "source_type": content.source_type,
                "quality_score": content.quality_score,
                "coherence_score": content.coherence_score,
                "word_count": content.word_count,
                "timestamp": content.timestamp
            }
        }
        dataset.append(entry)
    
    # Save dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Retrieval dataset saved to: {output_file}[/green]")
    
    # Print statistics
    avg_quality = sum(c.quality_score for c in scraped_content) / len(scraped_content)
    avg_coherence = sum(c.coherence_score for c in scraped_content) / len(scraped_content)
    total_words = sum(c.word_count for c in scraped_content)
    
    console.print(f"\n[bold blue]📊 Dataset Statistics:[/bold blue]")
    console.print(f"Total documents: [cyan]{len(dataset)}[/cyan]")
    console.print(f"Average quality: [green]{avg_quality:.1f}/10[/green]")
    console.print(f"Average coherence: [green]{avg_coherence:.1f}/10[/green]")
    console.print(f"Total words: [yellow]{total_words:,}[/yellow]")
    
    # Source type breakdown
    source_types = {}
    for content in scraped_content:
        source_types[content.source_type] = source_types.get(content.source_type, 0) + 1
    
    console.print(f"\n[bold blue]📈 Source Types:[/bold blue]")
    for source_type, count in source_types.items():
        console.print(f"{source_type}: [cyan]{count}[/cyan]")

class HighQualitySourceAgent:
    """Agent that targets high-quality sources for specific topics."""
    
    def __init__(self, topic: str = "React JavaScript"):
        self.topic = topic
        self.scraper = WebScraper()
        self.github_scraper = GitHubFileContentScraper()

    def scrape_github_repositories(self, max_repos: int = 10, max_files_per_repo: int = 30) -> List[Dict]:
        """Scrape GitHub repositories from path analysis."""
        console.print("[blue]Loading GitHub path analysis...[/blue]")
        
        path_analysis = self.github_scraper.load_path_analysis()
        if not path_analysis:
            console.print("[red]No path analysis data found[/red]")
            return []
        
        # Filter successful repositories and sort by quality
        successful_repos = [repo for repo in path_analysis if 'error' not in repo and 'quality_metrics' in repo]
        successful_repos.sort(key=lambda x: x['quality_metrics'].get('overall_quality', 0), reverse=True)
        
        console.print(f"[green]Found {len(successful_repos)} repositories to scrape[/green]")
        
        all_scraped_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scraping repository files...", total=min(len(successful_repos), max_repos))
            
            for repo_data in successful_repos[:max_repos]:
                scraped_files = self.github_scraper.scrape_repository_files(repo_data, max_files_per_repo, output_file)
                all_scraped_files.extend(scraped_files)
                progress.advance(task)
                
                # Rate limiting between repositories
                time.sleep(1)
        
        console.print(f"[green]Total files scraped: {len(all_scraped_files)}[/green]")
        return all_scraped_files

def create_github_file_dataset(scraped_files: List[Dict], output_file: str = "github_files_dataset.json"):
    """Create a dataset from scraped GitHub files."""
    # If files were saved incrementally, we don't need to save again
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        console.print(f"[green]Dataset already saved incrementally to {output_file}[/green]")
    else:
        console.print(f"[blue]Creating dataset with {len(scraped_files)} files...[/blue]")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_files, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Dataset saved to {output_file}[/green]")
    
    # Print statistics
    total_files = len(scraped_files)
    total_size = sum(len(f['content']) for f in scraped_files)
    avg_quality = sum(f['metadata']['quality_score'] for f in scraped_files) / total_files if total_files > 0 else 0
    
    console.print(f"[cyan]Dataset Statistics:[/cyan]")
    console.print(f"  Total files: {total_files}")
    console.print(f"  Total content size: {total_size:,} characters")
    console.print(f"  Average quality score: {avg_quality:.2f}")
    
    return output_file

if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "web"  # "web" or "github"
    
    if mode == "github":
        # GitHub file scraping mode
        max_repos = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        max_files_per_repo = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        output_file = sys.argv[4] if len(sys.argv) > 4 else "github_files_dataset.json"
        
        # Create scraping agent
        agent = HighQualitySourceAgent("React JavaScript")
        
        # Scrape GitHub repositories
        scraped_files = agent.scrape_github_repositories(max_repos, max_files_per_repo)
        
        if scraped_files:
            # Create GitHub files dataset
            create_github_file_dataset(scraped_files, output_file)
        else:
            console.print("[red]No GitHub files found![/red]")
    
    else:
        # Web scraping mode (original functionality)
        topic = sys.argv[2] if len(sys.argv) > 2 else "React JavaScript"
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        min_quality = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
        output_file = sys.argv[5] if len(sys.argv) > 5 else "retrieval_dataset.json"
        
        # Create scraping agent
        agent = HighQualitySourceAgent(topic)
        
        # Scrape content
        scraped_content = agent.scrape_high_quality_content(max_pages, min_quality)
        
        if scraped_content:
            # Create retrieval dataset
            create_retrieval_dataset(scraped_content, output_file)
        else:
            console.print("[red]No high-quality content found![/red]")
