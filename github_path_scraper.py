#!/usr/bin/env python3
"""
GitHub Repository Path Scraper with Quality Metrics

This script scrapes GitHub repositories to extract file paths and analyze
repository structure with quality metrics for React.js projects.
"""

import requests
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

@dataclass
class FilePathInfo:
    """Information about a file path in a repository."""
    path: str
    type: str  # 'file' or 'dir'
    size: Optional[int]
    extension: Optional[str]
    depth: int
    is_config: bool
    is_source: bool
    is_test: bool
    is_docs: bool

@dataclass
class RepoQualityMetrics:
    """Quality metrics for a GitHub repository."""
    total_files: int
    total_directories: int
    source_files: int
    test_files: int
    config_files: int
    documentation_files: int
    max_depth: int
    avg_depth: float
    file_extensions: Dict[str, int]
    structure_score: float
    completeness_score: float
    organization_score: float
    overall_quality: float

class GitHubPathScraper:
    """Scraper for GitHub repository file paths and structure analysis."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        
        # Source file extensions (for classification)
        self.source_extensions = {
            '.js', '.jsx', '.ts', '.tsx', '.vue', '.svelte',
            '.py', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift',
            '.kt', '.scala', '.clj', '.hs', '.elm', '.dart',
            '.ipynb'  # Jupyter notebooks
        }
        self.config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env'
        }
        
        self.config_files = {
            'package.json', 'tsconfig.json', 'webpack.config.js', 'babel.config.js',
            'eslintrc.js', '.eslintrc.json', 'prettier.config.js', '.prettierrc',
            'jest.config.js', 'vite.config.js', 'next.config.js', 'tailwind.config.js',
            'dockerfile', 'docker-compose.yml', '.gitignore', 'readme.md', 'license'
        }
        
        self.test_patterns = [
            r'.*test.*', r'.*spec.*', r'__tests__', r'tests?/', r'.*\.test\.',
            r'.*\.spec\.', r'cypress/', r'e2e/', r'integration/'
        ]
        
        self.docs_patterns = [
            r'docs?/', r'documentation/', r'readme', r'.*\.md$', r'.*\.rst$',
            r'.*\.txt$', r'changelog', r'contributing', r'license'
        ]

    def extract_repo_info(self, github_url: str) -> Tuple[str, str]:
        """Extract owner and repo name from GitHub URL."""
        parsed = urlparse(github_url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]
            return owner, repo
        else:
            raise ValueError(f"Invalid GitHub URL format: {github_url}")

    def get_default_branch(self, owner: str, repo: str) -> Optional[str]:
        """Get the default branch name for a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('default_branch', 'main')
        except requests.exceptions.RequestException:
            return None

    def get_repository_tree(self, owner: str, repo: str, branch: Optional[str] = None) -> List[Dict]:
        """Get the complete file tree of a repository."""
        # First, get the default branch if not specified
        if branch is None:
            branch = self.get_default_branch(owner, repo)
            if branch is None:
                console.print(f"[red]Could not determine default branch for {owner}/{repo}[/red]")
                return []
        
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('tree', [])
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error fetching repository tree for {owner}/{repo}: {e}[/red]")
            
            # Try common branch names as fallbacks
            fallback_branches = ['master', 'develop', 'dev', 'main']
            if branch in fallback_branches:
                fallback_branches.remove(branch)
            
            for fallback_branch in fallback_branches:
                try:
                    fallback_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{fallback_branch}?recursive=1"
                    response = self.session.get(fallback_url)
                    response.raise_for_status()
                    data = response.json()
                    console.print(f"[yellow]Using fallback branch '{fallback_branch}' for {owner}/{repo}[/yellow]")
                    return data.get('tree', [])
                except requests.exceptions.RequestException:
                    continue
            
            console.print(f"[red]All branch attempts failed for {owner}/{repo}[/red]")
            return []

    def get_file_content(self, owner: str, repo: str, file_path: str, branch: str = 'main') -> Optional[str]:
        """Get content of a specific file from GitHub repository using raw URL."""
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        
        try:
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            # Try with 'master' branch as fallback
            if branch == 'main':
                return self.get_file_content(owner, repo, file_path, 'master')
            return None
        except Exception:
            return None

    def clean_content(self, content: str) -> str:
        """Clean content by removing copyright headers and excessive comments."""
        if not content:
            return content
            
        lines = content.split('\n')
        cleaned_lines = []
        skip_copyright = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip copyright headers
            if '/*---------------------------------------------------------------------------------------------' in line:
                skip_copyright = True
                continue
            elif skip_copyright and '*/' in line:
                skip_copyright = False
                continue
            elif skip_copyright:
                continue
                
            # Skip single-line copyright comments
            if (line_stripped.startswith('//') or line_stripped.startswith('#')) and any(word in line_stripped.lower() for word in ['copyright', 'license', 'licensed under']):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def count_words(self, content: str) -> int:
        """Count words in file content."""
        if not content:
            return 0
        # Simple word counting - split by whitespace and filter out very short tokens
        words = [word.strip() for word in content.split() if len(word.strip()) > 2]
        return len(words)
    
    def calculate_quality_score(self, content: str, file_path: str) -> float:
        """Calculate content quality score (0-10) based on retrieval_scraper logic."""
        score = 5.0  # Base score
        
        # Word count scoring
        word_count = self.count_words(content)
        if 90 <= word_count <= 5000:
            score += 1.0
        elif word_count < 90:
            score -= 2.0
        
        # File extension scoring
        extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
        if extension in ['ts', 'tsx', 'js', 'jsx', 'py', 'rs', 'go']:
            score += 1.0
        
        # Code structure scoring
        if any(keyword in content.lower() for keyword in ['function', 'class', 'interface', 'import', 'export']):
            score += 1.0
        
        # Technical keywords
        tech_keywords = ['react', 'typescript', 'javascript', 'python', 'component', 'function', 'method', 'api']
        keyword_count = sum(1 for keyword in tech_keywords if keyword in content.lower())
        score += min(keyword_count * 0.3, 1.5)
        
        # Avoid spam indicators
        if content.count('TODO') > word_count * 0.1:  # Too many TODOs
            score -= 0.5
            
        return min(10.0, max(0.0, score))
    
    def calculate_coherence_score(self, content: str) -> float:
        """Calculate content coherence score (0-10)."""
        score = 5.0
        
        # Function/class structure
        lines = content.split('\n')
        structured_lines = sum(1 for line in lines if any(keyword in line for keyword in ['function', 'class', 'interface', 'def']))
        if structured_lines >= 2:
            score += 1.0
        
        # Consistent indentation
        indented_lines = sum(1 for line in lines if line.startswith('  ') or line.startswith('\t'))
        if indented_lines > len(lines) * 0.3:  # Good indentation
            score += 0.5
        
        # Import statements (indicates organized code)
        import_lines = sum(1 for line in lines if line.strip().startswith(('import', 'from', '#include', 'use ')))
        if import_lines >= 1:
            score += 0.5
            
        return min(10.0, max(0.0, score))

    def is_unwanted_file(self, path: str) -> bool:
        """Check if file should be skipped based on smart filtering rules."""
        path_lower = path.lower()
        filename = Path(path).name.lower()
        
        # Skip cache and temporary files
        cache_patterns = [
            'node_modules/', '.git/', '.vscode/', '.idea/', '__pycache__/',
            'dist/', 'build/', 'out/', '.next/', '.nuxt/', 'coverage/',
            '.cache/', 'tmp/', 'temp/', '.tmp/', '.temp/', 'logs/'
        ]
        
        # Skip build and dependency files
        build_patterns = [
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            'requirements.txt', 'pipfile.lock', 'poetry.lock',
            'composer.lock', 'gemfile.lock', 'cargo.lock'
        ]
        
        # Skip system and config files that are usually auto-generated
        system_patterns = [
            '.gitignore', '.gitattributes', '.dockerignore', '.eslintignore',
            '.prettierignore', '.editorconfig', '.env.example', '.env.local',
            'dockerfile', 'docker-compose.yml', 'makefile', 'rakefile'
        ]
        
        # Skip extension/plugin directories
        extension_patterns = [
            'extensions/', 'plugins/', 'addons/', 'modules/',
            '.github/', '.vscode/extensions/', 'chrome-extension/',
            'browser-extension/', 'vscode-extension/'
        ]
        
        # Skip minified and generated files
        generated_patterns = [
            '.min.js', '.min.css', '.bundle.js', '.chunk.js',
            '.generated.', '.auto.', '.compiled.'
        ]
        
        # Check all patterns
        all_patterns = cache_patterns + extension_patterns
        for pattern in all_patterns:
            if pattern in path_lower:
                return True
        
        # Check filename patterns
        filename_patterns = build_patterns + system_patterns + generated_patterns
        for pattern in filename_patterns:
            if pattern in filename or filename.endswith(pattern):
                return True
        
        return False

    def analyze_file_path(self, path_info: Dict, owner: str = None, repo: str = None) -> Optional[FilePathInfo]:
        """Analyze a single file path and extract information."""
        path = path_info['path']
        file_type = path_info['type']
        size = path_info.get('size', 0)
        
        # Smart filtering - skip unwanted files early
        if self.is_unwanted_file(path):
            console.print(f"[dim yellow]⏭️  Skipped {path} (unwanted file type)[/dim yellow]")
            return None
        
        # Calculate depth
        depth = len(Path(path).parts) - 1
        
        # Get extension
        extension = Path(path).suffix.lower() if file_type == 'blob' else None
        
        # Classify file type
        path_lower = path.lower()
        filename_lower = Path(path).name.lower()
        
        is_config = (
            extension in self.config_extensions or 
            filename_lower in self.config_files or
            any(pattern in path_lower for pattern in ['config', 'settings'])
        )
        
        is_source = extension in self.source_extensions if extension else False
        
        is_test = any(re.match(pattern, path_lower) for pattern in self.test_patterns)
        
        is_docs = any(re.match(pattern, path_lower) for pattern in self.docs_patterns)
        
        # For source files, check word count and save individually
        if is_source and owner and repo and file_type == 'blob':
            content = self.get_file_content(owner, repo, path)
            if content:
                # Clean content first
                cleaned_content = self.clean_content(content)
                word_count = self.count_words(cleaned_content)
                console.print(f"[cyan]📝 {path}: {word_count} words[/cyan]")
                if word_count < 90:
                    console.print(f"[yellow]⏭️  Skipped {path} (words: {word_count})[/yellow]")
                    return None  # Skip files with less than 90 words
                else:
                    # Calculate quality metrics
                    quality_score = self.calculate_quality_score(cleaned_content, path)
                    coherence_score = self.calculate_coherence_score(cleaned_content)
                    
                    console.print(f"[green]✅ Included {path} (words: {word_count}, quality: {quality_score:.1f}, coherence: {coherence_score:.1f})[/green]")
                    
                    # Save file immediately to single dataset file
                    file_data = {
                        'id': hashlib.md5(f"{owner}/{repo}/{path}".encode()).hexdigest()[:12],
                        'repository': f"{owner}/{repo}",
                        'file_path': path,
                        'content': cleaned_content,
                        'word_count': word_count,
                        'quality_score': quality_score,
                        'coherence_score': coherence_score,
                        'file_extension': extension,
                        'depth': depth,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_file_to_dataset(file_data, "specialized_repos_dataset.json")
            else:
                console.print(f"[red]❌ Could not fetch content for {path}[/red]")
                return None  # Skip files we can't fetch
        
        return FilePathInfo(
            path=path,
            type=file_type,
            size=size,
            extension=extension,
            depth=depth,
            is_config=is_config,
            is_source=is_source,
            is_test=is_test,
            is_docs=is_docs
        )

    def get_repository_files(self, owner: str, repo: str) -> List[Dict]:
        """Get all files from a GitHub repository using the Git Trees API."""
        try:
            # Get the default branch first
            repo_url = f"https://api.github.com/repos/{owner}/{repo}"
            response = self.session.get(repo_url, timeout=30)
            response.raise_for_status()
            
            repo_data = response.json()
            default_branch = repo_data.get('default_branch', 'main')
            
            # Get the tree recursively
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
            response = self.session.get(tree_url, timeout=30)
            response.raise_for_status()
            
            tree_data = response.json()
            files = tree_data.get('tree', [])
            
            # Filter and format files
            formatted_files = []
            for file_info in files:
                if not self.is_unwanted_file(file_info['path']):
                    formatted_files.append({
                        'path': file_info['path'],
                        'type': file_info['type'],  # 'blob' for files, 'tree' for directories
                        'size': file_info.get('size', 0)
                    })
            
            return formatted_files
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error fetching repository files: {e}[/red]")
            return []
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            return []

    def calculate_quality_metrics(self, file_paths: List[FilePathInfo]) -> RepoQualityMetrics:
        """Calculate quality metrics for the repository."""
        total_files = sum(1 for f in file_paths if f.type == 'blob')
        total_directories = sum(1 for f in file_paths if f.type == 'tree')
        
        source_files = sum(1 for f in file_paths if f.is_source)
        test_files = sum(1 for f in file_paths if f.is_test)
        config_files = sum(1 for f in file_paths if f.is_config)
        documentation_files = sum(1 for f in file_paths if f.is_docs)
        
        depths = [f.depth for f in file_paths if f.type == 'blob']
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Count file extensions
        extensions = {}
        for f in file_paths:
            if f.extension:
                extensions[f.extension] = extensions.get(f.extension, 0) + 1
        
        # Calculate quality scores
        structure_score = self._calculate_structure_score(file_paths, max_depth, avg_depth)
        completeness_score = self._calculate_completeness_score(
            total_files, source_files, test_files, config_files, documentation_files
        )
        organization_score = self._calculate_organization_score(file_paths)
        
        overall_quality = (structure_score + completeness_score + organization_score) / 3
        
        return RepoQualityMetrics(
            total_files=total_files,
            total_directories=total_directories,
            source_files=source_files,
            test_files=test_files,
            config_files=config_files,
            documentation_files=documentation_files,
            max_depth=max_depth,
            avg_depth=avg_depth,
            file_extensions=extensions,
            structure_score=structure_score,
            completeness_score=completeness_score,
            organization_score=organization_score,
            overall_quality=overall_quality
        )

    def _calculate_structure_score(self, file_paths: List[FilePathInfo], max_depth: int, avg_depth: float) -> float:
        """Calculate structure quality score (0-10)."""
        score = 10.0
        
        # Penalize excessive depth
        if max_depth > 8:
            score -= (max_depth - 8) * 0.5
        
        # Penalize high average depth
        if avg_depth > 4:
            score -= (avg_depth - 4) * 0.3
        
        # Check for common directory structure
        paths = [f.path for f in file_paths]
        has_src = any('src/' in path for path in paths)
        has_components = any('component' in path.lower() for path in paths)
        has_utils = any('util' in path.lower() or 'helper' in path.lower() for path in paths)
        
        if has_src:
            score += 0.5
        if has_components:
            score += 0.5
        if has_utils:
            score += 0.3
        
        return max(0, min(10, score))

    def _calculate_completeness_score(self, total: int, source: int, test: int, config: int, docs: int) -> float:
        """Calculate completeness quality score (0-10)."""
        if total == 0:
            return 0
        
        score = 0
        
        # Source files ratio
        source_ratio = source / total
        if source_ratio > 0.3:
            score += 3
        elif source_ratio > 0.1:
            score += 2
        elif source_ratio > 0:
            score += 1
        
        # Test coverage indicator
        test_ratio = test / max(source, 1)
        if test_ratio > 0.5:
            score += 3
        elif test_ratio > 0.2:
            score += 2
        elif test_ratio > 0:
            score += 1
        
        # Configuration files
        if config > 0:
            score += 2
        
        # Documentation
        if docs > 0:
            score += 2
        
        return min(10, score)

    def _calculate_organization_score(self, file_paths: List[FilePathInfo]) -> float:
        """Calculate organization quality score (0-10)."""
        paths = [f.path for f in file_paths]
        
        score = 5.0  # Base score
        
        # Check for organized structure
        has_separate_dirs = len(set(Path(p).parts[0] for p in paths if '/' in p)) > 1
        if has_separate_dirs:
            score += 1
        
        # Check for common React patterns
        react_patterns = [
            'components/', 'pages/', 'hooks/', 'utils/', 'services/',
            'styles/', 'assets/', 'public/', 'src/'
        ]
        
        pattern_matches = sum(1 for pattern in react_patterns 
                            if any(pattern in path for path in paths))
        score += min(3, pattern_matches * 0.5)
        
        # Penalize too many files in root
        root_files = sum(1 for f in file_paths if f.depth == 0 and f.type == 'blob')
        if root_files > 10:
            score -= (root_files - 10) * 0.1
        
        return max(0, min(10, score))

    def check_repository_exists(self, owner: str, repo: str) -> bool:
        """Check if repository exists and is accessible."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            response = self.session.get(url)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def scrape_repository(self, github_url: str, max_files: int = 50) -> Dict[str, Any]:
        """Scrape a single GitHub repository and return analysis."""
        try:
            owner, repo = self.extract_repo_info(github_url)
            console.print(f"[blue]Scraping repository: {owner}/{repo}[/blue]")
            
            # Check if repository exists first
            if not self.check_repository_exists(owner, repo):
                return {'error': 'Repository not found or not accessible', 'url': github_url}
            
            # Get repository file tree
            file_paths = self.get_repository_files(owner, repo)
            if not file_paths:
                return {'error': 'Could not fetch repository files', 'url': github_url}
            
            # Analyze each file path with limit
            analyzed_paths = []
            files_processed = 0
            
            for path_info in file_paths:
                if files_processed >= max_files:
                    console.print(f"[yellow]⏸️  Reached limit of {max_files} files for {owner}/{repo}[/yellow]")
                    break
                    
                analyzed = self.analyze_file_path(path_info, owner, repo)
                if analyzed:  # Only include non-None results
                    analyzed_paths.append(analyzed)
                    files_processed += 1
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(analyzed_paths)
            
            return {
                'url': github_url,
                'owner': owner,
                'repo': repo,
                'file_paths': [fp.__dict__ for fp in analyzed_paths],
                'quality_metrics': quality_metrics.__dict__,
                'total_files_analyzed': len(analyzed_paths),
                'files_limit_reached': files_processed >= max_files,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            console.print(f"[red]Error scraping {github_url}: {e}[/red]")
            return {
                'url': github_url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def save_file_to_dataset(self, file_data: Dict, output_file: str):
        """Save a single file to the main dataset progressively."""
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
                    
            console.print(f"[cyan]💾 Saved: {file_data['repository']}/{file_data['file_path']}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error saving file to dataset: {e}[/red]")

    def save_repository_incrementally(self, repo_data: Dict, output_file: str):
        """Save a single repository analysis incrementally."""
        import os
        
        # Check if file exists and has content
        file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        
        try:
            if file_exists:
                # Read existing data
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Append new repository
                existing_data.append(repo_data)
                
                # Write back
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            else:
                # Create new file with first entry
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([repo_data], f, indent=2, ensure_ascii=False)
                    
            console.print(f"[cyan]💾 Saved: {repo_data['owner']}/{repo_data['repo']}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error saving repository incrementally: {e}[/red]")

    def scrape_multiple_repositories(self, github_urls: List[str], output_file: str = 'github_paths_analysis.json') -> List[Dict]:
        """Scrape multiple GitHub repositories for path analysis."""
        results = []
        successful = []
        failed = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing repositories...", total=len(github_urls))
            
            for url in github_urls:
                result = self.scrape_repository(url)
                results.append(result)
                
                if 'error' in result:
                    failed.append(result)
                else:
                    successful.append(result)
                    # Save incrementally
                    self.save_repository_incrementally(result, output_file)
                
                progress.advance(task)
                
                # Rate limiting
                time.sleep(0.5)
        
        console.print(f"\n[green]Analysis saved progressively to {output_file}[/green]")
        console.print(f"\n[green]Successfully analyzed: {len(successful)} repositories[/green]")
        console.print(f"[red]Failed: {len(failed)} repositories[/red]")
        
        if successful:
            # Create summary table
            table = Table(title="Repository Quality Summary")
            table.add_column("Repository", style="cyan")
            table.add_column("Files", justify="right")
            table.add_column("Source", justify="right")
            table.add_column("Tests", justify="right")
            table.add_column("Quality", justify="right", style="green")
            
            for result in successful[:10]:  # Show top 10
                metrics = result['quality_metrics']
                table.add_row(
                    f"{result['owner']}/{result['repo']}",
                    str(metrics['total_files']),
                    str(metrics['source_files']),
                    str(metrics['test_files']),
                    f"{metrics['overall_quality']}/10"
                )
            
            console.print(table)

def load_github_token() -> Optional[str]:
    """Load GitHub token from config file or environment."""
    try:
        with open('hf_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config.get('github_token')
    except FileNotFoundError:
        import os
        return os.getenv('GITHUB_TOKEN')

def search_specialized_repositories(github_token: Optional[str] = None) -> List[str]:
    """Search GitHub for specialized repositories: TypeScript, ML, RL, Three.js."""
    all_urls = []
    
    # Define specialized search queries
    search_queries = [
        {
            'name': 'TypeScript',
            'query': 'typescript language:typescript stars:>500',
            'max_results': 25
        },
        {
            'name': 'Machine Learning',
            'query': 'machine-learning OR tensorflow OR pytorch OR "neural network" OR "deep learning" language:python language:typescript language:javascript stars:>50',
            'max_results': 30
        },
        {
            'name': 'Reinforcement Learning', 
            'query': 'reinforcement-learning OR "deep-q" OR "policy-gradient" OR "actor-critic" OR gym OR openai language:python language:typescript language:javascript stars:>10',
            'max_results': 25
        },
        {
            'name': 'Three.js',
            'query': 'three.js OR threejs OR "3d graphics" OR webgl OR "web graphics" language:typescript language:javascript stars:>20',
            'max_results': 30
        }
    ]
    
    # GitHub Search API endpoint
    search_url = "https://api.github.com/search/repositories"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    for search_config in search_queries:
        try:
            console.print(f"[blue]🔍 Searching for {search_config['name']} repositories...[/blue]")
            
            params = {
                'q': search_config['query'],
                'sort': 'stars',
                'order': 'desc',
                'per_page': min(search_config['max_results'], 100)
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            repositories = data.get('items', [])
            
            console.print(f"[green]Found {len(repositories)} {search_config['name']} repositories[/green]")
            
            for repo in repositories:
                repo_url = repo['html_url']
                stars = repo['stargazers_count']
                language = repo.get('language', 'Unknown')
                
                console.print(f"  ⭐ {stars:,} - {repo['full_name']} ({language})")
                all_urls.append(repo_url)
            
            # Rate limiting between searches
            time.sleep(2)
            
        except requests.exceptions.Timeout:
            console.print(f"[yellow]Timeout searching {search_config['name']}, retrying with shorter timeout...[/yellow]")
            try:
                response = requests.get(search_url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                repositories = data.get('items', [])
                console.print(f"[green]Found {len(repositories)} {search_config['name']} repositories (retry)[/green]")
                
                for repo in repositories:
                    repo_url = repo['html_url']
                    stars = repo['stargazers_count']
                    language = repo.get('language', 'Unknown')
                    console.print(f"  ⭐ {stars:,} - {repo['full_name']} ({language})")
                    all_urls.append(repo_url)
            except:
                console.print(f"[red]Failed to search {search_config['name']} after retry[/red]")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error searching {search_config['name']}: {e}[/red]")
            continue
        except Exception as e:
            console.print(f"[red]Unexpected error for {search_config['name']}: {e}[/red]")
            continue
    
    console.print(f"[cyan]Total repositories found: {len(all_urls)}[/cyan]")
    return all_urls

def main():
    """Main function to run the GitHub path scraper."""
    console.print("[bold blue]GitHub Repository Path Scraper[/bold blue]")
    
    # Load GitHub token
    github_token = load_github_token()
    if not github_token:
        console.print("[yellow]Warning: No GitHub token found. Rate limits may apply.[/yellow]")
    
    # Initialize scraper
    scraper = GitHubPathScraper(github_token)
    
    # Search for specialized repositories (TypeScript, ML, RL, Three.js)
    github_urls = search_specialized_repositories(github_token=github_token)
    
    if not github_urls:
        console.print("[red]No GitHub repositories found in search.[/red]")
        return
    
    # Scrape repositories
    results = scraper.scrape_multiple_repositories(
        github_urls, 
        output_file='react_github_paths_analysis.json'
    )
    
    console.print("[green]GitHub path scraping completed![/green]")

if __name__ == "__main__":
    main()
