# Synthetic Dataset Generator CLI

A powerful command-line tool for generating synthetic datasets using open-source language models like Qwen-3-4B-Instruct-2507.

## Features

- **Topic-based Generation**: Generate datasets focused on specific topics
- **Template-driven**: Use custom input/output templates for consistent formatting
- **Uniqueness Checking**: Automatically avoid duplicate entries
- **Multiple Models**: Support for various open-source models
- **JSON Output**: Standard dataset format compatible with training pipelines
- **Validation**: Built-in dataset validation tools

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Dataset

```bash
python main.py generate \
  --topic "Python programming tutorials" \
  --input-template "How do I {specific_task} in Python?" \
  --output-template "Here's how to {specific_task} in Python: {detailed_explanation}" \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --num-entries 10 \
  --output-file "python_dataset.json"
```

### Command Options

- `--topic, -t`: Overall topic for dataset generation (required)
- `--input-template, -i`: Template for user input messages (required)
- `--output-template, -o`: Template for assistant output messages (required)
- `--model, -m`: Model to use (default: Qwen/Qwen2.5-3B-Instruct)
- `--output-file, -f`: Output JSON file (default: dataset.json)
- `--num-entries, -n`: Number of entries to generate (default: 1)
- `--existing-file, -e`: Existing dataset file to avoid duplicates
- `--verbose, -v`: Enable verbose output

### Validate Dataset

```bash
python main.py validate --file dataset.json
```

## Dataset Format

The generated dataset follows this JSON structure:

```json
[
  {
    "content": "How do I create a list in Python?",
    "role": "user"
  },
  {
    "content": "Here's how to create a list in Python: You can create a list using square brackets...",
    "role": "assistant"
  }
]
```

## Template Variables

Templates support placeholder variables that the model will fill in:
- Use `{variable_name}` syntax in templates
- The model will generate appropriate content for each variable
- Variables help ensure variety and relevance

## Examples

### Code Tutorial Dataset
```bash
python main.py generate \
  -t "JavaScript web development" \
  -i "How can I {action} using {technology} in JavaScript?" \
  -o "To {action} using {technology} in JavaScript: {step_by_step_guide}" \
  -n 5
```

### Q&A Dataset
```bash
python main.py generate \
  -t "Machine learning concepts" \
  -i "What is {ml_concept} and when should I use it?" \
  -o "{ml_concept} is {definition}. You should use it when {use_cases}" \
  -n 10
```

### Troubleshooting Dataset
```bash
python main.py generate \
  -t "Software debugging" \
  -i "I'm getting {error_type} when {scenario}. How do I fix this?" \
  -o "This {error_type} typically occurs because {cause}. Here's how to fix it: {solution}" \
  -n 8
```

## Model Support

The CLI supports various open-source models:
- Qwen/Qwen2.5-3B-Instruct (default)
- Qwen/Qwen2.5-7B-Instruct
- microsoft/DialoGPT-medium
- facebook/blenderbot-400M-distill
- Any HuggingFace model compatible with text generation

## Hardware Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster generation
- **Storage**: 5GB+ for model downloads

## Tips for Better Results

1. **Specific Topics**: Use focused topics rather than broad ones
2. **Clear Templates**: Make templates specific but flexible
3. **Appropriate Model**: Choose model size based on your hardware
4. **Batch Generation**: Generate multiple entries at once for efficiency
5. **Validation**: Always validate your dataset after generation

## Troubleshooting

### Model Loading Issues
- Ensure sufficient RAM/VRAM
- Try smaller models if memory is limited
- Check internet connection for model downloads

### Generation Quality
- Adjust temperature and sampling parameters
- Use more specific templates
- Try different models for your use case

### Uniqueness Problems
- Use more varied templates
- Increase generation parameters diversity
- Check existing dataset for patterns

## Contributing

Feel free to submit issues and enhancement requests!
