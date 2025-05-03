# Repository to Easy Dataset Integration

This directory contains scripts to integrate external content sources with Easy Dataset.

## `repo_to_dataset.py`

This script downloads content from git repositories and/or URLs, processes it, and loads it into Easy Dataset for creating training datasets.

### Prerequisites

- Python 3.6+
- Easy Dataset running locally or accessible via network
- Git installed (for repository cloning)
- Internet access (for URL crawling)

### Installation

The script will automatically install required Python packages if they're not already installed:

- GitPython (for repository operations)
- Crawl4AI (for URL crawling)
- Playwright (for browser automation)
- Requests (for API communication)

### Usage

```bash
python repo_to_dataset.py --repos <repo1,repo2,...> --urls <url1,url2,...> --project-name "My Project"
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--repos` | Comma-separated list of repository URLs | None |
| `--urls` | Comma-separated list of URLs to crawl | None |
| `--output-dir` | Directory to save the output | `./output` |
| `--project-name` | Name for the Easy Dataset project | `Project-{timestamp}` |
| `--base-url` | Base URL for Easy Dataset API | `http://localhost:1717` |
| `--llm-provider` | LLM provider (openai, ollama, etc.) | `ollama` |
| `--llm-model` | LLM model name | `llama3` |
| `--api-key` | API key for the LLM provider (if required) | None |
| `--endpoint` | Custom API endpoint (if required) | None |
| `--language` | Language for generation (en, zh) | `en` |
| `--deep-crawl` | Perform deep crawling for URLs | False |
| `--max-pages` | Maximum number of pages to crawl in deep mode | 5 |
| `--export-format` | Export format (alpaca, sharegpt) | `alpaca` |
| `--export-file-type` | Export file type (json, jsonl) | `jsonl` |
| `--skip-export` | Skip exporting the dataset | False |

### Examples

1. Process a single GitHub repository:

```bash
python repo_to_dataset.py --repos https://github.com/user/repo --project-name "GitHub Repo Dataset"
```

2. Process multiple URLs with deep crawling:

```bash
python repo_to_dataset.py --urls https://docs.example.com/api,https://docs.example.com/guide --deep-crawl --max-pages 10 --project-name "API Documentation"
```

3. Process both repositories and URLs:

```bash
python repo_to_dataset.py --repos https://github.com/user/repo1,https://github.com/user/repo2 --urls https://docs.example.com/api --project-name "Combined Dataset"
```

4. Use OpenAI as the LLM provider:

```bash
python repo_to_dataset.py --repos https://github.com/user/repo --llm-provider openai --llm-model gpt-3.5-turbo --api-key YOUR_API_KEY
```

5. Generate a dataset in ShareGPT format:

```bash
python repo_to_dataset.py --repos https://github.com/user/repo --export-format sharegpt --export-file-type json
```

### Output

The script creates the following output:

1. A directory structure in the specified output directory:
   - `repos/`: Contains the extracted repository content
   - `urls/`: Contains the crawled URL content
   - `combined_content.md`: Combined content from all sources
   - `dataset.jsonl` or `dataset.json`: The exported dataset (if not skipped)

2. A project in Easy Dataset with:
   - The uploaded content
   - Generated questions
   - Generated answers
   - Dataset ready for export

### Workflow

1. The script processes repositories and URLs to extract content
2. The content is combined into a single markdown file
3. A new project is created in Easy Dataset
4. The content is uploaded and split into chunks
5. Questions are generated from the chunks
6. Answers are generated for the questions
7. The dataset is exported in the specified format (if not skipped)

### Troubleshooting

- If the script fails to connect to Easy Dataset, make sure it's running and accessible at the specified URL
- If repository cloning fails, check that the repository URL is correct and accessible
- If URL crawling fails, check that the URL is accessible and doesn't require authentication
- If question or answer generation fails, check that the LLM provider is configured correctly
