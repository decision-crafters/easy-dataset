# FastMCP to Easy Dataset Integration

This directory contains a script to download and process the FastMCP repository and related documentation into an Easy Dataset project.

## `fastmcp_to_dataset.sh`

This shell script automates the process of:
1. Cloning the FastMCP repository (https://github.com/jlowin/fastmcp.git)
2. Crawling related documentation URLs
3. Processing the content into an Easy Dataset project
4. Generating questions and answers
5. Exporting the dataset in the specified format

### Prerequisites

- Python 3.6+
- Easy Dataset running locally (on port 1717)
- Git installed
- Internet access

### Usage

```bash
./fastmcp_to_dataset.sh [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir DIR` | Output directory | `./fastmcp_output` |
| `-n, --project-name NAME` | Project name | `FastMCP Documentation` |
| `-p, --provider PROVIDER` | LLM provider (openai, ollama, etc.) | `ollama` |
| `-m, --model MODEL` | LLM model name | `llama3` |
| `-k, --api-key KEY` | API key for the LLM provider | None |
| `-e, --endpoint URL` | Custom API endpoint | None |
| `-l, --language LANG` | Language for generation (en, zh) | `en` |
| `-f, --format FORMAT` | Export format (alpaca, sharegpt) | `alpaca` |
| `-t, --file-type TYPE` | Export file type (json, jsonl) | `jsonl` |
| `-s, --skip-export` | Skip exporting the dataset | False |
| `-h, --help` | Show help message | N/A |

### Examples

1. Basic usage with default settings:

```bash
./fastmcp_to_dataset.sh
```

2. Specify output directory and project name:

```bash
./fastmcp_to_dataset.sh --output-dir ./my_fastmcp_data --project-name "FastMCP Training Data"
```

3. Use OpenAI as the LLM provider:

```bash
./fastmcp_to_dataset.sh --provider openai --model gpt-3.5-turbo --api-key YOUR_API_KEY
```

4. Generate a dataset in ShareGPT format:

```bash
./fastmcp_to_dataset.sh --format sharegpt --file-type json
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

### Included URLs

The script processes the following URLs by default:

1. Repository: https://github.com/jlowin/fastmcp.git
2. Documentation: https://gofastmcp.com
3. MCP Protocol: https://modelcontextprotocol.io
4. MCP Server Guide: https://modelcontextprotocol.io/quickstart/server

### Troubleshooting

- If the script fails to connect to Easy Dataset, make sure it's running and accessible at http://localhost:1717
- If repository cloning fails, check your internet connection and that Git is installed
- If URL crawling fails, check your internet connection
- If question or answer generation fails, check that the LLM provider is configured correctly
