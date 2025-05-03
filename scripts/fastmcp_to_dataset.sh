#!/bin/bash
# FastMCP to Easy Dataset Integration Script
#
# This script clones the FastMCP repository and crawls related documentation URLs,
# then processes them into an Easy Dataset project for training an LLM.

set -e  # Exit on any error

# Default values
OUTPUT_DIR="./fastmcp_output"
PROJECT_NAME="FastMCP Documentation"
LLM_PROVIDER="ollama"
LLM_MODEL="llama3"
LANGUAGE="en"
EXPORT_FORMAT="alpaca"
EXPORT_FILE_TYPE="jsonl"

# Display help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Process FastMCP repository and related documentation into an Easy Dataset project."
    echo
    echo "Options:"
    echo "  -o, --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
    echo "  -n, --project-name NAME   Project name (default: $PROJECT_NAME)"
    echo "  -p, --provider PROVIDER   LLM provider (default: $LLM_PROVIDER)"
    echo "  -m, --model MODEL         LLM model (default: $LLM_MODEL)"
    echo "  -k, --api-key KEY         API key for the LLM provider"
    echo "  -e, --endpoint URL        Custom API endpoint"
    echo "  -l, --language LANG       Language for generation (en, zh) (default: $LANGUAGE)"
    echo "  -f, --format FORMAT       Export format (alpaca, sharegpt) (default: $EXPORT_FORMAT)"
    echo "  -t, --file-type TYPE      Export file type (json, jsonl) (default: $EXPORT_FILE_TYPE)"
    echo "  -s, --skip-export         Skip exporting the dataset"
    echo "  -q, --skip-questions      Skip generating questions"
    echo "  -a, --skip-answers        Skip generating answers"
    echo "  -u, --skip-upload         Skip uploading to Easy Dataset (just process content)"
    echo "  -b, --skip-browser        Skip opening browser for prompts tab"
    echo "  -c, --chunk-size SIZE     Chunk size for manual chunking (default: 1500)"
    echo "  -g, --generate-prompts    Generate prompts using Ollama"
    echo "  -d, --domain DOMAIN       Domain or topic for prompt generation (default: FastMCP)"
    echo "  -h, --help                Show this help message"
    echo
    echo "Example:"
    echo "  $0 --output-dir ./my_output --model gpt-4 --api-key YOUR_API_KEY"
    echo
    echo "Content-only mode (no Easy Dataset upload):"
    echo "  $0 --output-dir ./my_output --skip-upload"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -p|--provider)
            LLM_PROVIDER="$2"
            shift 2
            ;;
        -m|--model)
            LLM_MODEL="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -e|--endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        -f|--format)
            EXPORT_FORMAT="$2"
            shift 2
            ;;
        -t|--file-type)
            EXPORT_FILE_TYPE="$2"
            shift 2
            ;;
        -s|--skip-export)
            SKIP_EXPORT="--skip-export"
            shift
            ;;
        -q|--skip-questions)
            SKIP_QUESTIONS="--skip-questions"
            shift
            ;;
        -a|--skip-answers)
            SKIP_ANSWERS="--skip-answers"
            shift
            ;;
        -u|--skip-upload)
            SKIP_UPLOAD="--skip-upload"
            shift
            ;;
        -b|--skip-browser)
            SKIP_BROWSER="--skip-browser"
            shift
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -g|--generate-prompts)
            GENERATE_PROMPTS="--generate-prompts"
            shift
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Prepare command arguments
REPO_URL="https://github.com/jlowin/fastmcp.git"
RELATED_URLS="https://gofastmcp.com,https://modelcontextprotocol.io,https://modelcontextprotocol.io/quickstart/server"

# Build the command
CMD="python /root/easy-dataset/scripts/repo_to_dataset.py"
CMD+=" --repos $REPO_URL"
CMD+=" --urls $RELATED_URLS"
CMD+=" --output-dir $OUTPUT_DIR"
CMD+=" --project-name \"$PROJECT_NAME\""
CMD+=" --llm-provider $LLM_PROVIDER"
CMD+=" --llm-model $LLM_MODEL"
CMD+=" --language $LANGUAGE"
CMD+=" --export-format $EXPORT_FORMAT"
CMD+=" --export-file-type $EXPORT_FILE_TYPE"
CMD+=" --max-pages 5"

# Add optional arguments if provided
if [ ! -z "$API_KEY" ]; then
    CMD+=" --api-key $API_KEY"
fi

if [ ! -z "$ENDPOINT" ]; then
    CMD+=" --endpoint $ENDPOINT"
fi

if [ ! -z "$SKIP_EXPORT" ]; then
    CMD+=" $SKIP_EXPORT"
fi

if [ ! -z "$SKIP_QUESTIONS" ]; then
    CMD+=" $SKIP_QUESTIONS"
fi

if [ ! -z "$SKIP_ANSWERS" ]; then
    CMD+=" $SKIP_ANSWERS"
fi

if [ ! -z "$CHUNK_SIZE" ]; then
    CMD+=" --chunk-size $CHUNK_SIZE"
fi

if [ ! -z "$GENERATE_PROMPTS" ]; then
    CMD+=" $GENERATE_PROMPTS"
fi

if [ ! -z "$DOMAIN" ]; then
    CMD+=" --domain \"$DOMAIN\""
fi

if [ ! -z "$SKIP_BROWSER" ]; then
    CMD+=" $SKIP_BROWSER"
fi

if [ ! -z "$SKIP_UPLOAD" ]; then
    CMD+=" $SKIP_UPLOAD"
else
    # Check if Easy Dataset is running (only if we're not skipping upload)
    echo "Checking if Easy Dataset is running..."
    if ! curl -s http://localhost:1717 > /dev/null; then
        echo "Error: Easy Dataset is not running. Please start it before running this script."
        echo "If you just want to process content without uploading to Easy Dataset, use the --skip-upload option."
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print summary
echo "========================================"
echo "FastMCP to Easy Dataset Integration"
echo "========================================"
echo "Repository: $REPO_URL"
echo "Related URLs: $RELATED_URLS"
echo "Output directory: $OUTPUT_DIR"
echo "Project name: $PROJECT_NAME"
echo "LLM provider: $LLM_PROVIDER"
echo "LLM model: $LLM_MODEL"
echo "Language: $LANGUAGE"
echo "Export format: $EXPORT_FORMAT"
echo "Export file type: $EXPORT_FILE_TYPE"

# Show skip options if provided
if [ ! -z "$SKIP_UPLOAD" ]; then
    echo "Mode: Content processing only (no upload to Easy Dataset)"
else
    if [ ! -z "$SKIP_QUESTIONS" ]; then
        echo "Skip: Question generation"
    fi
    if [ ! -z "$SKIP_ANSWERS" ]; then
        echo "Skip: Answer generation"
    fi
    if [ ! -z "$SKIP_EXPORT" ]; then
        echo "Skip: Dataset export"
    fi
fi

# Show prompt generation options if provided
if [ ! -z "$GENERATE_PROMPTS" ]; then
    echo "Generate prompts: Yes"
    if [ ! -z "$DOMAIN" ]; then
        echo "Domain: $DOMAIN"
    else
        echo "Domain: FastMCP (default)"
    fi

    if [ ! -z "$SKIP_BROWSER" ]; then
        echo "Skip browser: Yes"
    else
        echo "Skip browser: No (will open prompts tab in browser)"
    fi
fi

# Show chunk size if provided
if [ ! -z "$CHUNK_SIZE" ]; then
    echo "Chunk size: $CHUNK_SIZE characters"
fi
echo "========================================"

# Run the command
echo "Starting processing..."
echo "$CMD"
eval "$CMD"

# Print completion message
echo "========================================"
echo "Processing complete!"
echo "========================================"
echo "Content saved to: $OUTPUT_DIR/combined_content.md"

if [ -z "$SKIP_UPLOAD" ]; then
    if [ -z "$SKIP_EXPORT" ] && [ -z "$SKIP_QUESTIONS" ] && [ -z "$SKIP_ANSWERS" ]; then
        echo "Dataset exported to: $OUTPUT_DIR/dataset.$EXPORT_FILE_TYPE"
        echo "You can now use this dataset for training your LLM."
    else
        echo "Project created in Easy Dataset."
        if [ -z "$SKIP_QUESTIONS" ]; then
            echo "Questions generated for the content."
            if [ -z "$SKIP_ANSWERS" ]; then
                echo "Answers generated for the questions."
            fi
        fi
    fi
else
    echo "Content processing completed without uploading to Easy Dataset."
    echo "You can use the combined content file for other purposes."
fi
echo "========================================"
