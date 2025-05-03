#!/usr/bin/env python3
"""
Repository to Easy Dataset Integration Script

This script downloads content from git repositories and/or URLs,
processes it, and loads it into Easy Dataset for creating training datasets.

Usage:
    python repo_to_dataset.py --repos <repo1,repo2,...> --urls <url1,url2,...> --project-name "My Project"

Example:
    python repo_to_dataset.py \
        --repos https://github.com/user/repo1,https://github.com/user/repo2 \
        --urls https://docs.example.com/api,https://docs.example.com/guide \
        --project-name "API Documentation Dataset"
"""

import os
import sys
import json
import argparse
import logging
import asyncio
import tempfile
import shutil
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing required packages, install if missing
try:
    from git import Repo
except ImportError:
    logger.info("Installing GitPython...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
    from git import Repo

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except ImportError:
    logger.info("Installing Crawl4AI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "crawl4ai"])

    # Install playwright for Crawl4AI
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        except Exception as e:
            logger.warning(f"Error installing playwright browsers: {e}. Will try to continue anyway.")
    except Exception as e:
        logger.warning(f"Error installing playwright: {e}. Will try to continue anyway.")

    # Now try importing again
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

class EasyDatasetClient:
    """Client for interacting with the easy-dataset API."""

    def __init__(self, base_url: str = "http://localhost:1717"):
        """Initialize the easy-dataset client with the specified base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self.project_id = None
        self.available_endpoints = self._discover_endpoints()
        self.prompts = {
            "globalPrompt": "",
            "questionPrompt": "",
            "answerPrompt": "",
            "labelPrompt": "",
            "domainTreePrompt": ""
        }

    def _discover_endpoints(self) -> Dict[str, bool]:
        """
        Discover which API endpoints are available in this Easy Dataset installation.

        Returns:
            Dictionary mapping endpoint names to boolean values indicating availability
        """
        endpoints = {
            "text-split": False,
            "split": False,
            "chunks": False,
            "text-chunks": False,
            "model": False,
            "settings": False,
            "llm-config": False
        }

        # Create a temporary project to test endpoints
        try:
            temp_project_name = f"temp-project-{int(time.time())}"
            response = self.session.post(
                f"{self.base_url}/api/projects",
                json={
                    "name": temp_project_name,
                    "description": "Temporary project for endpoint discovery"
                }
            )
            response.raise_for_status()
            temp_project_id = response.json().get("id")

            if not temp_project_id:
                logger.warning("Failed to create temporary project for endpoint discovery")
                return endpoints

            # Test each endpoint
            for endpoint in endpoints.keys():
                try:
                    # Just send a HEAD request to check if the endpoint exists
                    response = self.session.head(
                        f"{self.base_url}/api/projects/{temp_project_id}/{endpoint}"
                    )
                    # If we get a 404, the endpoint doesn't exist
                    # Any other status code means the endpoint exists but might require specific parameters
                    endpoints[endpoint] = response.status_code != 404
                except requests.exceptions.RequestException:
                    # If we get an exception, assume the endpoint doesn't exist
                    endpoints[endpoint] = False

            # Clean up the temporary project
            try:
                response = self.session.delete(
                    f"{self.base_url}/api/projects/{temp_project_id}"
                )
            except requests.exceptions.RequestException:
                # If we can't delete the project, just log a warning
                logger.warning(f"Failed to delete temporary project {temp_project_id}")

            logger.info(f"Discovered available endpoints: {[ep for ep, available in endpoints.items() if available]}")
            return endpoints
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to discover endpoints: {e}")
            return endpoints

    def create_project(self, name: str, description: str = "") -> str:
        """
        Create a new project in easy-dataset.

        Args:
            name: The name of the project
            description: Optional description of the project

        Returns:
            The project ID
        """
        logger.info(f"Creating project: {name}")

        try:
            response = self.session.post(
                f"{self.base_url}/api/projects",
                json={
                    "name": name,
                    "description": description
                }
            )
            response.raise_for_status()
            data = response.json()
            self.project_id = data.get("id")
            logger.info(f"Project created with ID: {self.project_id}")
            return self.project_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create project: {e}")
            if hasattr(e, 'response') and e.response and e.response.text:
                logger.error(f"Response: {e.response.text}")
            raise

    def configure_llm(self, provider: str, model: str, api_key: str = None, endpoint: str = None) -> bool:
        """
        Configure the LLM settings for the project.

        Args:
            provider: The LLM provider (e.g., 'openai', 'ollama')
            model: The model name
            api_key: API key (if required)
            endpoint: API endpoint (if custom)

        Returns:
            True if successful, False otherwise
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return False

        logger.info(f"Configuring LLM: {provider}/{model}")

        # Build the model configuration
        model_config = {
            "provider": provider,
            "name": model,
            "temperature": 0.3,  # Lower temperature for more consistent outputs
            "maxTokens": 2048
        }

        if api_key:
            model_config["apiKey"] = api_key

        if endpoint:
            model_config["endpoint"] = endpoint
        elif provider == "openai":
            model_config["endpoint"] = "https://api.openai.com/v1"
        elif provider == "ollama":
            model_config["endpoint"] = "http://localhost:11434/api"

        # Store the model configuration for later use
        self.model_config = model_config

        # Try to configure the LLM, but don't fail if it doesn't work
        try:
            # Try different API endpoints for Easy Dataset
            try:
                # First try the settings endpoint
                response = self.session.patch(
                    f"{self.base_url}/api/projects/{self.project_id}/settings",
                    json={
                        "llmConfig": model_config
                    }
                )
                response.raise_for_status()
                logger.info("LLM configuration updated successfully via settings endpoint")
                return True
            except requests.exceptions.RequestException:
                # If that fails, try the model endpoint
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/projects/{self.project_id}/model",
                        json=model_config
                    )
                    response.raise_for_status()
                    logger.info("LLM configuration updated successfully via model endpoint")
                    return True
                except requests.exceptions.RequestException:
                    # If that fails, try the llm-config endpoint
                    try:
                        response = self.session.post(
                            f"{self.base_url}/api/projects/{self.project_id}/llm-config",
                            json=model_config
                        )
                        response.raise_for_status()
                        logger.info("LLM configuration updated successfully via llm-config endpoint")
                        return True
                    except requests.exceptions.RequestException:
                        # If all API endpoints fail, just log a warning and continue
                        logger.warning("Could not configure LLM via API. Will use model config in requests instead.")
                        return True
        except Exception as e:
            logger.warning(f"Failed to configure LLM: {e}")
            logger.warning("Continuing without LLM configuration. Will use model config in requests instead.")
            return True  # Return True to continue with the process

    def upload_file(self, content_path: str) -> str:
        """
        Upload a file to the project without splitting it.

        Args:
            content_path: Path to the content file

        Returns:
            File ID if successful, empty string otherwise
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return ""

        logger.info(f"Uploading file: {content_path}")

        try:
            # Upload the file
            with open(content_path, 'rb') as f:
                file_name = os.path.basename(content_path)
                response = self.session.post(
                    f"{self.base_url}/api/projects/{self.project_id}/files",
                    headers={
                        'x-file-name': file_name
                    },
                    data=f.read()
                )
                response.raise_for_status()
                file_data = response.json()
                file_name = file_data.get("fileName", file_name)
                file_id = file_data.get("id", "")

            logger.info(f"File uploaded: {file_name}")
            return file_id if file_id else file_name
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload file: {e}")
            if hasattr(e, 'response') and e.response and e.response.text:
                logger.error(f"Response: {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return ""

    def split_content(self, file_id: str) -> List[str]:
        """
        Split a file into chunks.

        Args:
            file_id: ID or name of the file to split

        Returns:
            List of chunk IDs
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return []

        logger.info(f"Splitting file: {file_id}")

        try:
            # Try to split the content using various methods
            chunk_ids = []

            # Method 1: Try the text-split endpoint with form data if available
            if self.available_endpoints.get("text-split", False):
                try:
                    logger.info("Trying text-split endpoint with form data...")

                    # Use form data instead of JSON
                    form_data = {
                        "fileName": file_id,
                        "minChars": "500",
                        "maxChars": "2000"
                    }

                    response = self.session.post(
                        f"{self.base_url}/api/projects/{self.project_id}/text-split",
                        data=form_data,
                        headers={
                            "Content-Type": "application/x-www-form-urlencoded"
                        }
                    )
                    response.raise_for_status()
                    split_data = response.json()

                    # Get the chunk IDs - handle different response formats
                    chunks = []
                    if isinstance(split_data, list) and len(split_data) > 0:
                        # Format: [{chunks: [...]}]
                        for item in split_data:
                            if "chunks" in item and isinstance(item["chunks"], list):
                                chunks.extend(item["chunks"])
                    elif isinstance(split_data, dict):
                        # Format: {chunks: [...]}
                        if "chunks" in split_data and isinstance(split_data["chunks"], list):
                            chunks = split_data["chunks"]
                        # Format: {fileResult: {chunks: [...]}}
                        elif "fileResult" in split_data and "chunks" in split_data["fileResult"]:
                            chunks = split_data["fileResult"]["chunks"]

                    # Extract chunk IDs
                    for chunk in chunks:
                        if isinstance(chunk, dict) and "id" in chunk:
                            chunk_ids.append(chunk["id"])
                        elif isinstance(chunk, str):
                            chunk_ids.append(chunk)

                    if chunk_ids:
                        logger.info(f"Content split into {len(chunk_ids)} chunks using text-split endpoint")
                        return chunk_ids
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to use text-split endpoint with form data: {e}")

                    # Try with multipart/form-data
                    try:
                        logger.info("Trying text-split endpoint with multipart/form-data...")

                        import io
                        from requests_toolbelt.multipart.encoder import MultipartEncoder

                        # Use multipart/form-data
                        form_data = MultipartEncoder(
                            fields={
                                "fileName": file_id,
                                "minChars": "500",
                                "maxChars": "2000"
                            }
                        )

                        response = self.session.post(
                            f"{self.base_url}/api/projects/{self.project_id}/text-split",
                            data=form_data,
                            headers={
                                "Content-Type": form_data.content_type
                            }
                        )
                        response.raise_for_status()
                        split_data = response.json()

                        # Get the chunk IDs - handle different response formats
                        chunks = []
                        if isinstance(split_data, list) and len(split_data) > 0:
                            # Format: [{chunks: [...]}]
                            for item in split_data:
                                if "chunks" in item and isinstance(item["chunks"], list):
                                    chunks.extend(item["chunks"])
                        elif isinstance(split_data, dict):
                            # Format: {chunks: [...]}
                            if "chunks" in split_data and isinstance(split_data["chunks"], list):
                                chunks = split_data["chunks"]
                            # Format: {fileResult: {chunks: [...]}}
                            elif "fileResult" in split_data and "chunks" in split_data["fileResult"]:
                                chunks = split_data["fileResult"]["chunks"]

                        # Extract chunk IDs
                        for chunk in chunks:
                            if isinstance(chunk, dict) and "id" in chunk:
                                chunk_ids.append(chunk["id"])
                            elif isinstance(chunk, str):
                                chunk_ids.append(chunk)

                        if chunk_ids:
                            logger.info(f"Content split into {len(chunk_ids)} chunks using text-split endpoint with multipart/form-data")
                            return chunk_ids
                    except Exception as e:
                        logger.warning(f"Failed to use text-split endpoint with multipart/form-data: {e}")
            else:
                logger.info("text-split endpoint not available, skipping...")

            # Method 2: Try the split endpoint if available
            if self.available_endpoints.get("split", False):
                try:
                    logger.info("Trying split endpoint...")

                    # Set up the model configuration
                    model_config = getattr(self, 'model_config', {
                        "provider": "ollama",
                        "name": "llama3",
                        "temperature": 0.3,
                        "maxTokens": 2048
                    })

                    # Change to send JSON instead of form data
                    json_payload = {
                        "fileName": file_id,
                        "model": model_config, # Send model config as nested JSON object
                        "language": "en"
                    }

                    # Initiate the split via POST
                    logger.info(f"Initiating POST request to /split for file {file_id}")
                    response = self.session.post(
                        f"{self.base_url}/api/projects/{self.project_id}/split",
                        json=json_payload,
                        headers={
                            "Content-Type": "application/json"
                        },
                        timeout=10 # Set a short timeout for the POST, we don't need the response
                    )
                    # We don't necessarily need to raise_for_status() here if the server
                    # processes asynchronously and doesn't return immediate success.
                    # Log status for info.
                    logger.info(f"POST /split initiated, status code: {response.status_code}")

                    # Wait for the server to process the split in the background
                    sleep_duration = 60 # seconds
                    logger.info(f"Waiting {sleep_duration} seconds for server to process split...")
                    time.sleep(sleep_duration)

                    # Now fetch the results using GET /split
                    logger.info(f"Fetching split results via GET request for project {self.project_id}")
                    get_response = self.session.get(
                         f"{self.base_url}/api/projects/{self.project_id}/split"
                    )
                    get_response.raise_for_status() # Raise if GET fails
                    split_data = get_response.json()


                    # Get the chunk IDs - handle different response formats
                    chunks = []
                    if isinstance(split_data, list) and len(split_data) > 0:
                        # Format: [{chunks: [...]}]
                        for item in split_data:
                            if "chunks" in item and isinstance(item["chunks"], list):
                                chunks.extend(item["chunks"])
                    elif isinstance(split_data, dict):
                        # Format: {chunks: [...]}
                        if "chunks" in split_data and isinstance(split_data["chunks"], list):
                            chunks = split_data["chunks"]
                        # Format: {fileResult: {chunks: [...]}}
                        elif "fileResult" in split_data and "chunks" in split_data["fileResult"]:
                            chunks = split_data["fileResult"]["chunks"]

                    # Extract chunk IDs
                    chunk_ids = []
                    for chunk in chunks:
                        if isinstance(chunk, dict) and "id" in chunk:
                            chunk_ids.append(chunk["id"])
                        elif isinstance(chunk, str):
                            chunk_ids.append(chunk)

                    if chunk_ids:
                        logger.info(f"Content split into {len(chunk_ids)} chunks using split endpoint")
                        return chunk_ids
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to use split endpoint: {e}")
            else:
                logger.info("split endpoint not available, skipping...")

            # Method 3: Use the file directly as a single chunk
            logger.info("Using the file directly as a single chunk...")
            return [file_id]
        except Exception as e:
            logger.error(f"Failed to split content: {e}")
            # Return the file ID as a fallback
            logger.info("Using the file directly as a single chunk (fallback)...")
            return [file_id]

    def upload_content(self, content_path: str) -> List[str]:
        """
        Upload content to the project and split it into chunks.

        Args:
            content_path: Path to the content file

        Returns:
            List of chunk IDs
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return []

        # First, upload the file
        file_id = self.upload_file(content_path)
        if not file_id:
            return []

        # Then, split the content
        return self.split_content(file_id)

    def generate_prompts(self, domain: str, language: str = "en") -> bool:
        """
        Generate prompts for the project using Ollama.

        Args:
            domain: The domain or topic of the project
            language: Language for prompt generation ('en' for English)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating prompts for domain: {domain}")

        try:
            # Use Ollama to generate prompts
            import json
            import subprocess

            # Global prompt
            global_prompt_request = f"""
            Generate a global prompt for an LLM that will be used to generate questions and answers about {domain}.
            The prompt should guide the LLM to generate high-quality, accurate, and relevant content.
            Keep the prompt concise (max 200 characters) and focused on quality and accuracy.
            Return only the prompt text without any explanations or formatting.
            """

            # Question prompt
            question_prompt_request = f"""
            Generate a prompt specifically for generating questions about {domain}.
            The prompt should guide the LLM to generate diverse, relevant, and thought-provoking questions.
            Questions should be clear, specific, and cover different aspects of {domain}.
            Keep the prompt concise (max 200 characters) and focused.
            Return only the prompt text without any explanations or formatting.
            """

            # Answer prompt
            answer_prompt_request = f"""
            Generate a prompt specifically for generating answers about {domain}.
            The prompt should guide the LLM to generate accurate, comprehensive, and well-structured answers.
            Answers should be based on facts, provide sufficient detail, and be easy to understand.
            Keep the prompt concise (max 200 characters) and focused.
            Return only the prompt text without any explanations or formatting.
            """

            # Generate prompts using Ollama
            global_prompt = self._call_ollama(global_prompt_request)
            question_prompt = self._call_ollama(question_prompt_request)
            answer_prompt = self._call_ollama(answer_prompt_request)

            # Update the prompts
            self.prompts["globalPrompt"] = global_prompt
            self.prompts["questionPrompt"] = question_prompt
            self.prompts["answerPrompt"] = answer_prompt

            logger.info("Prompts generated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}")
            return False

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama to generate a response using stdin for the prompt.

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            The generated response
        """
        try:
            import subprocess
            import json
            import sys

            # Ensure ollama command exists (optional, but good practice)
            # import shutil
            # if not shutil.which("ollama"):
            #     logger.error("Ollama command not found in PATH.")
            #     return ""

            # Call Ollama using subprocess.Popen and pass prompt via stdin
            # Use the model specified in the configuration, defaulting to llama3
            model_name = getattr(self, 'model_config', {}).get('name', 'llama3')
            cmd = ["ollama", "run", model_name] # Removed prompt from here

            logger.debug(f"Calling Ollama command: {' '.join(cmd)}")
            logger.debug(f"Passing prompt via stdin (length: {len(prompt)} chars)")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8' # Ensure consistent encoding
            )

            # Send the prompt via stdin and close it
            stdout, stderr = process.communicate(input=prompt)

            if process.returncode != 0:
                # Log stderr more verbosely
                error_message = f"Ollama call failed with return code {process.returncode}."
                if stderr:
                    error_message += f" Stderr: {stderr.strip()}"
                logger.error(error_message)
                # Also log stdout if it contains error info (sometimes happens)
                if stdout:
                     logger.error(f"Ollama stdout on error: {stdout.strip()}")
                return ""

            # Get the response text from stdout
            response_text = stdout.strip()

            # Clean up the response (remove potential extra quotes if ollama adds them)
            # This might need adjustment based on actual ollama output format
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1].strip()
            elif response_text.startswith("'") and response_text.endswith("'"):
                 response_text = response_text[1:-1].strip()

            logger.debug(f"Ollama response received (length: {len(response_text)} chars)")
            return response_text
        except FileNotFoundError:
             logger.error("Ollama command not found. Please ensure Ollama is installed and in your PATH.")
             return ""
        except Exception as e:
            # Catch potential encoding errors or other subprocess issues
            logger.error(f"Failed to call Ollama: {e}")
            # Log exception type for better debugging
            logger.error(f"Exception type: {type(e).__name__}")
            return ""

    def update_prompts(self) -> bool:
        """
        Update the prompts for the project.

        Returns:
            True if successful, False otherwise
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return False

        logger.info("Updating prompts for the project")

        try:
            # Method 1: Try to update the prompts using the config endpoint
            try:
                logger.info("Trying to update prompts using config endpoint...")
                response = self.session.put(
                    f"{self.base_url}/api/projects/{self.project_id}/config",
                    json={
                        "prompts": self.prompts
                    }
                )
                response.raise_for_status()
                logger.info("Prompts updated successfully using config endpoint")
                return True
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to update prompts using config endpoint: {e}")

                # Method 2: Try to update the prompts using the settings endpoint
                try:
                    logger.info("Trying to update prompts using settings endpoint...")
                    response = self.session.patch(
                        f"{self.base_url}/api/projects/{self.project_id}/settings",
                        json={
                            "prompts": self.prompts
                        }
                    )
                    response.raise_for_status()
                    logger.info("Prompts updated successfully using settings endpoint")
                    return True
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to update prompts using settings endpoint: {e}")

                    # Method 3: Try to update each prompt individually
                    try:
                        logger.info("Trying to update prompts individually...")
                        success = True

                        for prompt_key, prompt_value in self.prompts.items():
                            if not prompt_value:
                                continue

                            try:
                                response = self.session.put(
                                    f"{self.base_url}/api/projects/{self.project_id}/prompts/{prompt_key}",
                                    json={
                                        "value": prompt_value
                                    }
                                )
                                response.raise_for_status()
                                logger.info(f"Prompt {prompt_key} updated successfully")
                            except requests.exceptions.RequestException as e:
                                logger.warning(f"Failed to update prompt {prompt_key}: {e}")
                                success = False

                        if success:
                            logger.info("All prompts updated successfully individually")
                            return True
                        else:
                            logger.warning("Some prompts failed to update individually")
                            return False
                    except Exception as e:
                        logger.warning(f"Failed to update prompts individually: {e}")

                        # Method 4: Try to update using form data
                        try:
                            logger.info("Trying to update prompts using form data...")

                            # Convert prompts to form data
                            form_data = {}
                            for key, value in self.prompts.items():
                                if value:
                                    form_data[f"prompts.{key}"] = value

                            response = self.session.patch(
                                f"{self.base_url}/api/projects/{self.project_id}/settings",
                                data=form_data,
                                headers={
                                    "Content-Type": "application/x-www-form-urlencoded"
                                }
                            )
                            response.raise_for_status()
                            logger.info("Prompts updated successfully using form data")
                            return True
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"Failed to update prompts using form data: {e}")

                            # Method 5: Try to update using direct browser automation
                            try:
                                logger.info("Trying to update prompts using direct browser automation...")

                                # Create a JavaScript file to update the prompts
                                js_file_path = os.path.join(os.getcwd(), "update_prompts.js")
                                with open(js_file_path, 'w', encoding='utf-8') as f:
                                    f.write(f"""
                                    // This script updates the prompts for project {self.project_id}
                                    const projectId = "{self.project_id}";
                                    const prompts = {json.dumps(self.prompts)};

                                    // Function to update prompts
                                    async function updatePrompts() {{
                                        try {{
                                            // Method 1: Try to update using the API
                                            const response = await fetch(`/api/projects/${{projectId}}/config`, {{
                                                method: 'PUT',
                                                headers: {{
                                                    'Content-Type': 'application/json'
                                                }},
                                                body: JSON.stringify({{ prompts }})
                                            }});

                                            if (response.ok) {{
                                                console.log('Prompts updated successfully');
                                                return;
                                            }}

                                            // Method 2: Try to update using the settings API
                                            const response2 = await fetch(`/api/projects/${{projectId}}/settings`, {{
                                                method: 'PATCH',
                                                headers: {{
                                                    'Content-Type': 'application/json'
                                                }},
                                                body: JSON.stringify({{ prompts }})
                                            }});

                                            if (response2.ok) {{
                                                console.log('Prompts updated successfully via settings');
                                                return;
                                            }}

                                            // Method 3: Try to update the UI directly
                                            // This is a fallback method that might not work in all cases
                                            console.log('Trying to update prompts via UI...');

                                            // Wait for the form elements to be available
                                            setTimeout(() => {{
                                                // Update each prompt field
                                                for (const [key, value] of Object.entries(prompts)) {{
                                                    if (!value) continue;

                                                    // Find the textarea for this prompt
                                                    const textarea = document.querySelector(`textarea[name="prompts.${{key}}"]`);
                                                    if (textarea) {{
                                                        textarea.value = value;

                                                        // Trigger change event
                                                        const event = new Event('input', {{ bubbles: true }});
                                                        textarea.dispatchEvent(event);
                                                    }}
                                                }}

                                                // Find and click the save button
                                                const saveButton = document.querySelector('button[type="submit"]');
                                                if (saveButton) {{
                                                    saveButton.click();
                                                    console.log('Save button clicked');
                                                }}
                                            }}, 2000);
                                        }} catch (error) {{
                                            console.error('Error updating prompts:', error);
                                        }}
                                    }}

                                    // Run the update function when the page loads
                                    window.addEventListener('load', updatePrompts);

                                    // Also run it now in case the page is already loaded
                                    if (document.readyState === 'complete') {{
                                        updatePrompts();
                                    }}
                                    """)

                                logger.info(f"Created JavaScript file to update prompts: {js_file_path}")
                                logger.info("Please open the prompts tab in your browser and paste the contents of this file in the console")

                                # Open the prompts tab in the browser
                                import webbrowser
                                prompts_url = f"{self.base_url}/projects/{self.project_id}/settings?tab=prompts"
                                logger.info(f"Opening prompts tab in browser: {prompts_url}")
                                webbrowser.open(prompts_url)

                                return True
                            except Exception as e:
                                logger.warning(f"Failed to update prompts using direct browser automation: {e}")
        except Exception as e:
            logger.error(f"Failed to update prompts: {e}")

        # If all methods fail, return False
        logger.error("All methods to update prompts failed")
        return False

    def generate_questions(self, chunk_ids: List[str], language: str = "en") -> int:
        """
        Generate questions for the specified chunks.

        Args:
            chunk_ids: List of chunk IDs to generate questions for
            language: Language for question generation ('en' for English)

        Returns:
            Number of questions generated
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return 0

        logger.info(f"Generating questions for {len(chunk_ids)} chunks")

        try:
            # Set up the model configuration
            model_config = getattr(self, 'model_config', {
                "provider": "ollama",
                "name": "llama3",
                "temperature": 0.3,
                "maxTokens": 2048,
                "endpoint": "http://localhost:11434/api"
            })

            # Try to generate questions using the API
            try:
                # Generate questions
                response = self.session.post(
                    f"{self.base_url}/api/projects/{self.project_id}/generate-questions",
                    json={
                        "chunkIds": chunk_ids,
                        "language": language if language == "en" else "中文",
                        "model": model_config
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Count total questions generated
                total_questions = 0

                # Handle different response formats
                if "totalQuestions" in result:
                    total_questions = result.get("totalQuestions", 0)
                elif "results" in result and isinstance(result["results"], list):
                    # Sum up questions from each chunk
                    for chunk_result in result["results"]:
                        if "questions" in chunk_result and isinstance(chunk_result["questions"], list):
                            total_questions += len(chunk_result["questions"])
                        elif "total" in chunk_result:
                            total_questions += chunk_result["total"]

                logger.info(f"Generated {total_questions} questions using API")
                return total_questions
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to generate questions using API: {e}")
                if hasattr(e, 'response') and e.response and e.response.text:
                    logger.warning(f"Response: {e.response.text}")

                # If the API fails, try to generate questions directly using Ollama
                logger.info("Trying to generate questions directly using Ollama...")
                return self._generate_questions_directly(chunk_ids, language)
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return 0

    def _get_file_content(self, file_id: str) -> str:
        """
        Get the content of a file.

        Args:
            file_id: ID or name of the file

        Returns:
            Content of the file
        """
        try:
            # Try to get the file content from the API
            try:
                response = self.session.get(
                    f"{self.base_url}/api/projects/{self.project_id}/files/{file_id}"
                )
                response.raise_for_status()

                # Check if the response is JSON
                try:
                    file_data = response.json()
                    if isinstance(file_data, dict) and "content" in file_data:
                        return file_data["content"]
                except:
                    # If not JSON, return the raw content
                    return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to get file content from API: {e}")

                # Try to get the file content directly
                try:
                    # Try to download the file
                    response = self.session.get(
                        f"{self.base_url}/api/projects/{self.project_id}/files/{file_id}/download"
                    )
                    response.raise_for_status()
                    return response.text
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to download file: {e}")

                    # Try to get the file from the local filesystem
                    try:
                        file_path = os.path.join(os.getcwd(), file_id)
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                return f.read()
                        else:
                            # Try with the output directory
                            file_path = os.path.join(os.getcwd(), "my_fastmcp_data", file_id)
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    return f.read()
                    except Exception as e:
                        logger.warning(f"Failed to get file content from local filesystem: {e}")

            # If all methods fail, return empty string
            logger.error(f"Failed to get content for file {file_id}")
            return ""
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return ""

    def _generate_questions_directly(self, chunk_ids: List[str], language: str = "en") -> int:
        """
        Generate questions directly using Ollama without relying on the API.

        Args:
            chunk_ids: List of chunk IDs to generate questions for
            language: Language for question generation ('en' for English)

        Returns:
            Number of questions generated
        """
        try:
            # Get the chunks content
            chunks_content = {}
            for chunk_id in chunk_ids:
                # First try to get the chunk content from the API
                try:
                    response = self.session.get(
                        f"{self.base_url}/api/projects/{self.project_id}/chunks/{chunk_id}"
                    )
                    response.raise_for_status()
                    chunk_data = response.json()

                    if isinstance(chunk_data, dict) and "content" in chunk_data:
                        chunks_content[chunk_id] = chunk_data["content"]
                        # Got content, move to next chunk_id
                        continue
                    else:
                        logger.warning(f"Unexpected chunk data format for chunk {chunk_id}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to get chunk content for chunk {chunk_id}: {e}")

                # If chunk API fails, try file API (assuming chunk_id might be file_id)
                try:
                    logger.info(f"Trying to get content for {chunk_id} using file endpoints...")
                    file_content = self._get_file_content(chunk_id)
                    if file_content:
                        chunks_content[chunk_id] = file_content
                        # Got content, move to next chunk_id
                        continue
                    else:
                         logger.warning(f"Failed to get content for {chunk_id} using file endpoints.")
                except Exception as e:
                    logger.warning(f"Error getting content for {chunk_id} using file endpoints: {e}")

                # If still no content after all API attempts
                if chunk_id not in chunks_content:
                     logger.error(f"Failed to retrieve content for chunk/file {chunk_id} using API methods.")


            if not chunks_content:
                logger.error("No chunk content could be retrieved for question generation.")
                return 0

            # Generate questions for each chunk
            total_questions = 0
            questions_dir = os.path.join(os.getcwd(), "questions")
            os.makedirs(questions_dir, exist_ok=True)

            for chunk_id, content in chunks_content.items():
                # Generate questions using Ollama
                logger.info(f"Generating questions for chunk {chunk_id} using Ollama...")

                # Prepare the prompt
                prompt = f"""
                Generate 5 high-quality questions based on the following text.
                The questions should be diverse, relevant, and thought-provoking.
                Return the questions as a JSON array of strings.

                Text:
                {content}

                Questions (JSON array):
                """

                # Call Ollama
                questions_json = self._call_ollama(prompt)

                # Parse the questions
                try:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\[.*\]', questions_json, re.DOTALL)
                    if json_match:
                        questions_json = json_match.group(0)

                    questions = json.loads(questions_json)

                    if not isinstance(questions, list):
                        logger.warning(f"Unexpected questions format for chunk {chunk_id}: {questions}")
                        questions = []
                except Exception as e:
                    logger.warning(f"Failed to parse questions for chunk {chunk_id}: {e}")
                    questions = []

                # Save the questions
                if questions:
                    # Save the questions to a file
                    questions_file_path = os.path.join(questions_dir, f"{chunk_id}-questions.json")
                    with open(questions_file_path, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, ensure_ascii=False, indent=2)

                    # Try to save the questions to the API
                    try:
                        response = self.session.post(
                            f"{self.base_url}/api/projects/{self.project_id}/chunks/{chunk_id}/questions",
                            json=questions
                        )
                        response.raise_for_status()
                        logger.info(f"Saved {len(questions)} questions for chunk {chunk_id} to API")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to save questions for chunk {chunk_id} to API: {e}")
                        logger.info(f"Questions saved to file: {questions_file_path}")

                    total_questions += len(questions)

            logger.info(f"Generated {total_questions} questions directly using Ollama")
            return total_questions
        except Exception as e:
            logger.error(f"Failed to generate questions directly: {e}")
            return 0

    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Get all questions in the project.

        Returns:
            List of questions
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return []

        try:
            # First try to get all questions at once
            try:
                response = self.session.get(
                    f"{self.base_url}/api/projects/{self.project_id}/questions"
                )
                response.raise_for_status()
                questions_data = response.json()

                # Handle different response formats
                questions = []

                # Format 1: Direct list of questions
                if isinstance(questions_data, list):
                    questions = questions_data
                # Format 2: {chunkId: ..., questions: [...]}
                elif isinstance(questions_data, dict) and "questions" in questions_data:
                    questions = questions_data["questions"]
                # Format 3: [{chunkId: ..., questions: [...]}, ...]
                elif isinstance(questions_data, list) and len(questions_data) > 0 and "questions" in questions_data[0]:
                    for chunk_data in questions_data:
                        chunk_id = chunk_data.get("chunkId")
                        chunk_questions = chunk_data.get("questions", [])

                        # Add chunk ID to each question if not already present
                        for question in chunk_questions:
                            if isinstance(question, dict) and "chunkId" not in question:
                                question["chunkId"] = chunk_id

                        questions.extend(chunk_questions)

                if questions:
                    logger.info(f"Found {len(questions)} questions with direct API call")
                    return questions
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to get questions with direct API call: {e}")

            # If no questions found, try getting chunks first and then questions for each chunk
            logger.info("No questions found with direct API call, trying to get chunks first...")

            # Get all chunks
            try:
                chunks_response = self.session.get(
                    f"{self.base_url}/api/projects/{self.project_id}/split"
                )
                chunks_response.raise_for_status()
                chunks_data = chunks_response.json()

                # Extract chunk IDs
                chunk_ids = []
                if "chunks" in chunks_data and isinstance(chunks_data["chunks"], list):
                    chunk_ids = chunks_data["chunks"]
                elif "fileResult" in chunks_data and "chunks" in chunks_data["fileResult"]:
                    chunk_ids = [chunk["id"] for chunk in chunks_data["fileResult"]["chunks"] if "id" in chunk]

                # Get questions for each chunk
                questions = []
                for chunk_id in chunk_ids:
                    try:
                        chunk_questions_response = self.session.get(
                            f"{self.base_url}/api/projects/{self.project_id}/chunks/{chunk_id}/questions"
                        )
                        chunk_questions_response.raise_for_status()
                        chunk_questions_data = chunk_questions_response.json()

                        # Extract questions
                        if "questions" in chunk_questions_data:
                            chunk_questions = chunk_questions_data["questions"]

                            # Add chunk ID to each question if not already present
                            for question in chunk_questions:
                                if isinstance(question, dict) and "chunkId" not in question:
                                    question["chunkId"] = chunk_id

                            questions.extend(chunk_questions)
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to get questions for chunk {chunk_id}: {e}")
                        continue

                if questions:
                    logger.info(f"Found {len(questions)} questions from chunks")
                    return questions
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to get chunks: {e}")

            # If still no questions found, try to get files and generate questions for them
            logger.info("No questions found from chunks, trying to get files...")

            # Get all files
            try:
                files_response = self.session.get(
                    f"{self.base_url}/api/projects/{self.project_id}/files"
                )
                files_response.raise_for_status()
                files_data = files_response.json()

                # Extract file IDs
                file_ids = []
                if isinstance(files_data, list):
                    for file in files_data:
                        if isinstance(file, dict) and "id" in file:
                            file_ids.append(file["id"])
                        elif isinstance(file, str):
                            file_ids.append(file)
                elif isinstance(files_data, dict) and "files" in files_data:
                    for file in files_data["files"]:
                        if isinstance(file, dict) and "id" in file:
                            file_ids.append(file["id"])
                        elif isinstance(file, str):
                            file_ids.append(file)

                # Generate questions for each file
                questions = []
                for file_id in file_ids:
                    # Only process markdown files
                    if not file_id.endswith('.md'):
                        continue

                    logger.info(f"Generating questions for file: {file_id}")

                    # Get the file content
                    file_content = self._get_file_content(file_id)
                    if not file_content:
                        logger.warning(f"Failed to get content for file {file_id}")
                        continue

                    # Generate questions using Ollama
                    prompt = f"""
                    Generate 5 high-quality questions based on the following text.
                    The questions should be diverse, relevant, and thought-provoking.
                    Return the questions as a JSON array of strings.

                    Text:
                    {file_content[:10000]}  # Limit to first 10000 characters

                    Questions (JSON array):
                    """

                    # Call Ollama
                    questions_json = self._call_ollama(prompt)

                    # Parse the questions
                    try:
                        # Try to extract JSON from the response
                        import re
                        json_match = re.search(r'\[.*\]', questions_json, re.DOTALL)
                        if json_match:
                            questions_json = json_match.group(0)

                        questions_list = json.loads(questions_json)

                        if not isinstance(questions_list, list):
                            logger.warning(f"Unexpected questions format for file {file_id}: {questions_list}")
                            continue

                        # Create question objects
                        for i, question_text in enumerate(questions_list):
                            question_id = f"{file_id}-q{i+1}"
                            questions.append({
                                "id": question_id,
                                "chunkId": file_id,
                                "question": question_text
                            })

                        # Save the questions to a file
                        questions_dir = os.path.join(os.getcwd(), "questions")
                        os.makedirs(questions_dir, exist_ok=True)
                        questions_file_path = os.path.join(questions_dir, f"{file_id}-questions.json")
                        with open(questions_file_path, 'w', encoding='utf-8') as f:
                            json.dump(questions, f, ensure_ascii=False, indent=2)

                        logger.info(f"Generated {len(questions_list)} questions for file {file_id}")
                    except Exception as e:
                        logger.warning(f"Failed to parse questions for file {file_id}: {e}")
                        continue

                if questions:
                    logger.info(f"Generated {len(questions)} questions from files")
                    return questions
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to get files: {e}")

            # If still no questions found, try to load from files
            logger.info("No questions found from API, trying to load from files...")
            questions_dir = os.path.join(os.getcwd(), "questions")
            if os.path.exists(questions_dir):
                questions = self._load_questions_from_files(questions_dir)
                if questions:
                    logger.info(f"Loaded {len(questions)} questions from files")
                    return questions

            # If all methods fail, return an empty list
            logger.warning("No questions found using any method")
            return []
        except Exception as e:
            logger.error(f"Failed to get questions: {e}")
            return []

    def generate_answers(self, language: str = "en") -> int:
        """
        Generate answers for all questions in the project.

        Args:
            language: Language for answer generation ('en' for English)

        Returns:
            Number of answers generated
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return 0

        logger.info("Generating answers for all questions")

        try:
            # Get all questions
            questions = self.get_questions()

            if not questions:
                logger.warning("No questions found to generate answers for")

                # Try to load questions from files
                questions_dir = os.path.join(os.getcwd(), "questions")
                if os.path.exists(questions_dir):
                    logger.info("Trying to load questions from files...")
                    questions = self._load_questions_from_files(questions_dir)

                if not questions:
                    logger.error("No questions found to generate answers for")
                    return 0

            # Set up the model configuration
            model_config = getattr(self, 'model_config', {
                "provider": "ollama",
                "name": "llama3",
                "temperature": 0.3,
                "maxTokens": 2048,
                "endpoint": "http://localhost:11434/api"
            })

            # Try to generate answers using the API
            try:
                # Generate answers for each question
                answers_generated = 0
                for question in questions:
                    # Handle different question formats
                    question_id = None
                    chunk_id = None
                    question_text = None

                    if isinstance(question, dict):
                        # Format 1: {id: "...", chunkId: "...", question: "..."}
                        question_id = question.get("id")
                        chunk_id = question.get("chunkId")
                        question_text = question.get("question", "")

                        # Format 2: {question: "...", chunkId: "..."}
                        if not question_id and "question" in question:
                            question_id = question.get("question")

                    if not question_id or not chunk_id:
                        logger.warning(f"Skipping question with missing ID or chunk ID: {question}")
                        continue

                    logger.info(f"Generating answer for question: {question_text[:30] if question_text else question_id[:30]}...")

                    try:
                        response = self.session.post(
                            f"{self.base_url}/api/projects/{self.project_id}/datasets",
                            json={
                                "questionId": question_id,
                                "chunkId": chunk_id,
                                "language": language if language == "en" else "中文",
                                "model": model_config
                            }
                        )
                        response.raise_for_status()
                        answers_generated += 1

                        # Add a small delay to avoid overwhelming the API
                        time.sleep(0.5)
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to generate answer for question {question_id}: {e}")
                        if hasattr(e, 'response') and e.response and e.response.text:
                            logger.warning(f"Response: {e.response.text}")
                        continue

                logger.info(f"Generated {answers_generated} answers using API")
                return answers_generated
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to generate answers using API: {e}")
                if hasattr(e, 'response') and e.response and e.response.text:
                    logger.warning(f"Response: {e.response.text}")

                # If the API fails, try to generate answers directly using Ollama
                logger.info("Trying to generate answers directly using Ollama...")
                return self._generate_answers_directly(questions, language)
        except Exception as e:
            logger.error(f"Failed to generate answers: {e}")
            return 0

    def _load_questions_from_files(self, questions_dir: str) -> List[Dict[str, Any]]:
        """
        Load questions from files in the questions directory.

        Args:
            questions_dir: Directory containing question files

        Returns:
            List of questions
        """
        questions = []

        try:
            # Get all JSON files in the questions directory
            question_files = [f for f in os.listdir(questions_dir) if f.endswith('-questions.json')]

            for file_name in question_files:
                try:
                    # Extract chunk ID from file name
                    chunk_id = file_name.replace('-questions.json', '')

                    # Load questions from file
                    file_path = os.path.join(questions_dir, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_questions = json.load(f)

                    # Add questions to the list
                    if isinstance(file_questions, list):
                        for i, question_text in enumerate(file_questions):
                            question_id = f"{chunk_id}-q{i+1}"
                            questions.append({
                                "id": question_id,
                                "chunkId": chunk_id,
                                "question": question_text
                            })
                except Exception as e:
                    logger.warning(f"Failed to load questions from file {file_name}: {e}")

            logger.info(f"Loaded {len(questions)} questions from files")
            return questions
        except Exception as e:
            logger.warning(f"Failed to load questions from files: {e}")
            return []

    def _generate_answers_directly(self, questions: List[Dict[str, Any]], language: str = "en") -> int:
        """
        Generate answers directly using Ollama without relying on the API.

        Args:
            questions: List of questions to generate answers for
            language: Language for answer generation ('en' for English)

        Returns:
            Number of answers generated
        """
        try:
            # Get the chunks content
            chunks_content = {}

            # First, collect all unique chunk IDs
            chunk_ids = set()
            for question in questions:
                if isinstance(question, dict) and "chunkId" in question:
                    chunk_ids.add(question["chunkId"])

            # Then, get the content for each chunk
            for chunk_id in chunk_ids:
                try:
                    # Try to get the chunk content from the API
                    response = self.session.get(
                        f"{self.base_url}/api/projects/{self.project_id}/chunks/{chunk_id}"
                    )
                    response.raise_for_status()
                    chunk_data = response.json()

                    if isinstance(chunk_data, dict) and "content" in chunk_data:
                        chunks_content[chunk_id] = chunk_data["content"]
                        # Got content, move to next chunk_id
                        continue
                    else:
                        logger.warning(f"Unexpected chunk data format for chunk {chunk_id}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to get chunk content for chunk {chunk_id}: {e}")

                    # If chunk API fails, try file API (assuming chunk_id might be file_id)
                    try:
                        logger.info(f"Trying to get content for {chunk_id} using file endpoints...")
                        file_content = self._get_file_content(chunk_id)
                        if file_content:
                             chunks_content[chunk_id] = file_content
                             # Got content, move to next chunk_id
                             continue
                        else:
                            logger.warning(f"Failed to get content for {chunk_id} using file endpoints.")
                    except Exception as e_file:
                        logger.warning(f"Error getting content for {chunk_id} using file endpoints: {e_file}")

                # If still no content after all API attempts
                if chunk_id not in chunks_content:
                     logger.error(f"Failed to retrieve content for chunk/file {chunk_id} using API methods.")


            if not chunks_content:
                logger.error("No chunk content could be retrieved for question generation.")
                return 0

            # Generate answers for each question
            answers_generated = 0
            answers_dir = os.path.join(os.getcwd(), "answers")
            os.makedirs(answers_dir, exist_ok=True)

            for question in questions:
                # Handle different question formats
                question_id = None
                chunk_id = None
                question_text = None

                if isinstance(question, dict):
                    # Format 1: {id: "...", chunkId: "...", question: "..."}
                    question_id = question.get("id")
                    chunk_id = question.get("chunkId")
                    question_text = question.get("question", "")

                    # Format 2: {question: "...", chunkId: "..."}
                    if not question_id and "question" in question:
                        question_id = question.get("question")

                if not question_id or not chunk_id:
                    logger.warning(f"Skipping question with missing ID or chunk ID: {question}")
                    continue

                # Skip if chunk content is not available
                if chunk_id not in chunks_content:
                    logger.warning(f"Skipping question {question_id} because chunk content is not available")
                    continue

                # Generate answer using Ollama
                logger.info(f"Generating answer for question: {question_text[:30]}...")

                # Prepare the prompt
                prompt = f"""
                Answer the following question based on the provided context.
                Your answer should be accurate, comprehensive, and well-structured.

                Context:
                {chunks_content[chunk_id]}

                Question:
                {question_text}

                Answer:
                """

                # Call Ollama
                answer = self._call_ollama(prompt)

                # Save the answer
                if answer:
                    # Create a dataset entry
                    dataset_entry = {
                        "questionId": question_id,
                        "chunkId": chunk_id,
                        "question": question_text,
                        "answer": answer
                    }

                    # Save the answer to a file
                    answer_file_path = os.path.join(answers_dir, f"{question_id}-answer.json")
                    with open(answer_file_path, 'w', encoding='utf-8') as f:
                        json.dump(dataset_entry, f, ensure_ascii=False, indent=2)

                    # Try to save the answer to the API
                    try:
                        response = self.session.post(
                            f"{self.base_url}/api/projects/{self.project_id}/datasets",
                            json={
                                "questionId": question_id,
                                "chunkId": chunk_id,
                                "answer": answer,
                                "language": language if language == "en" else "中文"
                            }
                        )
                        response.raise_for_status()
                        logger.info(f"Saved answer for question {question_id} to API")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to save answer for question {question_id} to API: {e}")
                        logger.info(f"Answer saved to file: {answer_file_path}")

                    answers_generated += 1

                    # Add a small delay to avoid overwhelming Ollama
                    time.sleep(1)

            logger.info(f"Generated {answers_generated} answers directly using Ollama")
            return answers_generated
        except Exception as e:
            logger.error(f"Failed to generate answers directly: {e}")
            return 0

    def export_dataset(self, output_path: str, format: str = "alpaca", file_type: str = "jsonl") -> bool:
        """
        Export the dataset to a file.

        Args:
            output_path: Path to save the exported dataset
            format: Dataset format ('alpaca' or 'sharegpt')
            file_type: File type ('json' or 'jsonl')

        Returns:
            True if successful, False otherwise
        """
        if not self.project_id:
            logger.error("No project selected. Create or select a project first.")
            return False

        logger.info(f"Exporting dataset to: {output_path}")

        try:
            # Try to get datasets from the API
            try:
                # Get all datasets
                response = self.session.get(
                    f"{self.base_url}/api/projects/{self.project_id}/datasets"
                )
                response.raise_for_status()
                datasets_data = response.json()

                # Handle different response formats
                datasets = []

                # Format 1: Direct list of datasets
                if isinstance(datasets_data, list):
                    datasets = datasets_data
                # Format 2: {datasets: [...]}
                elif isinstance(datasets_data, dict) and "datasets" in datasets_data:
                    datasets = datasets_data["datasets"]

                if datasets:
                    logger.info(f"Found {len(datasets)} datasets from API")
                else:
                    logger.warning("No datasets found from API")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to get datasets from API: {e}")
                datasets = []

            # If no datasets found from API, try to load from local files
            if not datasets:
                logger.info("Trying to load datasets from local files...")
                datasets = self._load_datasets_from_files()

            if not datasets:
                logger.warning("No datasets found to export")
                return False

            # Convert to the specified format
            entries = []

            if format == "alpaca":
                for dataset in datasets:
                    # Extract question and answer based on different possible formats
                    question = None
                    answer = None

                    # Format 1: {question: "...", answer: "..."}
                    if "question" in dataset and "answer" in dataset:
                        question = dataset.get("question", "")
                        answer = dataset.get("answer", "")
                    # Format 2: {questionId: "...", answer: "..."}
                    elif "questionId" in dataset and "answer" in dataset:
                        question = dataset.get("questionId", "")
                        answer = dataset.get("answer", "")
                    # Format 3: Nested question object
                    elif "question" in dataset and isinstance(dataset["question"], dict):
                        question_obj = dataset["question"]
                        if "question" in question_obj:
                            question = question_obj.get("question", "")
                        answer = dataset.get("answer", "")

                    if not question or not answer:
                        continue

                    entry = {
                        "instruction": question,
                        "input": "",
                        "output": answer
                    }
                    entries.append(entry)
            elif format == "sharegpt":
                for dataset in datasets:
                    # Extract question and answer based on different possible formats
                    question = None
                    answer = None

                    # Format 1: {question: "...", answer: "..."}
                    if "question" in dataset and "answer" in dataset:
                        question = dataset.get("question", "")
                        answer = dataset.get("answer", "")
                    # Format 2: {questionId: "...", answer: "..."}
                    elif "questionId" in dataset and "answer" in dataset:
                        question = dataset.get("questionId", "")
                        answer = dataset.get("answer", "")
                    # Format 3: Nested question object
                    elif "question" in dataset and isinstance(dataset["question"], dict):
                        question_obj = dataset["question"]
                        if "question" in question_obj:
                            question = question_obj.get("question", "")
                        answer = dataset.get("answer", "")

                    if not question or not answer:
                        continue

                    entry = {
                        "conversations": [
                            {"role": "human", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    entries.append(entry)

            # Write to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if file_type == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            else:  # json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported {len(entries)} entries to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return False

    def _load_datasets_from_files(self) -> List[Dict[str, Any]]:
        """
        Load datasets from files in the answers directory.

        Returns:
            List of datasets
        """
        datasets = []

        try:
            # Get all JSON files in the answers directory
            answers_dir = os.path.join(os.getcwd(), "answers")
            if not os.path.exists(answers_dir):
                logger.warning(f"Answers directory not found: {answers_dir}")
                return []

            answer_files = [f for f in os.listdir(answers_dir) if f.endswith('-answer.json')]

            for file_name in answer_files:
                try:
                    # Load dataset from file
                    file_path = os.path.join(answers_dir, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)

                    # Add dataset to the list
                    if isinstance(dataset, dict) and "question" in dataset and "answer" in dataset:
                        datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to load dataset from file {file_name}: {e}")

            logger.info(f"Loaded {len(datasets)} datasets from files")
            return datasets
        except Exception as e:
            logger.warning(f"Failed to load datasets from files: {e}")
            return []

def clone_repository(repo_url: str, target_dir: str) -> bool:
    """
    Clone a git repository to the target directory.

    Args:
        repo_url: URL of the repository to clone
        target_dir: Directory to clone the repository to

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Cloning repository: {repo_url}")

    try:
        Repo.clone_from(repo_url, target_dir)
        logger.info(f"Repository cloned to: {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        return False

def extract_repo_content(repo_dir: str, output_file: str) -> bool:
    """
    Extract content from a repository and save it as markdown.

    Args:
        repo_dir: Directory containing the cloned repository
        output_file: Path to save the extracted content

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Extracting content from repository: {repo_dir}")

    try:
        # Create a list of file extensions to include
        include_extensions = [
            '.md', '.txt', '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h',
            '.html', '.css', '.json', '.yaml', '.yml', '.sh', '.bash', '.go',
            '.rs', '.rb', '.php', '.swift', '.kt', '.cs', '.jsx', '.tsx'
        ]

        # Create a list of directories to exclude
        exclude_dirs = [
            '.git', 'node_modules', 'venv', 'env', '.env', '__pycache__',
            'build', 'dist', 'target', 'bin', 'obj', '.idea', '.vscode'
        ]

        # Walk through the repository and collect file contents
        content_sections = []

        for root, dirs, files in os.walk(repo_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)

                # Skip files with excluded extensions
                if not any(file.endswith(ext) for ext in include_extensions):
                    continue

                # Skip large files
                if os.path.getsize(file_path) > 1024 * 1024:  # 1MB
                    logger.warning(f"Skipping large file: {rel_path}")
                    continue

                try:
                    # Try to read the file as text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()

                    # Add the file content to the sections
                    content_sections.append(f"# File: {rel_path}\n\n```{os.path.splitext(file)[1][1:]}\n{file_content}\n```\n\n")
                except UnicodeDecodeError:
                    logger.warning(f"Skipping binary file: {rel_path}")
                except Exception as e:
                    logger.warning(f"Error reading file {rel_path}: {e}")

        # Write the content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Repository Content\n\n")
            f.write(f"Repository: {os.path.basename(repo_dir)}\n\n")
            f.write("---\n\n")
            f.write("\n".join(content_sections))

        logger.info(f"Repository content extracted to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract repository content: {e}")
        return False

async def crawl_url(url: str, output_file: str, deep_crawl: bool = False, max_pages: int = 5) -> bool:
    """
    Crawl a URL and save the content as markdown.

    Args:
        url: URL to crawl
        output_file: Path to save the crawled content
        deep_crawl: Whether to perform deep crawling
        max_pages: Maximum number of pages to crawl in deep mode

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Crawling URL: {url}")

    try:
        # Configure the browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=True,
        )

        # Configure the crawler
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0)
            ),
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Use regular crawling - deep crawling not supported in this version
            result = await crawler.arun(
                url=url,
                config=run_config
            )

            # Save the markdown output
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Content from {url}\n\n")
                f.write(result.markdown.fit_markdown)

            logger.info(f"URL content crawled to: {output_file}")
            return True
    except Exception as e:
        logger.error(f"Error crawling URL {url}: {e}")

        # Create a minimal file with the error message
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Error crawling {url}\n\n")
            f.write(f"An error occurred while crawling this URL: {str(e)}\n")

        return False

async def process_urls(urls: List[str], output_dir: str, deep_crawl: bool = False, max_pages: int = 5) -> List[str]:
    """
    Process multiple URLs and save the content as markdown.

    Args:
        urls: List of URLs to process
        output_dir: Directory to save the output
        deep_crawl: Whether to perform deep crawling
        max_pages: Maximum number of pages to crawl in deep mode

    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for i, url in enumerate(urls):
        output_file = os.path.join(output_dir, f"url_{i+1}.md")
        success = await crawl_url(url, output_file, deep_crawl, max_pages)
        if success:
            output_files.append(output_file)

    return output_files

def process_repositories(repos: List[str], output_dir: str) -> List[str]:
    """
    Process multiple repositories and save the content as markdown.

    Args:
        repos: List of repository URLs to process
        output_dir: Directory to save the output

    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for i, repo_url in enumerate(repos):
        # Create a temporary directory for the repository
        repo_dir = os.path.join(output_dir, f"repo_{i+1}")
        os.makedirs(repo_dir, exist_ok=True)

        # Clone the repository
        if clone_repository(repo_url, repo_dir):
            # Extract the repository content
            output_file = os.path.join(output_dir, f"repo_{i+1}.md")
            if extract_repo_content(repo_dir, output_file):
                output_files.append(output_file)

            # Clean up the repository directory
            shutil.rmtree(repo_dir)

    return output_files

def combine_markdown_files(input_files: List[str], output_file: str) -> bool:
    """
    Combine multiple markdown files into a single file.

    Args:
        input_files: List of input file paths
        output_file: Output file path

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Combining {len(input_files)} markdown files into: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(f"# Combined Content\n\n")

            for i, file_path in enumerate(input_files):
                logger.info(f"Adding content from: {file_path}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        content = in_f.read()

                    out_f.write(f"## Source {i+1}: {os.path.basename(file_path)}\n\n")
                    out_f.write(content)
                    out_f.write("\n\n---\n\n")
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")

        logger.info(f"Combined content saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to combine markdown files: {e}")
        return False

async def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Process repositories and URLs for Easy Dataset")
    parser.add_argument("--repos", help="Comma-separated list of repository URLs")
    parser.add_argument("--urls", help="Comma-separated list of URLs to crawl")
    parser.add_argument("--output-dir", default="./output", help="Directory to save the output")
    parser.add_argument("--project-name", default=f"Project-{int(time.time())}", help="Name for the Easy Dataset project")
    parser.add_argument("--base-url", default="http://localhost:1717", help="Base URL for Easy Dataset API")
    parser.add_argument("--llm-provider", default="ollama", help="LLM provider (openai, ollama, etc.)")
    parser.add_argument("--llm-model", default="llama3", help="LLM model name")
    parser.add_argument("--api-key", help="API key for the LLM provider (if required)")
    parser.add_argument("--endpoint", help="Custom API endpoint (if required)")
    parser.add_argument("--language", default="en", choices=["en", "zh"], help="Language for generation (en, zh)")
    parser.add_argument("--deep-crawl", action="store_true", help="Perform deep crawling for URLs")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of pages to crawl in deep mode")
    parser.add_argument("--export-format", default="alpaca", choices=["alpaca", "sharegpt"], help="Export format")
    parser.add_argument("--export-file-type", default="jsonl", choices=["json", "jsonl"], help="Export file type")
    parser.add_argument("--skip-export", action="store_true", help="Skip exporting the dataset")
    parser.add_argument("--skip-questions", action="store_true", help="Skip generating questions")
    parser.add_argument("--skip-answers", action="store_true", help="Skip generating answers")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to Easy Dataset (just process content)")
    parser.add_argument("--skip-browser", action="store_true", help="Skip opening browser for prompts tab")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for manual chunking")
    parser.add_argument("--generate-prompts", action="store_true", help="Generate prompts using Ollama")
    parser.add_argument("--domain", default="FastMCP", help="Domain or topic for prompt generation")

    args = parser.parse_args()

    # Check if at least one source is provided
    if not args.repos and not args.urls:
        logger.error("No repositories or URLs provided. Please specify at least one source.")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process repositories
    repo_files = []
    if args.repos:
        repos = [repo.strip() for repo in args.repos.split(",") if repo.strip()]
        logger.info(f"Processing {len(repos)} repositories...")
        repo_files = process_repositories(repos, os.path.join(args.output_dir, "repos"))

    # Process URLs
    url_files = []
    if args.urls:
        urls = [url.strip() for url in args.urls.split(",") if url.strip()]
        logger.info(f"Processing {len(urls)} URLs...")
        url_files = await process_urls(
            urls,
            os.path.join(args.output_dir, "urls"),
            args.deep_crawl,
            args.max_pages
        )

    # Combine all content
    all_files = repo_files + url_files
    if not all_files:
        logger.error("No content was successfully processed.")
        return 1

    combined_file = os.path.join(args.output_dir, "combined_content.md")
    if not combine_markdown_files(all_files, combined_file):
        return 1

    # If we're just processing content, we're done
    if args.skip_upload:
        logger.info(f"Content processing completed. Combined content saved to: {combined_file}")
        return 0

    # Check if Easy Dataset is running
    try:
        response = requests.get(args.base_url)
        if response.status_code >= 400:
            logger.error(f"Easy Dataset is not running at {args.base_url}")
            logger.error("Please start Easy Dataset before running this script")
            return 1
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Easy Dataset at {args.base_url}")
        logger.error("Please start Easy Dataset before running this script")
        return 1

    # Initialize the Easy Dataset client
    client = EasyDatasetClient(base_url=args.base_url)

    # Set the chunk size
    client.chunk_size = args.chunk_size

    # Create a project
    project_id = client.create_project(args.project_name)
    if not project_id:
        return 1

    # Configure the LLM - continue even if this fails
    client.configure_llm(args.llm_provider, args.llm_model, args.api_key, args.endpoint)

    # Upload the content first (without splitting)
    logger.info(f"Uploading content from: {combined_file}")
    file_id = client.upload_file(combined_file)
    if not file_id:
        logger.error("Failed to upload content. Check if the API endpoints are correct.")
        logger.info("The combined content is still available at: " + combined_file)
        return 1

    # Generate and update prompts if requested
    if args.generate_prompts:
        logger.info(f"Generating prompts for domain: {args.domain}")
        if client.generate_prompts(args.domain, args.language):
            logger.info("Prompts generated successfully")
            # Update the prompts in the project
            if client.update_prompts():
                logger.info("Prompts updated in the project")

                # Open the prompts tab in the browser if requested
                if not args.skip_browser:
                    import webbrowser
                    prompts_url = f"{args.base_url}/projects/{client.project_id}/settings?tab=prompts"
                    logger.info(f"Opening prompts tab in browser: {prompts_url}")
                    try:
                        webbrowser.open(prompts_url)
                    except Exception as e:
                        logger.warning(f"Failed to open browser: {e}")
            else:
                logger.warning("Failed to update prompts in the project")
        else:
            logger.warning("Failed to generate prompts")

    # Now try to split the content
    chunk_ids = client.split_content(file_id)
    if not chunk_ids:
        logger.error("Failed to split content. Check if the API endpoints are correct.")
        logger.info("The content was uploaded but could not be split.")
        logger.info("You can manually split the content at: " + f"{args.base_url}/projects/{client.project_id}")

        # If we can't split the content, we can still continue with the process
        # by using the file directly
        if not args.skip_questions and not args.skip_answers:
            logger.info("Continuing with question and answer generation using the uploaded file...")
            chunk_ids = [file_id]

    # Generate questions (if not skipped)
    if not args.skip_questions:
        question_count = client.generate_questions(chunk_ids, args.language)
        if question_count == 0:
            logger.warning("No questions were generated. Check if the API endpoints are correct.")
            logger.warning("Continuing with the process...")
    else:
        logger.info("Skipping question generation as requested")

    # Generate answers (if not skipped)
    if not args.skip_answers and not args.skip_questions:
        answer_count = client.generate_answers(args.language)
        if answer_count == 0:
            logger.warning("No answers were generated. Check if the API endpoints are correct.")
            logger.warning("Continuing with the process...")
    else:
        logger.info("Skipping answer generation as requested")

    # Export the dataset (if not skipped)
    if not args.skip_export and not args.skip_questions and not args.skip_answers:
        export_path = os.path.join(args.output_dir, f"dataset.{args.export_file_type}")
        if not client.export_dataset(export_path, format=args.export_format, file_type=args.export_file_type):
            logger.warning("Failed to export dataset. Check if the API endpoints are correct.")
            logger.warning("The combined content is still available at: " + combined_file)
    elif args.skip_export:
        logger.info("Skipping dataset export as requested")

    logger.info("Processing completed successfully")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Combined content: {combined_file}")

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
