Image Generation Script with Google AI

This script, genimages.py, facilitates the generation of images using the Google Gemini API. It reads prompts from an input file, sends them to a specified Google AI model, and saves the generated images to a designated output directory.

SCRIPT_AUTHOR="Igor Brzezek"
SCRIPT_VERSION="1.0.1"
SCRIPT_DATE="2025-12-01"
AUTHOR_EMAIL="igor.brzezek@gmail.com"
AUTHOR_GITHUB="https://github.com/igorbrzezek"

Purpose and Application

The primary goal of this script is to automate the process of generating multiple images from textual prompts using Google's powerful generative AI models. This is particularly useful for:
- Content Creation: Quickly generating visual assets for articles, presentations, or social media.
- Prototyping: Creating diverse image concepts for design projects.
- Research and Development: Experimenting with different prompts and models to understand AI image generation capabilities.
- Batch Processing: Generating a large number of images without manual intervention.

Features
- Supports reading prompts from .txt and .json files.
- Customizable output directory and image format (.png or .jpg).
- Option to override original filenames with a sequential pattern.
- Configurable delay between image generation tasks to manage API rate limits.
- Ability to specify the Google AI model for generation.
- Option to list available image generation models.
- Colored Output: Visual feedback for errors (Red), success (Green), and warnings (Yellow).
- Overwrite Control: Interactive confirmation or forced overwrite of existing files.
- Summary Statistics: Detailed report at the end of execution (counts, time, average file size). Optional list of failed images.
- Retry Logic: Automatically retries failed generations a specified number of times.
- Resolution Control: Support for presets (1k, 2k, 4k) or custom dimensions.
- Debug mode for detailed output.
- Batch Mode: Run silently without messages and force overwrite.
- API Key Setup: Option to set the API key directly via command line argument.
- Price Estimation: Estimate the cost of generation before running the task.

Prerequisites
Before running the script, ensure you have:
- Python 3.8 or higher installed.
- A Google Cloud Project with the Gemini API enabled.
- An API key for the Google Gemini API.

Installation

Required Libraries
The script requires the google-generativeai, python-dotenv, and Pillow libraries.

Installing on Linux/macOS

1. Open a terminal.
2. Install pip (if not already installed):
   sudo apt update
   sudo apt install python3-pip # For Debian/Ubuntu
   # For macOS, pip is usually included with Python installed via Homebrew:
   # brew install python
   # You might need to use 'pip3' instead of 'pip' if 'pip' refers to Python 2.
3. Install the required Python libraries:
   pip install google-generativeai python-dotenv Pillow

Installing on Windows

1. Open Command Prompt or PowerShell.
2. Install pip (if not already installed):
   - Python usually comes with pip. You can check by running pip --version.
   - If pip is missing, you might need to reinstall Python or add Python to your system's PATH.
3. Install the required Python libraries:
   pip install google-generativeai python-dotenv Pillow

API Key Configuration

The script requires a Google Gemini API key to authenticate with the Google AI services. You can configure this in two ways:

1. Using an Environment Variable (Recommended):
   Set the GEMINI_API_KEY environment variable with your API key.

   On Linux/macOS:
   export GEMINI_API_KEY="YOUR_API_KEY_HERE"
   # To make it permanent, add this line to your shell's profile file (e.g., ~/.bashrc, ~/.zshrc)

   On Windows (Command Prompt):
   set GEMINI_API_KEY="YOUR_API_KEY_HERE"
   # To make it permanent, use:
   # setx GEMINI_API_KEY "YOUR_API_KEY_HERE"

   On Windows (PowerShell):
   $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
   # To make it permanent, you might need to add it to your system environment variables
   # via System Properties -> Advanced -> Environment Variables.

2. Using a Custom Environment Variable Name:
   If you prefer to use a different environment variable name, you can specify it with the --keyenv argument.
   For example, if your API key is stored in MY_GEMINI_KEY:
   export MY_GEMINI_KEY="YOUR_API_KEY_HERE"
   python genimages.py -i prompts.txt -t txt --keyenv MY_GEMINI_KEY

Usage

python genimages.py [OPTIONS]

Basic Usage

To generate images, you must provide an input file with prompts and specify its type.

python genimages.py -i prompts.json

This command will read prompts from prompts.json, generate images using the default model (gemini-3-pro-image-preview), and save them in a new directory named prompts/ (derived from the input filename) in JPG format.

Input File Formats

Use the -t or --type option to specify the format of your input file. Default is 'json'.

Text File (.txt)
Each prompt should be on a separate line. Empty lines are ignored. The script attempts to extract the desired filename from a specific phrase within the prompt: Nazwa pliku: XXX (Polish for "Filename: XXX"). If this exact phrase is not found, a default filename like image_001 will be used.

Example prompts.txt content:
A majestic lion in a savanna at sunset. Nazwa pliku: lion_sunset
An astronaut riding a horse on Mars. Nazwa pliku: mars_horse
A futuristic city at night with flying cars. Nazwa pliku: future_city

JSON File (.json)
The JSON file should be a dictionary where keys are desired filenames (without extension) and values are the prompts.

Example prompts.json content:
{
  "lion_sunset": "A majestic lion in a savanna at sunset.",
  "mars_horse": "An astronaut riding a horse on Mars.",
  "future_city": "A futuristic city at night with flying cars."
}

Output Options

- -o or --output-dir: Specify a custom output directory.
  python genimages.py -i prompts.txt -t txt -o my_generated_images
  This will save images in the my_generated_images/ directory.

- --imgpat: Override original filenames with a sequential pattern.
  python genimages.py -i prompts.txt -t txt --imgpat "my_image"
  This will generate files like my_image_001.jpg, my_image_002.jpg, etc.

- --fmt: Choose the output image format (png or jpg). Default is jpg.
  python genimages.py -i prompts.json --fmt png
  This will save images in PNG format.

- --delay: Set a delay (in seconds) between image generation tasks to avoid API rate limits. Default is 1 second.
  python genimages.py -i prompts.txt -t txt --delay 5
  This will wait 5 seconds between each image generation request.

Model Selection

- --model: Specify the Google AI model to use for image generation. The default is gemini-3-pro-image-preview.
  python genimages.py -i prompts.txt -t txt --model gemini-1.5-flash-latest

Listing Available Models

To see a list of available image generation models, use the --list option. This requires your API key to be configured.

python genimages.py --list

Price Estimation
 
- --testprice: Estimate the cost of generating images without actually running the generation task. This helps in understanding the potential cost based on the number of prompts and the selected model.
  python genimages.py -i prompts.json --testprice
 
Advanced Options
 
 - --color: Enable colored output for better visibility of errors (Red), success (Green), and warnings (Yellow).
  python genimages.py -i prompts.txt -t txt --color

- --overwrite: Force overwriting of existing files without asking for confirmation. By default, the script asks interactively.
  python genimages.py -i prompts.txt -t txt --overwrite

- --stat: Print a summary of the generation process at the end (counts, time, average file size).
  - Use '--stat err' to also list failed images in red.
  python genimages.py -i prompts.txt -t txt --stat
  python genimages.py -i prompts.txt -t txt --stat err

- --ret: Set the number of retries for failed generations. Default is 3.
  python genimages.py -i prompts.txt -t txt --ret 5

- --res: Set the resolution of the generated images.
  - Presets: 1k (1024x1024), 2k (2048x2048), 4k (4096x4096).
  - Custom: WIDTH HEIGHT (e.g., 1920 1080).
  - Default: 1408 768.
  python genimages.py -i prompts.json --res 2k
  python genimages.py -i prompts.json --res 1920 1080

- --genkeyenv: Create a local system variable named GEMINI_API_KEY with the provided value.
  python genimages.py --genkeyenv YOUR_API_KEY

- -b or --batch: Batch mode. Runs silently without any messages and forces overwrite.
  python genimages.py -i prompts.json --batch

Debug Mode

- --debug: Enable debug output for more detailed information, including full prompt text and warnings.
  python genimages.py -i prompts.json --debug

Examples

1. Generate images from a text file, save to a custom directory, use PNG format, and show summary:
   python genimages.py -i my_prompts.txt -t txt -o output_pngs --fmt png --stat

2. Generate images from a JSON file, with a custom filename pattern, 3-second delay, and colored output:
   python genimages.py -i my_prompts.json --imgpat "design_concept" --delay 3 --color

3. Generate images with 2k resolution and force overwrite:
   python genimages.py -i prompts.json --res 2k --overwrite

4. List available models and then generate an image using a specific model:
   python genimages.py --list
   # (After reviewing the list, choose a model, e.g., 'gemini-1.5-pro-latest')
   python genimages.py -i single_prompt.txt -t txt --model gemini-1.5-pro-latest

5. Set API key and run in batch mode:
   python genimages.py --genkeyenv YOUR_KEY -i prompts.json --batch

6. Estimate the cost of generation:
  python genimages.py -i prompts.json --testprice --model gemini-1.5-pro-latest

Cross-Platform Compatibility

The script is designed to be compatible with both Linux and Windows operating systems.

- File Paths: The pathlib module is used for handling file paths, ensuring that paths are constructed correctly regardless of the operating system's path separator (/ on Linux/macOS, \ on Windows).
- Environment Variables: The os.getenv function is used for retrieving environment variables, which works consistently across platforms.
- Command Execution: The installation instructions provide platform-specific commands for installing Python and libraries. The script itself does not execute external shell commands in a way that would differ significantly between OSes.

Potential Issues and Solutions

- Python Interpreter: Ensure that python or python3 command correctly points to your Python installation on both systems. On Windows, it might be py or python.
- Encoding: The script uses utf-8 encoding for reading prompt files, which is standard and should work across platforms. If you encounter encoding issues, ensure your input files are saved with UTF-8 encoding.
- PIL (Pillow) Library: Pillow handles image saving and loading internally, abstracting away OS-specific image handling.

No modifications are needed for the genimages.py script itself to run on both Linux and Windows, as it uses standard Python libraries and practices that are cross-platform compatible. Therefore, a genimages_ng.py file is not required.
