import os
import argparse
import json
import re
import time
import sys
from pathlib import Path
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

SCRIPT_AUTHOR="Igor Brzezek"
SCRIPT_VERSION="1.0.1"
SCRIPT_DATE="2025-12-01"
AUTHOR_EMAIL="igor.brzezek@gmail.com"
AUTHOR_GITHUB="https://github.com/igorbrzezek"

# Import Google Generative AI SDK
try:
    import google.generativeai as genai
except ImportError:
    print("Error: The 'google-genai' module is not installed. Please install it: pip install google-genai")
    exit(1)

# API configuration will be done inside main() after parsing arguments.
# Load dotenv once at the start.
load_dotenv()

# Placeholder for model pricing (per image)
# NOTE: These prices are illustrative. Check the official Google Cloud pricing page for up-to-date information.
MODEL_PRICING = {
    "gemini-3-pro-image-preview": 0.020,
    "gemini-1.5-pro-latest": 0.020,
    "gemini-1.5-flash-latest": 0.015,
    "imagen-3.0": 0.020, # Example, not a real gemini model name from the script
}

class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"

def cprint(text, color=None, use_color=False, end='\n'):
    if use_color and color:
        print(f"{color}{text}{Colors.RESET}", end=end)
    else:
        print(text, end=end)

def save_image_from_response(response, filepath: Path, format: str, overwrite: bool, use_color: bool, target_size: tuple[int, int] = None) -> tuple[str, Path | None]:
    """
    Saves the image from the API response to the specified file path and format.

    :param response: The response from model.generate_content().
    :param filepath: Path object to save the file.
    :param format: The saving format ('png' or 'jpeg').
    :param target_size: Optional tuple (width, height) to resize the image.
    :return: A tuple (status, path). Status is 'SUCCESS', 'SKIPPED', or 'ERROR'.
    """
    # Pillow uses 'JPEG' for 'jpg' format
    save_format = 'JPEG' if format.lower() == 'jpg' else 'PNG'
    
    try:
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_data = BytesIO(part.inline_data.data)
                    img = Image.open(image_data)
                    
                    # Resize if target_size is specified
                    if target_size:
                        try:
                            # Try using the newer Resampling enum (Pillow >= 9.1.0)
                            resample_method = Image.Resampling.LANCZOS
                        except AttributeError:
                            # Fallback for older Pillow versions
                            resample_method = Image.LANCZOS
                        img = img.resize(target_size, resample_method)

                    # Ensure the target directory exists
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Enforce the correct extension based on the selected format
                    new_filepath = filepath.with_suffix(f".{format.lower()}")
                    
                    # Check for overwrite
                    if new_filepath.exists() and not overwrite:
                        cprint(f"   [WARNING] File '{new_filepath.name}' already exists.", Colors.YELLOW, use_color, end=' ')
                        user_input = input("Overwrite? [y/N]: ").strip().lower()
                        if user_input != 'y':
                            cprint(f"   [SKIPPED] Image generation skipped for: {new_filepath.name}", Colors.YELLOW, use_color)
                            return 'SKIPPED', None

                    # Save in the selected format
                    img.save(new_filepath, format=save_format)
                    
                    cprint(f"   [SUCCESS] Image saved as: {new_filepath.name} ({save_format})", Colors.GREEN, use_color)
                    return 'SUCCESS', new_filepath
        cprint(f"   [ERROR] No image data found in the response for the prompt.", Colors.RED, use_color)
        return 'ERROR', None
    except Exception as e:
        cprint(f"   [ERROR] An error occurred while saving image {filepath.name}: {e}", Colors.RED, use_color)
        return 'ERROR', None

def read_prompts_from_file(input_filepath: Path, file_format: str, use_color: bool) -> list[tuple[str, str]]:
    """Reads the list of prompts and corresponding filenames from the input file."""
    prompts = []
    print(f"\nLoading prompts from file: {input_filepath.name} in {file_format.upper()} format...")
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            if file_format == 'txt':
                for i, line in enumerate(f):
                    line = line.strip()
                    if line: 
                        # Try to extract filename from "Nazwa pliku: XXX" which is assumed to be present
                        match = re.search(r'Nazwa pliku:\s*([\w\d\._-]+)', line, re.IGNORECASE)
                        default_filename = f"image_{i+1:03d}"
                        filename = match.group(1).split('.')[0] if match else default_filename 
                        
                        prompts.append((filename, line))
                        
            elif file_format == 'json':
                data = json.load(f)
                if isinstance(data, dict):
                    for filename, prompt_text in data.items():
                        if filename and prompt_text:
                            base_name = filename.split('.')[0]
                            prompts.append((base_name, prompt_text))
                else:
                    raise ValueError("Invalid JSON format: Expected an object (dictionary).")

    except FileNotFoundError:
        cprint(f"\n[ERROR] Input file not found: {input_filepath}", Colors.RED, use_color)
        return []
    except json.JSONDecodeError as e:
        cprint(f"\n[ERROR] JSON file parsing error: {e}", Colors.RED, use_color)
        return []
    except Exception as e:
        cprint(f"\n[ERROR] An unexpected error occurred while loading the file: {e}", Colors.RED, use_color)
        return []
        
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"Generates images using the Google Gemini API.\n\n"
            f"Script: genimages.py\n"
            f"Author: {SCRIPT_AUTHOR} ({AUTHOR_EMAIL})\n"
            f"GitHub: {AUTHOR_GITHUB}\n"
            f"Version: {SCRIPT_VERSION}\n"
            f"Date: {SCRIPT_DATE}\n\n"
            "This script, `genimages.py`, facilitates the generation of images using the Google Gemini API. "
            "It reads prompts from an input file, sends them to a specified Google AI model, and saves the generated images to a designated output directory.\n\n"
            "Purpose and Application:\n"
            "The primary goal of this script is to automate the process of generating multiple images from textual prompts using Google's powerful generative AI models. This is particularly useful for:\n"
            "- Content Creation: Quickly generating visual assets for articles, presentations, or social media.\n"
            "- Prototyping: Creating diverse image concepts for design projects.\n"
            "- Research and Development: Experimenting with different prompts and models to understand AI image generation capabilities.\n"
            "- Batch Processing: Generating a large number of images without manual intervention.\n\n"
            "Features:\n"
            "- Supports reading prompts from .txt and .json files.\n"
            "- Customizable output directory and image format (.png or .jpg).\n"
            "- Option to override original filenames with a sequential pattern.\n"
            "- Configurable delay between image generation tasks to manage API rate limits.\n"
            "- Ability to specify the Google AI model for generation.\n"
            "- Option to list available image generation models.\n"
            "- Debug mode for detailed output.\n"
            "- Colored output for better visibility of errors and successes.\n"
            "- Interactive or forced overwrite of existing files.\n"
            "- Summary statistics at the end of execution (optional error list).\n"
            "- Retry logic for failed generations.\n"
            "- Custom image resolution support (presets or custom dimensions).\n"
            "- Batch mode for silent execution.\n"
            "- Option to provide API key directly via command line.\n"
            "- Price estimation mode to calculate costs before generation.\n\n"
            "Usage:\n"
            "python genimages.py [OPTIONS]\n\n"
            "Basic Usage:\n"
            "To generate images, you must provide an input file with prompts.\n"
            "python genimages.py -i prompts.txt\n"
            "This command will read prompts from prompts.txt, generate images using the default model (gemini-3-pro-image-preview), and save them in a new directory named prompts/ (derived from the input filename) in JPG format.\n\n"
            "Input File Formats:\n"
            "Use the -t or --type option to specify the format of your input file.\n\n"
            "Text File (.txt):\n"
            "Each prompt should be on a separate line. Empty lines are ignored. The script attempts to extract the desired filename from a specific phrase within the prompt: Nazwa pliku: XXX (Polish for \"Filename: XXX\"). If this exact phrase is not found, a default filename like image_001 will be used.\n\n"
            "Example prompts.txt content:\n"
            "A majestic lion in a savanna at sunset. Nazwa pliku: lion_sunset\n"
            "An astronaut riding a horse on Mars. Nazwa pliku: mars_horse\n"
            "A futuristic city at night with flying cars. Nazwa pliku: future_city\n\n"
            "JSON File (.json):\n"
            "The JSON file should be a dictionary where keys are desired filenames (without extension) and values are the prompts.\n\n"
            "Example prompts.json content:\n"
            "{\n"
            "  \"lion_sunset\": \"A majestic lion in a savanna at sunset.\",\n"
            "  \"mars_horse\": \"An astronaut riding a horse on Mars.\",\n"
            "  \"future_city\": \"A futuristic city at night with flying cars.\"\n"
            "}\n\n"
            "Output Options:\n"
            "- -o or --output-dir: Specify a custom output directory.\n"
            "  python genimages.py -i prompts.txt -o my_generated_images\n"
            "  This will save images in the my_generated_images/ directory.\n\n"
            "- --imgpat: Override original filenames with a sequential pattern.\n"
            "  python genimages.py -i prompts.txt --imgpat \"my_image\"\n"
            "  This will generate files like my_image_001.jpg, my_image_002.jpg, etc.\n\n"
            "- --fmt: Choose the output image format (png or jpg). Default is jpg.\n"
            "  python genimages.py -i prompts.txt --fmt png\n"
            "  This will save images in PNG format.\n\n"
            "- --delay: Set a delay (in seconds) between image generation tasks to avoid API rate limits. Default is 1 second.\n"
            "  python genimages.py -i prompts.txt --delay 5\n"
            "  This will wait 5 seconds between each image generation request.\n\n"
            "Model Selection:\n"
            "- --model: Specify the Google AI model to use for image generation. The default is gemini-3-pro-image-preview.\n"
            "  python genimages.py -i prompts.txt --model gemini-1.5-flash-latest\n\n"
            "Listing Available Models:\n"
            "To see a list of available image generation models, use the --list option. This requires your API key to be configured.\n"
            "python genimages.py --list\n\n"
            "Debug Mode:\n"
            "- --debug: Enable debug output for more detailed information, including full prompt text and warnings.\n"
            "  python genimages.py -i prompts.txt --debug\n\n"
            "Examples:\n"
            "1. Generate images from a text file, save to a custom directory, and use PNG format:\n"
            "   python genimages.py -i my_prompts.txt -o output_pngs --fmt png\n\n"
            "2. Generate images from a JSON file, with a custom filename pattern and a 3-second delay:\n"
            "   python genimages.py -i my_prompts.json -t json --imgpat \"design_concept\" --delay 3\n\n"
            "3. List available models and then generate an image using a specific model:\n"
            "   python genimages.py --list\n"
            "   # (After reviewing the list, choose a model, e.g., 'gemini-1.5-pro-latest')\n"
            "   python genimages.py -i single_prompt.txt --model gemini-1.5-pro-latest\n\n"
            "4. Estimate the cost of generation without running the task:\n"
            "   python genimages.py -i my_prompts.json -t json --testprice\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False # Disable default help
    )
    
    # Custom help argument
    parser.add_argument(
        '--help', action='store_true',
        help='Show full help message and exit.'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        help=(
            "Path to the input file containing the list of prompts to generate.\n"
            "The file format must be specified with the -t option."
        )
    )
    
    parser.add_argument(
        '-t', '--type',
        choices=['txt', 'json'],
        default='json',
        help=(
            "Input file format (default: 'json'):\n"
            "  - 'txt': Each prompt on a separate line. Empty lines are ignored.\n"
            "    Filename is extracted from the phrase 'Nazwa pliku: XXX' in the prompt.\n"
            "  - 'json': File in the dictionary format {'filename': 'prompt_text'}."
        )
    )
    
    parser.add_argument(
        '-o', '--output-dir', 
        type=str, 
        default=None, 
        help=(
            "Optional target DIRECTORY name for the generated images.\n"
            "  - If provided: files are saved in the given DIRECTORY.\n"
            "  - If missing: files are saved in a directory named after the input file\n"
            "    without its extension (e.g., 'prompts.txt' results in './prompts/')."
        )
    )
    
    parser.add_argument(
        '--imgpat', 
        type=str, 
        default=None, 
        help=(
            "Filename PATTERN that overrides the original names and sequentially numbers the images.\n"
            "  - Format: PATTERN_NUMBER (e.g., 'lab_imgpat_001.jpg').\n"
            "  - Numbering starts from 1, with leading zeros (three positions)."
        )
    )
    
    parser.add_argument(
        '--fmt', 
        choices=['png', 'jpg'], 
        default='jpg', 
        help=(
            "Output format for the generated images (default: 'jpg').\n"
            "  - 'png': Lossless compression format, better for diagrams and sharp-edged graphics.\n"
            "  - 'jpg': Lossy compression format, better for photographs."
        )
    )

    parser.add_argument(
        '--delay', 
        type=int, 
        default=1, 
        help=(
            "Delay (in seconds) between submitting consecutive image generation tasks.\n"
            "  - Default value: 1 second. Helps in avoiding API rate limits."
        )
    )

    parser.add_argument(
        '--model', 
        type=str, 
        default='gemini-3-pro-image-preview', 
        help=(
            "The AI model (M) to use for image generation.\n"
            "  - Default: 'gemini-3-pro-image-preview'."
        )
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="Lists only available image generation models and exits."
    )
    
    parser.add_argument(
        '--keyenv', 
        type=str, 
        default='GEMINI_API_KEY', 
        help=(
            "The name of the environment variable (KE) containing the Gemini API key.\n"
            "  - Default: 'GEMINI_API_KEY'."
        )
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enables debug output, including warnings and detailed process information."
    )

    parser.add_argument(
        '--color',
        action='store_true',
        help="Enables colored output (Red=Error, Green=Success, Yellow=Warning)."
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Forces overwriting of existing files without asking for confirmation."
    )

    parser.add_argument(
        '--stat',
        nargs='?',
        const='basic',
        default=None,
        help="Prints a summary. Use '--stat err' to also list failed images."
    )

    parser.add_argument(
        '--ret',
        type=int,
        default=3,
        help="Number of retries for failed generations (default: 3)."
    )

    parser.add_argument(
        '--res',
        nargs='+',
        default=['1408', '768'],
        help="Sets the resolution of the generated images. Accepts '1k', '2k', '4k' or 'WIDTH HEIGHT' (default: '1408 768')."
    )

    parser.add_argument(
        '--genkeyenv',
        type=str,
        help="Creates a local system variable named GEMINI_API_KEY with the provided value (API key)."
    )

    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help="Batch mode: runs without any messages and forces overwrite."
    )

    parser.add_argument(
        '--testprice',
        action='store_true',
        help="Estimates the cost of generating images without actually generating them."
    )
 
    # --- Custom Help Logic ---
    if '--help' in sys.argv[1:]:
        parser.parse_args(['--help']) # Parse only --help to trigger its action
        parser.print_help()
        sys.exit(0)
    elif '-h' in sys.argv[1:]:
        # Remove '-h' from sys.argv to prevent argparse from complaining
        sys.argv.remove('-h')
        parser.parse_args([]) # Parse without -h
        parser.print_usage()
        print("\nFor full help, use: python genimages.py --help")
        sys.exit(0)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) # Parse args, show help if no args
    
    # --- API Key Retrieval and Configuration ---
    if args.genkeyenv:
        os.environ['GEMINI_API_KEY'] = args.genkeyenv

    KEY_ENV_VAR = args.keyenv
    API_KEY = os.getenv(KEY_ENV_VAR)
    genai.configure(api_key=API_KEY)
    
    # --- LIST MODELS LOGIC (Early Exit) ---
    if args.list:
        if not API_KEY:
             cprint(f"\n[ERROR] The API key environment variable '{KEY_ENV_VAR}' is not set.", Colors.RED, args.color)
             print("Cannot list models without an API key.")
             return
        
        print("\n--- Available Image Generation Models ---")
        try:
            models = genai.list_models()
            
            # Filtering Logic: Look for 'image', 'vision', or 'preview' in model names
            IMAGE_MODEL_KEYWORDS = ['image', 'vision', 'preview']
            
            image_models = [
                m for m in models
                if any(keyword in m.name.lower() for keyword in IMAGE_MODEL_KEYWORDS)
            ]

            if not image_models:
                print("No specialized image generation models found.")
                
            for m in image_models:
                max_tokens = m.output_token_limit if m.output_token_limit is not None else 'N/A'
                print(f"- {m.name} (Max Tokens: {max_tokens})")
            print("---------------------------------------")
            return

        except Exception as e:
            cprint(f"[ERROR] Failed to list models: {e}", Colors.RED, args.color)
            return

    # --- TEST PRICE LOGIC (Early Exit) ---
    if args.testprice:
        if not args.input:
            parser.error("-i/--input is required when using the --testprice option.")

        model_name = args.model
        if model_name not in MODEL_PRICING:
            cprint(f"\n[ERROR] Pricing information for model '{model_name}' is not available.", Colors.RED, args.color)
            cprint("Cannot estimate cost. Available models for pricing:", Colors.YELLOW, args.color)
            for m in MODEL_PRICING:
                print(f"- {m}")
            return

        input_file = Path(args.input)
        prompt_list = read_prompts_from_file(input_file, args.type, args.color)

        if not prompt_list:
            cprint("No prompts found for processing. Exiting.", Colors.YELLOW, args.color)
            return

        model_name = args.model
        price_per_image = MODEL_PRICING.get(model_name)

        print("\n--- Price Estimation ---")
        cprint(f"Model: {model_name}", use_color=args.color)

        if price_per_image is None:
            cprint(f"[WARNING] No pricing information available for model '{model_name}'.", Colors.YELLOW, args.color)
            cprint("Cannot estimate cost.", Colors.YELLOW, args.color)
            return

        cprint(f"Estimated cost per image: ${price_per_image:.4f}", use_color=args.color)
        print("-" * 30)

        total_cost = 0
        for i, (original_basename, prompt_text) in enumerate(prompt_list):
            total_cost += price_per_image
            if args.debug:
                print(f"Image {i+1:03d} ({original_basename}): ${price_per_image:.4f} -> Prompt: {prompt_text[:80]}...")
            else:
                print(f"Image {i+1:03d} ({original_basename}): ${price_per_image:.4f}")


        print("-" * 30)
        cprint(f"Total prompts: {len(prompt_list)}", use_color=args.color)
        cprint(f"Total estimated cost: ${total_cost:.4f}", Colors.GREEN, args.color)
        print("------------------------\n")
        cprint("NOTE: This is an estimate. Prices may vary based on the model version and other factors.", Colors.YELLOW, args.color)
        return

    # --- PROCEED WITH GENERATION LOGIC ---
    
    # --- Validation Checks ---
    if not args.input:
        parser.error("-i/--input is required when not using the --list option.")

    # --- Generation Config & Resolution Parsing ---
    generation_config = {}
    
    # Parse Resolution
    width, height = 1024, 1024 # Default fallback
    
    if args.res:
        if len(args.res) == 1:
            val = args.res[0].lower()
            if val == '1k':
                width, height = 1024, 1024
            elif val == '2k':
                width, height = 2048, 2048
            elif val == '4k':
                width, height = 4096, 4096
            else:
                parser.error(f"Invalid resolution preset '{args.res[0]}'. Use '1k', '2k', '4k' or specify 'WIDTH HEIGHT'.")
        elif len(args.res) == 2:
            try:
                width = int(args.res[0])
                height = int(args.res[1])
            except ValueError:
                parser.error("Resolution dimensions must be integers.")
        else:
            parser.error("Invalid number of arguments for --res. Use '1k', '2k', '4k' or 'WIDTH HEIGHT'.")

    if not API_KEY:
         cprint(f"\n[ERROR] The API key environment variable '{KEY_ENV_VAR}' is not set.", Colors.RED, args.color)
         print("Please set it before running image generation.")
         return
    
    # --- Model Initialization for Generation ---
    model = None
    try:
        model = genai.GenerativeModel(args.model)
    except Exception as e:
        cprint(f"\n[ERROR] Failed to load model '{args.model}': {e}", Colors.RED, args.color)
        model = None
         
    if not model:
        cprint("\n[ERROR] Model initialization failed. Check the model name and API configuration.", Colors.RED, args.color)
        return
        
    # Conditional Warning based on --debug flag
    if args.delay < 0:
        if args.debug:
            cprint("[WARNING] Delay value must be non-negative. Setting to 0.", Colors.YELLOW, args.color)
        args.delay = 0 # This ensures the delay is reset regardless of the --debug flag

    input_file = Path(args.input)
    
    # --- Determine Output Directory ---
    if args.output_dir:
        output_directory = Path(args.output_dir)
    else:
        output_directory = input_file.parent / input_file.stem

    if args.batch:
        args.overwrite = True
        sys.stdout = open(os.devnull, 'w')

    print(f"Using API Key from environment variable: {KEY_ENV_VAR}")
    print(f"Using AI Model: {args.model}")
    print(f"Target output directory: {output_directory}")
    print(f"Output format: {args.fmt.upper()}")
    print(f"Delay between tasks: {args.delay} second(s)")
    
    # Load prompts
    prompt_list = read_prompts_from_file(input_file, args.type, args.color)
    
    if not prompt_list:
        cprint("No prompts found for processing. Exiting.", Colors.YELLOW, args.color)
        return

    # --- Stats Initialization ---
    stats = {
        'total': len(prompt_list),
        'success': 0,
        'error': 0,
        'skipped': 0,
        'total_size': 0,
        'failed_images': []
    }
    start_time = time.time()

    if args.debug:
        print(f"Resolution set to: {width}x{height}")

    # --- Main Generation Loop ---
    for i, (original_basename, prompt_text) in enumerate(prompt_list):
        
        # 1. Determine Filename
        if args.imgpat:
            filename = f"{args.imgpat}_{i+1:03d}.{args.fmt.lower()}"
        else:
            filename = f"{original_basename}.{args.fmt.lower()}"
            
        output_filepath = output_directory / filename

        print(f"\n--- Generating image {i+1}/{len(prompt_list)} ---")
        print(f"-> Target file: {output_filepath.name}")
        
        # Optionally show full prompt text in debug mode
        if args.debug:
            print(f"-> Prompt (Full): {prompt_text}")
        else:
            print(f"-> Prompt: {prompt_text[:80]}...")

        # 2. Generate Image with Retry Logic
        generation_success = False
        for attempt in range(args.ret):
            if attempt > 0:
                cprint(f"   [RETRY] Attempt {attempt+1}/{args.ret}...", Colors.YELLOW, args.color)
            
            try:
                # Note: generation_config is not used for resolution as it's not supported by all models/libraries in this way.
                # Resolution is handled via post-processing in save_image_from_response.
                response = model.generate_content(prompt_text)
                
                # 3. Save Image
                status, saved_path = save_image_from_response(response, output_filepath, args.fmt, args.overwrite, args.color, target_size=(width, height))
                
                if status == 'SUCCESS' and saved_path:
                    stats['success'] += 1
                    stats['total_size'] += saved_path.stat().st_size
                    generation_success = True
                    break
                elif status == 'SKIPPED':
                    stats['skipped'] += 1
                    generation_success = True
                    break
                else:
                    # Status is ERROR, try again if attempts remain
                    pass
            
            except Exception as e:
                cprint(f"   [API ERROR] Failed to generate image for prompt: {e}", Colors.RED, args.color)
                # Try again if attempts remain
        
        if not generation_success:
            stats['error'] += 1
            stats['failed_images'].append(filename)
            cprint(f"   [FAILED] Could not generate image after {args.ret} attempts.", Colors.RED, args.color)
            
        # 4. Introduce delay if not the last item
        if i < len(prompt_list) - 1 and args.delay > 0:
            print(f"Waiting for {args.delay}s before the next task...")
            time.sleep(args.delay)

    # --- Summary ---
    if args.stat:
        duration = time.time() - start_time
        avg_size = stats['total_size'] / stats['success'] if stats['success'] > 0 else 0
        avg_size_mb = avg_size / (1024 * 1024)
        
        print("\n" + "="*30)
        print("       GENERATION SUMMARY")
        print("="*30)
        print(f"Total Prompts:   {stats['total']}")
        cprint(f"Successful:      {stats['success']}", Colors.GREEN, args.color)
        cprint(f"Errors:          {stats['error']}", Colors.RED, args.color)
        cprint(f"Skipped:         {stats['skipped']}", Colors.YELLOW, args.color)
        print(f"Total Time:      {duration:.2f}s")
        print(f"Avg Image Size:  {avg_size_mb:.2f} MB")
        print("="*30 + "\n")

        if args.stat == 'err' and stats['failed_images']:
            cprint("Failed Images:", Colors.RED, args.color)
            for img in stats['failed_images']:
                cprint(f"- {img}", Colors.RED, args.color)
            print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Script interrupted by user (CTRL+C). Exiting gracefully.")
        sys.exit(0)