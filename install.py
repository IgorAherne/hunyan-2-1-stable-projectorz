import subprocess
import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass

def get_current_script_dir() -> Path:
    """Helper to get the directory of the current script, handling potential errors."""
    try:
        # Resolve to handle symlinks and get the absolute path
        return Path(__file__).parent.resolve()
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive execution)
        return Path(os.getcwd()).resolve()

def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """
    Check internet connectivity and return more detailed error information.
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        # Handle different types of URL errors
        reason = getattr(e, 'reason', str(e))
        if isinstance(reason, socket.gaierror):
            return False, f"DNS resolution failed: {reason}"
        elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def get_git_env() -> dict:
    """
    Return a copy of the current environment configured to use the *portable* Git.
    """
    env = os.environ.copy()
    
    # Determine the path to portable Git relative to the 'code' directory
    CODE_DIR = get_current_script_dir()
    # Use resolve() for robust absolute path calculation
    PORTABLE_GIT_BASE = (CODE_DIR / ".." / "tools" / "git").resolve()
    
    # Prepend the portable Git folders to PATH
    git_paths = [
        str(PORTABLE_GIT_BASE / "mingw64" / "bin"),
        str(PORTABLE_GIT_BASE / "cmd"),
        str(PORTABLE_GIT_BASE / "usr" / "bin"),
        str(PORTABLE_GIT_BASE / "mingw64" / "libexec" / "git-core"),
    ]
    existing_path = env.get("PATH", "")
    # Ensure paths are correctly concatenated
    env["PATH"] = ";".join(git_paths) + (";" + existing_path if existing_path else "")
    
    # Set SSL cert variables
    ca_bundle = PORTABLE_GIT_BASE / "mingw64" / "etc" / "ssl" / "certs" / "ca-bundle.crt"
    if ca_bundle.exists():
        env["GIT_SSL_CAINFO"] = str(ca_bundle)
        env["SSL_CERT_FILE"]  = str(ca_bundle)
    
    return env

def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
    """
    Run a command with retry logic, using the portable Git environment.
    """
    last_error = None
    env = get_git_env()
    
    # Standardize pip install command format
    if cmd.startswith('pip install'):
        args = cmd[11:]
        # Use sys.executable (the activated venv python) and ensure no caching/isolation
        cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'

    # Add progress bar for pip commands
    if "pip install" in cmd and "--progress-bar" not in cmd:
        cmd += " --progress-bar=on"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                # Check connectivity before retry
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Connection check failed: {error_msg}")
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue # Proceed to the next attempt
            
            if "pip install" in cmd:
                # For pip, stream stdout for progress bar, capture stderr
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
            else:
                # For others, capture both
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                return result
            
            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")
            
        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))
        
        if attempt < max_retries - 1:
            print(f"Waiting {RETRY_DELAY} seconds before next attempt...")
            time.sleep(RETRY_DELAY)
    
    # If we get here, all retries failed
    raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")

def download_file(url: str, dest_path: Path, desc: str):
    """
    Download a file using urllib.request (streaming) with retry logic.
    """
    print(f"Downloading {desc}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"File already exists: {dest_path}. Skipping download.")
        return

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                print(f"\nRetry download attempt {attempt + 1}/{MAX_RETRIES}...")

            # Use a User-Agent to avoid potential blocking by servers
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            # Use shutil.copyfileobj for efficient streaming download
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"Successfully downloaded {desc}.")
            return
        except Exception as e:
            print(f"Download failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            # Clean up partial download
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
    
    raise InstallationError(f"Failed to download {desc} after {MAX_RETRIES} attempts.")

def install_dependencies():
    """Install all required dependencies with improved error handling."""
    CODE_DIR = get_current_script_dir()
    
    try:
        # Initial connectivity check
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            print("Please check your connection and try again.")
            sys.exit(1)
        
        # List of packages to install with pip
        # Based on the Hunyuan 2.1 requirements (Torch 2.5.1 with CUDA 12.4)
        packages = [
            (f"pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128", "Installing PyTorch 2.8 with CUDA 12.8"),
            ("pip install -r requirements.txt", "Installing dependencies from requirements.txt"),
            # Ensuring key packages needed for Gradio/API are installed/updated
            (f"pip install huggingface_hub accelerate gradio", "Installing HuggingFace Hub, Accelerate, and Gradio"),
        ]

        # Local wheel files (Placeholders)
        # Assuming wheels will be placed in 'code/whl' and compiled for Python 3.11 (cp311)
        wheel_files = {
            "custom_rasterizer": "whl/custom_rasterizer-0.1-cp311-cp311-win_amd64.whl",
            "differentiable_renderer": "whl/differentiable_renderer_mesh_painter-0.1-cp311-cp311-win_amd64.whl",
            # Add other wheels as needed
        }

        # Install packages (with retry)
        for cmd, desc in packages:
            print(f"\n--- {desc} ---")
            run_command_with_retry(cmd, desc)
        
        # Install local wheels
        print("\n--- Installing Custom Wheels ---")

        for name, wheel_path_str in wheel_files.items():
            # Path is relative to the 'code' directory
            wheel = CODE_DIR / wheel_path_str
            if not wheel.exists():
                # We treat missing wheels as errors since they are supposed to be bundled
                raise InstallationError(f"Required wheel file not found: {wheel}. Please ensure the 'whl' folder is present and complete.")
            
            print(f"Installing {name}...")
            run_command_with_retry(f"pip install {wheel}", f"Installing {name} from local wheel")

        # Download required models
        print("\n--- Downloading Models ---")
        # Real-ESRGAN model (required by the installation instructions)
        esrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        # Destination path relative to the 'code' directory structure of the repo
        esrgan_dest = CODE_DIR / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        download_file(esrgan_url, esrgan_dest, "RealESRGAN_x4plus.pth")

        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        print("\nSuggestions:")
        print("1. Check your internet connection.")
        print("2. Verify your firewall/antivirus isn't blocking connections.")
        print("3. Ensure the installer package is complete (check the 'whl' folder).")
        print("4. If network issues persist, try running the installer again.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during installation: {str(e)}")
        sys.exit(1)

def verify_installation():
    """Verify that critical packages were installed correctly."""
    try:
        import torch
        import gradio # Check if Gradio is installed
        print(f"\nVerification successful.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
        return True
    except ImportError as e:
        print(f"Verification failed (missing dependency): {str(e)}")
        return False
    except Exception as e:
        print(f"Verification failed (runtime error): {str(e)}")
        return False

if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified successfully!")
    else:
        print("\nInstallation completed but verification failed. Check the logs above for errors.")
        sys.exit(1)
