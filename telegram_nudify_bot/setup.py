#!/usr/bin/env python3
"""
Setup Script for Telegram Nudify Bot
Automates initial configuration and model downloading
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotSetup:
    """Setup class for the Telegram Nudify Bot"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.models_dir = self.project_dir / "models"
        self.temp_dir = self.project_dir / "temp"
        self.output_dir = self.project_dir / "output"
        self.logs_dir = self.project_dir / "logs"
    
    def print_banner(self):
        """Print setup banner"""
        print("🤖 Telegram Nudify Bot Setup")
        print("=" * 40)
        print("Setting up the AI-powered image processing bot...")
        print()
    
    def check_python_version(self):
        """Check Python version"""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            print(f"   Current version: {sys.version}")
            return False
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True, capture_output=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        directories = [
            self.models_dir,
            self.temp_dir,
            self.output_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def setup_environment(self):
        """Setup environment file"""
        print("\n⚙️  Setting up environment...")
        
        env_file = self.project_dir / ".env"
        env_example = self.project_dir / "env.example"
        
        if env_file.exists():
            print("⚠️  .env file already exists")
            response = input("Do you want to overwrite it? (y/N): ")
            if response.lower() != 'y':
                print("   Skipping environment setup")
                return True
        
        if not env_example.exists():
            print("❌ env.example not found")
            return False
        
        try:
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("   Please edit .env with your configuration:")
            print("   - Add your Telegram bot token")
            print("   - Configure model paths")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env: {e}")
            return False
    
    def download_models(self):
        """Download required models"""
        print("\n📥 Downloading models...")
        
        models_to_download = {
            "stable-diffusion-v1-5": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
            "u2net": "https://huggingface.co/xuebinqin/U-2-Net"
        }
        
        for model_name, model_url in models_to_download.items():
            model_path = self.models_dir / model_name
            if model_path.exists():
                print(f"⚠️  Model {model_name} already exists")
                continue
            
            print(f"📥 Downloading {model_name}...")
            try:
                # Use git to clone the model
                subprocess.run([
                    "git", "clone", model_url, str(model_path)
                ], check=True, capture_output=True)
                print(f"✅ Downloaded {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download {model_name}: {e}")
                print(f"   You can manually download it from: {model_url}")
    
    def check_gpu(self):
        """Check GPU availability"""
        print("\n🔧 Checking GPU...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU available: {gpu_name}")
                print(f"   GPU count: {gpu_count}")
                return True
            else:
                print("⚠️  No GPU available - will use CPU (slower)")
                return False
        except ImportError:
            print("⚠️  PyTorch not installed - cannot check GPU")
            return False
    
    def test_installation(self):
        """Test the installation"""
        print("\n🧪 Testing installation...")
        
        try:
            # Test imports
            import torch
            import diffusers
            import transformers
            import rembg
            from PIL import Image
            import telegram
            
            print("✅ All required packages imported successfully")
            
            # Test GPU
            if torch.cuda.is_available():
                print("✅ GPU is available")
            else:
                print("⚠️  GPU not available - using CPU")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for user"""
        print("\n" + "="*50)
        print("🎉 Setup Complete!")
        print("="*50)
        print()
        print("📋 Next Steps:")
        print("1. Edit .env file with your configuration:")
        print("   - Add your Telegram bot token from @BotFather")
        print("   - Configure model paths if needed")
        print()
        print("2. Start the bot:")
        print("   # Using Docker (recommended):")
        print("   docker-compose up -d")
        print()
        print("   # Using Python directly:")
        print("   python main.py")
        print()
        print("3. Test the bot:")
        print("   - Send /start to your bot")
        print("   - Send a photo to test processing")
        print()
        print("📚 Documentation:")
        print("   - README.md for detailed instructions")
        print("   - Check logs/ directory for logs")
        print()
        print("⚠️  Remember: This is for educational purposes only!")
    
    def run(self):
        """Run the complete setup"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            sys.exit(1)
        
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Setup failed at dependency installation")
            sys.exit(1)
        
        # Create directories
        self.create_directories()
        
        # Setup environment
        if not self.setup_environment():
            print("❌ Setup failed at environment setup")
            sys.exit(1)
        
        # Check GPU
        self.check_gpu()
        
        # Download models (optional)
        response = input("\nDo you want to download models now? (y/N): ")
        if response.lower() == 'y':
            self.download_models()
        
        # Test installation
        if not self.test_installation():
            print("❌ Setup failed at testing")
            sys.exit(1)
        
        # Print next steps
        self.print_next_steps()

def main():
    """Main function"""
    setup = BotSetup()
    setup.run()

if __name__ == "__main__":
    main() 
