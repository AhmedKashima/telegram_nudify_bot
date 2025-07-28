import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Telegram Nudify Bot"""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/stable-diffusion-v1-5')
    NSFW_MODEL_PATH = os.getenv('NSFW_MODEL_PATH', '/app/models/nudify-v1')
    U2NET_MODEL_PATH = os.getenv('U2NET_MODEL_PATH', '/app/models/u2net')
    
    # Processing Configuration
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', 1024))
    GENERATION_STEPS = int(os.getenv('GENERATION_STEPS', 50))
    GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', 7.5))
    STRENGTH = float(os.getenv('STRENGTH', 0.75))
    
    # GPU Configuration
    USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
    DEVICE = os.getenv('DEVICE', 'cuda' if USE_GPU else 'cpu')
    
    # Storage Configuration
    TEMP_DIR = os.getenv('TEMP_DIR', '/app/temp')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/app/output')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/app/logs/bot.log')
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        
        # Create necessary directories
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        ) 