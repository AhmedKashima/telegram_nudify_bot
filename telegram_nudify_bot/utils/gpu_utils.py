import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class GPUManager:
    """Utility class for GPU management and detection"""
    
    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if GPU is available"""
        try:
            return torch.cuda.is_available()
        except Exception as e:
            logger.error(f"Error checking GPU availability: {e}")
            return False
    
    @staticmethod
    def get_gpu_info() -> dict:
        """Get GPU information"""
        info = {
            'available': False,
            'count': 0,
            'name': None,
            'memory': None
        }
        
        try:
            if torch.cuda.is_available():
                info['available'] = True
                info['count'] = torch.cuda.device_count()
                
                if info['count'] > 0:
                    # Get first GPU info
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    
                    info['name'] = gpu_name
                    info['memory'] = f"{gpu_memory / 1024**3:.2f} GB"
                    
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return info
    
    @staticmethod
    def get_device() -> str:
        """Get the best available device"""
        if GPUManager.check_gpu_availability():
            return 'cuda'
        else:
            return 'cpu'
    
    @staticmethod
    def get_device_info() -> Tuple[str, Optional[str]]:
        """Get device and device name"""
        device = GPUManager.get_device()
        device_name = None
        
        if device == 'cuda':
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception as e:
                logger.error(f"Error getting CUDA device name: {e}")
        
        return device, device_name
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage"""
        usage = {
            'gpu_available': False,
            'gpu_memory_allocated': 0,
            'gpu_memory_reserved': 0,
            'cpu_memory': 0
        }
        
        try:
            if torch.cuda.is_available():
                usage['gpu_available'] = True
                usage['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                usage['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
        
        return usage
    
    @staticmethod
    def optimize_for_inference():
        """Optimize PyTorch for inference"""
        try:
            # Set to evaluation mode
            torch.set_grad_enabled(False)
            
            # Optimize for inference
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
            logger.info("PyTorch optimized for inference")
            
        except Exception as e:
            logger.error(f"Error optimizing for inference: {e}")
    
    @staticmethod
    def print_system_info():
        """Print system information"""
        logger.info("=== System Information ===")
        
        # GPU Info
        gpu_info = GPUManager.get_gpu_info()
        if gpu_info['available']:
            logger.info(f"GPU: {gpu_info['name']}")
            logger.info(f"GPU Memory: {gpu_info['memory']}")
            logger.info(f"GPU Count: {gpu_info['count']}")
        else:
            logger.warning("No GPU available - using CPU")
        
        # Device Info
        device, device_name = GPUManager.get_device_info()
        logger.info(f"Using device: {device}")
        if device_name:
            logger.info(f"Device name: {device_name}")
        
        # Memory Usage
        memory_usage = GPUManager.get_memory_usage()
        if memory_usage['gpu_available']:
            logger.info(f"GPU Memory Allocated: {memory_usage['gpu_memory_allocated']:.2f} GB")
            logger.info(f"GPU Memory Reserved: {memory_usage['gpu_memory_reserved']:.2f} GB")
        
        logger.info("==========================") 