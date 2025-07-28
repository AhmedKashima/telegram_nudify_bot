import os
import io
import logging
from PIL import Image, ImageOps
import numpy as np
import cv2
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        """Validate if the image data is valid"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    @staticmethod
    def load_image(image_data: bytes) -> Optional[Image.Image]:
        """Load image from bytes"""
        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        # Calculate new size
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    @staticmethod
    def pad_image_to_square(image: Image.Image, size: int = 512) -> Image.Image:
        """Pad image to square format"""
        # Calculate padding
        width, height = image.size
        max_dim = max(width, height)
        
        # Create new square image with padding
        new_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        
        # Calculate position to center the image
        x_offset = (max_dim - width) // 2
        y_offset = (max_dim - height) // 2
        
        # Paste the original image
        new_image.paste(image, (x_offset, y_offset))
        
        # Resize to target size
        new_image = new_image.resize((size, size), Image.Resampling.LANCZOS)
        return new_image
    
    @staticmethod
    def image_to_tensor(image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array for processing"""
        return np.array(image)
    
    @staticmethod
    def tensor_to_image(tensor: np.ndarray) -> Image.Image:
        """Convert numpy array back to PIL image"""
        # Ensure values are in valid range
        tensor = np.clip(tensor, 0, 255).astype(np.uint8)
        return Image.fromarray(tensor)
    
    @staticmethod
    def save_image(image: Image.Image, filepath: str) -> bool:
        """Save image to file"""
        try:
            image.save(filepath, 'JPEG', quality=95)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
        """Convert PIL image to bytes"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format, quality=95)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            return b''
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Apply basic image enhancement"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """Get image information"""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'width': image.width,
            'height': image.height
        } 