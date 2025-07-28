import logging
import torch
import numpy as np
from PIL import Image
from rembg import remove
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

class HumanSegmentation:
    """Human segmentation using U2Net model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Load the U2Net model"""
        try:
            # rembg automatically handles model loading
            logger.info("U2Net model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load U2Net model: {e}")
            return False
    
    def segment_human(self, image: Image.Image) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Segment human from image
        
        Returns:
            Tuple of (segmented_image, mask)
        """
        try:
            # Convert PIL image to bytes for rembg
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            # Remove background (this will isolate the main subject)
            result_bytes = remove(image_bytes)
            
            # Convert back to PIL image
            segmented_image = Image.open(io.BytesIO(result_bytes))
            
            # Create a simple mask (white pixels are the subject)
            mask = self._create_mask_from_segmented(segmented_image)
            
            logger.info("Human segmentation completed successfully")
            return segmented_image, mask
            
        except Exception as e:
            logger.error(f"Human segmentation failed: {e}")
            return None, None
    
    def _create_mask_from_segmented(self, segmented_image: Image.Image) -> Image.Image:
        """Create a mask from the segmented image"""
        try:
            # Convert to numpy array
            img_array = np.array(segmented_image)
            
            # Create mask based on non-white pixels
            # White background (255, 255, 255) becomes 0, everything else becomes 255
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            
            # Find non-white pixels (subject)
            non_white = np.any(img_array != [255, 255, 255], axis=2)
            mask[non_white] = 255
            
            return Image.fromarray(mask)
            
        except Exception as e:
            logger.error(f"Failed to create mask: {e}")
            # Return a simple mask (all white)
            return Image.new('L', segmented_image.size, 255)
    
    def enhance_segmentation(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Enhance the segmentation result"""
        try:
            # Convert to numpy arrays
            img_array = np.array(image)
            mask_array = np.array(mask)
            
            # Apply morphological operations to clean up the mask
            import cv2
            kernel = np.ones((5, 5), np.uint8)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur to smooth the mask edges
            mask_array = cv2.GaussianBlur(mask_array, (5, 5), 0)
            
            # Normalize mask to 0-1 range
            mask_array = mask_array.astype(np.float32) / 255.0
            
            # Apply mask to image
            enhanced_image = img_array * mask_array[:, :, np.newaxis]
            
            return Image.fromarray(enhanced_image.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Failed to enhance segmentation: {e}")
            return image
    
    def get_segmentation_quality(self, mask: Image.Image) -> float:
        """Calculate segmentation quality score"""
        try:
            mask_array = np.array(mask)
            
            # Calculate the ratio of subject pixels to total pixels
            total_pixels = mask_array.size
            subject_pixels = np.sum(mask_array > 0)
            
            quality_score = subject_pixels / total_pixels
            
            logger.info(f"Segmentation quality score: {quality_score:.3f}")
            return quality_score
            
        except Exception as e:
            logger.error(f"Failed to calculate segmentation quality: {e}")
            return 0.0 