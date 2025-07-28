import logging
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import pipeline
from typing import Optional, Tuple
import os
import time

from .segmentation import HumanSegmentation
from utils.gpu_utils import GPUManager
from utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

class NudifyPipeline:
    """Main pipeline for nudify image processing"""
    
    def __init__(self, config):
        self.config = config
        # self.device = GPUManager.get_device()
        self.device = "cpu"
        self.segmentation = HumanSegmentation()
        self.img2img_pipeline = None
        self.nsfw_pipeline = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading models...")
            
            # Load segmentation model
            if not self.segmentation.load_model():
                raise Exception("Failed to load segmentation model")
            
            # Load Stable Diffusion img2img pipeline
            model_path = self.config.MODEL_PATH
            if os.path.exists(model_path):
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
                logger.info(f"Loaded img2img pipeline from {model_path}")
            else:
                # Fallback to online model
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
                logger.info("Loaded online img2img pipeline")
            
            # Load NSFW pipeline if available
            nsfw_model_path = self.config.NSFW_MODEL_PATH
            if os.path.exists(nsfw_model_path):
                self.nsfw_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    nsfw_model_path,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.nsfw_pipeline = self.nsfw_pipeline.to(self.device)
                logger.info(f"Loaded NSFW pipeline from {nsfw_model_path}")
            else:
                logger.warning(f"NSFW model not found at {nsfw_model_path}")
            
            # Optimize for inference
            GPUManager.optimize_for_inference()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def process_image(self, image: Image.Image) -> Optional[Image.Image]:
        """Main processing pipeline"""
        try:
            start_time = time.time()
            logger.info("Starting image processing...")
            
            # Step 1: Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Step 2: Segment human
            segmented_image, mask = self._segment_human(processed_image)
            if segmented_image is None:
                raise Exception("Human segmentation failed")
            
            # Step 3: Generate nudified version
            result_image = self._generate_nudified(segmented_image, mask)
            if result_image is None:
                raise Exception("Nudify generation failed")
            
            # Step 4: Post-process result
            final_image = self._postprocess_image(result_image)
            
            processing_time = time.time() - start_time
            logger.info(f"Image processing completed in {processing_time:.2f} seconds")
            
            return final_image
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess the input image"""
        try:
            # Resize image
            resized_image = ImageProcessor.resize_image(
                image, 
                self.config.MAX_IMAGE_SIZE
            )
            
            # Enhance image
            enhanced_image = ImageProcessor.enhance_image(resized_image)
            
            # Pad to square if necessary
            if enhanced_image.size[0] != enhanced_image.size[1]:
                enhanced_image = ImageProcessor.pad_image_to_square(enhanced_image)
            
            logger.info(f"Image preprocessed: {enhanced_image.size}")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _segment_human(self, image: Image.Image) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Segment human from image"""
        try:
            segmented_image, mask = self.segmentation.segment_human(image)
            
            if segmented_image is not None and mask is not None:
                # Enhance segmentation
                enhanced_image = self.segmentation.enhance_segmentation(segmented_image, mask)
                
                # Check segmentation quality
                quality = self.segmentation.get_segmentation_quality(mask)
                if quality < 0.1:  # Very low quality
                    logger.warning("Low segmentation quality detected")
                
                return enhanced_image, mask
            else:
                raise Exception("Segmentation returned None")
                
        except Exception as e:
            logger.error(f"Human segmentation failed: {e}")
            return None, None
    
    def _generate_nudified(self, image: Image.Image, mask: Image.Image) -> Optional[Image.Image]:
        """Generate nudified version using Stable Diffusion"""
        try:
            # Choose pipeline (NSFW if available, otherwise standard)
            pipeline = self.nsfw_pipeline if self.nsfw_pipeline else self.img2img_pipeline
            
            if pipeline is None:
                raise Exception("No pipeline available")
            
            # Prepare prompt
            prompt = "nude woman, beautiful, high quality, detailed"
            negative_prompt = "clothes, clothing, dressed, low quality, blurry"
            
            # Generate image
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=self.config.STRENGTH,
                guidance_scale=self.config.GUIDANCE_SCALE,
                num_inference_steps=self.config.GENERATION_STEPS,
                num_images_per_prompt=1
            )
            
            if result.images and len(result.images) > 0:
                generated_image = result.images[0]
                
                # Apply mask to preserve original pose and environment
                final_image = self._apply_mask(generated_image, image, mask)
                
                logger.info("Nudify generation completed")
                return final_image
            else:
                raise Exception("No images generated")
                
        except Exception as e:
            logger.error(f"Nudify generation failed: {e}")
            return None
    
    def _apply_mask(self, generated_image: Image.Image, original_image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply mask to blend generated and original image"""
        try:
            # Convert images to numpy arrays
            gen_array = np.array(generated_image)
            orig_array = np.array(original_image)
            mask_array = np.array(mask).astype(np.float32) / 255.0
            
            # Ensure mask has correct dimensions
            if len(mask_array.shape) == 2:
                mask_array = mask_array[:, :, np.newaxis]
            
            # Blend images using mask
            # Generated image for subject (masked areas), original for background
            blended_array = gen_array * mask_array + orig_array * (1 - mask_array)
            
            # Convert back to PIL image
            blended_image = Image.fromarray(blended_array.astype(np.uint8))
            
            return blended_image
            
        except Exception as e:
            logger.error(f"Mask application failed: {e}")
            return generated_image
    
    def _postprocess_image(self, image: Image.Image) -> Image.Image:
        """Post-process the generated image"""
        try:
            # Enhance final result
            enhanced_image = ImageProcessor.enhance_image(image)
            
            # Ensure proper size
            if enhanced_image.size[0] > self.config.MAX_IMAGE_SIZE or enhanced_image.size[1] > self.config.MAX_IMAGE_SIZE:
                enhanced_image = ImageProcessor.resize_image(enhanced_image, self.config.MAX_IMAGE_SIZE)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image postprocessing failed: {e}")
            return image
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear GPU memory
            GPUManager.clear_gpu_memory()
            
            # Delete pipeline references
            if self.img2img_pipeline:
                del self.img2img_pipeline
            if self.nsfw_pipeline:
                del self.nsfw_pipeline
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {e}")
    
    def get_status(self) -> dict:
        """Get pipeline status"""
        return {
            'device': self.device,
            'segmentation_loaded': self.segmentation is not None,
            'img2img_loaded': self.img2img_pipeline is not None,
            'nsfw_loaded': self.nsfw_pipeline is not None,
            'gpu_available': GPUManager.check_gpu_availability(),
            'memory_usage': GPUManager.get_memory_usage()
        } 