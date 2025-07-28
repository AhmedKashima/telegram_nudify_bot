#!/usr/bin/env python3
"""
Telegram Nudify Bot
Main bot application for processing images
"""
import torch  # если ещё не импортирован
device = "cpu"  # или "c
import asyncio
import logging
import os
import time
from typing import Optional
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import TelegramError

from config import Config
from pipeline.nudify_pipeline import NudifyPipeline
from utils.image_utils import ImageProcessor
from utils.gpu_utils import GPUManager

# Setup logging
Config.setup_logging()
logger = logging.getLogger(__name__)

class NudifyBot:
    """Main Telegram bot class"""
    
    def __init__(self):
        self.config = Config
        self.pipeline = None
        self.application = None
        self.processing_users = set()  # Track users currently processing
        
        # Validate configuration
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    async def start(self):
        """Start the bot"""
        try:
            # Initialize pipeline
            logger.info("Initializing nudify pipeline...")
            self.pipeline = NudifyPipeline(self.config)
            
            # Create application
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
            
            # Add handlers
            self._add_handlers()
            
            # Print system info
            GPUManager.print_system_info()
            
            # Start bot
            logger.info("Starting Telegram bot...")
            await self.application.initialize()
            await self.application.start()
            await self.application.run_polling()
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise
    
    def _add_handlers(self):
        """Add message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        logger.info(f"User {user.id} ({user.username}) started the bot")
        
        welcome_message = f"""
🤖 **Nudify Bot - Welcome!**

👤 **User:** @{user.username or 'Unknown'}
🕐 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**How to use:**
1. Send me a photo (JPEG or PNG)
2. I'll process it using AI
3. You'll receive the result

**Commands:**
• `/start` - Show this message
• `/help` - Show help
• `/status` - Check system status

⚠️ **Note:** This bot is for educational purposes only.
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
🤖 **Nudify Bot - Help**

**How to use:**
1. Send a photo to the bot
2. Wait for processing (may take 30-60 seconds)
3. Receive the processed result

**Supported formats:**
• JPEG (.jpg, .jpeg)
• PNG (.png)

**Processing steps:**
1. Image validation and preprocessing
2. Human segmentation using U2Net
3. AI-powered image generation
4. Post-processing and enhancement

**System requirements:**
• GPU recommended for faster processing
• Stable Diffusion models
• U2Net segmentation model

**Commands:**
• `/start` - Initialize bot
• `/help` - Show this help
• `/status` - Check system status

⚠️ **Educational use only!**
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get pipeline status
            pipeline_status = self.pipeline.get_status() if self.pipeline else {}
            
            # Get GPU info
            gpu_info = GPUManager.get_gpu_info()
            memory_usage = GPUManager.get_memory_usage()
            
            status_message = f"""
🔧 **System Status**

**GPU Information:**
• Available: {'✅ Yes' if gpu_info['available'] else '❌ No'}
• Device: {pipeline_status.get('device', 'Unknown')}
• Memory: {gpu_info.get('memory', 'Unknown')}

**Model Status:**
• Segmentation: {'✅ Loaded' if pipeline_status.get('segmentation_loaded') else '❌ Not loaded'}
• Img2Img: {'✅ Loaded' if pipeline_status.get('img2img_loaded') else '❌ Not loaded'}
• NSFW Model: {'✅ Loaded' if pipeline_status.get('nsfw_loaded') else '❌ Not loaded'}

**Memory Usage:**
• GPU Allocated: {memory_usage.get('gpu_memory_allocated', 0):.2f} GB
• GPU Reserved: {memory_usage.get('gpu_memory_reserved', 0):.2f} GB

**Processing:**
• Active users: {len(self.processing_users)}
• Status: {'🟢 Ready' if not self.processing_users else '🟡 Busy'}
            """
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Status command failed: {e}")
            await update.message.reply_text("❌ Failed to get system status")
    
    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages"""
        user = update.effective_user
        user_id = user.id
        
        # Check if user is already processing
        if user_id in self.processing_users:
            await update.message.reply_text("⏳ You already have a processing request. Please wait for the current one to complete.")
            return
        
        # Check if pipeline is available
        if not self.pipeline:
            await update.message.reply_text("❌ Processing pipeline is not available. Please try again later.")
            return
        
        try:
            # Add user to processing set
            self.processing_users.add(user_id)
            
            # Send processing message
            processing_message = await update.message.reply_text("🔄 Processing your image... This may take 30-60 seconds.")
            
            # Get photo
            photo = update.message.photo[-1]  # Get highest quality
            file = await context.bot.get_file(photo.file_id)
            
            # Download image
            image_data = await file.download_as_bytearray()
            
            # Validate image
            if not ImageProcessor.validate_image(image_data):
                await update.message.reply_text("❌ Invalid image format. Please send a JPEG or PNG image.")
                return
            
            # Load image
            image = ImageProcessor.load_image(image_data)
            if image is None:
                await update.message.reply_text("❌ Failed to load image. Please try again.")
                return
            
            # Process image
            logger.info(f"Processing image for user {user_id}")
            result_image = self.pipeline.process_image(image)
            
            if result_image is None:
                await update.message.reply_text("❌ Image processing failed. Please try again with a different image.")
                return
            
            # Convert result to bytes
            result_bytes = ImageProcessor.image_to_bytes(result_image)
            
            # Send result
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=result_bytes,
                caption="✅ Processing completed! Here's your result."
            )
            
            # Update processing message
            await processing_message.edit_text("✅ Processing completed!")
            
            logger.info(f"Successfully processed image for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing image for user {user_id}: {e}")
            await update.message.reply_text("❌ An error occurred during processing. Please try again.")
            
        finally:
            # Remove user from processing set
            self.processing_users.discard(user_id)
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "status":
            await self._status_command(update, context)
    
    async def stop(self):
        """Stop the bot"""
        try:
            logger.info("Stopping bot...")
            
            # Cleanup pipeline
            if self.pipeline:
                self.pipeline.cleanup()
            
            # Stop application
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            
            logger.info("Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

async def main():
    """Main function"""
    bot = None
    try:
        # Create and start bot
        bot = NudifyBot()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        # Cleanup
        if bot:
            await bot.stop()

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main()) 