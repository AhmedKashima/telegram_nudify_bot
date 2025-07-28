# Telegram Nudify Bot

A sophisticated Telegram bot that uses AI to process images with human segmentation and Stable Diffusion. This bot combines U2Net for human segmentation with Stable Diffusion for image-to-image generation.

## ⚠️ DISCLAIMER

**This project is for educational purposes only.**
- Use responsibly and ethically
- Respect privacy and consent
- Do not use for malicious purposes
- Follow local laws and regulations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │  Image Pipeline │    │  AI Models      │
│                 │    │                 │    │                 │
│ • Photo Handler │◄──►│ • Preprocessing │◄──►│ • U2Net         │
│ • Commands      │    │ • Segmentation  │    │ • Stable Diff   │
│ • Status Check  │    │ • Generation    │    │ • NSFW Models   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
telegram_nudify_bot/
├── main.py                    # Main bot application
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── setup.py                   # Automated setup script
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── env.example                # Environment template
├── README.md                  # This file
├── pipeline/
│   ├── __init__.py
│   ├── nudify_pipeline.py    # Main processing pipeline
│   └── segmentation.py       # Human segmentation
├── utils/
│   ├── __init__.py
│   ├── image_utils.py        # Image processing utilities
│   └── gpu_utils.py          # GPU management
├── models/                    # AI models directory
├── temp/                      # Temporary files
├── output/                    # Generated images
└── logs/                      # Log files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended)
- Docker & Docker Compose (for containerized deployment)
- Telegram Bot Token (from @BotFather)

### 1. Automated Setup

```bash
git clone git@github.com:AhmedKashima/telegram_nudify_bot.git
cd telegram_nudify_bot

# Run automated setup
python setup.py
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

### 3. Docker Setup (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

## ⚙️ Configuration

Edit the `.env` file with your settings:

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Model Configuration
MODEL_PATH=/app/models/stable-diffusion-v1-5
NSFW_MODEL_PATH=/app/models/nudify-v1
U2NET_MODEL_PATH=/app/models/u2net

# Processing Configuration
MAX_IMAGE_SIZE=1024
GENERATION_STEPS=50
GUIDANCE_SCALE=7.5
STRENGTH=0.75

# GPU Configuration
USE_GPU=true
DEVICE=cuda

# Storage Configuration
TEMP_DIR=/app/temp
OUTPUT_DIR=/app/output

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/bot.log
```

## 🤖 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize the bot and show welcome message |
| `/help` | Show help and usage instructions |
| `/status` | Check system status and GPU information |

## 📱 Usage

1. **Start the bot**: Send `/start` to initialize
2. **Send a photo**: Upload a JPEG or PNG image
3. **Wait for processing**: The bot will process the image (30-60 seconds)
4. **Receive result**: Get the processed image back

## 🔧 Technical Details

### Processing Pipeline

1. **Image Validation**: Check format and size
2. **Preprocessing**: Resize, enhance, and normalize
3. **Human Segmentation**: Use U2Net to isolate subjects
4. **AI Generation**: Apply Stable Diffusion img2img
5. **Post-processing**: Enhance and finalize result
6. **Mask Application**: Blend generated and original

### AI Models Used

- **U2Net**: Human segmentation and background removal
- **Stable Diffusion**: Image-to-image generation
- **NSFW Models**: Specialized models for adult content

### GPU Optimization

- Automatic GPU detection and utilization
- Memory management and cleanup
- Optimized inference settings
- Fallback to CPU if GPU unavailable

## 🐳 Docker Deployment

### Prerequisites

- Docker and Docker Compose
- NVIDIA Docker runtime
- NVIDIA GPU drivers

### Quick Start

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### GPU Support

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 📊 Monitoring

### Log Files

- `logs/bot.log` - Main application logs
- Docker logs: `docker-compose logs -f`

### Health Checks

```bash
# Check bot status
curl http://localhost:8000/health

# Check GPU status
docker exec telegram-nudify-bot python -c "import torch; print(torch.cuda.is_available())"
```

### System Status

Use `/status` command in Telegram to check:
- GPU availability and memory
- Model loading status
- Active processing users
- System resources

## 🔒 Security Features

- **Input Validation**: Check image format and size
- **User Rate Limiting**: Prevent abuse
- **Error Handling**: Graceful failure recovery
- **Resource Management**: Memory and GPU cleanup
- **Logging**: Comprehensive activity tracking

## 🛠️ Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Models Not Found**
   ```bash
   # Download models manually
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/stable-diffusion-v1-5
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in config
   GENERATION_STEPS=30
   MAX_IMAGE_SIZE=512
   ```

4. **Bot Not Responding**
   ```bash
   # Check token
   echo $TELEGRAM_BOT_TOKEN
   
   # Check logs
   tail -f logs/bot.log
   ```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python main.py

# Docker debug
docker-compose logs -f
```

## 📚 Advanced Configuration

### Custom Models

```bash
# Add custom NSFW model
MODEL_PATH=/app/models/your-custom-model

# Use different segmentation model
U2NET_MODEL_PATH=/app/models/custom-u2net
```

### Performance Tuning

```bash
# Faster processing (lower quality)
GENERATION_STEPS=20
GUIDANCE_SCALE=5.0

# Higher quality (slower)
GENERATION_STEPS=100
GUIDANCE_SCALE=10.0
```

### Memory Optimization

```bash
# Reduce memory usage
MAX_IMAGE_SIZE=512
USE_GPU=false  # Use CPU instead
```

## 🔄 Development

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with hot reload
python main.py
```

### Testing

```bash
# Run tests
python -m pytest tests/

# Test specific components
python -c "from pipeline.nudify_pipeline import NudifyPipeline; print('Pipeline OK')"
```

## 📄 License

This project is for educational purposes only. Use responsibly and ethically.

## 🤝 Contributing

Contributions are welcome for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates

## ⚖️ Legal Notice

- This software is for educational purposes only
- Users are responsible for compliance with local laws
- No warranty is provided
- Use at your own risk

---

**Remember**: Always use AI tools responsibly and respect privacy and consent. 
