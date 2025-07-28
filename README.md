# 🛡️ Educational Telegram Nudify Bot (Local Only)

### FOR RESEARCH USE ONLY  
This bot is strictly intended for:  
🔬 Educational purposes  
🔐 Private use  
⚖️ Legal applications on local machines  

❌ **Do not deploy or distribute publicly**  

## 📦 Features
- Accepts image files (JPEG, PNG) via Telegram
- Applies segmentation with U2Net (via rembg)
- Runs img2img nudification using Stable Diffusion (local .ckpt or .safetensors)
- Sends result back to the user
- Access is restricted to authorized Telegram user IDs
- Built with Docker + NVIDIA GPU support
- Fully offline, no external API calls

## 📁 Project Structure

📁 telegram_nudify_bot/
├── 📄 config.py
├── 📄 docker-compose.yml
├── 📄 Dockerfile
├── 📄 env.example
├── 📂 logs/
├── 📄 main.py
├── 📂 models/
├── 📂 output/
├── 📂 pipeline/
├── 📂 __pycache__/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📂 temp/
├── 📂 utils/
├── 📂 venv/
└── 📂 wheels/


### 1. 📥 Clone & Enter Directory
```bash
git clone git@github.com:AhmedKashima/telegram_nudify_bot.git
cd telegram_nudify_bot
```

## 🔐 Create .env
```bash
BOT_TOKEN=your_telegram_bot_token
AUTHORIZED_USERS=123456789,987654321
MODEL_PATH=/app/models/nudify-model.ckpt
```

## 📦 Install Dependencies (for local dev)
```bash
pip install -r requirements.txt
```

## � Run with Docker (GPU only)
```bash
docker compose up --build
```

## 🔄 Workflow Overview
User Submission
📤 Sends photo (JPEG/PNG)

Bot Processing
├─ 🔒 User ID verification
├─ 🖼️ Image validation
├─ ✂️ Background segmentation (rembg/U2Net)
├─ 🎨 Stable Diffusion img2img processing
├─ 💧 
└─ 📤 Returns processed image

