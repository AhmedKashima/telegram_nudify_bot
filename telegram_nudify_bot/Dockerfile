# # Use NVIDIA CUDA base image
# # FROM nvidia/cuda:11.8-devel-ubuntu20.04

# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# # Set working directory
# WORKDIR /app


# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev \
#     git \
#     wget \
#     curl \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create symbolic link for python
# RUN ln -s /usr/bin/python3 /usr/bin/python

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Install additional dependencies for GPU support
# RUN pip3 install --no-cache-dir \
#     torch==2.1.1+cu118 \
#     torchvision==0.16.1+cu118 \
#     --index-url https://download.pytorch.org/whl/cu118

# # Create necessary directories
# RUN mkdir -p /app/models /app/temp /app/output /app/logs

# # Copy application code
# COPY . .

# # Create non-root user
# RUN useradd -m -u 1000 botuser && \
#     chown -R botuser:botuser /app

# # Switch to non-root user
# USER botuser

# # Set environment variables for the application
# ENV PYTHONPATH=/app
# ENV CUDA_VISIBLE_DEVICES=0

# # Expose port (if needed for web interface)
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# # Default command
# CMD ["python3", "main.py"] 



# COPY wheels/ /wheels/

# RUN pip3 install --no-cache-dir \
#     /wheels/torch-2.1.1+cu118-cp38-cp38-linux_x86_64.whl \
#     /wheels/torchvision-0.16.1+cu118-cp38-cp38-linux_x86_64.whl

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Копируем wheel файлы в контейнер (если используешь wheels)
COPY wheels/ /wheels/

# Устанавливаем PyTorch и torchvision из локальных wheel
RUN pip3 install --no-cache-dir \
    /wheels/torch-2.1.1+cu118-cp38-cp38-linux_x86_64.whl \
    /wheels/torchvision-0.16.1+cu118-cp38-cp38-linux_x86_64.whl

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем остальные зависимости из requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Создаем нужные папки
RUN mkdir -p /app/models /app/temp /app/output /app/logs

COPY . .

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app

USER botuser

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

CMD ["python3", "main.py"]
