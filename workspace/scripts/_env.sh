export UV_CACHE_DIR=~/.cache/uv XDG_CONFIG_HOME=~/uvconfig PATH=~/.local/bin:$PATH \
       HOME=/home/ubuntu PYTHONPATH=/home/ubuntu/simple-inference-autoresearch \
       HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p ~/uvconfig
cd /home/ubuntu/simple-inference-autoresearch
