# ========================================
# Setup on the remote machine
# sudo apt-get update
# sudo apt-get install ca-certificates curl gnupg -y
# 
# # Add GPG key and Repository
# sudo install -m 0755 -d /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# 
# # Install
# sudo apt-get update
# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
# 
# # Permissions (Fixes the "Permission Denied" error later)
# sudo usermod -aG docker $USER
# newgrp docker
# ========================================
FROM python:3.10-slim

# Install system dependencies for Pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the locust file and the dataset
COPY locust.py .
# Ensure the dataset is in the same folder when building
COPY data.parquet ./data.parquet

# Entrypoint will be overridden by the shell script, 
# but we provide the base command here.
ENTRYPOINT ["locust"]
