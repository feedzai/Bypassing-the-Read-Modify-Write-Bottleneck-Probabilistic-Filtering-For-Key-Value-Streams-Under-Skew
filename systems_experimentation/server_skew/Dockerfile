# ==============================================================================
# DEPLOYMENT & OPERATIONAL INSTRUCTIONS
# ==============================================================================
#
# 0. SERVER SETUP (Run these once on a fresh AWS Ubuntu instance):
#    # Update and install Docker
#    sudo apt-get update
#    sudo apt-get install ca-certificates curl gnupg -y
#    
#    # Add Docker's official GPG key & repository
#    sudo install -m 0755 -d /etc/apt/keyrings
#    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
#    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
#    
#    # Install Docker Engine and Compose Plugin
#    sudo apt-get update
#    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
#
#    # Grant your user permission (Requires log out/in or 'newgrp docker')
#    sudo usermod -aG docker $USER
#    newgrp docker
#
# 1. PRE-REQUISITES:
#    - AWS EC2 Security Group: Allow Inbound Port 80 (HTTP) and Port 22 (SSH).
#
# 2. FILE STRUCTURE:
#    ├── flask_server.py                     # Ensure 'delete contents' logic in cleanup_kv_stores
#    ├── temporal_aggregations.py    # Logic dependency
#    ├── requirements.txt           # flask, numpy, rocksdict, gunicorn
#    ├── Dockerfile
#    └── docker-compose.yml         # Port 80:8000 and ./data volumes
#
# 3. DEPLOYMENT COMMANDS:
#    docker compose up -d --build
#
# 4. TEST CURL COMMANDS:
#
#    # A. Check Statistics
#    curl -X GET http://localhost/api/v1/stats
#
#    # B. Test from other machine
#    curl -X POST http://public-ip-of-machine/api/v1/stats
#
# ==============================================================================
FROM python:3.11-slim as builder

# Install system dependencies required for building rocksdict/rocksdb
RUN apt-get update && apt-get install -y \
    build-essential \
    libgflags-dev \
    libsnappy-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final Run stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed python packages from the builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local/bin are usable
ENV PATH=/root/.local/bin:$PATH

# Expose the port (we'll use 8000 for Gunicorn)
EXPOSE 8000

# Use Gunicorn for production (replace 'app:app' with 'your_filename:app')
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "flask_server:app"]