# Systems Experimentation: Data Server & Aggregation Module

## 1. Overview
To ensure the reproducibility of the benchmarks reported in the paper, this
server is containerized to isolate the execution environment. This ensures
consistent library versions for `rocksdict` and `numpy`, and prevents host-level
dependency conflicts.

## 2. Host Machine Setup
These instructions are optimized for a fresh **AWS Ubuntu 22.04+** instance. Run these once to prepare the environment.

### Phase A: Docker Engine Installation
```bash
# Update and install base dependencies
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg -y

# Add Docker’s official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \$(. /etc/os-release && echo \"\$VERSION_CODENAME\") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

# Install Docker Engine and Compose Plugin

```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
```

# Grant user permissions (Required to run docker without sudo)

```
sudo usermod -aG docker $USER
newgrp docker
```

# Deploy

To deploy the server use:

```
mkdir -p data && chmod 777 data
docker compose up -d --build
docker compose logs -f # For testing
```

From another machine try:

```
curl -X POST http://10.0.2.7/api/v1/reset \
     -H "Content-Type: application/json" \
     -d '{"threshold_minutes": 5, "fixed_filtering_rate": 1.0, "batched_n_updates": 1}'
```
