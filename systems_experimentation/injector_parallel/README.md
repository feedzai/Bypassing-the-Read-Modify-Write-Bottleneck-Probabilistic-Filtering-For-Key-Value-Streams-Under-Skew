## Scope

This is the injector that must be setup to replicate the parallel experiments.

## Changes

The setup and usage is simmilar to the other injectors however, the number fo
workers must be matched to the number of workers in the server. Do this in the
following lines of the ~run_experiments.sh~:

```python
# IMPORTANT: Match this to the number of workers you launched on the server!
NUM_WORKERS=<MATCH THIS TO THE NUMBER OF WORKERS IN THE SERVER>
```

## 1. Injector Machine Setup
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

## 2. Run the load tests

To run the load tests simply use:

```
chmod +x run_experiments.sh
./run_experiments.sh
```

This runs a script to sequentially test all endpoints. This must be performed
for all the different number of workers the server is launched with.

