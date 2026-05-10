#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y python3 python3-venv python3-pip ca-certificates

python3 -m venv /opt/copilot-sdk-venv
/opt/copilot-sdk-venv/bin/pip install --upgrade pip
/opt/copilot-sdk-venv/bin/pip install github-copilot-sdk

cat >> ~/.bashrc <<'EOF'
export PATH="/opt/copilot-sdk-venv/bin:$PATH"
EOF