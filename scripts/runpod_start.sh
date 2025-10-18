#!/bin/bash
# RunPod Start Script - Inicia servidor Whisper Stream
#
# Uso: bash scripts/runpod_start.sh

cd /workspace/stt-stream

echo "🚀 Iniciando Whisper Stream..."
poetry run python -m server.main --config server-config.yaml
