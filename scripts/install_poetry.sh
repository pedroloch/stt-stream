#!/bin/bash
# Instalar Poetry e configurar PATH
# Uso: bash scripts/install_poetry.sh

set -e

echo "📦 Instalando Poetry..."

# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Adicionar ao PATH
export PATH="/root/.local/bin:$PATH"

# Adicionar permanentemente ao bashrc
if ! grep -q '/.local/bin' ~/.bashrc; then
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    echo "✅ PATH adicionado ao ~/.bashrc"
fi

# Reload bashrc
source ~/.bashrc 2>/dev/null || true

# Verificar instalação
if command -v poetry &> /dev/null; then
    echo "✅ Poetry instalado com sucesso!"
    poetry --version
else
    echo "❌ Erro: Poetry não encontrado"
    echo "Execute manualmente:"
    echo "  export PATH=\"/root/.local/bin:\$PATH\""
    echo "  poetry --version"
    exit 1
fi
