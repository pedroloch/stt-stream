#!/bin/bash
set -e

echo "🚀 Instalando Whisper Stream"
echo "=============================="
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detectar OS e arquitetura
OS="$(uname -s)"
ARCH="$(uname -m)"

echo -e "${BLUE}📊 Sistema:${NC} $OS $ARCH"
echo ""

# 1. Verificar Python
echo -e "${BLUE}🐍 Verificando Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 não encontrado${NC}"
    echo "Instale Python 3.11+ primeiro"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PYTHON_VERSION encontrado${NC}"

# Verificar versão mínima (3.11)
REQUIRED_MAJOR=3
REQUIRED_MINOR=11
CURRENT_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
CURRENT_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$CURRENT_MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$CURRENT_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$CURRENT_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo -e "${YELLOW}⚠️  Python $PYTHON_VERSION < 3.11 (recomendado)${NC}"
    echo "Considere atualizar para Python 3.11+"
fi

echo ""

# 2. Instalar uv se não existir
echo -e "${BLUE}📦 Verificando uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Adicionar ao PATH para esta sessão
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ Falha ao instalar uv${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✅ uv instalado${NC}"
echo ""

# 3. Detectar hardware e instalar dependências apropriadas
echo -e "${BLUE}🔍 Detectando hardware...${NC}"

# Executar script de detecção se já instalado, senão fazer detecção simples
if [ -f "server/utils/hardware_detector.py" ]; then
    # Tenta usar o detector do projeto
    BACKEND=""

    if [[ "$ARCH" == "arm64" ]] && [[ "$OS" == "Darwin" ]]; then
        echo -e "${GREEN}🍎 Apple Silicon detectado${NC}"
        BACKEND="mlx"
    elif command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}🎮 NVIDIA GPU detectada${NC}"
        BACKEND="cuda"
    else
        echo -e "${YELLOW}💻 CPU mode${NC}"
        BACKEND="cpu"
    fi

    echo ""

    # 4. Instalar dependências Python
    echo -e "${BLUE}📦 Instalando dependências Python...${NC}"

    if [ "$BACKEND" == "mlx" ]; then
        echo "Instalando com suporte MLX (Apple Silicon)..."
        uv pip install -e ".[mlx]"
    elif [ "$BACKEND" == "cuda" ]; then
        echo "Instalando com suporte CUDA (NVIDIA)..."
        uv pip install -e ".[cuda]"
    else
        echo "Instalando modo CPU..."
        uv pip install -e .
    fi

    echo -e "${GREEN}✅ Dependências Python instaladas${NC}"
    echo ""
fi

# 5. Instalar sox (para cliente de áudio)
echo -e "${BLUE}🎙️  Verificando sox (captura de áudio)...${NC}"
if ! command -v sox &> /dev/null; then
    echo "sox não encontrado. Instalando..."

    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            brew install sox
        else
            echo -e "${YELLOW}⚠️  Homebrew não encontrado${NC}"
            echo "Instale sox manualmente: https://sourceforge.net/projects/sox/"
        fi
    elif [[ "$OS" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y sox
        elif command -v yum &> /dev/null; then
            sudo yum install -y sox
        else
            echo -e "${YELLOW}⚠️  Gerenciador de pacotes não suportado${NC}"
            echo "Instale sox manualmente"
        fi
    fi
fi

if command -v sox &> /dev/null; then
    echo -e "${GREEN}✅ sox instalado${NC}"
else
    echo -e "${YELLOW}⚠️  sox não foi instalado (necessário para cliente de áudio)${NC}"
fi
echo ""

# 6. Verificar Bun
echo -e "${BLUE}🥟 Verificando Bun (para cliente)...${NC}"
if ! command -v bun &> /dev/null; then
    echo "Bun não encontrado. Deseja instalar? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        curl -fsSL https://bun.sh/install | bash

        # Adicionar ao PATH
        export BUN_INSTALL="$HOME/.bun"
        export PATH="$BUN_INSTALL/bin:$PATH"
    else
        echo -e "${YELLOW}⚠️  Bun não instalado (necessário para cliente de exemplo)${NC}"
    fi
fi

if command -v bun &> /dev/null; then
    echo -e "${GREEN}✅ Bun instalado${NC}"

    # Instalar dependências do cliente
    echo "Instalando dependências do cliente..."
    bun install
    echo -e "${GREEN}✅ Dependências do cliente instaladas${NC}"
else
    echo -e "${YELLOW}⚠️  Cliente Bun não será instalado${NC}"
fi
echo ""

# 7. Criar arquivos de config se não existirem
echo -e "${BLUE}⚙️  Configuração...${NC}"

if [ ! -f "server-config.yaml" ]; then
    cp server-config.example.yaml server-config.yaml
    echo -e "${GREEN}✅ server-config.yaml criado${NC}"
fi

if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo -e "${GREEN}✅ config.yaml criado${NC}"
fi

echo ""

# 8. Testar instalação
echo -e "${BLUE}🧪 Testando instalação...${NC}"
python3 scripts/check-hardware.py

echo ""
echo "=============================="
echo -e "${GREEN}✅ Instalação completa!${NC}"
echo "=============================="
echo ""
echo "Para iniciar:"
echo ""
echo -e "${YELLOW}Servidor:${NC}"
echo "  uv run python -m server.main --config server-config.yaml"
echo ""
echo -e "${YELLOW}Cliente (em outro terminal):${NC}"
echo "  bun start"
echo ""
echo -e "${BLUE}Próximos passos:${NC}"
echo "  - Edite server-config.yaml para configurar modelo/idioma"
echo "  - Edite config.yaml para configurar URL do servidor"
echo "  - Veja docs/ para mais informações"
echo ""
