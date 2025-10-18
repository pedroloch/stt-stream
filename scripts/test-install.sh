#!/bin/bash
# Script para testar se a instalação funcionou

echo "🧪 Testando instalação do Whisper Stream"
echo "========================================"
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ERRORS=0

# Test 1: Python imports
echo "1. Testando imports Python..."
python3 -c "from server.utils.hardware_detector import HardwareDetector; print('  ✅ hardware_detector')" || ((ERRORS++))
python3 -c "from server.config import Config; print('  ✅ config')" || ((ERRORS++))
python3 -c "from server.backends.factory import BackendFactory; print('  ✅ factory')" || ((ERRORS++))
python3 -c "from server.whisper_processor import WhisperProcessor; print('  ✅ whisper_processor')" || ((ERRORS++))
python3 -c "from server.websocket_handler import WebSocketHandler; print('  ✅ websocket_handler')" || ((ERRORS++))
python3 -c "from server.main import WhisperServer; print('  ✅ main')" || ((ERRORS++))

echo ""

# Test 2: Run hardware detection
echo "2. Testando detecção de hardware..."
python3 scripts/check-hardware.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✅ Hardware detection OK${NC}"
else
    echo -e "  ${RED}❌ Hardware detection FAILED${NC}"
    ((ERRORS++))
fi

echo ""

# Test 3: Check config files
echo "3. Verificando arquivos de configuração..."
if [ -f "server-config.example.yaml" ]; then
    echo "  ✅ server-config.example.yaml"
else
    echo "  ❌ server-config.example.yaml NOT FOUND"
    ((ERRORS++))
fi

if [ -f "config.example.yaml" ]; then
    echo "  ✅ config.example.yaml"
else
    echo "  ❌ config.example.yaml NOT FOUND"
    ((ERRORS++))
fi

echo ""

# Summary
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Todos os testes passaram!${NC}"
    echo ""
    echo "Você pode iniciar o servidor com:"
    echo "  uv run python -m server.main --config server-config.yaml"
    exit 0
else
    echo -e "${RED}❌ $ERRORS teste(s) falharam${NC}"
    echo ""
    echo "Verifique a instalação:"
    echo "  ./scripts/install.sh"
    exit 1
fi
