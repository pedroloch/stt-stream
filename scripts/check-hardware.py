#!/usr/bin/env python3
"""
Script para testar detecção de hardware

Uso:
    python scripts/check-hardware.py
"""

import sys
import logging

# Setup logging simples
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("="*60)
    print("🔍 WHISPER STREAM - HARDWARE DETECTION TEST")
    print("="*60)
    print()

    try:
        # Importar módulos do servidor
        from server.utils.hardware_detector import HardwareDetector

        logger.info("✅ Módulos do servidor importados com sucesso")

        # Criar detector
        detector = HardwareDetector(logger)

        # Detectar hardware
        logger.info("Detectando hardware...")
        hardware = detector.detect()

        # Mostrar info
        detector.print_hardware_info(hardware)

        # Recomendação
        print("\n" + "="*60)
        print("💡 RECOMENDAÇÃO")
        print("="*60)

        if hardware.type == "apple_silicon":
            print("✅ Apple Silicon detectado!")
            print("   Instale com: uv pip install -e '.[mlx]'")
            print("   Backend recomendado: mlx")
        elif hardware.type == "cuda":
            print("✅ NVIDIA GPU detectada!")
            print("   Instale com: uv pip install -e '.[cuda]'")
            print("   Backend recomendado: cuda")
        else:
            print("ℹ️  CPU detectado (sem GPU)")
            print("   Instale com: uv pip install -e .")
            print("   Backend recomendado: cpu")
            print("   Nota: Será mais lento que GPU")

        print("="*60)
        print()

        return 0

    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulos: {e}")
        logger.error("\n💡 Certifique-se de ter instalado o projeto:")
        logger.error("   uv pip install -e .")
        return 1

    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
