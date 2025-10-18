#!/usr/bin/env python3
"""
Verifica dependências instaladas e compatibilidade de plataforma

Uso:
    python scripts/check_deps.py
    poetry run python scripts/check_deps.py
"""

import sys
from typing import Dict


def check_dependencies() -> Dict[str, str]:
    """
    Verifica todas as dependências do projeto

    Returns:
        Dicionário com status de cada dependência
    """
    deps = {}

    # ============================================
    # Base Dependencies
    # ============================================
    try:
        import aiohttp
        deps['aiohttp'] = f'✅ v{aiohttp.__version__}'
    except ImportError:
        deps['aiohttp'] = '❌ NÃO INSTALADO'

    try:
        import numpy
        deps['numpy'] = f'✅ v{numpy.__version__}'
    except ImportError:
        deps['numpy'] = '❌ NÃO INSTALADO'

    try:
        import yaml
        deps['pyyaml'] = '✅'
    except ImportError:
        deps['pyyaml'] = '❌ NÃO INSTALADO'

    # ============================================
    # Whisper Backends
    # ============================================
    try:
        import faster_whisper
        deps['faster-whisper'] = f'✅ v{faster_whisper.__version__}'
    except ImportError:
        deps['faster-whisper'] = '❌ NÃO INSTALADO'

    try:
        import whisperx
        deps['whisperx'] = '✅ (git)'
    except ImportError:
        deps['whisperx'] = '⚠️  opcional (Linux CUDA apenas)'

    try:
        import mlx_whisper
        deps['mlx-whisper'] = '✅'
    except ImportError:
        deps['mlx-whisper'] = '⚠️  opcional (Mac Apple Silicon apenas)'

    # ============================================
    # CUDA Dependencies
    # ============================================
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = f"CUDA {torch.version.cuda}" if cuda_available else "CPU only"
        deps['torch'] = f'✅ v{torch.__version__} ({cuda_info})'
    except ImportError:
        deps['torch'] = '⚠️  opcional (GPU/VAD)'

    try:
        import torchaudio
        deps['torchaudio'] = f'✅ v{torchaudio.__version__}'
    except ImportError:
        deps['torchaudio'] = '⚠️  opcional (VAD/diarization)'

    try:
        import nvidia.cudnn
        import os
        cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
        deps['nvidia-cudnn-cu12'] = f'✅ @ {cudnn_path}'
    except ImportError:
        deps['nvidia-cudnn-cu12'] = '⚠️  opcional (Linux CUDA para faster-whisper)'

    # ============================================
    # Diarization Backends
    # ============================================
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
        import nemo
        deps['nemo-toolkit'] = f'✅ v{nemo.__version__} (Sortformer)'
    except ImportError:
        deps['nemo-toolkit'] = '⚠️  opcional (Sortformer diarization)'

    try:
        import pyannote.audio
        deps['pyannote-audio'] = '✅ (Pyannote diarization)'
    except ImportError:
        deps['pyannote-audio'] = '⚠️  opcional (Pyannote diarization)'

    return deps


def print_report(deps: Dict[str, str]) -> None:
    """Print formatted dependency report"""
    print("=" * 70)
    print("📦 WHISPER STREAM - DEPENDENCY CHECK")
    print("=" * 70)
    print()

    # Categorizar dependências
    required = ['aiohttp', 'numpy', 'pyyaml', 'faster-whisper']
    backends = ['whisperx', 'mlx-whisper']
    cuda = ['torch', 'torchaudio', 'nvidia-cudnn-cu12']
    diarization = ['nemo-toolkit', 'pyannote-audio']

    def print_section(title: str, dep_names: list):
        print(f"\n{title}:")
        print("-" * 70)
        for name in dep_names:
            if name in deps:
                status = deps[name]
                print(f"  {name:30s} {status}")

    print_section("⭐ REQUIRED (Base)", required)
    print_section("🔧 BACKENDS (Optional)", backends)
    print_section("🎮 CUDA/GPU (Optional)", cuda)
    print_section("👥 DIARIZATION (Optional)", diarization)

    print()
    print("=" * 70)

    # Check if required deps are installed
    missing_required = [dep for dep in required if '❌' in deps.get(dep, '❌')]

    if missing_required:
        print(f"\n❌ ERRO: Dependências obrigatórias faltando: {', '.join(missing_required)}")
        print("   Execute: poetry install")
        sys.exit(1)
    else:
        print("\n✅ Todas dependências obrigatórias instaladas!")

        # Check optional
        optional_installed = []
        if '✅' in deps.get('torch', ''):
            optional_installed.append('CUDA/GPU')
        if '✅' in deps.get('whisperx', ''):
            optional_installed.append('WhisperX')
        if '✅' in deps.get('nemo-toolkit', ''):
            optional_installed.append('Sortformer')
        if '✅' in deps.get('pyannote-audio', ''):
            optional_installed.append('Pyannote')

        if optional_installed:
            print(f"⭐ Extras instalados: {', '.join(optional_installed)}")

    print()


def main():
    """Main entry point"""
    deps = check_dependencies()
    print_report(deps)


if __name__ == "__main__":
    main()
