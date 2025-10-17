# 🧪 Testing Guide - Whisper Stream

Guia completo para testes do projeto.

## 📋 Resumo de Testes

### ✅ Testes Unitários (31 testes)
- **test_hardware_detector.py**: 14 testes
- **test_config.py**: 17 testes

Todos passando em 0.07s

## 🚀 Como Rodar os Testes

### Setup Inicial

```bash
# 1. Instalar projeto em modo development
uv pip install -e ".[dev]"

# Ou com Poetry
poetry install --with dev

# 2. Instalar pytest se não estiver
pip install pytest pytest-asyncio pytest-cov
```

### Rodar Todos os Testes

```bash
# Com pytest (recomendado)
pytest

# Com coverage
pytest --cov=server --cov-report=html

# Verbose mode
pytest -v

# Apenas unit tests
pytest tests/unit/

# Apenas integration tests
pytest tests/integration/
```

### Rodar Testes Específicos

```bash
# Um arquivo específico
pytest tests/unit/test_config.py

# Uma classe específica
pytest tests/unit/test_config.py::TestConfig

# Um teste específico
pytest tests/unit/test_config.py::TestConfig::test_from_file
```

##📊 Resultados dos Testes

```
============================= test session starts ==============================
platform darwin -- Python 3.10.9, pytest-7.1.2
collected 31 items

tests/unit/test_config.py ..................                            [ 54%]
tests/unit/test_hardware_detector.py ..............                     [100%]

============================== 31 passed in 0.07s ===============================
```

## 🧪 TDD Workflow

Este projeto usa **Test-Driven Development (TDD)**:

### 1. Escrever Teste Primeiro

```python
# tests/unit/test_new_feature.py
def test_new_feature():
    """Test: Nova feature deve fazer X"""
    result = new_feature()
    assert result == expected_value
```

### 2. Rodar Teste (deve falhar)

```bash
pytest tests/unit/test_new_feature.py
# ❌ FAILED - new_feature not found
```

### 3. Implementar Feature

```python
# server/new_feature.py
def new_feature():
    return expected_value
```

### 4. Rodar Teste Novamente (deve passar)

```bash
pytest tests/unit/test_new_feature.py
# ✅ PASSED
```

### 5. Refatorar se Necessário

## 🔍 Testes Manuais

### 1. Teste de Instalação

```bash
# Testar que instalação funciona
./scripts/test-install.sh
```

**Esperado**:
```
✅ Todos os testes passaram!
```

### 2. Teste de Hardware Detection

```bash
# Testar detecção de hardware
python scripts/check-hardware.py
```

**Esperado em Apple Silicon**:
```
Tipo: APPLE_SILICON
Dispositivo: Apple M1 Pro
Backend recomendado: mlx
```

**Esperado em NVIDIA GPU**:
```
Tipo: CUDA
Dispositivo: NVIDIA RTX 3090
Backend recomendado: cuda
```

**Esperado em CPU**:
```
Tipo: CPU
Dispositivo: CPU
Backend recomendado: cpu
```

### 3. Teste do Servidor

```bash
# Iniciar servidor (terminal 1)
python -m server.main --config server-config.example.yaml

# Testar health endpoint (terminal 2)
curl http://localhost:9090/health
```

**Esperado**:
```json
{
  "status": "healthy",
  "processor_ready": true
}
```

### 4. Teste do WebSocket

```bash
# Instalar wscat
npm install -g wscat

# Conectar ao WebSocket
wscat -c ws://localhost:9090/ws
```

**Esperado**:
```json
{
  "type": "connected",
  "message": "Conectado ao servidor Whisper Stream",
  ...
}
```

### 5. Teste Cliente Bun

```bash
# Instalar deps (se não fez ainda)
bun install

# Rodar cliente
bun start
```

**Esperado**:
```
🔍 Verificando servidor...
✅ Servidor está online
🔌 Conectando ao WebSocket...
✅ WebSocket conectado
```

## 📝 Escrevendo Novos Testes

### Exemplo de Teste Unitário

```python
"""tests/unit/test_my_module.py"""

import pytest
from server.my_module import MyClass

class TestMyClass:
    """Testes para MyClass"""

    def test_initialization(self):
        """Test: MyClass deve inicializar corretamente"""
        obj = MyClass()
        assert obj is not None

    def test_method_returns_correct_value(self):
        """Test: method() deve retornar valor correto"""
        obj = MyClass()
        result = obj.method()
        assert result == "expected"

    def test_method_with_parameter(self):
        """Test: method(param) deve aceitar parâmetro"""
        obj = MyClass()
        result = obj.method("input")
        assert result == "processed_input"

    def test_method_raises_error(self):
        """Test: method() deve lançar erro em caso inválido"""
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Exemplo de Teste com Mocks

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    """Test: Função deve usar dependência externa"""

    mock_dep = MagicMock()
    mock_dep.get_data.return_value = "mocked_data"

    with patch('module.dependency', mock_dep):
        result = function_that_uses_dependency()

        assert result == "expected_result"
        mock_dep.get_data.assert_called_once()
```

### Exemplo de Teste Async

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test: Função assíncrona deve funcionar"""
    result = await async_function()
    assert result is not None
```

## 🐛 Debugging Testes

### Ver Output Completo

```bash
# Mostrar prints
pytest -s

# Mostrar traceback completo
pytest --tb=long

# Parar no primeiro erro
pytest -x

# Modo verbose + traceback
pytest -vvs --tb=short
```

### Rodar Teste Específico em Debug

```python
# Adicionar breakpoint no teste
def test_something():
    import pdb; pdb.set_trace()  # Breakpoint aqui
    result = function()
    assert result == expected
```

```bash
pytest tests/unit/test_something.py -s
```

## 📊 Coverage

### Gerar Relatório de Coverage

```bash
# Coverage HTML
pytest --cov=server --cov-report=html

# Abrir relatório
open htmlcov/index.html

# Coverage no terminal
pytest --cov=server --cov-report=term-missing

# Coverage com branches
pytest --cov=server --cov-branch
```

**Meta**: >= 80% coverage

## ✅ Checklist Antes de Commit

```bash
# 1. Rodar todos os testes
pytest

# 2. Verificar coverage
pytest --cov=server

# 3. Linting (se configurado)
black server/
ruff check server/

# 4. Teste manual rápido
./scripts/test-install.sh
```

## 🔧 CI/CD (Futuro)

Configurar GitHub Actions para rodar testes automaticamente:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=server
```

## 📚 Recursos

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Última atualização**: 2025-01-17

**Status**: 31/31 testes passing ✅
