# Traffic Control AI System

Sistema inteligente de controle de trafego usando SUMO com Q-Learning e algoritmos heuristicos.

## Quick Start

### Instalacao

```bash
# Instalar dependencias Python
pip install -r requirements.txt

# Instalar SUMO (Simulation of Urban Mobility)
# Ubuntu/Debian:
sudo apt install sumo sumo-tools sumo-doc

# macOS:
brew install sumo

# Windows/Outras plataformas:
# https://sumo.dlr.de/docs/Installing/
```

### Uso Basico

```bash
# Demo interativo (modo padrao)
python main.py

# Treinar agente Q-Learning (50 episodios)
python main.py --mode train --agent qlearning --episodes 50

# Treino rapido sem GUI
python main.py --mode train --agent qlearning --episodes 50 --no-gui --fast

# Testar modelo treinado
python main.py --mode test --load-model models/modelo_treinado.json
```

## Features

- Simulacao SUMO com intersecao de 4 vias realista
- Agente Q-Learning com deteccao automatica de GPU (AMD/NVIDIA/CPU)
- Agente heuristico baseline para comparacao
- Visualizacao em tempo real via SUMO GUI
- Metricas de desempenho: tempo de espera, throughput, velocidade media, emissoes

## Dataset Gerado Automaticamente

O projeto **nao usa datasets externos**. Os dados sao gerados pela simulacao SUMO durante execucao:

- Cada episodio gera centenas de linhas de dados (state, action, reward)
- Metricas coletadas: filas, tempos de espera, velocidades, fases do semaforo
- Dados salvos automaticamente em `data/` e `models/`

## Estrutura do Projeto

```
traffic_control/
├── src/
│   ├── agents/          # Q-Learning e heuristico
│   ├── environments/    # Ambiente SUMO
│   ├── utils/           # GPU acceleration, persistencia
│   └── visualization/   # Plotagem de resultados
├── sumo_config/         # Configuracoes SUMO (rede, rotas)
├── examples/            # Modelo exemplo
├── main.py              # Aplicacao principal
└── requirements.txt     # Dependencias
```

## GPU Support (Opcional)

Deteccao automatica de GPU:
- **AMD**: ROCm (se instalado)
- **NVIDIA**: CUDA (se instalado)
- **Fallback**: CPU (sempre funciona)

Para instalar PyTorch com suporte GPU:
```bash
# AMD ROCm 6.3 (recomendado para RX 9070 XT / gfx1201)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Ou use o script automatico:
./install_pytorch_rocm7.sh

# NVIDIA CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Para AMD Radeon RX 9070 XT (gfx1201):**
- ROCm 7 oficialmente suporta esta GPU
- Use PyTorch com ROCm 6.3+ ou nightly builds
- Execute: `./install_pytorch_rocm7.sh` para instalacao automatica

## Argumentos Disponiveis

```
--mode          Modo de execucao: train, test, demo (default: demo)
--agent         Tipo de agente: qlearning, heuristic (default: heuristic)
--episodes      Numero de episodios (default: 10)
--no-gui        Rodar sem GUI do SUMO (mais rapido)
--load-model    Carregar modelo pre-treinado
--fast          Modo treino rapido (episodios de 30min, steps 2s)
```

## Troubleshooting

**SUMO nao encontrado:**
```bash
# Verifique instalacao
sumo --version

# Adicione ao PATH se necessario (Linux)
export SUMO_HOME=/usr/share/sumo
```

**Erro de importacao traci:**
```bash
# Reinstale sumo-tools
pip install traci
```
# traffic_control
