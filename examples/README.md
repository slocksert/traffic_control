# Examples

## Pre-trained Model

`example_model.json` - Modelo base para demonstracao

## Como Usar

```bash
# Carregar modelo exemplo
python main.py --mode demo --load-model examples/example_model.json

# Treinar seu proprio modelo
python main.py --mode train --agent qlearning --episodes 50

# Modelos treinados serao salvos em models/
```
