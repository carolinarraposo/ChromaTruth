<img alt="UBI Logo" height="150" src="informatica-ubi.jpg" width="150"/>

# ChromaTruth : Deteção de Deepfakes através de Inconsistências na Reconstrução Cromática

Projeto Final de Investigação realizado no âmbito da disciplina de **Redes Neuronais e Aprendizagem Profunda 2025/2026**  

---
## Introdução

**ChromaTruth** é um sistema de deteção de deepfakes baseado na ideia de que imagens manipuladas apresentam **inconsistências cromáticas** difíceis de reconstruir. 

O método utiliza:
- **Stable Diffusion Inpainting** para gerar deepfakes realistas;
- Uma **UNet** treinada apenas com imagens reais;
- **Mapas de erro e XAI** para identificar regiões manipuladas;
- **ROC/AUC** para avaliar o desempenho do detetor.


--- 
##  Estrutura do Projeto

````
ChromaTruth/
├── README.md
│
├── data/
│   ├── real/                         # Imagens reais (rostos)
│   └── fake/                         # Deepfakes gerados (Fase 1)
│                  
├── src/
│   ├── deepfakes.py                  # Geração de deepfakes (adversário)
│   ├── modelo.py                     # Criação e treino do modelo (UNet)
│   ├── xai.py                        # Deteção e Explicabilidade (XAI)
│   └── main.py
│
├── models/
│       ├── unet_chroma_truth.pth     # pesos treinados da Unet
│
├── outputs/                          # resultados de cada fase (imagens)
│   ├── fase1_preview/
│   ├── fase2_treino/
│   └── fase3_xai/
│
└── requirements.txt
````

## Como utilizar

### Clonar projeto
```bash
git clone https://github.com/carolinarraposo/ChromaTruth.git
cd ChromaTruth
```

### Instalação das dependências
O projeto foi testado com Python 3.10+

```bash
pip install -r requirements.txt
```

### Execução

O fluxo completo do projeto é controlado pelo ficheiro main.py.
Cada fase pode ser ativada ou desativada diretamente no código.

Para correr o projeto:

```bash
python src/main.py
```
O script executa apenas as fases que estiverem marcadas como True:

```python
RUN_DEEPFAKES = False       # Fase 1 — Geração de deepfakes
PREVIEW_DEEPFAKES = False   # Pré-visualização das máscaras e fakes
RUN_TRAINING = False        # Fase 2 — Treino da UNet
RUN_XAI = False             # Fase 3 — Deteção + XAI + ROC
```

### Resultados 

Após a execução, estes são os resultados guardados automaticamente nas pastas ``models`` e ``outputs``:

- ``unet_chroma_truth.pth``    — parâmetros treinados da rede UNet
- ``preview_{idx}.png``        — exemplos de máscaras e fakes
- ``learning_curve.png``       — evolução da loss durante o treino
- ``roc_curve.png``            — curva ROC/AUC
- ``heatmaps e overlays``      — explicabilidade das regiões manipuladas


