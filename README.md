<img alt="UBI Logo" height="150" src="informatica-ubi.jpg" width="150"/>

# ChromaTruth : Deteção de Deepfakes através de Inconsistências na Reconstrução Cromática


Projeto desenvolvido por **Carolina Raposo**  
Universidade da Beira Interior — Inteligência Artificial e Ciência de Dados  
Disciplina: Redes Neuronais e Aprendizagem Profunda (2025/2026)


---
## Introdução

**ChromaTruth** é um sistema de deteção de deepfakes baseado na ideia de que imagens manipuladas apresentam **inconsistências cromáticas** difíceis de reconstruir. 

O método utiliza:
- **Stable Diffusion Inpainting** para gerar deepfakes realistas;
- Uma **UNet** treinada apenas com imagens reais;
- **Mapas de erro e XAI** para identificar regiões manipuladas;
- **ROC/AUC** para avaliar o desempenho do detetor.

As imagens reais utilizadas provêm do dataset CelebA-HQ, amplamente utilizado em investigação de visão computacional.

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
├── ChromaTruth.ipynb                 # Notebook com toda a implementação
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

### Opção 1: Abrir diretamente no Google Colab

Clique no botão abaixo para abrir o notebook no Colab:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10xpJ3Qx8e-2f3ermHExFmcoR7dVirtmr?usp=sharing)


### Opção 2: Clonar projeto
```bash
git clone https://github.com/carolinarraposo/ChromaTruth.git
cd ChromaTruth
```

#### Instalação das dependências

O projeto foi desenvolvido e testado em **Google Colab**.  
Para execução local:

```bash
pip install -r requirements.txt
```

#### Execução

Para executar o projeto, basta abrir o ficheiro ChromaTruth.ipynb e executar as células sequencialmente.

O notebook contém:

- Fase 1 — Geração de deepfakes
- Fase 2 — Treino da UNet
- Fase 3 — Deteção, XAI, ROC/AUC e geração de heatmaps

### Resultados 

Após a execução, estes são os resultados guardados automaticamente nas pastas ``models`` e ``outputs``:

- ``unet_chroma_truth.pth``   — parâmetros treinados da rede UNet
- ``preview_{idx}.png``       — exemplos de máscaras e fakes
- ``learning_curve.png``      — evolução da loss durante o treino
- ``roc_curve.png``           — curva ROC/AUC
- ``*_bg.png``                — reconstrução RGB a partir do canal L
- ``*_overlay.png``           — heatmaps das regiões manipuladas


