# Classificador de Dígitos Manuscritos com CNN + Edge AI
 
> Projeto desenvolvido para o processo seletivo **Intensivo Maker | AI** – Etapa Prática de Machine Learning e Edge AI.
---

## —﹒Identidade do Candidato ﹒  <img width="40" height="40" alt="light board sticker _ Sparkles Effects Sticker - Find   Share on GIPHY" src="https://github.com/user-attachments/assets/eb3221b1-c5fc-4645-9125-9413ff64b102" />              


- **Nome completo:** *Ana Aisha Tomaz de Morais*
- Curso: Engenharia de Software - UFCA (3° Semestre)                                                                                                                                                                          
- **GitHub:** [*@aishatomaz*](https://github.com/aishatomaz)
- **EMAIL:** *aisha.tomaz@aluno.ufca.edu.br*

---

## Ꮺ Resumo da Arquitetura do Modelo
 
O modelo implementado em `train_model.py` é uma **Rede Neural Convolucional (CNN)** construída com a API Sequential do Keras, projetada para classificar imagens 28×28 pixels do dataset MNIST em 10 classes (dígitos de 0 a 9).
 
### Arquitetura camada a camada
 
```
Input: (28, 28, 1) — imagem em escala de cinza normalizada
        │
        ▼
Conv2D(32 filtros, 3×3, ReLU)      ← extrai padrões simples (bordas, curvas)
MaxPooling2D(2×2)                  ← reduz dimensionalidade pela metade
        │
        ▼
Conv2D(64 filtros, 3×3, ReLU)      ← extrai padrões mais complexos
MaxPooling2D(2×2)                  ← nova redução espacial
        │
        ▼
Flatten()                          ← achata o tensor para vetor 1D
Dense(64, ReLU)                    ← camada totalmente conectada
Dropout(0.3)                       ← regularização — evita overfitting
        │
        ▼
Dense(10, Softmax)                 ← saída: probabilidade para cada dígito (0–9)
```
 
**Total de parâmetros treináveis:** ~93.000
 
**Configuração de treinamento:**
- Otimizador: `Adam`
- Função de perda: `sparse_categorical_crossentropy`
- Épocas: `5`
- Batch size: `128`
- Validação: `10%` dos dados de treino (`validation_split=0.1`)
---
 
## Ꮺ Bibliotecas Utilizadas
 
| Biblioteca        | Versão mínima | Uso no projeto                                               |
|------------------|---------------|--------------------------------------------------------------|
| `tensorflow`     | ≥ 2.12        | Construção, treinamento e conversão do modelo CNN            |
| `keras`          | (incluso no TF)| API de alto nível para definição das camadas                |
| `numpy`          | última estável | Manipulação de arrays e pré-processamento dos dados          |
| `os`             | stdlib Python | Leitura do tamanho dos arquivos gerados no relatório         |
 
As dependências estão declaradas no arquivo `requirements.txt` e são instaladas automaticamente pelo pipeline de CI:
 
```bash
pip install -r requirements.txt
```
 
---
 
## Ꮺ Técnica de Otimização do Modelo
 
O script `optimize_model.py` aplica **Dynamic Range Quantization**, a técnica de otimização padrão do TensorFlow Lite para Edge AI.
 
### O que essa técnica faz
 
Os pesos da rede, originalmente armazenados em `float32` (4 bytes por valor), são convertidos para `int8` (1 byte por valor) durante a conversão para `.tflite`. Isso é feito de forma **pós-treinamento**, sem necessidade de re-treinar o modelo.
 
### Por que foi escolhida
 
É a técnica com melhor custo-benefício para Edge AI: reduz o modelo em ~75% do tamanho com impacto mínimo na acurácia, não exige um dataset de calibração (ao contrário da Full Integer Quantization) e é nativamente suportada pelo TFLite Runtime em microcontroladores e SBCs.
 
### Pipeline de conversão
 
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # ← Dynamic Range Quantization
tflite_model = converter.convert()
```
 
### Relatório gerado pelo `optimize_model.py`
 
| Métrica                     | Valor esperado  |
|-----------------------------|-----------------|
| Tamanho original (`.h5`)    | ~350 KB         |
| Tamanho otimizado (`.tflite`)| ~85 KB         |
| Redução de tamanho          | ~75%            |
 
> O modelo `.tflite` resultante é adequado para deployment em dispositivos com memória limitada, como microcontroladores com TFLite Micro ou Raspberry Pi.
 
---
 
## Ꮺ Resultados Obtidos
 
O modelo foi treinado por **5 épocas** em CPU, respeitando as restrições do ambiente de CI.
 
| Métrica                        | Resultado        |
|-------------------------------|------------------|
| Acurácia no treino (época 5)   | ~99%             |
| Acurácia de validação (época 5)| ~99%             |
| **Acurácia final no teste**    | **~99%**         |
| Loss final no teste            | ~0.03            |
 
O pipeline do GitHub Actions executa com sucesso os dois steps de validação:
- ✦ `model.h5` gerado após o treinamento
- ✦ `model.tflite` gerado após a otimização
- ✦ Etapa de finalização com mensagem `Desafio executado com sucesso!`
---
 
## Ꮺ Comentários Adicionais
 
**Decisões técnicas importantes:**
 
A escolha de **2 blocos Conv2D + MaxPooling** (ao invés de 3) foi intencional: para o MNIST, que é um dataset relativamente simples (imagens pequenas em escala de cinza), uma arquitetura mais profunda não traria ganhos significativos de acurácia e aumentaria desnecessariamente o tempo de treinamento em CPU — conflitando com as restrições do CI.
 
O **Dropout(0.3)** foi adicionado antes da camada de saída para reduzir overfitting, já que a rede tem capacidade suficiente para memorizar os dados de treino em poucas épocas.
 
**Fluxo completo implementado:**
 
```
MNIST (Keras)
     │
     ▼
Pré-processamento (reshape + normalização /255)
     │
     ▼
Treinamento CNN (5 épocas, Adam, batch 128)
     │
     ▼
Avaliação no conjunto de teste
     │
     ▼
Salvamento → model.h5
     │
     ▼
Dynamic Range Quantization
     │
     ▼
model.tflite  ← pronto para Edge AI
```
 
જ **Aprendizados:**
 
O maior aprendizado foi entender que *Edge AI não é sobre ter o modelo mais preciso*, mas sobre o equilíbrio entre acurácia, tamanho e latência de inferência. A conversão para `.tflite` com quantização torna o modelo viável para rodar em dispositivos como ESP32-S3 com PSRAM ou Raspberry Pi, onde memória e consumo de energia são limitantes reais. Ao compreender o conteúdo abordado, o processo se torna mais simples.
 
જ **Limitações:**
 
O modelo foi treinado e avaliado apenas no MNIST, que é um conjunto controlado. Em cenários reais de dígitos manuscritos com variações de iluminação, inclinação e ruído de fundo, seria necessário aplicar técnicas de data augmentation (`ImageDataGenerator`) para melhorar a generalização.
