# TSI_TASI_Ativ03
# Geração de Embeddings com OllamaEmbeddings

Este repositório contém um exemplo de como gerar embeddings de texto utilizando o modelo `Ollama` através da biblioteca `langchain_community`. O código converte um texto em um vetor numérico (embedding) que representa semanticamente o conteúdo do texto. Esses vetores podem ser utilizados para medir similaridade semântica entre textos.

## Visão Geral

Embeddings são representações numéricas de palavras ou textos, onde a distância entre os vetores reflete a semelhança semântica entre os textos. O modelo `OllamaEmbeddings`, fornecido pela biblioteca `langchain_community`, é utilizado para gerar esses vetores a partir de um texto.

### Objetivo do Código:
- **Converter um texto em um vetor de embeddings**.
- **Exibir o tamanho do vetor de embeddings gerado**.
- **Mostrar uma amostra dos primeiros valores do vetor gerado**.

### 1. Importação da Biblioteca

```python
from langchain_community.embeddings import OllamaEmbeddings
