# Projeto de Transcrição e Resumo Automático de Áudio

Este projeto utiliza uma combinação de bibliotecas para transcrever áudios, resumir o texto transcrito, e gerar uma saída de áudio com o resumo. O processo envolve três etapas principais:

1. **Transcrição de Áudio**: O áudio é transcrito para texto usando o modelo `Whisper` da OpenAI.
2. **Resumo do Texto**: O texto transcrito é resumido usando o modelo `OllamaLLM` com o modelo `Llama3.2`.
3. **Geração de Áudio**: O resumo gerado é convertido de volta para áudio utilizando o modelo `Bark`.

## Requisitos

- Python 3.x
- Bibliotecas Python:
  - whisper
  - langchain_ollama
  - bark
  - IPython

## Instalação

Para instalar as dependências necessárias, siga os passos abaixo:

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio
