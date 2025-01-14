# TSI_TASI_Ativ05
# Construção de um Sistema de Recuperação de Informações usando VectorDB e RetrievalQA
Este projeto implementa um sistema de perguntas e respostas (Q&A) utilizando a biblioteca LangChain e o serviço Ollama. O código carrega um documento PDF, divide o conteúdo em fragmentos, gera embeddings semânticos e usa o modelo de linguagem Ollama para responder perguntas baseadas no conteúdo do PDF.

## Funcionalidades
- Funcionalidade
- Carrega um arquivo PDF e divide seu conteúdo em páginas.
- Divide o texto em fragmentos menores para facilitar o processamento.
- Cria embeddings para os fragmentos de texto usando o modelo Ollama.
- Indexa os embeddings com FAISS, permitindo buscas rápidas e eficientes.
- Realiza uma busca de similaridade para encontrar os fragmentos mais relevantes para a consulta fornecida.
- Gera uma resposta com o modelo de linguagem Ollama, com base nos fragmentos recuperados.
- Exibe a resposta para a consulta.

### Pré-requisitos:
Antes de executar o código, é necessário ter instalado os seguintes requisitos:

- Python 3.x ou superior.
- Ollama: Plataforma para geração de embeddings e modelos de linguagem.
- LangChain: Biblioteca que facilita a criação de pipelines de processamento de linguagem natural (PLN).
- FAISS: Biblioteca para busca eficiente em grandes volumes de dados.

### Depedencia:
```python
pip install langchain langchain_community faiss-cpu
