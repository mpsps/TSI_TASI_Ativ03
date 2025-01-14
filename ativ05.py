from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Carregando o PDF
pdf_path = 'info.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Dividindo o texto em partes menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,    # Tamanho dos chunks
    chunk_overlap=50,   # Sobreposição entre os chunks
    length_function=len
)
texts = text_splitter.split_documents(pages)

# Verificando o primeiro texto
# print(texts[0])

# Criando o banco de dados FAISS com embeddings do Ollama
db = FAISS.from_documents(texts, OllamaEmbeddings(model="mxbai-embed-large"))

# Realizando a busca por similaridade
query = "O que é hardware?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# Buscando os 5 documentos mais relevantes
# docs = db.similarity_search(query, k=5)

# Imprimindo os 5 documentos mais relevantes
#for i, doc in enumerate(docs):
#   print(f"Chunk {i + 1}: {doc.page_content}")

# Criando o modelo LLM do Ollama
model = OllamaLLM(model="llama3.2-vision:latest")

# Recuperando o retriever com 5 documentos mais relevantes
# retriever = db.as_retriever(search_kwargs={"k": 5})
retriever = db.as_retriever()

# Criando a cadeia de QA
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

# Exemplo de uso da cadeia de QA
response = qa_chain.invoke(query)
print("QA Response:", response)
