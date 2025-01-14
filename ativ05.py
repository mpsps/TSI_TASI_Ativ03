from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Carregar o PDF
pdf_path = 'info.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Dividir o texto em chunks (fragmentos menores)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,    # Tamanho dos chunks
    chunk_overlap=50,   # Sobreposição entre os chunks
    length_function=len
)
texts = text_splitter.split_documents(pages)

# Criar embeddings com Ollama
db = FAISS.from_documents(texts, OllamaEmbeddings(model="mxbai-embed-large"))

# Realizar a busca de similaridade
query = "O que é hardware?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# Carregar o modelo de linguagem para responder a pergunta
model = OllamaLLM(model="llama3.2-vision:latest")

# Configurar o recuperador e a cadeia de Q&A
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

# Invocar a resposta
response = qa_chain.invoke(query)
print("Resposta da Q&A:", response)
