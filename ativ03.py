import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

# Define os documentos
documents = [ "Variáveis em Python e os tipos de dados básicos como int, float e string.", 
             "Estruturas de controle em Python: if, else, e elif para tomada de decisões.", 
             "Loops em Python, como for e while, para repetição de tarefas.", 
             "Funções em Python: como definir e chamar funções usando a palavra-chave def."]

# Define o modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
document_embeddings = embeddings.embed_documents(documents)

# Mostra o tamanho dos embeddings
embedding_size = len(document_embeddings[0])
print(f"Tamanho dos embeddings: {embedding_size}")

# Realiza uma busca de similaridade para uma consulta dada
query = "Como criar funções em Python?"
query_embedding = embeddings.embed_query(query)

# Calcula os scores de similaridade
similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Encontra o documento mais similar
most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Documento mais similar à consulta '{query}':")
print(most_similar_document)