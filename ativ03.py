from langchain_community.embeddings import OllamaEmbeddings

# criação de embeddings
embeddingsText = "Estoicismo é uma escola e doutrina filosófica surgida na Grécia Antiga, que preza a fidelidade ao nconhecimento e o foco em tudo aquilo que pode ser controlado pela própria pessoa. Despreza todos os tipos de sentimentos externos, como a paixão e os desejos extremos."

# modelo de embeddings do Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# geração do embeddings
texto_embeddings = embeddings.embed_query(embeddingsText)

# tamanho do embeddings
tamanho_vetor = len(texto_embeddings)
print(f"------------ Tamanho dos embeddings: {tamanho_vetor} ------------ ")

# exibição dos embeddings gerados
print("------------ Os embeddings gerados: -----------------")
for i in texto_embeddings[:10]:
    print(i)
    
print("-----------------------------------------------------")