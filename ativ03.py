from langchain_community.embeddings import OllamaEmbeddings

# texto que será usado para criação de embeddings
texto = "A UML, Linguagem Unificada de Modelagem, é uma linguagem gráfica para visualização, especificação, construção e documentação de artefatos de sistemas complexos de software."

# escolhendo o modelo de embeddings (que ira gera um associado a cada palavra)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# essa parte do codigo ira gera os embeddings de cada palavras, que definirar o contexto atraves de um numero
texto_embeddings = embeddings.embed_query(texto)

# Esta parte serve para vê o tamanho
tamanho_vetor = len(texto_embeddings)
print(f"Tamanho dos embeddings: {tamanho_vetor}")

# por ultimo ira exibi os 5 primeiros embeddings que foram gerados por veio do texto
print("\n 5 primeiros embeddings são:")
for i in texto_embeddings[:5]:
    print(i)