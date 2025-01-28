import whisper
model = whisper.load_model("small")
result = model.transcribe("/Users/Magdiel/Documents/GitHub/TSI_TASI/apanhar.mp3")
print(f"\nTexto {result['text']}")

from langchain_ollama.llms import OllamaLLM
model = OllamaLLM(model="llama3.2:latest")
resposta = model.invoke("Faça um resumo do texto:  "+result['text']) 
print(f"\n\nResumo do Ollama: {resposta}")

from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

preload_models()

text_prompt = f"""
      [PORTUGUESE-BR] {resposta} 
 """
audio_array = generate_audio(text_prompt)
Audio(audio_array, rate=SAMPLE_RATE)