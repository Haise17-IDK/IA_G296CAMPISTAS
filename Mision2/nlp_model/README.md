python 3.13

librería Hugging Face Transformers.
https://huggingface.co/

```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

```
🔎 1. AutoModelForQuestionAnswering

Es una clase de la familia AutoModel, que carga automáticamente un modelo preentrenado para la tarea de Question Answering.

Este tipo de modelo está entrenado para:
👉 Recibir un contexto (un párrafo de texto).
👉 Recibir una pregunta sobre ese contexto.
👉 Devolver la respuesta exacta dentro del texto.

Ejemplo de modelos que funcionan con esto:

distilbert-base-cased-distilled-squad

bert-large-uncased-whole-word-masking-finetuned-squad

deepset/roberta-base-squad2

🔎 2. AutoTokenizer

Es el encargado de convertir texto en tokens (números que el modelo entiende).

También sabe cómo volver a convertir los resultados del modelo en palabras.

Usa el tokenizer correcto según el modelo que cargues (BERT, RoBERTa, DistilBERT, etc.).


```
model_id = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
```
Este modelo sirve para Pregunta y Respuesta en español (Question Answering).
Si le das un contexto y una pregunta en español, te devuelve la respuesta dentro del texto.