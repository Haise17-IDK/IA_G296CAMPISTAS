python 3.13

librería Hugging Face Transformers.
https://huggingface.co/
1. pipeline

pipeline es una función de alto nivel que simplifica el uso de modelos de NLP (Procesamiento de Lenguaje Natural) ya entrenados.

Sirve para realizar tareas comunes como análisis de sentimientos, traducción, clasificación de texto, generación de texto, etc., sin necesidad de cargar y configurar el modelo manualmente.
```
clasificador = pipeline ('sentiment-analysis')
```

| Libreria     | comando                  |
| ------------ | ------------------------ |
| transformers | pip install transformers |
| torch        | pip install torch        |
## ejemplo 1
ejemplo1.py 
```
from transformers import pipeline

#crear un pepeline de análisis de sentimiento
clasificador = pipeline ('sentiment-analysis')
clasificador2 = pipeline ('sentiment-analysis')
# análisar un sentencia
resultado= clasificador("Me encanta usar la librería trasnformers de hugging face!")
print(resultado)
resultado2=clasificador2("no me gusta trabajar los sabado")
print(resultado2)

```
## ejemplo 2 
Analisis_de_Sentimiento.ipynb

```
clasificacion_2 = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')

```
pipeline('sentiment-analysis', ...)

Igual que antes, le dices a Hugging Face que quieres un pipeline de análisis de sentimientos.

model='finiteautomata/beto-sentiment-analysis'

Aquí le indicas el modelo que debe usar.

En este caso es el modelo finiteautomata/beto-sentiment-analysis, que está basado en BETO (un BERT entrenado para español 🇪🇸).

Este modelo fue ajustado específicamente para análisis de sentimientos en texto en español.


```
ner_1 = pipeline('ner')
```
usa otra funcionalidad de Hugging Face Transformers: ahora no es análisis de sentimientos, sino NER (Named Entity Recognition) 👉 reconocimiento de entidades con nombre.
¿Qué es NER?

NER significa Reconocimiento de Entidades Nombradas.
Es una tarea de NLP donde el modelo identifica y clasifica palabras o frases importantes dentro de un texto, como:

👤 Personas

🏙️ Lugares

🏢 Organizaciones

📅 Fechas

💰 Cantidades, etc.
```
ner_2 = pipeline('ner', grouped_entities=True)
```
¿Qué cambia con grouped_entities=True?

Con este parámetro, el pipeline intenta unir las palabras que pertenecen a la misma entidad.
```
ner_3 = pipeline('ner', grouped_entities=True, model='Babelscape/wikineural-multilingual-ner')
```
model='Babelscape/wikineural-multilingual-ner'

Aquí le dices qué modelo usar: Wikineural Multilingual NER, desarrollado por Babelscape.

📌 Este modelo es multilingüe, entrenado con datos en varios idiomas (incluido español 🇪🇸, inglés, alemán, francés, italiano, portugués, etc.).
👉 Eso significa que lo puedes usar para hacer NER en distintos idiomas sin necesidad de cambiar de modelo.