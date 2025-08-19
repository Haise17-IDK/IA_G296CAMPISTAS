# Importar las librerias
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import gradio as gr
#es necesario si el modelo requiere acceso
login()
model_id="meta-llama/Llama-3.2-1B-Instruct"
#cargar el modelo preentrenado
model=AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 #formato más eficiente para acelerar inferencia en GPU
)
# cargamos el tokenizer correspondiente del modelo
tokenizer= AutoTokenizer.from_pretrained(model_id)
#Detectamos si tenemos GPU disponible, de lo contrario usamos CPU
if torch.cuda.is_avaible():
    device =torch.device("cuda")
    print(f"Usando GPU: {torch.cuda.get_device_name(device)}")
else:
    device=torch.device("cpu")
    print("Usando CPU")
#Movemos el modelo al dispositivo detectado (GPU oCPU)
model =model.to(device)