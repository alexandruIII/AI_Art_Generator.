import gradio as gr
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2

# Încarcă modelul Candy
sess = ort.InferenceSession("candy-9.onnx")
input_name = sess.get_inputs()[0].name

def aplica_stil(imagine):
    img = imagine.convert("RGB")
    img = np.array(img)

    # Redimensionează imaginea la ce suportă modelul (verifică cu .shape)
    img_resized = cv2.resize(img, (224, 224))  # exemplu: 224x224

    # Normalizează dacă e cazul (comentăm pentru test)
    img_np = img_resized.astype(np.float32).transpose(2, 0, 1)
    img_np = img_np[np.newaxis, :]
    # img_np /= 255.0  # comentăm, dacă modelul lucrează cu valori 0–255

    # Rulează modelul
    output = sess.run(None, {input_name: img_np})[0]
    output = output.squeeze().transpose(1, 2, 0)
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Convertim din BGR în RGB dacă e cazul
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(output)

    return output_img

# Interfață Gradio
gr.Interface(
    fn=aplica_stil,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🎨 Aplicație AI - Stil Artistic Candy (ONNX)",
    description="Încarcă o imagine și aplică stilul pictural Candy.",
).launch()
