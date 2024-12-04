import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import requests
from io import BytesIO
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from ultralytics import YOLO
import base64


app = Flask(__name__)
imagenes_dir = 'imagenes-gemini-temp'
# Cargar el modelo de TensorFlow/Keras
# Cargar el modelo YOLOv8 entrenado
yolo_model = YOLO('modelo\yolov8n.pt')  
# Cargar el modelo de clasificación cuantizado (.tflite)
classification_model = tf.keras.models.load_model('modelo/modelo_final.h5')
PROCESSED_IMAGES_DIR = 'processed_images'

# Directorio público para guardar imágenes procesadas
app.config['PROCESSED_IMAGES_DIR'] = 'static/processed_images'
os.makedirs(app.config['PROCESSED_IMAGES_DIR'], exist_ok=True)

def preprocess_with_yolo(image):
    # Simula la función YOLOv8 (reemplazar con la implementación real)
    return image.convert('L')  # Convertir a escala de grises como ejemplo


genai.configure(api_key="AIzaSyBU70jTJ_Nb7LTcEohDa_OhSd6A0gvABH8")
generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 32,
            "max_output_tokens": 1024,
           # "response_mime_type": "text/plain",
        }
model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config       
        )

#### regions solo yolov8#####
def preprocess_with_yolo(image):
    """
    Procesa una imagen utilizando YOLO para detectar objetos y devuelve la
    región de interés más relevante redimensionada a 224x224 en escala de grises.
    """
    img_array = np.array(image.convert('RGB'))
    results = yolo_model.predict(img_array, imgsz=224)
    detections = results[0].boxes.data

    if len(detections) == 0:
        raise ValueError("No se detectaron objetos en la imagen con YOLO.")

    # Seleccionar el primer objeto detectado (puedes personalizar esto si necesitas más lógica)
    x1, y1, x2, y2, confidence, class_id = map(int, detections[0].tolist())

    # Validar coordenadas y recortar ROI (región de interés)
    roi = safe_crop(image, x1, y1, x2, y2)

    # Redimensionar la ROI a 224x224
    roi_resized = roi.resize((224, 224))

    # Convertir a escala de grises
    roi_gray = roi_resized.convert('L')  # 'L' es el modo de escala de grises en PIL

    return roi_gray

@app.route('/yolov8', methods=['POST'])
def yolov8_preprocess():
    """
    Procesa una imagen con YOLO y devuelve la URL de la imagen procesada.
    """
    image_url = request.form.get('image_url', '')
    if not image_url:
        return jsonify({'status': 'Error', 'message': 'No image URL provided'}), 400

    try:
        # Descargar y cargar la imagen
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Preprocesar la imagen con YOLO
        roi_gray = preprocess_with_yolo(image)

        # Convertir a RGB si es necesario
        if roi_gray.mode != "RGB":
            roi_rgb = roi_gray.convert("RGB")
        else:
            roi_rgb = roi_gray

        # Guardar la imagen procesada
        processed_image_name = 'processed_image.jpg'
        processed_image_path = os.path.join(app.config['PROCESSED_IMAGES_DIR'], processed_image_name)
        roi_rgb.save(processed_image_path)

        # Devolver la URL pública de la imagen
        image_url = f'/static/processed_images/{processed_image_name}'
        return jsonify({
            'status': 'Success',
            'message': 'Imagen procesada exitosamente.',
            'processed_image_url': image_url
        }), 200

    except ValueError as ve:
        return jsonify({'status': 'Error', 'message': str(ve)}), 400
    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 500

# Ruta para servir imágenes estáticas (Flask lo maneja automáticamente desde 'static/')
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

#### regions solo yolov8#####

#### regions solo Modelo#####

@app.route('/classify', methods=['POST'])
def classify_image():
    """
    Toma la imagen procesada y la pasa al modelo para obtener la clasificación.
    """
    try:
        # Ruta de la imagen procesada
        processed_image_path = os.path.join(app.config['PROCESSED_IMAGES_DIR'], 'processed_image.jpg')

        if not os.path.exists(processed_image_path):
            return jsonify({'status': 'Error', 'message': 'No processed image found'}), 404

        # Cargar y preprocesar la imagen para el modelo
        image = Image.open(processed_image_path).resize((224, 224))  # Ajusta el tamaño si tu modelo lo requiere
        image_array = tf.keras.utils.img_to_array(image) / 255.0  # Normalización
        image_array = tf.expand_dims(image_array, axis=0)  # Agregar batch dimension

        # Obtener la predicción del modelo
        predictions = classification_model.predict(image_array)
        predicted_class = predictions.argmax(axis=-1)[0]  # Obtener la clase con mayor probabilidad

        return jsonify({
            'status': 'Success',
            'message': 'Clasificación realizada exitosamente.',
            'predicted_class': int(predicted_class),  # Devolver como entero
            'predictions': predictions.tolist()  # Devolver las probabilidades completas
        }), 200

    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 500

#### regions solo Modelo#####

#### regions solo GEMINI#####



#### regions solo GEMINI#####



def process_image(image):
    # Convertir la imagen a RGB
    image = image.convert('RGB')
    # Redimensionar al tamaño esperado
    image = image.resize((224, 224))
    # Convertir a un array NumPy
    img_array = np.array(image, dtype=np.float32)
    # Normalizar los valores entre 0 y 1
    img_array = img_array / 255.0
    # Expandir las dimensiones para agregar batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def safe_crop(image, x1, y1, x2, y2):
    width, height = image.size  # Obtener dimensiones de la imagen
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return image.crop((x1, y1, x2, y2))

# Modificar la función detect_accident para usar el modelo .h5
def detect_accident(image):
    img_array = np.array(image)
    results = yolo_model.predict(img_array, imgsz=224)
    detections = results[0].boxes.data

    if len(detections) == 0:
        return "No se detectaron accidentes en la imagen."

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = map(int, detection.tolist())

        # Verificar que las coordenadas sean válidas
        if x2 <= x1 or y2 <= y1:
            print(f"Coordenadas inválidas detectadas: {(x1, y1, x2, y2)}")
            continue

        # Validar las coordenadas para que no excedan los límites de la imagen
        roi = safe_crop(image, x1, y1, x2, y2)

        # Redimensionar la ROI
        roi_resized = roi.resize((224, 224))
        img_array = np.array(roi_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción del modelo
        print(f"Input shape del modelo de clasificación: {classification_model.input_shape}")
        predictions = classification_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        if confidence > 0.5:
            severity = ['Leve', 'Moderado', 'Grave'][predicted_class]
            return f"Accidente detectado con gravedad: {severity}"
    
    return "No se pudo determinar la gravedad del accidente."

@app.route('/')
def home():
    return render_template('index2.html')

###############################
@app.route('/subir_imagen', methods=['POST'])
def upload_image():
    print(request.form) 
    # Obtener la URL de la imagen del cuerpo de la solicitud
    image_url = request.form['image_url']
    if not image_url:
        return jsonify({'status': 'Error', 'message': 'No image URL provided'}), 400

    try:
        steps = []

        # Descargar la imagen
        steps.append('Descargando la imagen...')
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        # Procesar la imagen
        steps.append('Procesando la imagen...')
        processed_image = process_image(image)
        

        # Usar el servicio de Gemini para generar una descripción
        # Guardar la imagen en un archivo temporal para subirla a Gemin
        temp_image_path = os.path.join(imagenes_dir, 'temp_image.jpg')
        image.save(temp_image_path)
        imagen_prueba = Image.open(temp_image_path)

        def upload_to_gemini(path, mime_type=None):
            """Uploads the given file to Gemini.
            See https://ai.google.dev/gemini-api/docs/prompting_with_media
            """
            file = genai.upload_file(path, mime_type="image/jpeg")
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file

        steps.append('Usando el servicio de Gemini para generar la descripción...')
        
        files = [
            #upload_to_gemini(os.path.join(imagenes_dir, "file0.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file1.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file2.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file3.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file4.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file5.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file6.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file7.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file8.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file9.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file10.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file11.jpg"), mime_type="image/jpeg"),
            #upload_to_gemini(os.path.join(imagenes_dir, "file12.jpg"), mime_type="image/jpeg"),
        ]

        response = model.generate_content([
            "EL modelo tiene que describir según el formato establecido lo que observa de la imagen te dejo unos ejemplos , recuerda que la plantilla es completamente necesaria te dejo la plantilla acá y cada respuesta es un output diferente: "
            "Tipo de Colisión: Que tipo de colisión observas en la imagen "
            "Gravedad: que gravedad  clasificarías entre leve moderado y grave(no existen intermedios y no hay mas repeusta solo peude escoger una de esas 3 palabras)? "
            "Involucrados: que autos están involucrados en el accidente trata de ser lo más específico en cuanto a tipos de autos "
            "¿cómo describirías el accidente? "
            "¿Qué daños observas que haya dejado el accidente? "
            "Por último, si no determinas que sea un accidente, simplemente escribe 'no es un accidente' en los campos limitate a la plantilla no escribas alguna otra linea que no sea de la plantlla de 4 outputs tampoco no escribas preambulos solamente los 4 outputs que son los mencionados aca y agregales ** antes de cada nombre de output y al finalizar tambien ",
            imagen_prueba
        ])
        gemini_output = response.text.strip()
        steps.append(f'Descripción de Gemini: {gemini_output}')

        # response = model.generate_content([
        # "EL modelo tiene que describir según el formato establecido lo que observa de la imagen te dejo unos ejemplos , recuerda que la plantilla es completamente necesaria te dejo la plantilla acá y cada respuesta es un output diferente  :1er output:Tipo de Colisión: Que tipo de colisión observas en la imagen y que gravedad le casificarias entre leve moderado y grave?2do output: Involucrados:que autos estan involucrados en el accidetne trat de ser lo mas especifico en cuanto a tipos de autos 3er output : como describirias el accidente? 4to output : que daños observas que haya dejado el accidente por ultimo si es que no determinas que sea un accidente simplemente escribe no es un accidente en los campos"
        # "ImagenAccidente ",
        # files[0],
        # "Tipo colisión Colisión lateral\nGravedad: Leve"
        # "Involucrados Dos vehiculos de color negro una camioneta y auto"
        # "Descripción general del accidente En una intersección con pasos peatonales señalizados, un sedán negro y un SUV oscuro colisionaron en ángulo. El sedán impactó con su parte frontal el lado derecho delantero del SUV, sugiriendo que uno de los vehículos no cedió el paso."
        # "Daños El SUV tiene daños en el lado derecho delantero, incluyendo el parachoques, el faro y posiblemente la suspensión. El sedán presenta daños en el parachoques delantero, capó y sistema de luces, con potencial daño estructural."
        # "ImagenAccidente ",
        # files[1],
        # "Tipo colisión Volcadura\nGravedad: Grave",
        # "Involucrados Vehiculo de color blanco"
        # "Descripción general del accidente En una carretera, un vehículo blanco ha volcado y está completamente invertido sobre su techo. Los daños y la posición del vehículo sugieren que perdió el control, posiblemente debido a un choque o maniobra brusca."
        # "Daños El vehículo blanco presenta daños severos en el techo, el capó, y las partes delanteras y traseras. Es probable que haya daños significativos en el sistema de suspensión, así como posibles daños estructurales serios que requerirán una evaluación completa y reparaciones extensas."
        # "ImagenAccidente ",
        # files[2],
        # "Tipo colisión Colisión Frontal\nGravedad: Moderada"
        # "Involucrados Dos vehiculos de color negro una camioneta y auto"
        # "Descripción general del accidente En una intersección urbana, un sedán gris y un coche blanco colisionaron. El impacto se produjo en la parte frontal del coche blanco y en el lado izquierdo delantero del sedán gris, indicando que uno de los vehículos no respetó una señal de tráfico."
        # "Daños El coche blanco presenta daños graves en la parte frontal, afectando el parachoques, capó y posiblemente el motor. El sedán gris tiene daños significativos en el lado izquierdo delantero, incluyendo el parachoques, aleta y sistema de suspensión.",
        # imagen_prueba])
        # gemini_output = response.text.strip()

        # steps.append(f'Descripción de Gemini: {gemini_output}')
        
        return jsonify({'status': 'Success', 'steps': steps, 'gemini_output': gemini_output}), 200
    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 500
    



if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)