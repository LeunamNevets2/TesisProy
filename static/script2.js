document.getElementById('image-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const imageUrl = document.getElementById('image-url').value;
    if (!imageUrl) {
        alert('Por favor, ingresa una URL de imagen.');
        return;
    }

    // Limpiar el contenedor de resultados
    const stepsContainer = document.getElementById('steps');
    const resultContainer = document.getElementById('result');
    stepsContainer.innerHTML = '';
    resultContainer.innerHTML = '';
    stepsContainer.style.display = 'block'; // Mostrar pasos de procesamiento

    try {
        // Mostrar la imagen original proporcionada
        resultContainer.innerHTML += `<h3>Imagen Original:</h3>`;
        resultContainer.innerHTML += `<img id="original-image" src="${imageUrl}" alt="Imagen original proporcionada" style="max-width: 100%; margin-bottom: 20px;">`;

        // Paso 1: Procesar la imagen con YOLO
        stepsContainer.innerHTML += '<p>1. Procesando la imagen con YOLO...</p>';
        const yoloResponse = await fetch('/yolov8', {
            method: 'POST',
            body: new URLSearchParams({ image_url: imageUrl })
        });
        const yoloData = await yoloResponse.json();

        if (yoloData.status !== 'Success') {
            throw new Error('Error en el procesamiento de YOLO: ' + yoloData.message);
        }

        stepsContainer.innerHTML += '<p>Imagen procesada con YOLO correctamente.</p>';
        resultContainer.innerHTML += `<h3>Imagen Procesada por YOLO:</h3>`;
        resultContainer.innerHTML += `<img id="processed-image" src="${yoloData.processed_image_url}" alt="Imagen procesada por YOLO" style="max-width: 100%; margin-bottom: 20px;">`;

        // Paso 2: Clasificar la imagen con el modelo
        stepsContainer.innerHTML += '<p>2. Clasificando la imagen...</p>';
        const classifyResponse = await fetch('/classify', {
            method: 'POST',
            body: new URLSearchParams({ image_url: imageUrl })
        });
        const classifyData = await classifyResponse.json();

        if (classifyData.status !== 'Success') {
            throw new Error('Error en la clasificación: ' + classifyData.message);
        }

        stepsContainer.innerHTML += '<p>Clasificación realizada exitosamente.</p>';
        resultContainer.innerHTML += `<p><strong>Clase Predicha:</strong> ${classifyData.predicted_class}</p>`;
        resultContainer.innerHTML += `<p><strong>Probabilidades:</strong> ${JSON.stringify(classifyData.predictions)}</p>`;

        // Paso 3: Generar la descripción con Gemini
        stepsContainer.innerHTML += '<p>3. Generando la descripción con Gemini...</p>';
        const geminiResponse = await fetch('/subir_imagen', {
            method: 'POST',
            body: new URLSearchParams({ image_url: imageUrl })
        });
        const geminiData = await geminiResponse.json();

        if (geminiData.status !== 'Success') {
            throw new Error('Error en Gemini: ' + geminiData.message);
        }

        stepsContainer.innerHTML += '<p>Descripción generada exitosamente por Gemini.</p>';
        resultContainer.innerHTML += `<h3>Descripción Generada:</h3>`;

        // Procesar y mostrar la descripción de Gemini
        const geminiOutput = geminiData.gemini_output;
        
        // Dividir el texto correctamente por los subtítulos (usando el delimitador '**')
        const sections = geminiOutput.split(/\*\*(.*?)\*\*/).filter(Boolean); // Filtrar valores vacíos

        let html = '<ul>';
        for (let i = 0; i < sections.length; i += 2) {
            let title = sections[i].trim();
            let content = sections[i + 1].trim();
            if (title && content) {
                html += `<li><b>${title}</b> ${content}</li>`;
            }
        }
        html += '</ul>';

        resultContainer.innerHTML += html;

    } catch (error) {
        stepsContainer.innerHTML += `<p class="error">Error: ${error.message}</p>`;
    }
});
