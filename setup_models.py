import os
import requests
from tqdm import tqdm
import sys

def download_file(url, destination_folder, file_name):
    """
    Descarga un archivo desde una URL a una carpeta de destino con una barra de progreso.
    Si el archivo ya existe, no lo descarga de nuevo.
    """
    # Asegurarse de que la carpeta de destino exista
    os.makedirs(destination_folder, exist_ok=True)

    file_path = os.path.join(destination_folder, file_name)

    # Verificar si el archivo ya existe
    if os.path.exists(file_path):
        print(f"✅ El archivo '{file_name}' ya existe en la ubicación correcta. No se necesita descarga.")
        return

    try:
        print(f"⬇️  Descargando '{file_name}'...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza un error si la descarga falla (e.g., 404)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        with open(file_path, 'wb') as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        print(f"✅ Descarga completada: '{file_path}'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar el archivo: {e}", file=sys.stderr)
        print("Por favor, verifica tu conexión a internet o la URL del archivo.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # URL del modelo de expresión facial de DeepFace
    MODEL_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
    # Carpeta de destino (directorio .deepface en la carpeta del usuario)
    DESTINATION_FOLDER = os.path.join(os.path.expanduser('~'), '.deepface', 'weights')
    
    download_file(MODEL_URL, DESTINATION_FOLDER, "facial_expression_model_weights.h5")