# 🧠 Detección de Emociones en Tiempo Real con DeepFace

[![Python](https://img.shields.io/badge/Built%20with-Python%203.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/ML%20Library-DeepFace-red?style=for-the-badge&logo=tensorflow)](https://github.com/serengil/deepface)
[![Status](https://img.shields.io/badge/Status-Funcional-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-GPL%203.0-brightgreen?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0.html)

---

## 🎯 Descripción General

Este proyecto implementa una **aplicación de escritorio en Python** que utiliza la cámara web para detectar y analizar las emociones de un ser humano en tiempo real. Se apoya en la potente librería **DeepFace** para el reconocimiento de emociones y **OpenCV** para la captura y visualización de video. La interfaz visual, con colores adaptados a cada emoción detectada, muestra la emoción predominante en español y ofrece **frases motivacionales o reflexivas** personalizadas según el estado anímico, diseñadas para fomentar el bienestar.

La aplicación está optimizada para el rendimiento al procesar solo un subconjunto de frames, lo que permite una experiencia fluida. Es una herramienta interesante para explorar la inteligencia artificial aplicada a la percepción humana y cómo la tecnología puede interactuar de forma sensible con nuestras emociones.

---

## ✨ Características Destacadas

* **Detección de Emociones en Tiempo Real**: Utiliza DeepFace para identificar la emoción predominante (feliz, triste, enojado, sorprendido, neutral, miedo, disgusto) a partir del rostro del usuario.
* **Interfaz Visual Adaptativa**: El color de fondo de la ventana cambia dinámicamente para reflejar la emoción detectada, creando una experiencia visual inmersiva.
* **Mensajes Personalizados por Emoción**: Muestra frases relevantes y motivacionales en español que se actualizan periódicamente, ofreciendo un soporte o reflexión acorde a la emoción actual.
* **Optimización de Rendimiento**: Procesamiento de frames cada cierto intervalo (`FRAME_SKIP`) para reducir la carga de la CPU y GPU, garantizando una detección fluida.
* **Gestión Inteligente de Frases**: Un gestor de frases (`FraseManager`) asegura que las frases no se repitan consecutivamente para la misma emoción hasta que se hayan mostrado todas las opciones disponibles.
* **Soporte Multilenguaje (Español)**: Las emociones detectadas y las frases se muestran en español con tildes y caracteres especiales correctos.
* **Fácil de Usar**: Interfaz simple para iniciar la detección con solo ejecutar el script.

---

## ⚙️ Tecnologías Utilizadas

* **Lenguaje de Programación**: Python 3.8+
* **Visión por Computadora**: OpenCV (`cv2`)
* **Análisis Facial y Emocional**: DeepFace
* **Manipulación de Imágenes**: `numpy`
* **Interfaz Gráfica de Usuario (GUI)**: Implementada directamente con OpenCV para la visualización.
* **Manejo de Texto y Fuentes**: PIL (Pillow: `Image`, `ImageDraw`, `ImageFont`)
* **Otras Librerías**: `random`, `time`, `collections` (deque), `os`, `sys`, `locale`.

---

## 📂 Estructura del Repositorio

El proyecto se compone de un único archivo principal que contiene toda la lógica.

```
Deteccion_Emociones/
│
├── main.py                     # Script principal de la aplicación
├── Arial.ttf                   # Fuente (opcional, si se desea una fuente específica)
├── .gitignore                  # Archivos y carpetas excluidas del control de versiones
└── README.md                   # Documentación del proyecto
```

**Nota**: Para la visualización del texto con caracteres especiales y un formato específico, se ha incluido la capacidad de cargar una fuente `Arial.ttf` en el mismo directorio del script. Si esta fuente no está presente, el sistema usará una fuente por defecto.

---

## 🚀 Instrucciones de Ejecución

### Requisitos

* **Python 3.8+**
* **pip** (gestor de paquetes de Python)
* **Una cámara web funcional**

### Pasos para la ejecución

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/SebastianVega4/Captura_Emociones_Cam_web
    cd Deteccion_Emociones
    ```

2.  **Crear un entorno virtual (opcional pero recomendado)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate   # En Windows
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install opencv-python numpy deepface pillow
    ```
    (DeepFace descargará automáticamente sus modelos pre-entrenados la primera vez que se ejecute.)

4.  **Ejecutar la aplicación**:
    ```bash
    python main.py
    ```

5.  **Interacción**:
    * Asegúrate de que tu rostro sea visible para la cámara.
    * La ventana mostrará tu imagen con el fondo y texto adaptados a tu emoción detectada.
    * Presiona la tecla `'q'` para salir de la aplicación.

---

## 👨‍🎓 Autor

Desarrollado por **Sebastián Vega**

📧 *Sebastian.vegar2015@gmail.com*

🔗 [LinkedIn - Johan Sebastián Vega Ruiz](https://www.linkedin.com/in/johan-sebastian-vega-ruiz-b1292011b/)

---

## 📜 Licencia

Este repositorio se encuentra bajo la Licencia **GPL 3.0**.

**Permisos:**
* Uso comercial
* Modificación
* Distribución
* Uso privado

---

Facultad de Ingeniería — Ingeniería de Sistemas 🧩

**🏫 Universidad Pedagógica y Tecnológica de Colombia**
📍 Sogamoso, Boyacá 📍

© 2025 — Sebastian Vega
