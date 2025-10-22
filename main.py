import cv2
import numpy as np
from deepface import DeepFace
import random
import time
from collections import deque
import os
import sys
import locale
from PIL import Image, ImageDraw, ImageFont  # Importación necesaria para texto

# MediaPipe para detección y segmentación
try:
    import mediapipe as mp
except Exception:
    mp = None

# Configuración regional para caracteres especiales (UTF-8)
try:
    locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, 'spanish')

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Configuración para reducir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Verificación de versión de NumPy
if np.__version__.startswith('2.'):
    print("Advertencia: NumPy 2.x puede tener problemas de compatibilidad")
    np.float = float  
    np.int = int

# Mapeo de emociones inglés-español con tildes correctas
TRADUCCION_EMOCIONES = {
    "happy": "feliz",
    "sad": "triste",
    "angry": "enojado",
    "surprise": "sorprendido",
    "neutral": "neutral",
    "fear": "miedo",
    "disgust": "disgusto"
}

# Configuración de colores por emoción (en formato BGR)
COLORES = {
    "feliz": (0, 255, 255),       # Amarillo brillante
    "triste": (255, 0, 0),        # Azul intenso
    "enojado": (0, 0, 255),       # Rojo vivo
    "sorprendido": (204, 153, 0), # Azul verdoso
    "neutral": (128, 128, 128),   # Gris medio
    "miedo": (0, 255, 0),         # Verde
    "disgusto": (128, 0, 128),    # Morado
    "default": (50, 50, 50)       # Gris oscuro (inicial)
}

# Frases motivacionales por emoción (con tildes y caracteres especiales correctos)
FRASES = {
    "feliz": [
        "¡Tu sonrisa ilumina el día! Sigue así.",
        "La alegría que compartes es contagiosa.",
        "Los momentos felices crean recuerdos duraderos.",
        "Tu actitud positiva abre puertas increíbles.",
        "La felicidad está en los pequeños detalles.",
        "Comparte tu alegría, se multiplicará.",
        "El optimismo atrae cosas buenas."
    ],
    "triste": [
        "Los días difíciles son temporales, tú eres fuerte.",
        "Permítete sentir, luego recuerda tu resiliencia.",
        "La tristeza prepara el alma para más alegría.",
        "Esta emoción pasará, como todo en la vida.",
        "No estás solo, comparte lo que sientes.",
        "La tristeza puede ser el inicio de un crecimiento.",
        "Después de la lluvia siempre sale el sol."
    ],
    "enojado": [
        "Respira profundo, la calma es tu aliada.",
        "Transforma esa energía en acción positiva.",
        "La ira es una señal, no un destino final.",
        "Canaliza esa pasión hacia soluciones creativas.",
        "Tómate un momento antes de actuar.",
        "El enojo es como un carbón caliente.",
        "La paciencia es la mejor respuesta."
    ],
    "sorprendido": [
        "¡La sorpresa es el inicio del aprendizaje!",
        "Mantén la mente abierta a posibilidades.",
        "Lo inesperado trae oportunidades únicas.",
        "La vida te sorprende cuando menos lo esperas.",
        "La sorpresa mantiene viva la curiosidad.",
        "Aprovecha este asombro para aprender.",
        "Las mejores cosas llegan sin avisar."
    ],
    "neutral": [
        "La calma es terreno fértil para ideas.",
        "En la serenidad encuentras claridad.",
        "Tu equilibrio es una fortaleza poderosa.",
        "El silencio interior prepara grandes acciones.",
        "La neutralidad permite observar objetivamente.",
        "No sentir nada en particular está bien.",
        "Este estado es perfecto para reflexionar."
    ],
    "miedo": [
        "El valor no es ausencia de miedo, sino superarlo.",
        "Analiza lo que te asusta, perderá poder.",
        "Los miedos son más pequeños cuando los enfrentas.",
        "Respira hondo y da un paso adelante.",
        "El coraje crece cuando actuamos a pesar del miedo.",
        "Eres más fuerte que lo que te asusta."
    ],
    "disgusto": [
        "Las reacciones fuertes indican valores profundos.",
        "Tómate un momento para procesar lo que sientes.",
        "El disgusto puede ser una señal de autocuidado.",
        "Reconoce lo que te molesta y busca soluciones.",
        "Esta emoción protege tus límites personales."
    ]
}

class FraseManager:
    """Gestor de frases que evita repeticiones hasta agotar todas las opciones"""
    def __init__(self):
        self.frases_usadas = {emocion: deque(maxlen=len(opciones)) 
                            for emocion, opciones in FRASES.items()}
        self.ultima_emocion = None
        self.ultima_frase = None
        self.contador_frames = 0
        self.frames_por_frase = 60  # Cambiar frase cada 60 frames (~2 segundos)
    
    def obtener_frase(self, emocion):
        """Obtiene una frase aleatoria no usada recientemente para la emoción"""
        if emocion not in FRASES:
            return f"Emoción detectada: {emocion.capitalize()}"
        
        self.contador_frames += 1
        
        # Cambiar frase si la emoción cambió o si ha pasado el tiempo suficiente
        if (emocion != self.ultima_emocion or 
            self.contador_frames >= self.frames_por_frase):
            
            self.contador_frames = 0
            self.ultima_emocion = emocion
            
            # Obtener frases no usadas recientemente
            disponibles = [f for f in FRASES[emocion] 
                          if f not in self.frases_usadas[emocion]]
            
            # Si todas han sido usadas, reiniciamos para esta emoción
            if not disponibles:
                self.frases_usadas[emocion].clear()
                disponibles = FRASES[emocion]
            
            self.ultima_frase = random.choice(disponibles)
            self.frases_usadas[emocion].append(self.ultima_frase)
        
        return f"{self.ultima_frase} [Emoción: {emocion.capitalize()}]"

def procesar_frame(frame):
    """Detecta emociones en un frame y devuelve la predominante"""
    try:
        # Usamos DeepFace para analizar las emociones sobre el recorte
        resultado = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        # DeepFace puede devolver diccionario o lista dependiendo de la versión
        if resultado:
            if isinstance(resultado, list):
                res = resultado[0]
            else:
                res = resultado

            emocion_ingles = res.get('dominant_emotion', '').lower()
            return TRADUCCION_EMOCIONES.get(emocion_ingles, "default")
    except Exception as e:
        print(f"Error en detección: {str(e)}")
    return "default"


def detectar_caras_mediapipe(frame_rgb, mp_face_detection, min_detection_confidence=0.5):
    """Detecta caras usando MediaPipe Face Detection. Retorna lista de bboxes en pixeles (x, y, w, h)."""
    if mp is None or mp_face_detection is None:
        return []

    results = mp_face_detection.process(frame_rgb)
    h, w, _ = frame_rgb.shape
    bboxes = []
    if results.detections:
        for det in results.detections:
            bboxC = det.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            # Aumentar el bbox ligeramente para incluir mentón y frente
            pad_x = int(0.1 * bw)
            pad_y = int(0.2 * bh)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            bboxes.append((x1, y1, x2 - x1, y2 - y1))
    return bboxes


def smooth_bbox(bbox_deque, new_bbox, maxlen=5):
    """Agrega new_bbox a la deque y devuelve bbox suavizado por promedio aritmético."""
    bbox_deque.append(new_bbox)
    xs = [b[0] for b in bbox_deque]
    ys = [b[1] for b in bbox_deque]
    ws = [b[2] for b in bbox_deque]
    hs = [b[3] for b in bbox_deque]
    return (
        int(sum(xs) / len(xs)),
        int(sum(ys) / len(ys)),
        int(sum(ws) / len(ws)),
        int(sum(hs) / len(hs)),
    )

def dibujar_interfaz(fondo, frame_pequeno, texto, tiempo_transcurrido, emocion):
    """Dibuja todos los elementos de la interfaz en el fondo"""
    # Redimensionar el frame pequeño para que encaje exactamente en el espacio asignado
    frame_redimensionado = cv2.resize(frame_pequeno, (640, 480))
    
    # Insertar el frame de la cámara con un borde
    fondo[60:540, 80:720] = frame_redimensionado
    
    # Dibujar borde blanco alrededor del frame
    cv2.rectangle(fondo, (75, 55), (725, 545), (255, 255, 255), 2)
    
    # Barra superior con texto (fondo semitransparente)
    overlay = fondo.copy()
    cv2.rectangle(overlay, (0, 0), (800, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, fondo, 0.4, 0, fondo)
    
    # Barra inferior con información
    cv2.rectangle(fondo, (0, 550), (800, 600), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, fondo, 0.4, 0, fondo)
    
    # ========================================================
    # USAR FUENTE DESDE EL DIRECTORIO ACTUAL
    # ========================================================
    # Convertir imagen de OpenCV a formato PIL
    fondo_pil = Image.fromarray(cv2.cvtColor(fondo, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(fondo_pil)
    
    # Intentar cargar la fuente Arial desde el directorio actual
    try:
        fuente = ImageFont.truetype('Arial.ttf', 20)
    except IOError:
        # Si no se encuentra, usar fuente por defecto
        fuente = ImageFont.load_default()
        print("Advertencia: Fuente Arial.ttf no encontrada. Usando fuente por defecto.")
    
    # Dibujar texto superior (frase motivacional)
    draw.text((20, 30), texto, font=fuente, fill=(255, 255, 255))
    
    # Preparar textos de la barra inferior (fuente más pequeña)
    try:
        fuente_info = ImageFont.truetype('Arial.ttf', 15)
    except:
        fuente_info = ImageFont.load_default()
    
    estado = "Estado: Detección activa" if emocion != "default" else "Estado: Esperando detección"
    tiempo_texto = f"Tiempo: {tiempo_transcurrido}s"
    instrucciones = "Presione 'Q' para salir"
    
    # Dibujar textos de la barra inferior
    draw.text((20, 555), estado, font=fuente_info, fill=(255, 255, 255))
    draw.text((650, 555), tiempo_texto, font=fuente_info, fill=(255, 255, 255))
    
    # Calcular ancho del texto para centrarlo
    if hasattr(fuente_info, 'getsize'):
        ancho_texto = fuente_info.getsize(instrucciones)[0]
    else:
        ancho_texto = len(instrucciones) * 10  # Aproximación si no se puede calcular
    
    x_centrado = (800 - ancho_texto) // 2
    draw.text((x_centrado, 580), instrucciones, font=fuente_info, fill=(255, 255, 255))
    
    # Convertir de vuelta a formato OpenCV
    fondo = cv2.cvtColor(np.array(fondo_pil), cv2.COLOR_RGB2BGR)
    
    return fondo

def main():
    """Función principal que maneja el flujo de la aplicación"""
    # Prueba con diferentes índices de cámara por defecto 0
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"¡Cámara encontrada en índice {camera_index}!")
            break
    else:
        print("Error: No se pudo acceder a ninguna cámara")
        return
    
    # Configurar ventana con título correcto
    window_title = "Deteccion de Emociones"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 800, 600)
    
    # Inicializar variables
    emocion_actual = "default"
    ultimo_cambio = time.time()
    frame_count = 0
    FRAME_SKIP = 4  # Procesar 1 de cada 4 frames para mejor rendimiento
    frase_manager = FraseManager()

    # MediaPipe: detección y segmentación (opcional)
    use_segmentation = True if mp is not None else False
    mp_face_detection = None
    mp_selfie_segmentation = None
    if mp is not None:
        mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        if use_segmentation:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Tracker para bbox estable (CSRT funciona bien para caras)
    tracker = None
    tracker_initialized = False
    bbox_history = deque(maxlen=6)
    emotion_from_face = "default"
    analyze_every_n_frames = 8
    last_analyze_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        
        frame_count += 1
        
        # Procesar solo algunos frames para mejorar rendimiento
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicar segmentación para desenfocar el fondo (opcional y visual)
        if use_segmentation and mp_selfie_segmentation is not None:
            seg_results = mp_selfie_segmentation.process(frame_rgb)
            if seg_results and seg_results.segmentation_mask is not None:
                mask = seg_results.segmentation_mask > 0.1
                bg = cv2.GaussianBlur(frame, (55, 55), 0)
                frame = np.where(mask[:, :, None], frame, bg)

        # Actualizar tracker si ya está inicializado
        face_bbox = None
        if tracker_initialized and tracker is not None:
            ok, tracked_bbox = tracker.update(frame)
            if ok:
                tracked_bbox = tuple(map(int, tracked_bbox))
                face_bbox = tracked_bbox  # x,y,w,h
                smoothed = smooth_bbox(bbox_history, face_bbox)
                face_bbox = smoothed
            else:
                # Tracker perdió al sujeto
                tracker = None
                tracker_initialized = False

        # Si no hay tracker activo, intentar detectar con MediaPipe cada FRAME_SKIP frames
        if not tracker_initialized and frame_count % FRAME_SKIP == 0 and mp_face_detection is not None:
            bboxes = detectar_caras_mediapipe(frame_rgb, mp_face_detection)
            if bboxes:
                # Seleccionar la cara más grande
                bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = bboxes[0]
                face_bbox = (x, y, w, h)
                # Inicializar tracker con bbox detectado
                try:
                    tracker = cv2.TrackerCSRT_create()
                except Exception:
                    tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, face_bbox)
                tracker_initialized = True
                bbox_history.clear()
                bbox_history.append(face_bbox)

        # Si ya tenemos bbox (track o detect), analizar emoción sobre el recorte cada N frames
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Asegurar límites
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size != 0 and (frame_count - last_analyze_frame) >= analyze_every_n_frames:
                last_analyze_frame = frame_count
                try:
                    # DeepFace espera RGB o BGR dependiendo de versión; usar BGR funciona generalmente
                    nueva_emocion = procesar_frame(face_crop)
                    if nueva_emocion and nueva_emocion != emotion_from_face:
                        emotion_from_face = nueva_emocion
                        emocion_actual = emotion_from_face
                        ultimo_cambio = time.time()
                        print(f"Emoción detectada (face): {emocion_actual}")
                except Exception as e:
                    print(f"Error analizando cara: {e}")

        # Si no hay face_bbox y no hay tracker, ocasionalmente intentar detectar globalmente cada 30 frames
        if face_bbox is None and frame_count % 30 == 0:
            # Fallback: usar DeepFace sobre frame reducido (menos preciso pero evita quedarse en 'default')
            try:
                small = cv2.resize(frame, (320, 240))
                fallback_emocion = procesar_frame(small)
                if fallback_emocion and fallback_emocion != emocion_actual:
                    emocion_actual = fallback_emocion
                    ultimo_cambio = time.time()
                    print(f"Emoción detectada (fallback): {emocion_actual}")
            except Exception:
                pass
        
        # Calcular tiempo transcurrido desde el último cambio
        tiempo_transcurrido = int(time.time() - ultimo_cambio)
        
        # Crear fondo con el color de la emoción actual
        fondo = np.full((600, 800, 3), COLORES.get(emocion_actual, COLORES["default"]), dtype=np.uint8)

        # Redimensionar frame de la cámara
        frame_pequeno = cv2.resize(frame, (640, 480))

        # Dibujar bounding box estable en el frame pequeño si existe
        texto_superior = frase_manager.obtener_frase(emocion_actual)
        if face_bbox is not None:
            # Ajustar coords al frame_pequeno si se desea superponer (frame original -> frame_pequeno escala)
            scale_x = 640 / frame.shape[1]
            scale_y = 480 / frame.shape[0]
            bx = int(face_bbox[0] * scale_x)
            by = int(face_bbox[1] * scale_y)
            bw = int(face_bbox[2] * scale_x)
            bh = int(face_bbox[3] * scale_y)
            cv2.rectangle(frame_pequeno, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            label = f"{emocion_actual.capitalize()}"
            cv2.putText(frame_pequeno, label, (bx, max(15, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Obtener frase adecuada (cambia periódicamente)
        texto = texto_superior

        # Dibujar toda la interfaz
        fondo = dibujar_interfaz(fondo, frame_pequeno, texto, tiempo_transcurrido, emocion_actual)

        # Mostrar el resultado
        cv2.imshow(window_title, fondo)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()