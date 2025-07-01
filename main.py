import cv2
import numpy as np
from deepface import DeepFace
import random
import time
from collections import deque
import os
import sys

# Configuración para reducir logs de TensorFlow y compatibilidad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Verificación de versión de NumPy
if np.__version__.startswith('2.'):
    print("Advertencia: NumPy 2.x puede tener problemas de compatibilidad")
    # Configuración para compatibilidad
    np.float = float  
    np.int = int

# Configuración de encoding para caracteres especiales
sys.stdout.reconfigure(encoding='utf-8')

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
    "feliz": (0, 255, 255),      # Amarillo brillante
    "triste": (255, 0, 0),       # Azul intenso
    "enojado": (0, 0, 255),      # Rojo vivo
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
    
    def obtener_frase(self, emocion):
        """Obtiene una frase aleatoria no usada recientemente para la emoción"""
        if emocion not in FRASES:
            return f"Emoción detectada: {emocion.capitalize()}"
        
        # Solo cambiar frase si la emoción cambió
        if emocion != self.ultima_emocion:
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
        # Usamos DeepFace para analizar las emociones
        resultado = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False, 
            silent=True
        )
        
        if resultado and isinstance(resultado, list):
            emocion_ingles = resultado[0]['dominant_emotion'].lower()
            return TRADUCCION_EMOCIONES.get(emocion_ingles, "default")
    except Exception as e:
        print(f"Error en detección: {str(e)}")
    return "default"

def dibujar_interfaz(fondo, frame_pequeno, texto, tiempo_transcurrido, emocion):
    """Dibuja todos los elementos de la interfaz en el fondo"""
    # Insertar el frame de la cámara con un borde
    fondo[60:540, 80:720] = frame_pequeno
    cv2.rectangle(fondo, (75, 55), (725, 545), (255, 255, 255), 2)
    
    # Barra superior con texto (fondo semitransparente)
    overlay = fondo.copy()
    cv2.rectangle(overlay, (0, 0), (800, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, fondo, 0.4, 0, fondo)
    
    # Mostrar texto con encoding UTF-8
    texto = texto.encode('latin-1', 'replace').decode('latin-1')
    cv2.putText(fondo, texto, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Barra inferior con información
    cv2.rectangle(fondo, (0, 550), (800, 600), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, fondo, 0.4, 0, fondo)
    
    # Estado de detección
    estado = f"Estado: {'Detección activa' if emocion != 'default' else 'Esperando detección'}"
    cv2.putText(fondo, estado, (20, 570), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Tiempo desde última detección
    cv2.putText(fondo, f"Tiempo: {tiempo_transcurrido}s", (650, 570), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Instrucciones
    cv2.putText(fondo, "Presione 'Q' para salir", (350, 590), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """Función principal que maneja el flujo de la aplicación"""
    # Prueba con diferentes índices de cámara
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"¡Cámara encontrada en índice {camera_index}!")
            break
    else:
        print("Error: No se pudo acceder a ninguna cámara")
        return
    
    # Configurar ventana
    cv2.namedWindow("Detección de Emociones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Emociones", 800, 600)
    
    # Inicializar variables
    emocion_actual = "default"
    ultimo_cambio = time.time()
    frame_count = 0
    FRAME_SKIP = 5  # Procesar 1 de cada 5 frames para mejor rendimiento
    frase_manager = FraseManager()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        
        frame_count += 1
        
        # Procesar solo algunos frames para mejorar rendimiento
        if frame_count % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            nueva_emocion = procesar_frame(frame_rgb)
            
            if nueva_emocion and nueva_emocion != emocion_actual:
                emocion_actual = nueva_emocion
                ultimo_cambio = time.time()
                print(f"Emoción detectada: {emocion_actual}")
        
        # Calcular tiempo transcurrido desde el último cambio
        tiempo_transcurrido = int(time.time() - ultimo_cambio)
        
        # Crear fondo con el color de la emoción actual
        fondo = np.full((600, 800, 3), COLORES.get(emocion_actual, COLORES["default"]), dtype=np.uint8)
        
        # Redimensionar frame de la cámara
        frame_pequeno = cv2.resize(frame, (640, 480))
        
        # Obtener frase adecuada (solo cambia si cambia la emoción)
        texto = frase_manager.obtener_frase(emocion_actual)
        
        # Dibujar toda la interfaz
        dibujar_interfaz(fondo, frame_pequeno, texto, tiempo_transcurrido, emocion_actual)
        
        # Mostrar el resultado
        cv2.imshow("Detección de Emociones", fondo)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()