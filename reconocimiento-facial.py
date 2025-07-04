import os
os.environ["YOLO_VERBOSE"] = "False"  # Desactivar logs extensos de ultralytics para consola

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

# --- CONFIGURACIONES PRINCIPALES ---
RUTA_VIDEO_ENTRADA = "video.mp4"
RUTA_MODELO_YOLO = "cabezas.pt"
RUTA_VIDEO_SALIDA = "video_anotado.mp4"
CARPETA_CARAS_CONOCIDAS = "caras_conocidas"
UMBRAL_CONFIDENCIA_YOLO = 0.6
TOLERANCIA_RECONOCIMIENTO_CARA = 0.6
UMBRAL_CAMBIO_CENTRO = 50       # en píxeles, para detectar cambios bruscos en posición
UMBRAL_CAMBIO_TAMANO = 0.2     # proporción relativa para cambios bruscos en tamaño de caja


def cargar_caras_conocidas(directorio):
    """
    Carga las imágenes de un directorio que contiene fotos de personas conocidas,
    calcula las codificaciones faciales de cada imagen y devuelve dos listas:
    una con las codificaciones y otra con los nombres extraídos de los nombres
    de archivo (sin extensión).
    """
    codificaciones = []
    nombres = []

    for archivo in os.listdir(directorio):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            nombre_persona = os.path.splitext(archivo)[0]  # nombre sin extensión
            ruta_imagen = os.path.join(directorio, archivo)
            imagen = face_recognition.load_image_file(ruta_imagen)
            cods = face_recognition.face_encodings(imagen)

            if cods:
                # Se toma la primera codificación encontrada (cara principal)
                codificaciones.append(cods[0])
                nombres.append(nombre_persona)

    return codificaciones, nombres


def calcular_centro_y_tamano_caja(caja):
    """
    Dada una caja delimitadora en formato (x1, y1, x2, y2),
    calcula y devuelve el centro (cx, cy) y el tamaño (ancho, alto) de la caja.
    """
    x1, y1, x2, y2 = caja
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2
    ancho = x2 - x1
    alto = y2 - y1
    return centro_x, centro_y, ancho, alto


def hay_cambio_abrupto(caja_anterior, caja_actual,
                      umbral_centro=UMBRAL_CAMBIO_CENTRO,
                      umbral_tamano=UMBRAL_CAMBIO_TAMANO):
    """
    Compara dos cajas delimitadoras para determinar si ha habido un cambio brusco.
    Se considera cambio brusco si la distancia entre centros supera el umbral
    o si el cambio relativo en ancho o alto supera el umbral definido.
    """
    if caja_anterior is None:
        # Si no hay caja previa, consideramos cambio abrupto para forzar reconocimiento
        return True

    cx_ant, cy_ant, w_ant, h_ant = calcular_centro_y_tamano_caja(caja_anterior)
    cx_act, cy_act, w_act, h_act = calcular_centro_y_tamano_caja(caja_actual)

    distancia_centros = np.linalg.norm([cx_act - cx_ant, cy_act - cy_ant])
    cambio_ancho = abs(w_act - w_ant) / max(w_ant, 1)  # evitar división por cero
    cambio_alto = abs(h_act - h_ant) / max(h_ant, 1)

    if (distancia_centros > umbral_centro or
            cambio_ancho > umbral_tamano or
            cambio_alto > umbral_tamano):
        return True

    return False


class GestorReconocimientoCaras:
    """
    Clase encargada de gestionar el reconocimiento facial y el almacenamiento
    en caché de resultados para cada ID de seguimiento, evitando reconocimiento
    redundante innecesario y mejorando rendimiento.
    """

    def __init__(self, codificaciones_conocidas, nombres_conocidos):
        self.codificaciones_conocidas = codificaciones_conocidas
        self.nombres_conocidos = nombres_conocidos
        self.cache_id_a_nombre = {}  # mapa de track_id a nombre reconocido
        self.cache_id_a_caja = {}    # mapa de track_id a caja previa

    def obtener_nombre(self, frame, caja, track_id):
        """
        Decide si se debe realizar reconocimiento facial para un track_id dado.
        Si es necesario, realiza el reconocimiento, guarda el resultado en caché
        y devuelve el nombre reconocido. Si no, devuelve el nombre almacenado en caché.
        """
        caja_previa = self.cache_id_a_caja.get(track_id)
        reconocer_cara = False

        # Si no existe en caché, reconocimiento inmediato
        if track_id not in self.cache_id_a_nombre:
            reconocer_cara = True
        else:
            # Si hay cambio brusco en caja, forzar nuevo reconocimiento
            if hay_cambio_abrupto(caja_previa, caja):
                reconocer_cara = True
                self.cache_id_a_nombre.pop(track_id, None)  # eliminar nombre previo para actualizar

        nombre = "???"

        if reconocer_cara:
            x1, y1, x2, y2 = caja
            recorte = frame[y1:y2, x1:x2]

            if recorte.size != 0:
                # Convertir imagen BGR (OpenCV) a RGB (face_recognition)
                recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)

                # Detectar ubicaciones de caras y sus codificaciones
                ubicaciones = face_recognition.face_locations(recorte_rgb)
                codificaciones = face_recognition.face_encodings(recorte_rgb, ubicaciones)

                if codificaciones:
                    codificacion_objetivo = codificaciones[0]
                    coincidencias = face_recognition.compare_faces(
                        self.codificaciones_conocidas, codificacion_objetivo,
                        tolerance=TOLERANCIA_RECONOCIMIENTO_CARA)
                    distancias = face_recognition.face_distance(
                        self.codificaciones_conocidas, codificacion_objetivo)

                    if True in coincidencias:
                        indice_mejor = np.argmin(distancias)
                        nombre = self.nombres_conocidos[indice_mejor]

            self.cache_id_a_nombre[track_id] = nombre
        else:
            # Usar nombre cacheado para ese track_id
            nombre = self.cache_id_a_nombre.get(track_id, "???")

        # Actualizar caja previa para el track_id
        self.cache_id_a_caja[track_id] = caja

        return nombre


def inicializar_lectura_y_escritura_video(ruta_entrada, ruta_salida):
    """
    Inicializa la lectura del video de entrada y la escritura del video de salida,
    manteniendo el tamaño y fps original del video.
    Devuelve los objetos cv2.VideoCapture y cv2.VideoWriter.
    """
    captura = cv2.VideoCapture(ruta_entrada)
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30  # Si no se detecta fps, usar 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    return captura, escritor


def main():
    # Cargar caras conocidas y sus codificaciones
    cods_conocidas, nombres_conocidos = cargar_caras_conocidas(CARPETA_CARAS_CONOCIDAS)

    if not cods_conocidas:
        print(f"[ADVERTENCIA] No se cargaron caras conocidas desde '{CARPETA_CARAS_CONOCIDAS}'")

    # Cargar modelo YOLO para detección y seguimiento
    modelo_yolo = YOLO(RUTA_MODELO_YOLO)

    # Inicializar video de entrada y salida
    captura_video, escritor_video = inicializar_lectura_y_escritura_video(RUTA_VIDEO_ENTRADA, RUTA_VIDEO_SALIDA)

    # Crear gestor de reconocimiento facial con caché
    gestor_reconocimiento = GestorReconocimientoCaras(cods_conocidas, nombres_conocidos)

    while True:
        ret, frame = captura_video.read()
        if not ret:
            # Fin del video
            break

        # Realizar detección y seguimiento con YOLO (persistencia activada)
        resultados = modelo_yolo.track(frame, persist=True, conf=UMBRAL_CONFIDENCIA_YOLO)

        if resultados and resultados[0].boxes is not None:
            cajas = resultados[0].boxes
            ids = cajas.id.cpu().numpy() if cajas.id is not None else []
            coords = cajas.xyxy.cpu().numpy()

            for i, (x1, y1, x2, y2) in enumerate(coords):
                # Convertir a enteros para dibujo
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(ids[i]) if len(ids) > i else -1
                caja_actual = (x1, y1, x2, y2)

                # Obtener nombre reconocido o cacheado para este track_id
                nombre_persona = gestor_reconocimiento.obtener_nombre(frame, caja_actual, track_id)

                # Dibujar caja delimitadora verde y el nombre bajo la caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, nombre_persona, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Escribir frame modificado en video de salida
        escritor_video.write(frame)

        # Mostrar el frame en ventana (puede comentarse si se desea solo guardar video)
        cv2.imshow("Reconocimiento Facial", frame)

        # Salir si se presiona la tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Liberar recursos al finalizar el procesamiento
    captura_video.release()
    escritor_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
