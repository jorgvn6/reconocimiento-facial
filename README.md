# Identificación de Personas Mediante Reconocimiento Facial

## Reconocimiento Facial con YOLOv8 y face_recognition

Este proyecto demuestra cómo utilizar modelos de visión por computadora basados en YOLOv8 para detectar personas y reconocer sus rostros mediante la librería `face_recognition`. El sistema identifica personas conocidas en un video, anota sus nombres en pantalla y mejora el rendimiento utilizando seguimiento por ID y memoria caché.

## Objetivos

- Detectar personas en video usando YOLOv8 con seguimiento de objetos.
- Reconocer rostros en tiempo real a partir de imágenes de referencia.
- Asociar detecciones con un `track_id` y evitar reconocimientos redundantes.
- Anotar un video mostrando el nombre de cada persona reconocida.

## Ejemplo Visual

A continuación se muestra un fragmento del video procesado, donde se visualiza el sistema en funcionamiento:

![Demo del sistema](media/demo.gif)

## Modelos Utilizados

- `cabezas.pt`: Modelo personalizado entrenado con el dataset v14 [Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) de Roboflow. Detecta cabezas en imágenes.

## Cómo Entrenar el Modelo Personalizado (`cabezas.pt`)

Para entrenar tu propio modelo basado en YOLOv8, puedes utilizar el siguiente script en Python. Este código entrena un modelo a partir del archivo de configuración `data.yaml`, el cual debe contener las rutas a tus imágenes y las clases del dataset.

### Script de entrenamiento

```python
from ultralytics import YOLO

def main():
    model = YOLO("yolov8x.pt")  # También se puede usar yolov8n.pt o yolov8s.pt
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=32,
        device=0,       # GPU (usa "cpu" si no tienes GPU)
        cache=True      # Precarga los datos en RAM
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
```

### Consideraciones al entrenar

Versiones de YOLOv8 disponibles:

- `yolov8n.pt` (Nano): Muy rápido y ligero, ideal para dispositivos con recursos limitados, pero con menor precisión.

- `yolov8s.pt` (Small): Un equilibrio entre velocidad y precisión, recomendado para pruebas rápidas o entrenamiento en laptops.

- `yolov8x.pt` (Extra Large): Más lento y pesado, pero con mayor capacidad de detección. Requiere más memoria GPU.

En este proyecto se usó `yolov8x.pt` como base para entrenar `cabezas.pt`, ya que se priorizó la precisión sobre la velocidad.

La opción `cache=True` carga todo el dataset en memoria RAM antes de entrenar. Esto puede acelerar el entrenamiento si tienes suficiente RAM, pero también puede provocar:

- Errores de memoria si el dataset es muy grande.

- Resultados **no deterministas** si se reutiliza la caché sin reiniciar.

- Problemas si usas múltiples procesos de entrenamiento simultáneamente.

En el archivo `data.yaml` debe modificarse la ruta de las imágenes de entrenamiento para que esté acorde a la ruta real en la máquina donde se va a entrenar.

## Cómo Funciona el Reconocimiento

El sistema carga imágenes de personas conocidas desde una carpeta, calcula sus codificaciones faciales, y compara los rostros detectados en el video con estas referencias. Para mejorar el rendimiento:

- Se mantiene un mapa de `track_id` a nombre reconocido.
- Se actualiza el reconocimiento solo si hay un cambio abrupto en la posición o tamaño de la persona.
- Los resultados se anotan directamente sobre el video de salida.

## Resultados Esperados

Al ejecutar el sistema de detección y reconocimiento, se generan dos productos principales para su análisis y visualización:

1. **Video Anotado**  
   Un archivo de video donde cada persona detectada está rodeada por un recuadro verde. Debajo se muestra su nombre, o `???` si no fue reconocida. Este video permite verificar visualmente quién aparece en la escena.

2. **Reconocimiento Optimizado**  
   Gracias al uso de caché por `track_id`, el sistema evita hacer reconocimiento facial en cada cuadro, mejorando el rendimiento sin perder precisión.

