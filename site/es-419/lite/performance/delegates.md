# Delegados de TensorFlow Lite

## Introducción

**Los delegados** permiten la aceleración por hardware de los modelos TensorFlow Lite aprovechando aceleradores en el dispositivo como la GPU y el [Procesador Digital de Señales (DSP)](https://en.wikipedia.org/wiki/Digital_signal_processor).

De forma predeterminada, TensorFlow Lite utiliza kernels de CPU que están optimizados para el conjunto de instrucciones [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions). Sin embargo, la CPU es un procesador multipropósito que no está necesariamente optimizado para la aritmética pesada que se encuentra típicamente en los modelos de aprendizaje automático (por ejemplo, la matemática matricial involucrada en la convolución y las capas densas).

Por otro lado, la mayoría de los teléfonos móviles modernos contienen chips que manejan mejor estas operaciones pesadas. Utilizarlos para operaciones de redes neuronales aporta enormes prestaciones en términos de latencia y eficiencia energética. Por ejemplo, las GPU pueden proporcionar hasta [5 veces más velocidad](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html) en latencia, mientras que el [Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor) ha demostrado reducir el consumo de energía hasta un 75% en nuestros experimentos.

Cada uno de estos aceleradores tiene APIs asociadas que permiten realizar cálculos personalizados, como [OpenCL](https://www.khronos.org/opencl/) o [OpenGL ES](https://www.khronos.org/opengles/) para GPU móviles y el [SDK Qualcomm® Hexagon](https://developer.qualcomm.com/software/hexagon-dsp-sdk) para DSP. Normalmente, tendría que escribir mucho código personalizado para ejecutar una red neuronal a través de estas interfaces. Además, las cosas se complican aún más si se tiene en cuenta que cada acelerador tiene sus pros y sus contras y no puede ejecutar todas las operaciones de una red neuronal. La API Delegate de TensorFlow Lite resuelve este problema actuando como puente entre el runtime de TFLite y estas APIs de bajo nivel.

![runtime con delegados](images/delegate_runtime.png)

## Seleccionar un delegado

TensorFlow Lite soporta múltiples delegados, cada uno de los cuales está optimizado para cierta(s) plataforma(s) y tipos particulares de modelos. Por lo general, habrá varios delegados aplicables a su caso de uso, en función de dos criterios principales: la *plataforma* (¿Android o iOS?) a la que se dirija, y el *tipo de modelo* (¿de punto flotante o cuantizado?) que esté intentando acelerar.

### Delegados por plataforma

#### Multiplataforma (Android e iOS)

- **Delegado de GPU**: El delegado GPU puede usarse tanto en Android como en iOS. Está optimizado para ejecutar modelos basados en flotantes de 32 y 16 bits cuando se dispone de una GPU. También es compatible con modelos cuantizados de 8 bits y ofrece un rendimiento en la GPU a la par que sus versiones flotantes. Para más información sobre el delegado de GPU, consulte [TensorFlow Lite en GPU](gpu_advanced.md). Para ver tutoriales paso a paso sobre cómo usar el delegado GPU con Android e iOS, consulte el [Tutorial del delegado GPU de TensorFlow Lite](gpu.md).

#### Android

- **Delegado NNAPI para dispositivos Android más recientes**: El delegado NNAPI puede usarse para acelerar modelos en dispositivos Android con GPU, DSP y/o NPU disponibles. Está disponible en Android 8.1 (API 27+) o superior. Para obtener una visión general del delegado NNAPI, instrucciones paso a paso y las mejores prácticas, consulte [Delegado NNAPI de TensorFlow Lite](nnapi.md).
- **Delegado Hexagon para dispositivos Android antiguos**: El delegado Hexagon puede usarse para acelerar modelos en dispositivos Android con Qualcomm Hexagon DSP. Puede usarse en dispositivos con versiones antiguas de Android que no soportan NNAPI. Vea [Delegado Hexagon de TensorFlow Lite](hexagon_delegate.md) para más detalles.

#### iOS

- **Delegado de Core ML para iPhones y iPads más recientes**: Para los iPhones y iPads más recientes en los que está disponible Neural Engine, puede usar el delegado de Core ML para acelerar la inferencia para modelos de coma flotante de 32 o 16 bits. Neural Engine está disponible para dispositivos móviles Apple con SoC A12 o superior. Para obtener una descripción general del delegado Core ML e instrucciones paso a paso, consulte [Delegado Core ML de TensorFlow Lite](coreml_delegate.md).

### Delegados por tipo de modelo

Cada acelerador está diseñado teniendo en cuenta un determinado ancho de bits de los datos. Si le da un modelo de punto flotante a un delegado que sólo admite operaciones cuantizadas de 8 bits (como el delegado [Hexagon](hexagon_delegate.md)), rechazará todas sus operaciones y el modelo se ejecutará íntegramente en la CPU. Para evitar este tipo de sorpresas, la tabla siguiente ofrece una visión general del soporte de delegados en función del tipo de modelo:

**Tipo de modelo** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
Punto flotante (32 bits) | Sí | Sí | No | Sí
[Cuantización de float16 posentrenamiento](post_training_float16_quant.ipynb) | Sí | No | No | Sí
[Cuantización del rango dinámico postentrenamiento](post_training_quant.ipynb) | Sí | Sí | No | No
[Cuantización entera postentrenamiento](post_training_integer_quant.ipynb) | Sí | Sí | Sí | No
[Entrenamiento consciente de la cuantización](http://www.tensorflow.org/model_optimization/guide/quantization/training) | Sí | Sí | Sí | No

### Validación del rendimiento

La información de esta sección sirve de lineamientos aproximados para preseleccionar los delegados que podrían mejorar su aplicación. Sin embargo, es importante tener en cuenta que cada delegado tiene un conjunto predefinido de operaciones que admite, y puede tener un rendimiento diferente en función del modelo y el dispositivo; por ejemplo, el [delegado NNAPI](nnapi.md) puede elegir usar el Edge-TPU de Google en un teléfono Pixel y utilizar un DSP en otro dispositivo. Por lo tanto, se suele recomendar que haga algunos análisis comparativos para calibrar cuán útil es un delegado para sus necesidades. Esto también ayuda a justificar el aumento del tamaño binario asociado a la incorporación de un delegado al runtime de TensorFlow Lite.

TensorFlow Lite cuenta con amplias herramientas de evaluación del rendimiento y la precisión que pueden empoderar a los desarrolladores para que confíen en usar delegados en su aplicación. Estas herramientas se analizan en la siguiente sección.

## Herramientas para la evaluación

### Latencia y huella de memoria

La herramienta [benchmark](https://www.tensorflow.org/lite/performance/measurement) de TensorFlow Lite puede usarse con los parámetros adecuados para estimar el rendimiento del modelo, incluyendo la latencia media de inferencia, la sobrecarga de inicialización, la huella de memoria, etc. Esta herramienta admite múltiples indicadores para averiguar la mejor configuración de delegados para su modelo. Por ejemplo, se puede especificar `--gpu_backend=gl` con `--use_gpu` para medir la ejecución de la GPU con OpenGL. La [documentación detallada](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar) contiene la lista completa de parámetros de delegado admitidos.

Aquí tiene un ejemplo de ejecución para un modelo cuantizado con GPU a través de `adb`:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

Puede descargar la versión precompilada de esta herramienta para Android, arquitectura ARM de 64 bits [aquí](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk) ([más detalles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)).

### Precisión y corrección

Los delegados suelen realizar los cálculos con una precisión diferente a la de sus homólogos de la CPU. Resultante de ello, hay una (generalmente menor) compensación de precisión asociada con la utilización de un delegado para la aceleración de hardware. Tenga en cuenta que esto no es *siempre* cierto; por ejemplo, dado que la GPU usa precisión de punto flotante para ejecutar modelos cuantizados, puede haber una ligera mejora de la precisión (por ejemplo, &lt;1% de mejora Top-5 en la clasificación de imágenes ILSVRC).

TensorFlow Lite dispone de dos tipos de herramientas para medir la precisión con la que se comporta un delegado para un modelo determinado: *Basada en tareas* y *Agnóstica de tareas*. Todas las herramientas descritas en esta sección admiten los [parámetros avanzados de delegación](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar) utilizados por la herramienta de evaluación comparativa de la sección anterior. Tenga en cuenta que las subsecciones siguientes se centran en la *evaluación del delegado* (¿El delegado rinde igual que la CPU?) más que en la evaluación del modelo (¿El modelo en sí es bueno para la tarea?).

#### Evaluación basada en tareas

TensorFlow Lite dispone de herramientas para evaluar la corrección en dos tareas basadas en imágenes:

- [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/) (Clasificación de imágenes) con [precisión superior a K](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

- [Detección de objetos COCO (con recuadros delimitadores)](https://cocodataset.org/#detection-2020) con [Precisión media promedio (mAP)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

Puede encontrar los binarios precompilados de estas herramientas (Android, arquitectura ARM de 64 bits), junto con la documentación, aquí:

- [Clasificación de imágenes con ImageNet](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification) ([Más detalles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification))
- [Detección de objetos COCO](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection) ([Más detalles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection))

El ejemplo siguiente demuestra la [evaluación de la clasificación de imágenes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification) con NNAPI utilizando Edge-TPU de Google en un Pixel 4:

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

El resultado esperado es una lista de métricas Top-K del 1 al 10:

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### Evaluación agnóstica de tareas

Para tareas en las que no existe una herramienta de evaluación establecida en el dispositivo, o si está experimentando con modelos personalizados, TensorFlow Lite cuenta con la herramienta [Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) ([aquí](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff) la arquitectura binaria ARM de 64 bits, Android).

Inference Diff compara la ejecución de TensorFlow Lite (en términos de latencia y desviaciones del valor de salida) en dos configuraciones:

- Inferencia de CPU de un solo hilo
- Inferencia definida por el usuario: definida por [estos parámetros](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)

Para ello, la herramienta genera datos gaussianos aleatorios y los hace pasar por dos intérpretes TFLite: uno que ejecuta kernels de CPU de un solo hilo y otro parametrizado por los argumentos del usuario.

Mide la latencia de ambos, así como la diferencia absoluta entre los tensores de salida de cada intérprete, sobre una base por elemento.

Para un modelo con un único tensor de salida, la salida podría tener este aspecto:

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

Lo que esto significa es que para el tensor de salida en el índice `0`, los elementos de la salida de la CPU difieren de la salida del delegado en una media de `1.96e-05`.

Tenga en cuenta que la interpretación de estos números requiere un conocimiento más profundo del modelo y de lo que significa cada tensor de salida. Si se trata de una simple regresión que determina algún tipo de puntuación o de incorporación, la diferencia debería ser baja (de lo contrario, es un error con el delegado). Sin embargo, salidas como la de "clase de detección" de los modelos SSD son un poco más difíciles de interpretar. Por ejemplo, puede mostrar una diferencia usando esta herramienta, pero eso puede no significar que haya algo realmente mal con el delegado: considere dos clases (falsas): "TV (ID: 10)", "Monitor (ID:20)". - Si un delegado está ligeramente desviado de la regla de oro y muestra monitor en lugar de TV, la diferencia de salida para este tensor podría ser algo tan alto como 20-10 = 10.
