# Herramientas de desarrollo para Android

TensorFlow Lite ofrece una serie de herramientas para integrar modelos en apps Android. Aquí se describen las herramientas de desarrollo que se pueden usar para generar apps con Kotlin, Java y C++, así como el soporte para el desarrollo de TensorFlow Lite en Android Studio.

Nota clave: Generalmente, debería usar la librería [TensorFlow Lite Task](#task_library) para integrar TensorFlow Lite en su app Android, a menos que su caso de uso no esté soportado por dicha librería. Si no está soportado por la librería de tareas, use la librería [TensorFlow Lite](#lite_lib) y la librería [Support](#support_lib).

Para empezar a escribir código Android rápidamente, consulte el [Inicio rápido para Android](../android/quickstart)

## Herramientas para generar builds con Kotlin y Java

Las siguientes secciones describen herramientas de desarrollo para TensorFlow Lite que usan los lenguajes Kotlin y Java.

### Librería de tareas TensorFlow Lite {:#task_library}

La librería de tareas de TensorFlow Lite contiene un conjunto de librerías específicas de tareas potentes y fáciles de usar para que los desarrolladores de apps generen builds con TensorFlow Lite. Facilita interfaces de modelo listas para usar optimizadas para tareas populares de aprendizaje automático, como clasificación de imágenes, preguntar y responder, etc. Las interfaces de modelo están diseñadas específicamente para cada tarea con el fin de lograr el mejor rendimiento y usabilidad. La librería de tareas funciona en varias plataformas y es compatible con Java y C++.

Para usar la librería de tareas en su app Android, use el AAR de MavenCentral para la [llibrería de tareas de visión](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) , la [librería de tareas de texto](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) y la [librería de tareas de audio](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio) , respectivamente.

Puede especificarlo en sus dependencias `build.gradle` de la siguiente manera:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:+'
    implementation 'org.tensorflow:tensorflow-lite-task-text:+'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:+'
}
```

Si usa instantáneas de cada noche, asegúrese de añadir el [repositorio de instantáneas Sonatype](./lite_build#use_nightly_snapshots) a su proyecto.

Consulte la introducción en la [Descripción general de la librería de tareas TensorFlow Lite](../inference_with_metadata/task_library/overview.md) para más detalles.

### Librería TensorFlow Lite {:#lite_lib}

Use la librería TensorFlow Lite en su app Android añadiendo el [AAR alojado en MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite) a su proyecto de desarrollo.

Puede especificarlo en sus dependencias `build.gradle` de la siguiente manera:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:+'
}
```

Si usa instantáneas de cada noche, asegúrese de añadir el [repositorio de instantáneas Sonatype](./lite_build#use_nightly_snapshots) a su proyecto.

Este AAR incluye binarios para todas las [ABIs de Android](https://developer.android.com/ndk/guides/abis). Puede reducir el tamaño del binario de su aplicación incluyendo sólo las ABIs que necesite apoyar.

A menos que se dirija a un hardware específico, debería omitir las ABI `x86`, `x86_64`, y `arm32`. ABI en la mayoría de los casos. Puede configurar esto con la siguiente configuración de Gradle. Incluye específicamente sólo `armeabi-v7a` y `arm64-v8a`, y debería cubrir la mayoría de los dispositivos Android modernos.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

Para obtener más información sobre `abiFilters`, consulte [ABIs de Android](https://developer.android.com/ndk/guides/abis) en la documentación de Android NDK.

### Librería de soporte de TensorFlow Lite {:#support_lib}

La librería de soporte de TensorFlow Lite para Android facilita la integración de modelos en su aplicación. Ofrece APIs de alto nivel que ayudan a transformar los datos de entrada brutos en la forma requerida por el modelo, y a interpretar la salida del modelo, reduciendo la cantidad de código repetitivo necesario.

Admite formatos de datos comunes para entradas y salidas, incluidas imágenes y arreglos. También proporciona unidades de pre y postprocesamiento que realizan tareas como el redimensionamiento y el recorte de imágenes.

Use la librería de soporte en su app Android incluyendo la [librería TensorFlow Lite Support Library AAR alojada en MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support).

Puede especificarlo en sus dependencias `build.gradle` de la siguiente manera:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:+'
}
```

Si usa instantáneas de cada noche, asegúrese de añadir el [repositorio de instantáneas Sonatype](./lite_build#use_nightly_snapshots) a su proyecto.

Para obtener instrucciones sobre cómo empezar, consulte la [Librería de soporte de TensorFlow Lite para Android](../inference_with_metadata/lite_support.md).

### Versiones mínimas del SDK de Android para las librerías

Librería | `minSdkVersion` | Requisitos de dispositivo
--- | --- | ---
tensorflow-lite | 19 | El uso de NNAPI requiere
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 u OpenCL
:                             :                 : (usualmente unicamente        : |  |
:                             :                 : disponible en API 21 o superior   : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : la API relacionada requiere   : |  |
:                             :                 : API 26 o superior                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -

### Uso de Android Studio

Además de las librerías de desarrollo descritas anteriormente, Android Studio también ofrece soporte para integrar modelos TensorFlow Lite, como se describe a continuación.

#### Vinculación de modelos ML de Android Studio

La función de vinculación de modelos ML de Android Studio 4.1 y posteriores le permite importar archivos de modelos `.tflite` en su app Android existente, y generar clases de interfaz para facilitar la integración de su código con un modelo.

Para importar un modelo TensorFlow Lite (TFLite):

1. Haga clic con el botón derecho del ratón en el módulo en el que desea usar el modelo TFLite o haga clic en **Archivo &gt; Nuevo &gt; Otro &gt; Modelo TensorFlow Lite**.

2. Elija la ubicación de su archivo TensorFlow Lite. Tenga en cuenta que la herramienta configura la dependencia del módulo con la vinculación ML Model y añade automáticamente todas las dependencias necesarias al archivo `build.gradle` de su módulo Android.

    Nota: Marque la segunda casilla de verificación para importar TensorFlow GPU si desea usar [Aceleración GPU](../performance/gpu).

3. Haga clic en `Finish` para iniciar el proceso de importación. Una vez finalizada la importación, la herramienta muestra una pantalla que describe el modelo, incluidos sus tensores de entrada y salida.

4. Para empezar a usar el modelo, seleccione Kotlin o Java, copie y pegue el código en la sección **Código de muestra**.

Puede volver a la pantalla de información del modelo haciendo doble clic en el modelo TensorFlow Lite bajo el directorio `ml` en Android Studio. Para más información sobre cómo usar la función de vinculación de modelos de Android Studio, consulte las [notas de la versión](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models) de Android Studio. Para una visión general del uso de la vinculación de modelos en Android Studio, consulte las [instrucciones](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) de ejemplo del código.

## Herramientas para generar builds con C y C++

Las librerías C y C++ para TensorFlow Lite están destinadas principalmente a los desarrolladores que usan el kit de desarrollo nativo de Android (NDK) para generar sus apps. Hay dos formas de usar TFLite a través de C++ si genera su app con el NDK:

### API en C de TFLite

El enfoque *recomendado* para los desarrolladores que usan el NDK es usar esta API. Descargue el archivo [TensorFlow Lite AAR alojado en MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite), cámbiele el nombre a `tensorflow-lite-*.zip`, y descomprímalo. Debe incluir los cuatro archivos de cabecera en las carpetas `headers/tensorflow/lite/` y `headers/tensorflow/lite/c/` y la librería dinámica `libtensorflowlite_jni.so` correspondiente en la carpeta `jni/` de su proyecto NDK.

El archivo de cabecera `c_api.h` contiene documentación básica sobre cómo usar la API en C de TFLite.

### API en C++ de TFLite

Si desea usar TFLite mediante la API de C++, puede generar las librerías compartidas de C++:

armeabi-v7a de 32 bits :

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

arm64-v8a de 64 bits:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

Actualmente, no existe una forma directa de extraer todos los archivos de cabecera necesarios, por lo que deberá incluir todos los archivos de cabecera en `tensorflow/lite/` del repositorio de TensorFlow. Además, necesitará los archivos de cabecera de [FlatBuffers](https://github.com/google/flatbuffers) y [Abseil](https://github.com/abseil/abseil-cpp).
