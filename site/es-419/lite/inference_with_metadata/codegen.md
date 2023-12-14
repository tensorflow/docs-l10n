# Generar interfaces de modelos usando metadatos

Al usar [Metadatos de TensorFlow Lite](../models/convert/metadata), los desarrolladores pueden generar código contenedor para permitir la integración en Android. Para la mayoría de los desarrolladores, la interfaz gráfica de [Android Studio ML Model Binding](#mlbinding) es la más fácil de usar. Si necesita más personalización o está usando herramientas de línea de comandos, también está disponible [Codegen de TensorFlow Lite](#codegen).

## Usar Android Studio ML Model Binding {:#mlbinding}

Para los modelos TensorFlow Lite mejorados con [metadatos](../models/convert/metadata.md), los desarrolladores pueden usar Android Studio ML Model Binding para configurar automáticamente los ajustes del proyecto y generar clases contenedoras basadas en los metadatos del modelo. El código del contenedor elimina la necesidad de interactuar directamente con `ByteBuffer`. En su lugar, los desarrolladores pueden interactuar con el modelo TensorFlow Lite con objetos tipados como `Bitmap` y `Rect`.

Nota: Se requiere [Android Studio 4.1](https://developer.android.com/studio) o superior

### Importar un modelo TensorFlow Lite en Android Studio

1. Haga clic derecho en el módulo en el que desea usar el modelo TFLite o haga clic en `File` y, a continuación, en `New` &gt; `Other` &gt; `TensorFlow Lite Model` ![Haga clic derecho en los menús para acceder a la funcionalidad de importación de TensorFlow Lite](../images/android/right_click_menu.png)

2. Seleccione la ubicación de su archivo TFLite. Tenga en cuenta que la herramienta configurará la dependencia del módulo por usted con la vinculación ML Model y todas las dependencias se insertarán automáticamente en el archivo `build.gradle` de su módulo Android.

    Opcional: Seleccione la segunda casilla de verificación para importar TensorFlow GPU si desea usar Aceleración de GPU. ![Diálogo de importación para TFLite model](../images/android/import_dialog.png)

3. Haga clic en `Finish`.

4. La siguiente pantalla aparecerá después de que la importación se haya realizado correctamente. Para empezar a usar el modelo, seleccione Kotlin o Java, copie y pegue el código bajo la sección `Sample Code`. Puede volver a esta pantalla haciendo doble clic en el modelo TFLite bajo el directorio `ml` en Android Studio. ![Página de detalles del modelo en Android Studio](../images/android/model_details.png)

### Aceleración de la inferencia del modelo {:#acceleration}

ML Model Binding ofrece a los desarrolladores una forma de acelerar su código usando delegados y el número de hilos.

Nota: El intérprete de TensorFlow Lite debe crearse en el mismo hilo que cuando se ejecuta. De lo contrario, puede ocurrir un error: TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.

Paso 1. Verifique que el archivo `build.gradle` del módulo contenga la siguiente dependencia:

```java
    dependencies {
        ...
        // TFLite GPU delegate 2.3.0 or above is required.
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
    }
```

Paso 2. Detecte si la GPU que se ejecuta en el dispositivo es compatible con el delegado GPU de TensorFlow, si no, ejecute el modelo usando múltiples hilos de CPU:

<div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = if(compatList.isDelegateSupportedOnThisDevice) {
        // if the device has a supported GPU, add the GPU delegate
        Model.Options.Builder().setDevice(Model.Device.GPU).build()
    } else {
        // if the GPU is not supported, run on 4 threads
        Model.Options.Builder().setNumThreads(4).build()
    }

    // Initialize the model as usual feeding in the options object
    val myModel = MyModel.newInstance(context, options)

    // Run inference per sample code
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">
    import org.tensorflow.lite.support.model.Model
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Model.Options options;
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        options = Model.Options.Builder().setDevice(Model.Device.GPU).build();
    } else {
        // if the GPU is not supported, run on 4 threads
        options = Model.Options.Builder().setNumThreads(4).build();
    }

    MyModel myModel = new MyModel.newInstance(context, options);

    // Run inference per sample code
      </pre>
    </section>
    </devsite-selector>
</div>

## Generar interfaces de modelos con el generador de código de TensorFlow Lite {:#codegen}

Nota: El generador de código contenedor de TensorFlow Lite actualmente sólo es compatible con Android.

Para el modelo TensorFlow Lite mejorado con [metadatos](../models/convert/metadata.md), los desarrolladores pueden usar el generador de código envolvente TensorFlow Lite Android para crear código envolvente específico de la plataforma. El código contenedor elimina la necesidad de interactuar directamente con `ByteBuffer`. En su lugar, los desarrolladores pueden interactuar con el modelo de TensorFlow Lite con objetos tipados como `Bitmap` y `Rect`.

La utilidad del generador de código depende de lo completa que esté la entrada de metadatos del modelo TensorFlow Lite. Consulte la sección `<Codegen usage>` bajo los campos relevantes en [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs), para ver cómo la herramienta codegen analiza cada campo.

### Generar código contenedor

Deberá instalar las siguientes herramientas en su terminal:

```sh
pip install tflite-support
```

Una vez completado, el generador de código puede usarse con la siguiente sintaxis:

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

El código resultante se encontrará en el directorio de destino. Si está usando [Google Colab](https://colab.research.google.com/) u otro entorno remoto, quizá sea más fácil comprimir el resultado en un archivo zip y descargarlo a su proyecto de Android Studio:

```python
# Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
```

### Usar el código generado

#### Paso 1: Importar el código generado

Descomprima el código generado si es necesario en una estructura de directorios. Se supone que la raíz del código generado es `SRC_ROOT`.

Abra el proyecto de Android Studio en el que desea usar el modelo TensorFlow lite e importe el módulo generado mediante: And File -&gt; New -&gt; Import Module -&gt; seleccione `SRC_ROOT`

Usando el ejemplo anterior, el directorio y el módulo importado se llamarían `classify_wrapper`.

#### Paso 2: Actualice el archivo `build.gradle` de la app.

En el módulo de la app que consumirá el módulo de librería generado:

En la sección android, añada lo siguiente:

```build
aaptOptions {
   noCompress "tflite"
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.

En la sección dependencies, añada lo siguiente:

```build
implementation project(":classify_wrapper")
```

#### Paso 3: Usar el modelo

```java
// 1. Initialize the model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Set the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Run the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieve the result
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

### Acelerar la inferencia de modelos

El código generado permite a los desarrolladores acelerar su código mediante el uso de [delegados](../performance/delegates.md) y el número de hilos. Estos pueden configurarse al inicializar el objeto modelo, ya que toma tres parámetros:

- **`Context`**: Contexto de la Activity o Servicio Android
- (Opcional) **`Device`**: Delegado de aceleración TFLite, por ejemplo GPUDelegate o NNAPIDelegate
- (Opcional) **`numThreads`**: Número de hilos usados para ejecutar el modelo - el valor predeterminado es uno.

Por ejemplo, para usar un delegado NNAPI y hasta tres hilos, puede inicializar el modelo así:

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### Solución de problemas

Si obtiene un error 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed', inserte las siguientes líneas bajo la sección android del módulo de la app que usará el módulo de librería:

```build
aaptOptions {
   noCompress "tflite"
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.
