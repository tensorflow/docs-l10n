# Operadores seleccionados de TensorFlow

Dado que la librería de operadores incorporada en TensorFlow Lite sólo soporta un número limitado de operadores TensorFlow, no todos los modelos son convertibles. Si desea más detalles, consulte [compatibilidad de operadores](ops_compatibility.md).

Para permitir la conversión, los usuarios pueden habilitar el uso de [ciertas ops](op_select_allowlist.md) de TensorFlow en su modelo TensorFlow Lite. Sin embargo, la ejecución de modelos TensorFlow Lite con ops TensorFlow requiere hacer pull in del runtime básico de TensorFlow, lo que aumenta el tamaño binario del intérprete de TensorFlow Lite. Para Android, puede evitarse esto generando selectivamente sólo las ops TensorFlow necesarias. Para más detalles, consulte [reducir el tamaño del binario](../guide/reduce_binary_size.md).

Este documento describe cómo [convertir](#convert_a_model) y [ejecutar](#run_inference) un modelo TensorFlow Lite que contenga ops de TensorFlow en una plataforma de su elección. También analiza [las métricas de rendimiento y tamaño](#metrics) y [las limitaciones conocidas](#known_limitations).

## Convertir un modelo

El siguiente ejemplo muestra cómo generar un modelo TensorFlow Lite con ops TensorFlow seleccionadas.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## Ejecutar inferencia

Cuando se usa un modelo TensorFlow Lite que haya sido convertido con soporte para determinadas ops TensorFlow, el cliente debe usar también un runtime TensorFlow Lite que incluya la librería necesaria de ops TensorFlow.

### AAR de Android

Para reducir el tamaño del binario, genere sus propios archivos AAR como se indica en la sección [siguiente](#building-the-android-aar). Si el tamaño del binario no es una preocupación considerable, le recomendamos usar el [AAR precompilado con las ops de TensorFlow alojado en MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-select-tf-ops).

Puede especificarlo en sus dependencias `build.gradle` añadiéndolo junto al AAR estándar de TensorFlow Lite de la siguiente manera:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

Para usar instantáneas nocturnas, asegúrese de que ha añadido el [repositorio de instantáneas Sonatype](../android/lite_build.md#use_nightly_snapshots).

Una vez que haya añadido la dependencia, el delegado necesario para manejar las ops TensorFlow del grafo debería instalarse automáticamente para los grafos que las requieran.

*Nota*: La dependencia de las ops de TensorFlow es relativamente grande, por lo que probablemente querrá filtrar las ABIs x86 innecesarias en su archivo `.gradle` configurando sus `abiFilters`.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### Generar el AAR de Android

Para reducir el tamaño del binario u otros casos avanzados, también puede generar la librería manualmente. Suponiendo un [entorno de compilación de trabajo de TensorFlow Lite](../android/quickstart.md), genere el AAR de Android con las ops de TensorFlow seleccionadas de la siguiente manera:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

Esto generará el archivo AAR `bazel-bin/tmp/tensorflow-lite.aar` para las ops incorporadas y personalizadas de TensorFlow Lite; y generará el archivo AAR `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` para las ops de TensorFlow. Si no dispone de un entorno de compilación que funcione, también puede [compilar los archivos anteriores con docker](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker).

A partir de ahí, puede importar los archivos AAR directamente a su proyecto o publicar los archivos AAR personalizados en su repositorio local de Maven:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

Finalmente, en el archivo `build.gradle` de su app, asegúrese de que tiene la dependencia `mavenLocal()` y sustituya la dependencia estándar de TensorFlow Lite por la que tiene soporte para determinadas ops de TensorFlow:

```build
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### Usando CocoaPods

TensorFlow Lite proporciona CocoaPods TF ops seleccionados preconstruidos nocturnos para `arm64`, de los que puede depender junto a los `TensorFlowLiteSwift` o CocoaPods `TensorFlowLiteObjC`.

*Nota*:Si necesita usar ops de TF seleccionados en un simulador `x86_64`, puede generar usted mismo el framework de ops seleccionados. Consulte la sección [Usando Bazel + Xcode](#using_bazel_xcode) para más detalles.

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

Después de ejecutar `pod install`, necesita dar un Indicador de enlazador adicional para forzar la carga del framework de ops de TF seleccionados en su proyecto. En su proyecto Xcode, vaya a `Build settings`-&gt; `Other Linker Flags`, y añada:

Para versiones &gt;= 2.9.0:

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.xcframework/ios-arm64/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

Para versiones &lt; 2.9.0:

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

A continuación, debería poder ejecutar cualquier modelo convertido con el `SELECT_TF_OPS` en su app para iOS. Por ejemplo, puede modificar la [app iOS de Clasificación de imágenes](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) para probar la función de ops de TF seleccionados.

- Reemplace el archivo modelo por el convertido con `SELECT_TF_OPS` activado.
- Añada la dependencia `TensorFlowLiteSelectTfOps` al `Podfile` siguiendo las instrucciones.
- Añada el Indicador del enlazador adicional como se describe más arriba.
- Ejecute la app de ejemplo y vea si el modelo funciona correctamente.

#### Usando Bazel + Xcode

TensorFlow Lite con ops de TensorFlow seleccionados para iOS puede ser generado usando Bazel. Primero, siga las [instrucciones de compilación de iOS](build_ios.md) para configurar correctamente su espacio de trabajo Bazel y el archivo `.bazelrc`.

Una vez que haya configurado el espacio de trabajo con la compatibilidad con iOS activada, puede usar el siguiente comando para construir el marco de trabajo complementario select TF ops, que puede añadirse sobre el marco de trabajo normal `TensorFlowLiteC.framework`. Tenga en cuenta que el framework select TF ops no puede generarse para la arquitectura `i386`, por lo que deberá proporcionar explícitamente la lista de arquitecturas destino excluyendo `i386`.

```sh
bazel build -c opt --config=ios --ios_multi_cpus=arm64,x86_64 \
  //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework
```

Esto generará el framework bajo el directorio `bazel-bin/tensorflow/lite/ios/`. Puede añadir este nuevo framework a su proyecto Xcode siguiendo pasos similares descritos en la sección [Ajustes del proyecto Xcode](./build_ios.md#modify_xcode_project_settings_directly) de la guía de compilación de iOS.

Tras añadir el framework al proyecto de su app, deberá especificar un indicador de enlace adicional en el proyecto de su app para forzar la carga del framework de ops TF seleccionados. En su proyecto Xcode, vaya a `Ajustes de compilación` -&gt; `Otros indicadores del enlazador`, y añada:

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C/C++

Si está usando Bazel o [CMake](https://www.tensorflow.org/lite/guide/build_cmake) para generar el intérprete de TensorFlow Lite, puede habilitar el delegado Flex enlazando una librería compartida del delegado Flex de TensorFlow Lite. Puede generarlo con Bazel con el siguiente comando.

```
bazel build -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
```

Este comando genera la siguiente librería compartida en `bazel-bin/tensorflow/lite/delegates/flex`.

Plataforma | Nombre de la librería
--- | ---
Linux | `libtensorflowlite_flex.so`
macOS | `libtensorflowlite_flex.dylib`
Windows | `tensorflowlite_flex.dll`

Tenga en cuenta que el `TfLiteDelegate` necesario se instalará automáticamente al crear el intérprete en runtime siempre que la librería compartida esté enlazada. No es necesario instalar explícitamente la instancia del delegado, como suele ser necesario con otros tipos de delegado.

**Nota:** Esta función está disponible desde la versión 2.7.

### Python

TensorFlow Lite con ops de TF seleccionados se instalará automáticamente con el [paquete pip de TensorFlow](https://www.tensorflow.org/install/pip). También puede elegir instalar únicamente el [paquete pip de intérprete de TensorFlow Lite](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter).

Nota: TensorFlow Lite con ops de TF seleccionados está disponible en la versión pip del paquete TensorFlow desde 2.3 para Linux y 2.4 para otros entornos.

## Métricas

### Rendimiento

Cuando se usa una mezcla tanto de ops de TensorFlow integradas como seleccionadas, todas las mismas optimizaciones de TensorFlow Lite y ops de TensorFlow integradas optimizadas estarán disponibles y se podrán usar con el modelo convertido.

La siguiente tabla describe el tiempo promedio que se tarda en ejecutar la inferencia en MobileNet en un Pixel 2. Los tiempos indicados son un promedio de 100 ejecuciones. Estos objetivos se generaron para Android usando los indicadores: `--config=android_arm64 -c opt`.

Compilación | Tiempo (milisegundos)
--- | ---
Sólo ops incorporadas (`TFLITE_BUILTIN`) | 260.7
Usar sólo ops TF (`SELECT_TF_OPS`) | 264.5

### Tamaño del binario

La siguiente tabla describe el tamaño del binario de TensorFlow Lite para cada compilación. Estos objetivos se generaron para Android usando `--config=android_arm -c opt`.

Compilación | Tamaño del binario de C++ | Tamaño del APK para Android
--- | --- | ---
Sólo ops integradas | 796 KB | 561 KB
Ops integradas + Ops TF | 23.0 MB | 8.0 MB
Ops integradas + Ops TF (1) | 4.1 MB | 1.8 MB

(1) Estas librerías fueron generadas selectivamente para el [modelo i3d-kinetics-400](https://tfhub.dev/deepmind/i3d-kinetics-400/1) con 8 ops incorporadas de TFLite y 3 ops de Tensorflow. Para más detalles, consulte la sección [Reducir el tamaño del binario de TensorFlow Lite](../guide/reduce_binary_size.md).

## Limitaciones conocidas

- Tipos no soportados: Ciertas ops de TensorFlow pueden no soportar el conjunto completo de tipos de entrada/salida que suelen estar disponibles en TensorFlow.

## Actualizaciones

- Versión 2.6
    - Se ha mejorado la compatibilidad con los operadores basados en atributos GraphDef y las inicializaciones de recursos HashTable.
- Versión 2.5
    - Puede aplicar una optimización conocida como [cuantización posterior al entrenamiento](../performance/post_training_quantization.md).
- Versión 2.4
    - Se ha mejorado la compatibilidad con los delegados acelerados por hardware
