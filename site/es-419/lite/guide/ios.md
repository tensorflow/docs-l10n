# Inicio rápido en iOS

Para empezar a utilizar TensorFlow Lite en iOS, le recomendamos que explore el siguiente ejemplo:

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Ejemplo de clasificación de imágenes en iOS</a>

Para una explicación del código fuente, debería leer también [Clasificación de imágenes en iOS con TensorFlow Lite](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/README.md).

Esta app de ejemplo usa la [clasificación de imágenes](https://www.tensorflow.org/lite/examples/image_classification/overview) para clasificar continuamente lo que ve desde la cámara trasera del dispositivo, mostrando las clasificaciones más probables. Permite al usuario seleccionar entre un modelo de punto flotante o [cuantizado](https://www.tensorflow.org/lite/performance/post_training_quantization) y seleccionar el número de hilos sobre los que realizar la inferencia.

Nota: En [Ejemplos](https://www.tensorflow.org/lite/examples) encontrará otras aplicaciones en iOS que demuestran TensorFlow Lite en diversos casos de uso.

## Añadir TensorFlow Lite a su proyecto Swift u Objective-C

TensorFlow Lite ofrece librerías nativas para iOS escritas en [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) y [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). Empiece a escribir su propio código iOS usando el ejemplo de clasificación de imágenes [Swift](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) como punto de partida.

Las secciones siguientes muestran cómo añadir Swift u Objective-C de TensorFlow Lite a su proyecto:

### Desarrolladores en CocoaPods

En su `Podfile`, añada el pod de TensorFlow Lite. A continuación, ejecute `pod install`.

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### Especificación de versiones

Hay versiones estables y versiones nocturnas disponibles para los pods `TensorFlowLiteSwift` y `TensorFlowLiteObjC`. Si no especifica una restricción de versión como en los ejemplos anteriores, CocoaPods tomará la última versión estable predeterminada.

También puede especificar una restricción de versión. Por ejemplo, si desea depender de la versión 2.10.0, puede escribir la dependencia como:

```ruby
pod 'TensorFlowLiteSwift', '~> 2.10.0'
```

Esto asegurará que la última versión disponible 2.x.y del pod `TensorFlowLiteSwift` sea usada en su app. Alternativamente, si desea depender de las compilaciones nocturnas, puede escribir:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

A partir de la versión 2.4.0 y las últimas versiones nocturnas, de forma predeterminada los delegados [GPU](https://www.tensorflow.org/lite/performance/gpu) y [Core ML](https://www.tensorflow.org/lite/performance/coreml_delegate) se excluyen del pod para reducir el tamaño del binario. Puede incluirlos especificando subspec:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML', 'Metal']
```

Esto le permitirá usar las últimas características añadidas a TensorFlow Lite. Tenga en cuenta que una vez creado el archivo `Podfile.lock` cuando ejecute el comando `pod install` por primera vez, la versión de la librería nocturna quedará bloqueada a la versión de la fecha actual. Si desea actualizar la librería nocturna a la más reciente, deberá ejecutar el comando `pod update`.

Para más información sobre las distintas formas de especificar restricciones de versión, consulte [Especificación de versiones de pods](https://guides.cocoapods.org/using/the-podfile.html#specifying-pod-versions).

### Desarrolladores en Bazel

En su archivo `BUILD`, añada la dependencia `TensorFlowLite` a su destino.

#### Swift

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

#### Objective-C

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

#### API de C/C++

Como alternativa, puede usar la [API de C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) o la [API de C++](https://tensorflow.org/lite/api_docs/cc).

```python
# Using C API directly
objc_library(
  deps = [
      "//tensorflow/lite/c:c_api",
  ],
)

# Using C++ API directly
objc_library(
  deps = [
      "//tensorflow/lite:framework",
  ],
)
```

### Importar la librería

Para los archivos Swift, importe el módulo TensorFlow Lite:

```swift
import TensorFlowLite
```

Para los archivos Objective-C, importe la cabecera global:

```objectivec
#import "TFLTensorFlowLite.h"
```

O bien, el módulo si configuró `CLANG_ENABLE_MODULES = YES` en su proyecto Xcode:

```objectivec
@import TFLTensorFlowLite;
```

Nota: Los desarrolladores en CocoaPods que deseen importar el módulo Objective-C de TensorFlow Lite, también deben incluir `use_frameworks!` en su `Podfile`.
