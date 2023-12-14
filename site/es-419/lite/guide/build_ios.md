# Generar builds de TensorFlow Lite para iOS

Este documento describe cómo generar la librería de TensorFlow Lite para iOS por su cuenta. Normalmente, no necesita generar localmente la librería TensorFlow Lite para iOS. Si sólo desea usarla, la forma más sencilla es usar las versiones estables o nocturnas precompiladas de los CocoaPods de TensorFlow Lite. Consulte el [Inicio rápido de iOS](ios.md) para más detalles sobre cómo usarlas en sus proyectos iOS.

## Generación local

En algunos casos, es posible que desee usar una compilación local de TensorFlow Lite, por ejemplo, cuando desee realizar cambios locales en TensorFlow Lite y analizar esos cambios en su app iOS o prefiera usar un framework estático en lugar del dinámico que le proporcionamos. Para crear un framework universal iOS para TensorFlow Lite localmente, necesita construirlo usando Bazel en una máquina macOS.

### Instale Xcode

Si aún no lo ha hecho, tendrá que instalar Xcode 8 o posterior y sus herramientas, usando `xcode-select`:

```sh
xcode-select --install
```

Si se trata de una nueva instalación, deberá aceptar el acuerdo de licencia para todos los usuarios con el siguiente comando:

```sh
sudo xcodebuild -license accept
```

### Instale Bazel

Bazel es el principal sistema de compilación para TensorFlow. Instale Bazel siguiendo las [instrucciones de la página web de Bazel](https://docs.bazel.build/versions/master/install-os-x.html). Asegúrese de elegir una versión entre `_TF_MIN_BAZEL_VERSION` y `_TF_MAX_BAZEL_VERSION` en el archivo [`configure.py`](https://github.com/tensorflow/tensorflow/blob/master/configure.py) en la raíz del repositorio `tensorflow`.

### Configure WORKSPACE y .bazelrc

Ejecute el script `./configure` en el directorio raíz de TensorFlow checkout, y responda "Sí" cuando el script le pregunte si desea generar TensorFlow con soporte para iOS.

### Genere el framework dinámico TensorFlowLiteC (recomendado)

Nota: Este paso no es necesario si (1) está usando Bazel para su app, o (2) sólo quiere analizar cambios locales en las API de Swift u Objective-C. En dichos casos, pase a la sección siguiente [Utilice en su propia aplicación](#use_in_your_own_application).

Una vez que Bazel esté correctamente configurado con soporte para iOS, puede generar el framework `TensorFlowLiteC` con el siguiente comando.

```sh
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

Este comando generará el archivo `TensorFlowLiteC_framework.zip` bajo el directorio `bazel-bin/tensorflow/lite/ios/` dentro de su directorio raíz de TensorFlow. De forma predeterminada, el framework generado contiene un binario "gordo", que contiene armv7, arm64 y x86_64 (pero no i386). Para ver la lista completa de indicadores de compilación utilizados cuando se especifica `--config=ios_fat`, consulte la sección de configuración de iOS en el archivo [`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc).

### Genere el framework estático TensorFlowLiteC

De forma predeterminada, sólo distribuimos el framework dinámico a través de Cocoapods. Si desea usar el framework estático en su lugar, puede generar el framework estático `TensorFlowLiteC` con el siguiente comando:

```
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

El comando generará un archivo llamado `TensorFlowLiteC_static_framework.zip` bajo el directorio `bazel-bin/tensorflow/lite/ios/` dentro de su directorio raíz de TensorFlow. Este framework estático puede usarse exactamente igual que el dinámico.

### Genere selectivamente frameworks TFLite

Puede generar frameworks más pequeños que tengan como objetivo sólo un conjunto de modelos utilizando la generación selectiva, que omitirá las operaciones no utilizadas en su conjunto de modelos y sólo incluirá los kernels op necesarios para ejecutar el conjunto de modelos dado. El comando es el siguiente:

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

El comando anterior generará el framework estático `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip` para las ops incorporadas y personalizadas de TensorFlow Lite; y opcionalmente, genera el framework estático `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteSelectTfOps_framework.zip` si sus modelos contienen ops Select TensorFlow. Tenga en cuenta que puede usar el indicador `--target_archs` para especificar sus arquitecturas de implementación.

## Utilice en su propia aplicación

### Desarrolladores en CocoaPods

Existen tres CocoaPods para TensorFlow Lite:

- `TensorFlowLiteSwift`: Brinda las API Swift para TensorFlow Lite.
- `TensorFlowLiteObjC`: Aporta las API de Objective-C para TensorFlow Lite.
- `TensorFlowLiteC`: Pod base común, que incorpora el core runtime de TensorFlow Lite y expone las APIs base en C usadas por los dos pods mencionados anteriormente. No está pensado para ser usado directamente por los usuarios.

Como desarrollador, debe elegir el pod `TensorFlowLiteSwift` o `TensorFlowLiteObjC` en función del lenguaje en el que esté escrita su app, pero no ambos. Los pasos exactos para usar compilación local de TensorFlow Lite difieren, dependiendo de la parte exacta que desee compilar.

#### Usar las API locales de Swift u Objective-C

Si está usando CocoaPods y sólo desea analizar algunos cambios locales en las APIs [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) o [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc) de TensorFlow Lite, siga estos pasos.

1. Realice cambios en las API de Swift u Objective-C en su checkout de `tensorflow`.

2. Abra el archivo `TensorFlowLite(Swift|ObjC).podspec` y actualice esta línea: <br> `s.dependency 'TensorFlowLiteC', "#{s.version}"` <br> para que sea <br> `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"` <br> Esto es para asegurar que está generando sus APIs Swift u Objective-C en base a la última versión nocturna disponible de las APIs `TensorFlowLiteC` (generadas cada noche entre 1-4 a. m. hora del Pacífico) en lugar de la versión estable, que puede ser obsoleta en comparación con su checkout local de `tensorflow`. Alternativamente, podría elegir publicar su propia versión de `TensorFlowLiteC` y usar esa versión (vea la sección [Usar el core local de TensorFlow Lite](#using_local_tensorflow_lite_core) más abajo).

3. En el `Podfile` de su proyecto iOS, cambie la dependencia de la siguiente manera para que apunte a la ruta local de su directorio raíz `tensorflow`. <br> Para Swift: <br> `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir}'` <br> Para Objective-C: <br> `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. Actualice la instalación de su pod desde el directorio raíz de su proyecto iOS. <br> `$ pod update`

5. Vuelva a abrir el espacio de trabajo generado (`<project>.xcworkspace`) y recompile su app dentro de Xcode.

#### Usar el core local de TensorFlow Lite

Puede configurar un repositorio privado de especificaciones de CocoaPods y publicar su framework personalizado `TensorFlowLiteC` en su repositorio privado. Puede copiar este archivo [podspec](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/ios/TensorFlowLiteC.podspec) y modificar algunos valores:

```ruby
  ...
  s.version      = <your_desired_version_tag>
  ...
  # Note the `///`, two from the `file://` and one from the `/path`.
  s.source       = { :http => "file:///path/to/TensorFlowLiteC_framework.zip" }
  ...
  s.vendored_frameworks = 'TensorFlowLiteC.framework'
  ...
```

Después de crear su propio archivo `TensorFlowLiteC.podspec`, puede seguir las [instrucciones sobre el uso de CocoaPods privados](https://guides.cocoapods.org/making/private-cocoapods.html) para usarlo en su propio proyecto. También puede modificar el `TensorFlowLite(Swift|ObjC).podspec` para que apunte a su pod personalizado `TensorFlowLiteC` y usar el pod Swift u Objective-C en su proyecto de app.

### Desarrolladores en Bazel

Si está usando Bazel como herramienta principal de generación, puede simplemente añadir la dependencia `TensorFlowLite` a su destino en su archivo `BUILD`.

Para Swift:

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

Para Objective-C:

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

Cuando usted genera el proyecto de su app, cualquier cambio en la librería TensorFlow Lite será captado e incorporado a su app.

### Modifique directamente los ajustes del proyecto Xcode

Es muy recomendable usar CocoaPods o Bazel para añadir la dependencia de TensorFlow Lite en su proyecto. Si aún desea añadir el framework `TensorFlowLiteC` manualmente, tendrá que añadir el framework `TensorFlowLiteC` como framework incorporado a su proyecto de aplicación. Descomprima el archivo `TensorFlowLiteC_framework.zip` generado a partir de la compilación anterior para obtener el directorio `TensorFlowLiteC.framework`. Este directorio es el framework real que Xcode puede entender.

Una vez que haya preparado el `TensorFlowLiteC.framework`, primero tiene que añadirlo como binario incrustado al objetivo de su app. La sección exacta de ajustes del proyecto para esto puede diferir según su versión de Xcode.

- Xcode 11: Vaya a la pestaña "General" del editor de proyectos para el destino de su app y añada `TensorFlowLiteC.framework` en la sección "Frameworks, bibliotecas y contenido integrado".
- Xcode 10 e inferiores: Vaya a la pestaña "General" del editor de proyectos para el destino de su app y añada el `TensorFlowLiteC.framework` en "Binarios integrados". El framework también debería añadirse automáticamente en la sección 'Frameworks y librerías vinculadas'.

Cuando añada el framework como un binario incrustado, Xcode también actualizará la entrada "Rutas de búsqueda del framework" en la pestaña "Ajustes de compilación" para incluir el directorio padre de su framework. En caso de que esto no ocurra automáticamente, deberá añadir manualmente el directorio padre del directorio `TensorFlowLiteC.framework`.

Una vez realizados estos dos ajustes, debería poder importar y llamar a la API en C de TensorFlow Lite, definida por los archivos de cabecera bajo el directorio `TensorFlowLiteC.framework/Headers`.
