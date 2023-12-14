# Generar builds de TensorFlow Lite para Android

Este documento describe cómo generar la librería de TensorFlow Lite para Android por su cuenta. Normalmente, no necesita generar localmente la librería TensorFlow Lite para Android. Si sólo desea usarla, consulte el [Inicio rápido de Android](../android/quickstart.md) para más detalles sobre cómo usarlas en sus proyectos Android.

## Usar instantáneas nocturnas

Para usar instantáneas nocturnas, añada el siguiente repositorio a su configuración raíz de generación de builds de Gradle.

```build
allprojects {
    repositories {      // should be already there
        mavenCentral()  // should be already there
        maven {         // add this repo to use snapshots
          name 'ossrh-snapshot'
          url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
    }
}
```

## Generar builds de TensorFlow Lite localmente

En algunos casos, es posible que desee usar un build local de TensorFlow Lite. Por ejemplo, puede que esté construyendo un binario personalizado que incluya [operaciones seleccionadas de TensorFlow](https://www.tensorflow.org/lite/guide/ops_select), o puede que desee realizar cambios locales en TensorFlow Lite.

### Configurar el ambiente de generación de builds usando Docker

- Descargue el archivo Docker. Al descargar el archivo Docker, acepta las siguientes condiciones de servicio que rigen su uso:

*Haciendo clic para aceptar, usted acepta que todo uso de Android Studio y del Android Native Development Kit se regirá por el Acuerdo de licencia del kit de desarrollo de software de Android disponible en https://developer.android.com/studio/terms (dicha URL puede ser actualizada o modificada por Google ocasionalmente).*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} Puede descargar el archivo Docker <a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/tflite-android.Dockerfile">aquí</a> {% dynamic else %} Debe aceptar las condiciones del servicio para descargar el archivo. <button class="button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">Acknowledge</button> {% dynamic endif %}

<!-- mdformat on -->

- Puede cambiar opcionalmente la versión de Android SDK o NDK. Coloque el archivo Docker descargado en una carpeta vacía y genere su imagen de docker ejecutando:

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- Inicie el contenedor docker de forma interactiva montando su carpeta actual en /host_dir dentro del contenedor (tenga en cuenta que /tensorflow_src es el repositorio de TensorFlow dentro del contenedor):

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

Si usa PowerShell en Windows, sustituya "$PWD" por "pwd".

Si desea usar un repositorio TensorFlow en el host, monte ese directorio en su lugar (-v hostDir:/host_dir).

- Una vez dentro del contenedor, puede ejecutar lo siguiente para descargar herramientas y librerías adicionales de Android (nótese que puede ser necesario aceptar la licencia):

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

Ahora, debe pasar a la sección [Configurar WORKSPACE y .bazelrc](#configure_workspace_and_bazelrc) para establecer los ajustes de generación de builds.

Cuando termine de generar las librerías, puede copiarlas en /host_dir dentro del contenedor para poder acceder a ellas en el host.

### Configurar el ambiente de generación de builds sin Docker

#### Instalar los requisitos previos de Bazel y Android

Bazel es el principal sistema de generación para TensorFlow. Para generar con él, debe tenerlo junto con el NDK y el SDK de Android instalados en su sistema.

1. Instale la última versión del sistema de generación [Bazel](https://bazel.build/versions/master/docs/install.html).
2. Se requiere el Android NDK para generar el código nativo (C/C++) de TensorFlow Lite. La versión actual recomendada es la 21e, que puede encontrarse [aquí](https://developer.android.com/ndk/downloads/older_releases.html#ndk-21e-downloads).
3. El SDK de Android y las herramientas de generación pueden obtenerse [aquí](https://developer.android.com/tools/revisions/build-tools.html), o alternativamente como parte de [Android Studio](https://developer.android.com/studio/index.html). La API de herramientas de generación que sea de versión igual o mayor que la 23 es la  recomendada para la generación de TensorFlow Lite.

### Configure WORKSPACE y .bazelrc

Este es un paso de configuración único que se requiere para generar las librerías TF Lite. Ejecute el script `./configure` en el directorio raíz de comprobación de TensorFlow y responda "Sí" cuando el script le pida configurar interactivamente el `./WORKSPACE` para los las compilaciones de Android. El script intentará configurar los ajustes usando las siguientes variables de entorno:

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

Si estas variables no están establecidas, deben proporcionarse de forma interactiva en el prompt del script. Una configuración correcta debería producir entradas similares a las siguientes en el archivo `.tf_configure.bazelrc` de la carpeta raíz:

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r21e"
build --action_env ANDROID_NDK_API_LEVEL="26"
build --action_env ANDROID_BUILD_TOOLS_VERSION="30.0.3"
build --action_env ANDROID_SDK_API_LEVEL="30"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### Generar e instalar

Una vez que Bazel esté correctamente configurado, puede generar el AAR de TensorFlow Lite desde el directorio raíz de comprobación de la siguiente manera:

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --define=android_dexmerger_tool=d8_dexmerger \
  --define=android_incremental_dexing_tool=d8_dexbuilder \
  //tensorflow/lite/java:tensorflow-lite
```

Esto generará un archivo AAR en `bazel-bin/tensorflow/lite/java/`. Tenga en cuenta que esto construye un AAR "grueso" con varias arquitecturas diferentes; si no las necesita todas, use el subconjunto apropiado para su entorno de implementación.

Puede crear archivos AAR más pequeños cuyo objetivo sea sólo un conjunto de modelos de la siguiente manera:

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

El script anterior generará el archivo `tensorflow-lite.aar` y opcionalmente el archivo `tensorflow-lite-select-tf-ops.aar` si uno de los modelos está usando ops de Tensorflow. Para más detalles, consulte la sección [Reducir el tamaño binario de TensorFlow Lite](../guide/reduce_binary_size.md).

#### Añadir AAR directamente al proyecto

Mueva el archivo `tensorflow-lite.aar` a un directorio llamado `libs` en su proyecto. Modifique el archivo `build.gradle` de su app para que haga referencia al nuevo directorio y sustituya la dependencia existente de TensorFlow Lite por la nueva librería local, por ejemplo:

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    compile(name:'tensorflow-lite', ext:'aar')
}
```

#### Instalar AAR en el repositorio local de Maven

Ejecute el siguiente comando desde su directorio raíz de comprobación:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

En el archivo `build.gradle` de su app, asegúrese de que tiene la dependencia `mavenLocal()` y sustituya la dependencia estándar de TensorFlow Lite por la que tiene soporte para determinadas ops de TensorFlow:

```
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
}
```

Tenga en cuenta que la versión `0.1.100` aquí es puramente para pruebas/desarrollo. Con el AAR local instalado, puede usar las [APIs de inferencia Java TensorFlow Lite](../guide/inference.md) estándar en el código de su app.
