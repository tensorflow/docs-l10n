# Servicio de aceleración para Android (Beta)

Beta: El Servicio de Aceleración para Android está actualmente en Beta. Revise las secciones [Advertencias](#caveats) y [Términos y privacidad] (#terms_privacy) de esta página para obtener más detalles.

Usar procesadores especializados como GPUs, NPUs o DSPs para la aceleración por hardware puede mejorar drásticamente el rendimiento de la inferencia (hasta 10 veces más rápida en algunos casos) y la experiencia de usuario de su aplicación Android con ML. Sin embargo, ante la variedad de hardware y controladores que pueden tener sus usuarios, puede ser un reto elegir la configuración de aceleración por hardware óptima para el dispositivo de cada usuario. Además, habilitar una configuración incorrecta en un dispositivo puede crear una mala experiencia de usuario debido a una alta latencia o, en algunos casos poco frecuentes, errores en el runtime o problemas de precisión causados por incompatibilidades de hardware.

El servicio de aceleración para Android es una API que le ayuda a elegir la configuración de aceleración de hardware óptima para un determinado dispositivo de usuario y su modelo `.tflite`, minimizando al mismo tiempo el riesgo de errores en runtime o problemas de precisión.

El servicio de aceleración evalúa diferentes configuraciones de aceleración en los dispositivos de los usuarios mediante la ejecución de pruebas comparativas de inferencia interna con su modelo TensorFlow Lite. Estas pruebas suelen completarse en unos pocos segundos, según su modelo. Puede correr las pruebas comparativas una vez en cada dispositivo de usuario antes del momento de la inferencia, almacenar en caché el resultado y usarlo durante la inferencia. Estas pruebas comparativas se ejecutan fuera de proceso, lo que minimiza el riesgo de caídas de su app.

Indique su modelo, muestreo de datos y resultados esperados (entradas y salidas "de oro") y el servicio de aceleración ejecutará una prueba comparativa interna de inferencia TFLite para ofrecerle recomendaciones de hardware.

![imagen](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/lite/images/acceleration/acceleration_service.png?raw=true)

El servicio de aceleración forma parte de la pila ML personalizada de Android y funciona con [TensorFlow Lite en los servicios de Google Play](https://www.tensorflow.org/lite/android/play_services).

## Añada las dependencias a su proyecto

Añada las siguientes dependencias al archivo build.gradle de su aplicación:

```
implementation  "com.google.android.gms:play-services-tflite-
acceleration-service:16.0.0-beta01"
```

La API del servicio de aceleración funciona con [TensorFlow Lite en Google Play Services](https://www.tensorflow.org/lite/android/play_services). Si aún no está usando el runtime de TensorFlow Lite provisto a través de Play Services, deberá actualizar sus [dependencias](https://www.tensorflow.org/lite/android/play_services#1_add_project_dependencies_2).

## Cómo usar la API de servicio de aceleración

Para usar el servicio de aceleración, empiece por crear la configuración de aceleración que desea evaluar para su modelo (por ejemplo, GPU con OpenGL). Después cree una configuración de validación con su modelo, algunos datos de muestreo y la salida esperada del modelo. Finalmente llame a `validateConfig()` pasándole tanto su configuración de aceleración como la de validación.

![imagen](../images/acceleration/acceleration_service_steps.png)

### Crear configuraciones de aceleración

Las configuraciones de aceleración son representaciones de las configuraciones de hardware que se traducen en delegados durante el tiempo de ejecución. El servicio de aceleración usará entonces estas configuraciones internamente para realizar inferencias de prueba.

Por el momento, el servicio de aceleración permite evaluar configuraciones de GPU (convertidas en delegado de GPU durante el tiempo de ejecución) con el [GpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig) y la inferencia de CPU (con [CpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig)). Estamos trabajando para admitir más delegados para acceder a otro hardware en el futuro.

#### Configuración de la aceleración de GPU

Cree una configuración de aceleración de GPU como se indica a continuación:

```
AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder()
  .setEnableQuantizedInference(false)
  .build();
```

Debe especificar si su modelo usa o no la cuantización con [`setEnableQuantizedInference()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig.Builder#public-gpuaccelerationconfig.builder-setenablequantizedinference-boolean-value).

#### Configuración de la aceleración de CPU

Cree la aceleración de la CPU como se indica a continuación:

```
AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder()
  .setNumThreads(2)
  .build();
```

Utilice el método [`setNumThreads()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig.Builder#setNumThreads(int)) para definir el número de hilos que desea usar para evaluar la inferencia de la CPU.

### Crear configuraciones de validación

Las configuraciones de validación le permiten definir cómo desea que el servicio de aceleración evalúe las inferencias. Las usará para pasar:

- muestras de entrada,
- resultados esperados,
- lógica de validación de la precisión.

Asegúrese de dar muestras de entrada para las que espera un buen rendimiento de su modelo (también conocidas como muestras "de oro").

Cree una [`ValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidationConfig) con [`CustomValidationConfig.Builder`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder) como se indica a continuación:

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenOutputs(outputBuffer)
   .setAccuracyValidator(new MyCustomAccuracyValidator())
   .build();
```

Especifique el número de las muestras de oro con [`setBatchSize()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setBatchSize(int)). Pase las entradas de sus muestras de oro usando [`setGoldenInputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldeninputs-object...-value). Entregue la salida esperada para la entrada pasada con [`setGoldenOutputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldenoutputs-bytebuffer...-value).

Puede definir un tiempo máximo de inferencia con [`setInferenceTimeoutMillis()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setinferencetimeoutmillis-long-value) (5000 ms por default). Si la inferencia tarda más que el tiempo que ha definido, la configuración será rechazada.

Como opción, también puede crear un [`AccuracyValidator`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.AccuracyValidator) personalizado como se indica a continuación:

```
class MyCustomAccuracyValidator implements AccuracyValidator {
   boolean validate(
      BenchmarkResult benchmarkResult,
      ByteBuffer[] goldenOutput) {
        for (int i = 0; i < benchmarkResult.actualOutput().size(); i++) {
            if (!goldenOutputs[i]
               .equals(benchmarkResult.actualOutput().get(i).getValue())) {
               return false;
            }
         }
         return true;

   }
}
```

Asegúrese de definir una lógica de validación que funcione para su caso de uso.

Tenga en cuenta que si los datos de validación ya están incrustados en su modelo, puede usar [`EmbeddedValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/EmbeddedValidationConfig).

##### Generar salidas de validación

Las salidas doradas son opcionales y siempre que proporcione entradas de oro, el servicio de aceleración puede generar internamente las salidas de oro. También puede definir la configuración de aceleración utilizada para generar estas salidas de oro llamando a [`setGoldenConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setGoldenConfig(com.google.android.gms.tflite.acceleration.AccelerationConfig)):

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenConfig(customCpuAccelerationConfig)
   [...]
   .build();
```

### Validar la configuración de la aceleración

Una vez que haya creado una configuración de aceleración y otra de validación, podrá evaluarlas para su modelo.

Asegúrese con Play Services de que el runtime de TensorFlow Lite con Play Services está correctamente inicializado y de que el delegado de GPU está disponible para el dispositivo ejecutando:

```
TfLiteGpu.isGpuDelegateAvailable(context)
   .onSuccessTask(gpuAvailable -> TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(gpuAvailable)
        .build()
      )
   );
```

Instancie el [`AccelerationService`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService) llamando a [`AccelerationService.create()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#create(android.content.Context)).

A continuación, puede validar la configuración de aceleración para su modelo llamando a [`validateConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfig(com.google.android.gms.tflite.acceleration.Model,%20com.google.android.gms.tflite.acceleration.AccelerationConfig,%20com.google.android.gms.tflite.acceleration.ValidationConfig)):

```
InterpreterApi interpreter;
InterpreterOptions interpreterOptions = InterpreterApi.Options();
AccelerationService.create(context)
   .validateConfig(model, accelerationConfig, validationConfig)
   .addOnSuccessListener(validatedConfig -> {
      if (validatedConfig.isValid() && validatedConfig.benchmarkResult().hasPassedAccuracyTest()) {
         interpreterOptions.setAccelerationConfig(validatedConfig);
         interpreter = InterpreterApi.create(model, interpreterOptions);
});
```

También puede validar varias configuraciones llamando a [`validateConfigs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfigs(com.google.android.gms.tflite.acceleration.Model,%20java.lang.Iterable%3Ccom.google.android.gms.tflite.acceleration.AccelerationConfig%3E,%20com.google.android.gms.tflite.acceleration.ValidationConfig)) y pasando un objeto `Iterable<AccelerationConfig>` como parámetro.

`validateConfig()`devolverá un `Task<`[`ValidatedAccelerationConfigResult`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidatedAccelerationConfigResult)`>` de los servicios de Google Play [Task Api](https://developers.google.com/android/guides/tasks) que permite realizar tareas asíncronas. <br> Para obtener el resultado de la llamada de validación, añada una retrollamada [`addOnSuccessListener()`](https://developers.google.com/android/reference/com/google/android/gms/tasks/OnSuccessListener).

#### Usar una configuración validada en su intérprete

Tras comprobar si el `ValidatedAccelerationConfigResult` devuelto en la retrollamada es válido, puede establecer la configuración validada como configuración de aceleración para su intérprete llamando a `interpreterOptions.setAccelerationConfig()`.

#### Almacenamiento en caché de la configuración

Es poco probable que la configuración de aceleración óptima para su modelo cambie en el dispositivo. Así que una vez que reciba una configuración de aceleración satisfactoria, debería almacenarla en el dispositivo y dejar que su aplicación la recupere y la use para crear sus `InterpreterOptions` durante las siguientes sesiones en lugar de ejecutar otra validación. Los métodos `serialize()` y `deserialize()` de `ValidatedAccelerationConfigResult` facilitan el proceso de almacenamiento y recuperación.

### Aplicación de muestra

Para revisar una integración in situ del servicio de aceleración, eche un vistazo a la [aplicación de muestra](https://github.com/tensorflow/examples/tree/master/lite/examples/acceleration_service/android_play_services).

## Limitaciones

El servicio de aceleración tiene actualmente las siguientes limitaciones:

- Por el momento sólo se admiten configuraciones de aceleración por CPU y GPU,
- Sólo es compatible con TensorFlow Lite en los servicios de Google Play y no se puede usar si está utilizando la versión incluida de TensorFlow Lite,
- No es compatible con la librería de tareas [TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview) ya que no puede inicializar directamente [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) con el objeto `ValidatedAccelerationConfigResult`.
- El SDK del servicio de aceleración sólo admite el nivel 22 de API y superiores.

## Advertencias {:#caveats}

Revise atentamente las siguientes advertencias, especialmente si tiene previsto usar este SDK en producción:

- Antes de salir de la versión Beta y lanzar la versión estable para la API del servicio de aceleración, publicaremos un nuevo SDK que puede tener algunas diferencias con la versión Beta actual. Para poder seguir usando el servicio de aceleración, deberá migrar a este nuevo SDK y enviar una actualización a su app en el momento oportuno. No hacerlo puede provocar averías, ya que es posible que el SDK Beta deje de ser compatible con los servicios de Google Play al cabo de un tiempo.

- No existe ninguna garantía de que una característica específica dentro de la API del servicio de aceleración o la API en su totalidad llegue a estar disponible de forma general. Puede seguir en Beta indefinidamente, cerrarse o combinarse con otras características en paquetes diseñados para audiencias específicas de desarrolladores. Puede que con el tiempo algunas funciones de la API del servicio de aceleración o la API en su totalidad pasen a estar disponibles de forma general, pero no existe un calendario fijo para ello.

## Términos y privacidad {:#terms_privacy}

#### Términos del servicio

Usar las API del servicio de aceleración está sujeto a las [Condiciones de servicio de las API de Google](https://developers.google.com/terms/).<br> Además, las API del servicio de aceleración se encuentran actualmente en fase beta y, como tal, al usarlo usted reconoce los posibles problemas descritos en la sección "Advertencias" anterior y reconoce que el servicio de aceleración puede no funcionar siempre según lo especificado.

#### Privacidad

Cuando usa las API del servicio de aceleración, el procesamiento de los datos de entrada (por ejemplo, imáxgenes, vídeo, texto) se produce completamente en el dispositivo, y **el servicio de aceleración no envía esos datos a los servidores de Google**. Resultará que podrá usar nuestras API para procesar datos de entrada que no deben salir del dispositivo.<br> Las API del servicio de aceleración pueden ponerse en contacto con los servidores de Google ocasionalmente para recibir información sobre correcciones de errores, modelos actualizados y compatibilidad del acelerador de hardware. Las API del servicio de aceleración también envían a Google métricas sobre el rendimiento y la utilización de las API de su app. Google usa estos datos métricos para medir el rendimiento, depurar, mantener y mejorar las API y detectar usos indebidos o abusos, tal y como se describe con más detalle en nuestra [Política de privacidad](https://policies.google.com/privacy).<br> **Usted es responsable de informar a los usuarios de su app sobre el procesamiento por parte de Google de los datos de las métricas del servicio de aceleración, tal y como exige la legislación aplicable.**<br> Entre los datos que recopilamos se incluyen los siguientes:

- Información del dispositivo (como el fabricante, el modelo, la versión del sistema operativo y la compilación) y los aceleradores de hardware de ML disponibles (GPU y DSP). Usado para diagnósticos y análisis de uso.
- Información de la app (nombre del paquete / id de paquete, versión de la app). Usado para diagnósticos y análisis de uso.
- Configuración de la API (como el formato de imagen y la resolución). Usado para diagnósticos y análisis de uso.
- Tipo de evento (como inicializar, descargar modelo, actualizar, ejecutar, detección). Usado para diagnósticos y análisis de uso.
- Códigos de error. Usados para diagnósticos.
- Métricas de rendimiento. Usadas para diagnósticos.
- Identificadores por instalación que no identifican unívocamente a un usuario o dispositivo físico. Usados para el funcionamiento de la configuración remota y los análisis de uso.
- Direcciones IP del remitente de la solicitud de red. Se usan para diagnósticos de configuración remota. Las direcciones IP recopiladas se conservan temporalmente.

## Soporte y retroalimentación

Puede dar retroalimentación y obtener soporte a través del TensorFlow Issue Tracker. Le rogamos que notifique los problemas y las solicitudes de asistencia utilizando la [plantilla de problemas](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) para TensorFlow Lite en los servicios de Google Play.
