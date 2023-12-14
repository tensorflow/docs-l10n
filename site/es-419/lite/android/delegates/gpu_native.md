# Delegado de aceleración de GPU con API C/C++

Usar unidades de procesamiento gráfico (GPU) para ejecutar sus modelos de aprendizaje automático (ML) puede mejorar drásticamente el rendimiento y la experiencia de usuario de sus aplicaciones habilitadas para ML. En los dispositivos Android, puede habilitar la ejecución acelerada por GPU de sus modelos usando un [*delegado*](../../performance/delegates) y una de las siguientes API:

- API del Intérprete: [guía](./gpu)
- API de librería de tareas: [guía](./gpu_task)
- API nativa (C/C++): esta guía

Esta guía cubre los usos avanzados del delegado de la GPU para la API de C, la API de C++ y el uso de modelos cuantizados. Para saber más sobre cómo usar el delegado de la GPU para TensorFlow Lite, incluidas las mejores prácticas y técnicas avanzadas, consulte la página [Delegados de la GPU](../../performance/gpu).

## Habilite la aceleración de la GPU

Use el delegado GPU de TensorFlow Lite para Android en C o C++ creando el delegado con `TfLiteGpuDelegateV2Create()` y destruyéndolo con `TfLiteGpuDelegateV2Delete()`, como se muestra en el siguiente código de ejemplo:

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

Revise el código del objeto `TfLiteGpuDelegateOptionsV2` para crear una instancia de delegado con opciones personalizadas. Puede inicializar las opciones predeterminadas con `TfLiteGpuDelegateOptionsV2Default()` y luego modificarlas según sea necesario.

El delegado GPU de TensorFlow Lite para Android en C o C++ usa el sistema de generación [Bazel](https://bazel.io). Puede generar el delegado usando el siguiente comando:

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

Cuando se llama a `Interpreter::ModifyGraphWithDelegate()` o `Interpreter::Invoke()`, quien llama debe tener un `EGLContext` en el hilo actual y `Interpreter::Invoke()` debe llamarse desde el mismo `EGLContext`. Si no existe un `EGLContext`, el delegado crea uno internamente, pero entonces debe asegurarse de que `Interpreter::Invoke()` se llama siempre desde el mismo hilo en el que se llamó a `Interpreter::ModifyGraphWithDelegate()`.

## Modelos cuantizados  {:#quantized-models}

Las librerías de delegado de GPU de Android admiten modelos cuantizados de forma predeterminada. No es necesario realizar ningún cambio en el código para usar modelos cuantizados con la GPU delegada. En la siguiente sección se explica cómo desactivar el soporte cuantizado para realizar pruebas o con fines experimentales.

#### Deshabilite el soporte de modelos cuantizados

El siguiente código muestra cómo ***deshabilitar*** el soporte para modelos cuantizados.

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

Para obtener más información sobre la ejecución de modelos cuantizados con aceleración de GPU, consulte la descripción general de [Delegado de GPU](../../performance/gpu#quantized-models).
