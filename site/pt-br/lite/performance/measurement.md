# Medição de desempenho

## Ferramentas de benchmarking

No momento, as ferramentas de benchmarking do TensorFlow Lite  medem e calculam estatísticas para as seguintes métricas de desempenho importantes:

- Tempo de inicialização
- Tempo de inferência do estado de warmup
- Tempo de inferência do estado estacionário
- Uso da memória durante o tempo de inicialização
- Uso geral da memória

As ferramentas de benchmarking estão disponíveis como aplicativos de benchmarking para Android e iOS e como binários de linha de comando nativos, e todas elas compartilham a mesma lógica de medição de desempenho principal. Observe que as opções e os formatos de saída disponíveis são um pouco diferentes devido às diferenças no ambiente de runtime.

### Aplicativo de benchmarking do Android

Há duas opções para usar a ferramenta de benchmarking com o Android. Uma é um [binário de benchmarking nativo](#native-benchmark-binary) e a outra é um aplicativo de benchmarking do Android, um medidor melhor do desempenho do modelo no aplicativo. De qualquer forma, os números da ferramenta de benchmarking ainda diferem um pouco de quando a inferência é executada com o modelo no próprio aplicativo.

Esse aplicativo de benchmarking do Android não tem interface do usuário. Instale e execute usando o comando `adb` e recupere os resultados usando o comando `adb logcat`.

#### Baixe ou compile o aplicativo

Baixe os aplicativos de benchmarking do Android pré-criados e noturnos usando os links abaixo:

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk)

Para os aplicativos de benchmarking do Android compatíveis com [ops do TF](https://www.tensorflow.org/lite/guide/ops_select) pelo [delegado Flex](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex), use os links abaixo:

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex.apk)

Você também pode compilar o aplicativo do código-fonte seguindo estas [instruções](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android).

Observação: é necessário compilar o aplicativo do código-fonte se você quiser executar o APK de benchmarking do Android na CPU x86 ou no delegado Hexagon ou se o modelo tiver [determinados operadores do TF](../guide/ops_select) ou [operadores personalizados](../guide/ops_custom).

#### Prepare o benchmarking

Antes de executar o aplicativo de benchmarking, instale o aplicativo e envie o arquivo do modelo ao dispositivo da seguinte maneira:

```shell
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```

#### Execute o benchmarking

```shell
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/your_model.tflite \
              --num_threads=4"'
```

`graph` é um parâmetro obrigatório.

- `graph`: `string` <br> O caminho para o arquivo do modelo do TFLite.

Você pode especificar mais parâmetros opcionais para executar o benchmarking.

- `num_threads`: `int` (default=1) <br> O número de threads que devem ser usados ao executar o interpretador do TFLite.
- `use_gpu`: `bool` (default=false) <br> Use o [delegado de GPU](gpu).
- `use_nnapi`: `bool` (default=false) <br> Use o [delegado NNAPI](nnapi).
- `use_xnnpack`: `bool` (default=`false`) <br> Use o [delegado XNNPACK](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack).
- `use_hexagon`: `bool` (default=`false`) <br> Use o [delegado Hexagon](hexagon_delegate).

Dependendo do dispositivo usado, algumas dessas opções podem não estar disponíveis ou não fazer nenhum efeito. Consulte mais [parâmetros](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters) desempenho de que você pode executar com o aplicativo de benchmarking.

Veja os resultados usando o comando `logcat`:

```shell
adb logcat | grep "Inference timings"
```

Os resultados de benchmarking são relatados assim:

```
... tflite  : Inference timings in us: Init: 5685, First inference: 18535, Warmup (avg): 14462.3, Inference (avg): 14575.2
```

### Biblioteca de benchmarking nativa

A ferramenta de benchmarking também é fornecida como um `benchmark_model` de binário nativo. Você pode executar essa ferramenta a partir de uma linha de comando de shell em Linux, Mac, dispositivos embarcados e dispositivos Android.

#### Baixe ou compile o binário

Baixe os binários de linha de comando nativos pré-criados e noturnos usando os links abaixo:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model)

Para os binários pré-criados e noturnos que são compatíveis com [ops do TF](https://www.tensorflow.org/lite/guide/ops_select) pelo [delegado Flex](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex), use os links abaixo:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_plus_flex)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)

Para fazer o benchmarking com o [delegado Hexagon do TensorFlow Lite](https://www.tensorflow.org/lite/android/delegates/hexagon), também pré-criamos os arquivos `libhexagon_interface.so` necessários (veja [aqui](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md) mais detalhes sobre esse arquivo). Depois de baixar o arquivo da plataforma correspondente pelos links abaixo, renomeie-o como `libhexagon_interface.so`.

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_libhexagon_interface.so)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_libhexagon_interface.so)

Você também pode compilar o binário de benchmarking nativo a partir do [código-fonte](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) no seu computador.

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Para compilar com a toolchain do Android NDK, você precisa configurar o ambiente de build primeiro ao seguir este [guia](../android/lite_build#set_up_build_environment_without_docker) ou usar a imagem docker conforme descrito neste [guia](../android/lite_build#set_up_build_environment_using_docker).

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

Observação: é uma abordagem válida enviar e executar binários diretamente em um dispositivo Android para benchmarking, mas pode resultar em diferenças sutis (mas observáveis) no desempenho relativo à execução dentro de um aplicativo Android real. Em especial, o agendador do Android ajusta o comportamento com base nas prioridades de thread e processo, que diferem de uma atividade/aplicativo em primeiro plano e um binário em segundo plano regular executado pelo `adb shell ...`. Esse comportamento ajustado é mais evidente ao permitir a execução de CPU de vários threads com o TensorFlow Lite. Por isso, o aplicativo de benchmarking do Android é recomendado para a medição de desempenho.

#### Execute o benchmarking

Para executar o benchmarking no seu computador, execute o binário no shell.

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

Você pode usar o mesmo conjunto de [parâmetros](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters) conforme mencionado acima com o binário de linha de comando nativo.

#### Análise de perfil das ops do modelo

O binário do modelo de benchmarking também permite que você analise as ops do modelo e obtenha os tempos de execução de cada operador. Para fazer isso, passe a flag `--enable_op_profiling=true` a `benchmark_model` durante a invocação. Os detalhes são explicados [aqui](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators).

### Binário de benchmarking nativo para várias opções de desempenho em uma única execução

Um binário C++ conveniente e simples também é fornecido para o [benchmarking de várias opções de desempenho](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#benchmark-multiple-performance-options-in-a-single-run) em uma única execução. Esse binário é criado com base na ferramenta de benchmarking mencionada acima que só pode fazer o benchmarking de uma única opção de desempenho por vez. Eles compartilham o mesmo processo de build/instalação/execução, mas o nome de destino do BUILD desse binário é `benchmark_model_performance_options` e exige alguns parâmetros adicionais. Um parâmetro importante para esse binário é:

`perf_options_list`: `string` (default='all') <br> Uma lista separada por vírgulas das opções de desempenho do TFLite para o benchmarking.

Você pode obter binários pré-criados noturnos para essa ferramenta conforme listado abaixo:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_performance_options)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_performance_options)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_performance_options)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_performance_options)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)

### Aplicativo de benchmarking do iOS

Para realizar o benchmarking em um dispositivo iOS, você precisa compilar o aplicativo do [código-fonte](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios). Coloque o arquivo do modelo do TensorFlow Lite no diretório [benchmark_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/benchmark_data) da árvore de código-fonte e modifique o arquivo `benchmark_params.json`. Esses arquivos são empacotados no aplicativo e o aplicativo lê os dados do diretório. Acesse o [aplicativo de benchmarking do iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios) para instruções detalhadas.

## Benchmarks de desempenho para modelos conhecidos

Esta seção lista benchmarks de desempenho do TensorFlow ao executar modelos conhecidos em alguns dispositivos Android e iOS.

### Benchmarks de desempenho do Android

Esses números de benchmarking de desempenho foram gerados com o [binário de benchmarking nativo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

Para benchmarks do Android, a afinidade de CPU é definida para usar big cores no dispositivo para reduzir a variância (veja mais [detalhes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android)).

Ela supõe que os modelos foram baixados e descompactados no diretório `/data/local/tmp/tflite_models`. O binário de benchmark é criado usando [estas instruções](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android) e se supõe que esteja no diretório `/data/local/tmp`.

Para realizar o benchmark:

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

Para executar com o delegado NNAPI, defina `--use_nnapi=true`. Para executar com o delegado de GPU, defina `--use_gpu=true`.

Os valores de desempenho abaixo são medidos no Android 10.

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Dispositivo</th>
      <th>CPU, 4 threads</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
</td>
    <td>Pixel 3</td>
    <td>23,9 ms</td>
    <td>6,45 ms</td>
    <td>13,8 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>14,0 ms</td>
    <td>9,0 ms</td>
    <td>14,8 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>Pixel 3</td>
    <td>13,4 ms</td>
    <td>---</td>
    <td>6,0 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>5,0 ms</td>
    <td>---</td>
    <td>3,2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
</td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>---</td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>34,5 ms</td>
    <td>---</td>
    <td>99,0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>Pixel 3</td>
    <td>35,8 ms</td>
    <td>9,5 ms</td>
    <td>18,5 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>23,9 ms</td>
    <td>11,1 ms</td>
    <td>19,0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>Pixel 3</td>
    <td>422 ms</td>
    <td>99,8 ms</td>
    <td>201 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>272,6 ms</td>
    <td>87,2 ms</td>
    <td>171,1 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>Pixel 3</td>
    <td>486 ms</td>
    <td>93 ms</td>
    <td>292 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>324,1 ms</td>
    <td>97,6 ms</td>
    <td>186,9 ms</td>
  </tr>
 </table>

### Benchmarks de desempenho do iOS

Esses números de benchmarking de desempenho foram gerados com o [aplicativo de benchmarking do iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios).

Para realizar os benchmarks do iOS, o aplicativo foi modificado para incluir o modelo apropriado e `benchmark_params.json` foi modificado para definir o `num_threads` como 2. Para usar o delegado de GPU, as opções `"use_gpu" : "1"` e `"gpu_wait_type" : "aggressive"` foram adicionadas a `benchmark_params.json`.

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Dispositivo</th>
      <th>CPU, 2 threads</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
</td>
    <td>iPhone XS</td>
    <td>14,8 ms</td>
    <td>3,4 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>iPhone XS</td>
    <td>11 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
</td>
    <td>iPhone XS</td>
    <td>30,4 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>iPhone XS</td>
    <td>21,1 ms</td>
    <td>15,5 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>iPhone XS</td>
    <td>261,1 ms</td>
    <td>45,7 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>iPhone XS</td>
    <td>309 ms</td>
    <td>54,4 ms</td>
  </tr>
 </table>

## Rastreie internals do TensorFlow Lite

### Rastreie internals do TensorFlow Lite no Android

Observação: esse recurso está disponível a partir do TensorFlow Lite v2.4.

Os eventos internos do interpretador do TensorFlow Lite de um aplicativo Android podem ser capturados por [ferramentas de tracing do Android](https://developer.android.com/topic/performance/tracing). Eles são os mesmos eventos com a API [Trace](https://developer.android.com/reference/android/os/Trace) do Android, então os eventos capturados do código Java/Kotlin são vistos juntos com os eventos internos do TensorFlow Lite.

Alguns exemplos de eventos são:

- Invocação de operador
- Modificação de grafo por delegado
- Alocação de tensor

Entre as diferentes opções de captura de traces, este guia aborda o Android Studio CPU Profiler e o aplicativo System Tracing. Consulte a [ferramenta de linha de comando Perfetto](https://developer.android.com/studio/command-line/perfetto) ou a [ferramenta de linha de comando Systrace](https://developer.android.com/topic/performance/tracing/command-line) para mais opções.

#### Adicionando eventos de trace em código Java

Este é uma amostra de código do aplicativo de exemplo de [Classificação de imagens](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android). O interpretador do TensorFlow Lite é executado na seção `recognizeImage/runInference`. Essa etapa é opcional, mas é útil para ajudar a perceber onde é realizada a chamada de inferência.

```java
  Trace.beginSection("recognizeImage");
  ...
  // Runs the inference call.
  Trace.beginSection("runInference");
  tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
  Trace.endSection();
  ...
  Trace.endSection();

```

#### Ative o tracing do TensorFlow Lite

Para ativar o tracing do TensorFlow Lite, defina a propriedade `debug.tflite.trace` do sistema Android como 1 <br> antes de iniciar o aplicativo Android.

```shell
adb shell setprop debug.tflite.trace 1
```

Se essa propriedade for definida com o interpretador do TensorFlow inicializado, serão rastreados os principais eventos (por exemplo, invocação de operador) do interpretador.

Depois de capturar todos os traces, desative o tracing ao definir o valor da propriedade como 0.

```shell
adb shell setprop debug.tflite.trace 0
```

#### Android Studio CPU Profiler

Capture traces com o [Android Studio CPU Profiler](https://developer.android.com/studio/profile/cpu-profiler) ao seguir as etapas abaixo:

1. Selecione **Run &gt; Profile 'app'** (Executar &gt; Fazer o profiling do aplicativo) nos menus superiores.

2. Clique em qualquer lugar na linha do tempo da CPU quando aparecer a janela do Profiler.

3. Selecione "Trace System Calls" (Rastrear chamadas do sistema) entre os modos de profiling da CPU.

    ![Selecione 'Trace System Calls'](images/as_select_profiling_mode.png)

4. Pressione o botão "Record" (Gravar).

5. Pressione o botão "Stop" (Parar).

6. Investigue o resultado do tracing.

    ![Trace do Android Studio](images/as_traces.png)

Nesse exemplo, você pode ver a hierarquia dos eventos em um thread e as estatísticas para cada tempo do operador, além de conferir o fluxo dos dados do aplicativo inteiro entre os threads.

#### Aplicativo System Tracing

Capture traces sem o Android Studio ao seguir as etapas detalhadas no [aplicativo System Tracing](https://developer.android.com/topic/performance/tracing/on-device).

Nesse exemplo, os mesmos eventos do TFLite foram capturados e salvos no formato Perfetto ou Systrace dependendo da versão do dispositivo Android. Os arquivos dos traces capturados podem ser abertos na [interface de usuário Perfetto](https://ui.perfetto.dev/#!/).

![Trace Perfetto](images/perfetto_traces.png)

### Rastreie internals do TensorFlow Lite no iOS

Observação: esse recurso está disponível a partir do TensorFlow Lite v2.5.

Os eventos internos do interpretador do TensorFlow Lite de um aplicativo iOS podem ser capturados pela ferramenta [Instruments](https://developer.apple.com/library/archive/documentation/ToolsLanguages/Conceptual/Xcode_Overview/MeasuringPerformance.html#//apple_ref/doc/uid/TP40010215-CH60-SW1) incluída com Xcode. Eles são os eventos [signpost](https://developer.apple.com/documentation/os/logging/recording_performance_data) do iOS, então os eventos capturados do código Swift/Objective-C são vistos juntos com os eventos internos do TensorFlow Lite.

Alguns exemplos de eventos são:

- Invocação de operador
- Modificação de grafo por delegado
- Alocação de tensor

#### Ative o tracing do TensorFlow Lite

Defina a variável de ambiente `debug.tflite.trace` seguindo as etapas abaixo:

1. Selecione **Product &gt; Scheme &gt; Edit Scheme...** (Produto &gt; Esquema &gt; Editar esquema...) nos menus superiores do Xcode.

2. Clique em "Profile" (Perfil) no painel à esquerda.

3. Desmarque a caixa de seleção "Use the Run action's arguments and environment variables" (Usar os argumentos e as variáveis de ambiente da ação de execução).

4. Adicione `debug.tflite.trace` à seção "Environment Variables" (Variáveis de ambiente).

    ![Defina a variável de ambiente](images/xcode_profile_environment.png)

Se você quiser excluir os eventos do TensorFlow Lite ao analisar o perfil do aplicativo iOS, desative o tracing ao remover a variável de ambiente.

#### Instruments do XCode

Capture traces seguindo as etapas abaixo:

1. Selecione **Product &gt; Profile** (Produto &gt; Perfil) nos menus superiores do Xcode.

2. Clique em **Logging** nos modelos de profiling quando a ferramenta Instruments for inicializada.

3. Pressione o botão "Start" (Iniciar).

4. Pressione o botão "Stop" (Parar).

5. Clique em "os_signpost" para abrir os itens do subsistema de registros do SO.

6. Clique no subsistema de registros do SO "org.tensorflow.lite".

7. Investigue o resultado do tracing.

    ![Trace do Instruments do Xcode](images/xcode_traces.png)

Nesse exemplo, você pode ver a hierarquia de eventos e as estatísticas para cada tempo do operador.

### Usando os dados de tracing

Os dados de tracing permitem que você identifique os gargalos de desempenho.

Veja alguns exemplos de insights que você pode obter do profiler e possíveis soluções para melhorar o desempenho:

- Se o número de núcleos de CPU disponíveis for menor do que o número de threads de inferência, então a sobrecarga do agendamento da CPU pode levar a um desempenho inferior. Você pode reagendar outras tarefas intensivas da CPU no seu aplicativo para evitar a sobreposição com a inferência do seu modelo ou ajustar o número de threads do interpretador.
- Se os operadores não forem totalmente delegados, algumas partes do grafo do modelo são executas na CPU, em vez do acelerador de hardware esperado. Você pode substituir os operadores incompatíveis por operadores compatíveis semelhantes.
