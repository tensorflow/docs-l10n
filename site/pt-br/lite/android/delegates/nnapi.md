# Delegado NNAPI do TensorFlow Lite

A [API Android Neural Networks (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) está disponível em todos os dispositivos Android com o Android 8.1 (nível 27 da API) ou mais recente. Ela fornece aceleração para modelos do TensorFlow em dispositivos Android com aceleradores de hardware compatíveis, incluindo:

- Unidade de Processamento Gráfico (GPU)
- Processamento de Sinal Digital (DSP)
- Unidade de Processamento Neural (NPU)

O desempenho varia dependendo do hardware específico disponível no dispositivo.

Esta página descreve como usar o delegado NNAPI com o Interpretador do TensorFlow Lite no Java e no Kotlin. Para APIs C Android, consulte a [documentação do Kit de desenvolvimento nativo do Android](https://developer.android.com/ndk/guides/neuralnetworks).

## Testando o delegado NNAPI no seu próprio modelo

### Importe o gradle

O delegado NNAPI faz parte do interpretador Android do TensorFlow Lite, versão 1.14.0 ou mais recente. Você pode importá-lo para seu projeto ao adicionar o seguinte código ao arquivo gradle do seu módulo:

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### Inicialize o delegado NNAPI

Adicione o código para inicializar o delegado NNAPI antes de inicializar o interpretador do TensorFlow Lite.

Observação: embora a NNAPI seja compatível com o nível 27 da API (Android Oreo MR1), o suporte para as operações melhorou significativamente a partir do nível 28 da API (Android Pie). Como resultado, recomendamos que os desenvolvedores usem o delegado NNAPI para o Android Pie ou mais recente na maioria dos casos.

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## Práticas recomendadas

### Teste o desempenho antes de implantar

O desempenho do runtime pode variar significativamente devido à arquitetura do modelo, tamanho, operações, disponibilidade de hardware e utilização de hardware de runtime. Por exemplo, se um aplicativo utiliza muito a GPU para renderização, a aceleração de NNAPI pode não melhorar o desempenho devido à contenção de recursos. Recomendamos executar um teste simples de desempenho usando o registro de depuração para medir o tempo de inferência. Execute o teste em vários smartphones com diferentes chipsets (fabricantes ou modelos do mesmo fabricante) que representem sua base de usuários antes de ativar a NNAPI em produção.

Para desenvolvedores avançados, o TensorFlow Lite também oferece [uma ferramenta de benchmark de modelo para Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

### Crie uma lista de exclusão de dispositivos

Em produção, pode haver casos em que a NNAPI não tenha o desempenho esperado. Recomendamos que os desenvolvedores mantenham uma lista de dispositivos que não devem usar a aceleração de NNAPI em combinação com modelos específicos. É possível criar essa lista com base no valor de `"ro.board.platform"`, que você pode recuperar usando o seguinte fragmento de código:

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

Para desenvolvedores avançados, considere manter essa lista através de um sistema de configuração remota. A equipe do TensorFlow está trabalhando ativamente em maneiras de simplificar e automatizar a descoberta e aplicação da configuração de NNAPI ideal.

### Quantização

A quantização reduz o tamanho do modelo ao usar números inteiros de 8 bits ou floats de 16 bits em vez de floats de 32 bits para computação. Os tamanhos do modelo de números inteiros de 8 bits são um quarto das versões float de 32 bits. Os floats de 16 bits são metade do tamanho. A quantização pode melhorar o desempenho significativamente, embora o processo possa resultar no trade-off de alguma exatidão do modelo.

Há vários tipos de técnicas de quantização pós-treinamento disponíveis, mas, para máximo suporte e aceleração no hardware atual, recomendamos a [quantização de números inteiros completa](post_training_quantization#full_integer_quantization_of_weights_and_activations). Essa abordagem converte ambos o peso e as operações em números inteiros. Esse processo de quantização exige um dataset representativo para funcionar.

### Use modelos e operações compatíveis

Se o delegado NNAPI não for compatível com algumas das combinações de parâmetros ou operações em um modelo, o framework só executará as partes compatíveis do grafo no acelerador. O restante será executado na CPU, resultando na execução fragmentada. Devido ao alto custo da sincronização de CPU/acelerador, isso pode resultar em um desempenho mais lento comparado à execução da rede inteira só na CPU.

A NNAPI tem um melhor desempenho quando os modelos usam somente [operações compatíveis](https://developer.android.com/ndk/guides/neuralnetworks#model). Os modelos a seguir são compatíveis com a NNAPI:

- [Classificação de imagens MobileNet v1 (224x224) (download do modelo float)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [(download do modelo quantizado)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) <br> *(modelo de classificação de imagens feito para aplicativos de visão móveis e embarcados)*
- [Detecção de objetos MobileNet v2 SSD](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite) <br> *(modelo de classificação de imagens que detecta vários objetos com caixas delimitadoras)*
- [Detecção de objetos MobileNet v1(300x300) Single Shot Detector (SSD)](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(download)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [PoseNet para a estimativa de pose](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) <br> *(modelo de visão que estima as poses de uma ou mais pessoas em imagens ou vídeos)*

A aceleração de NNAPI também não é compatível quando o modelo contém saídas de tamanho dinâmico. Nesse caso, você verá um aviso assim:

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors \
with a graph that has dynamic-sized tensors.
```

### Ative a implementação da CPU da NNAPI

Um grafo que não pode ser processado completamente por um acelerador pode reverter à implementação da CPU da NNAPI. No entanto, como geralmente é menos eficaz do que o interpretador do TensorFlow, essa opção é desativada por padrão no delegado NNAPI para o Android 10 (nível 29 da API) ou superior. Para substituir esse comportamento, defina `setUseNnapiCpu` como `true` no objeto `NnApiDelegate.Options`.
