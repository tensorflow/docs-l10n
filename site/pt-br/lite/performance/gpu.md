# Delegados de GPU para o TensorFlow Lite

Usar unidades de processamento gráfico (GPUs) para executar seus modelos de aprendizado de máquina (ML) pode melhorar drasticamente o desempenho do seu modelo e a experiência do usuário dos seus aplicativos com ML. O TensorFlow Lite permite o uso de GPUs e outros processadores especializados pelo driver de hardware chamados [*delegados*](./delegates). O uso de GPUs com seus aplicativos de ML do TensorFlow Lite pode oferecer os seguintes benefícios:

- **Velocidade**: as GPUs são criadas para o alto desempenho de workloads maciçamente paralelas. Esse design faz com que sejam adequadas para redes neurais profundas, que consistem em um grande número de operadores, cada um trabalhando em tensores de entrada que podem ser processados em paralelo, o que geralmente resulta em menor latência. No melhor dos casos, a execução do seu modelo em uma GPU pode ser rápida o suficiente para permitir aplicativos em tempo real que antes não eram possíveis.
- **Eficiência energética**: as GPUs podem realizar computações de ML de maneira bastante eficiente e otimizada, geralmente consumindo menos energia e gerando menos calor do que a mesma tarefa executada em CPUs.

Este documento fornece uma visão geral da compatibilidade com GPUs no TensorFlow Lite e alguns usos avançados para os processadores de GPU. Para informações mais específicas sobre como implementar o suporte à GPU em plataformas específicas, veja os seguintes guias:

- [Suporte à GPU para Android](../android/delegates/gpu)
- [Suporte à GPU para iOS](../ios/delegates/gpu)

## Suporte a operações de ML de GPU {:#supported_ops}

Há algumas limitações em relação às operações de ML do TensorFlow, *ops*, que podem ser aceleradas pelo delegado de GPU do TensorFlow Lite. O delegado é compatível com as seguintes ops na precisão de float de 16 e 32 bits:

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGICAL_AND`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM only)`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

Por padrão, todas as ops só são compatíveis com a versão 1. Ao habilitar o [suporte à quantização](#quantized-models), são permitidas as versões apropriadas, por exemplo, ADD v2.

### Solução de problemas de suporte à GPU

Se algumas das ops não forem compatíveis com o delegado de GPU, o framework só executará uma parte do grafo na GPU e o restante na CPU. Devido ao alto custo da sincronização de CPU/GPU, um modo de execução dividida como esse geralmente resulta em um desempenho inferior em relação à execução da rede inteira só na CPU. Nesse caso, o aplicativo gera um aviso, como este:

```none
WARNING: op code #42 cannot be handled by this delegate.
```

Não há callback para falhas desse tipo, já que não é uma falha de runtime real. Ao testar a execução do seu modelo com o delegado de GPU, você deve ficar alerta a esses avisos. Um grande número de avisos assim pode indicar que o seu modelo não é o mais adequado para uso com a aceleração de GPU e talvez exija a refatoração do modelo.

## Modelos de exemplo

Os seguintes modelos de exemplo foram criados para aproveitar a aceleração de GPU com o TensorFlow Lite e são fornecidos para referência e testes:

- [Classificação de imagens MobileNet v1 (224x224)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html): um modelo de classificação de imagens criado para aplicativos de visão baseados em dispositivos móveis e embarcados. ([modelo](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5))
- [Segmentação DeepLab (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html): modelo de segmentação de imagens que atribui rótulos semânticos, como cachorro, gato e carro, a todos os pixels na imagem de entrada. ([modelo](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1))
- [Detecção de objetos MobileNet SSD](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html): um modelo de classificação de imagens que detecta vários objetos com caixas delimitadoras. ([modelo](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite))
- [PoseNet para a estimativa de pose](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection): um modelo de visão que estima as poses de pessoas em imagens ou vídeos. ([modelo](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1))

## Otimização para GPUs

As seguintes técnicas podem ajudar você a obter o melhor desempenho ao executar modelos em hardware de GPU usando o delegado de GPU do TensorFlow Lite:

- **Operações de reformulação**: algumas operações que são rápidas em uma CPU podem ter um alto custo para a GPU em dispositivos móveis. As operações de reformulação são especialmente caras para executar, incluindo `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH` e assim por diante. Você deve examinar com cuidado o uso dessas operações e considerar que só devem ser aplicadas para a exploração de dados ou para iterações iniciais do seu modelo. A remoção delas pode melhorar significativamente o desempenho.

- **Canais de dados de imagens**: na GPU, os dados de tensor são divididos em 4 canais. Portanto, uma computação em um tensor de formato `[B,H,W,5]` tem praticamente o mesmo desempenho em um tensor de formato `[B,H,W,8]`, mas um desempenho significativamente inferior em `[B,H,W,4]`. Se o hardware da câmera que você estiver usando for compatível com frames de imagens em RGBA, é muito mais rápido alimentar essa entrada de 4 canais, já que evita uma cópia na memória do RGB de 3 canais para o RGBX de 4 canais.

- **Modelos otimizados para dispositivos móveis**: para melhor desempenho, considere treinar novamente seu classificador com uma arquitetura de rede otimizada para dispositivos móveis. A otimização para a inferência no dispositivo pode reduzir consideravelmente a latência e o consumo de energia ao aproveitar recursos de hardware de dispositivos móveis.

## Suporte à GPU avançado

Você pode usar técnicas adicionais avançadas com processamento de GPU para melhorar ainda mais o desempenho dos seus modelos, incluindo a quantização e serialização. As seguintes seções descrevem essas técnicas em mais detalhes.

### Usando modelos quantizados {:#quantized-models}

Esta seção explica como o delegado de GPU acelera modelos quantizados de 8 bits, incluindo o seguinte:

- Modelos treinados com o [treinamento consciente de quantização](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- [Quantização de intervalo dinâmico](https://www.tensorflow.org/lite/performance/post_training_quant) pós-treinamento
- [Quantização de números inteiros](https://www.tensorflow.org/lite/performance/post_training_integer_quant) pós-treinamento

Para otimizar o desempenho, use modelos com tensores de saída e entrada de ponto flutuante.

#### Como isso funciona?

Como o back-end da GPU só aceita a execução de ponto flutuante, executamos modelos quantizados ao dar uma "visão de ponto flutuante" do modelo original. Em um nível superior, isso envolve os seguintes passos:

- Os *tensores constantes* (como pesos/biases) são desquantizados uma vez na memória de GPU. Essa operação acontece quando o delegado é ativado para o TensorFlow Lite.

- As *entradas e saídas* do programa de GPU, se forem quantizadas de 8 bits, são desquantizadas e quantizadas (respectivamente) para cada inferência. Essa operação é realizada na CPU usando os kernels otimizados do TensorFlow Lite.

- Os *simuladores de quantização* são inseridos entre as operações para imitar o comportamento quantizado. Essa abordagem é necessária para modelos em que as ops esperam ativações para seguir limites aprendidos durante a quantização.

Para mais informações sobre como ativar esse recurso com o delegado de GPU, veja o seguinte:

- Usando [modelos quantizados com a GPU no Android](../android/delegates/gpu#quantized-models)
- Usando [modelos quantizados com a GPU no iOS](../ios/delegates/gpu#quantized-models)

### Reduzindo o tempo de inicialização com a serialização {:#delegate_serialization}

O recurso de delegado de GPU permite o carregamento a partir de código de kernels pré-compilados e dados de modelo serializados e salvos no disco de execuções anteriores. Essa abordagem evita uma nova compilação e pode reduzir o tempo de inicialização em até 90%. Essa melhoria é obtida ao trocar o espaço em disco pela economia de tempo. Você pode ativar esse recurso com algumas opções de configurações, conforme mostrado nos exemplos de código a seguir:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

Ao usar o recurso de serialização, confira se o seu código segue estas regras de implementação:

- Armazene os dados de serialização em um diretório que não seja acessível a outros apps. Em dispositivos Android, use [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir()), que aponta a um local privado para o aplicativo atual.
- O token do modelo precisa ser exclusivo ao dispositivo para o modelo específico. Você pode computar um token de modelo ao gerar uma impressão digital a partir dos dados do modelo usando bibliotecas como [`farmhash::Fingerprint64`](https://github.com/google/farmhash).

Observação: o uso desse recurso de serialização exige o [SDK OpenCL](https://github.com/KhronosGroup/OpenCL-SDK).
