# Delegados do TensorFlow Lite

## Introdução

Os **delegados** permitem a aceleração de hardware de modelos do TensorFlow Lite ao usar aceleradores no dispositivo, como GPU e [Processador de Sinal Digital (DSP)](https://en.wikipedia.org/wiki/Digital_signal_processor).

Por padrão, o TensorFlow Lite usa kernels de CPU otimizados para o conjunto de instruções [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions). No entanto, a CPU é um processador multiuso que não é necessariamente otimizado para a aritmética pesada geralmente encontrada em modelos de aprendizado de máquina (por exemplo, a matemática de matriz envolvida em camadas densas e de convolução).

Por outro lado, a maioria dos smartphones modernos contém chips que lidam melhor com essas operações pesadas. Utilizá-los para as operações de redes neurais oferece grandes benefícios em termos de latência e eficiência energética. Por exemplo, as GPUs podem fornecer até [5x mais velocidade](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html) na latência, enquanto o [DSP Qualcomm® Hexagon](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor) reduziu o consumo de energia em até 75% em nossos experimentos.

Cada um desses aceleradores tem APIs associadas que permitem computações personalizadas, como [OpenCL](https://www.khronos.org/opencl/) ou [OpenGL ES](https://www.khronos.org/opengles/) para GPU móvel e [SDK Qualcomm® Hexagon](https://developer.qualcomm.com/software/hexagon-dsp-sdk) para DSP. Normalmente, você precisaria escrever muito código personalizado para executar uma rede neural por essas interfaces. As coisas se complicam ainda mais quando você considera que cada acelerador tem prós e contras e não pode executar todas as operações em uma rede neural. A API Delegate do TensorFlow Lite resolve esse problema ao servir como uma ponte entre o runtime do TFLite e essas APIs de nível inferior.

![runtime com delegados](images/delegate_runtime.png)

## Escolha um delegado

O TensorFlow Lite é compatível com vários delegados, sendo que cada um é otimizado para determinadas plataformas e tipos específicos de modelos. Geralmente, há vários delegados aplicáveis ao seu caso de uso, dependendo de dois critérios principais: a *Plataforma* (Android ou iOS?) segmentada e o *Tipo de modelo* (ponto flutuante ou quantizado?) que você está tentando acelerar.

### Delegados por plataforma

#### Multiplataforma (Android e iOS)

- **Delegado de GPU**: pode ser usado em ambos o Android e o iOS. É otimizado para executar modelos baseados em float de 32 e 16 bits com uma GPU disponível. Também é compatível com modelos quantizados de 8 bits e fornece um desempenho de GPU equivalente às versões de float. Para mais detalhes sobre o delegado de GPU, confira [TensorFlow Lite na GPU](gpu_advanced.md). Para tutoriais passo a passo sobre como usar o delegado de GPU com o Android e o iOS, confira o [Tutorial do delegado de GPU do TensorFlow Lite](gpu.md).

#### Android

- **Delegado NNAPI para dispositivos Android mais recentes**: pode ser usado para acelerar modelos em dispositivos Android com GPU, DSP e/ou NPU disponível. Está disponível no Android 8.1 (API 27+) ou mais recente. Para uma visão geral do delegado NNAPI, instruções passo a passo e práticas recomendadas, confira [delegado NNAPI do TensorFlow Lite](nnapi.md).
- **Delegado Hexagon para dispositivos Android mais antigos**: o delegado Hexagon pode ser usado para acelerar modelos em dispositivos Android com o DSP Qualcomm Hexagon. Ele pode ser usado em dispositivos com versões mais antigas do Android que não são compatíveis com a NNAPI. Confira mais detalhes em [delegado Hexagon do TensorFlow Lite](hexagon_delegate.md).

#### iOS

- **Delegado Core ML para iPhones e iPads mais recentes**: para iPhones e iPads mais recentes em que o Neural Engine estiver disponível, você pode usar o delegado Core para acelerar a inferência para modelos de ponto flutuante de 32 ou 16 bits. O Neural Engine está disponível em dispositivos móveis Apple com SoC A12 e mais recente. Para uma visão geral do delegado Core ML e instruções passo a passo, confira [delegado Core ML do TensorFlow Lite](coreml_delegate.md).

### Delegados por tipo de modelo

Cada acelerador é criado com uma determinada largura de bits de dados em mente. Se você fornecer um modelo de ponto flutuante a um delegado que só é compatível com operações quantizadas de 8 bits (como o [delegado Hexagon](hexagon_delegate.md)), ele rejeitará todas as operações, e o modelo será executado totalmente na CPU. Para evitar surpresas, a tabela abaixo oferece uma visão geral da compatibilidade de delegados com base no tipo de modelo:

**Tipo de modelo** | **GPU** | **NNAPI** | **Hexagon** | **Core ML**
--- | --- | --- | --- | ---
Ponto flutuante (32 bits) | Sim | Sim | Não | Sim
[Quantização float16 pós-treinamento](post_training_float16_quant.ipynb) | Sim | Não | Não | Sim
[Quantização de intervalo dinâmico pós-treinamento](post_training_quant.ipynb) | Sim | Sim | Não | Não
[Quantização de números inteiros pós-treinamento](post_training_integer_quant.ipynb) | Sim | Sim | Sim | Não
[Treinamento consciente de quantização](http://www.tensorflow.org/model_optimization/guide/quantization/training) | Sim | Sim | Sim | Não

### Valide o desempenho

As informações nesta seção servem como uma diretriz aproximada para selecionar os delegados que podem melhorar seu aplicativo. Porém, é importante observar que cada delegado tem um conjunto predefinido de operações compatíveis e pode apresentar desempenhos diferentes dependendo do modelo e dispositivo. Por exemplo, o [delegado NNAPI](nnapi.md) pode escolher usar o Edge-TPU do Google em um smartphone Pixel e utilizar o DSP em outro dispositivo. Portanto, é geralmente recomendável realizar benchmarking para avaliar a utilidade de um delegado para suas necessidades. Isso também ajuda a justificar o aumento no tamanho do binário associado à atribuição de um delegado ao runtime do TensorFlow Lite.

O TensorFlow Lite tem vastas ferramentas de avaliação do desempenho e da exatidão que podem dar confiança aos desenvolvedores para usar os delegados nos seus aplicativos. Essas ferramentas são abordadas na próxima seção.

## Ferramentas para avaliação

### Latência e consumo de memória

A [ferramenta de benchmarking](https://www.tensorflow.org/lite/performance/measurement) do TensorFlow Lite pode ser usada com parâmetros adequados para estimar o desempenho do modelo, incluindo a latência de inferência média, a sobrecarga de inicialização, o consumo de memória etc. Essa ferramenta é compatível com várias flags para encontrar a melhor configuração de delegado para seu modelo. Por exemplo, `--gpu_backend=gl` pode ser especificado com `--use_gpu` para medir a execução de GPU com o OpenGL. A lista completa de parâmetros de delegados compatíveis está definida na [documentação detalhada](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar).

Confira um exemplo de execução para um modelo quantizado com GPU por `adb`:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

Você pode baixar a versão pré-criada dessa ferramenta para a arquitetura ARM de 64 bits do Android [aqui](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk) ([mais detalhes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)).

### Exatidão e precisão

Os delegados geralmente realizam computações com uma precisão diferente do que a CPU. Como resultado, há um trade-off de exatidão (geralmente pequeno) associado ao uso de um delegado para a aceleração de hardware. Observe que isso *nem sempre* é verdade. Por exemplo, como a GPU usa a precisão de ponto flutuante para executar modelos quantizados, pode haver uma leve melhoria na precisão (por exemplo, melhoria top-5 de &lt;1% na classificação de imagens ILSVRC).

O TensorFlow Lite tem dois tipos de ferramentas para avaliar a exatidão do comportamento de um delegado para um modelo específico: *baseada na tarefa* e *agnóstica à tarefa*. Todas as ferramentas descritas nesta seção aceitam os [parâmetros de delegação avançados](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar) usados pela ferramenta de benchmarking na seção anterior. Observe que as subseções abaixo focam na *avaliação do delegado* (o delegado tem o mesmo desempenho que a CPU?), e não na avaliação do modelo (o próprio modelo é bom para a tarefa?).

#### Avaliação baseada na tarefa

O TensorFlow Lite tem ferramentas para avaliar a exatidão de duas tarefas baseadas em imagens:

- [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/) (classificação de imagens) com [exatidão top-K](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

- [Detecção de objetos COCO (com caixas delimitadoras)](https://cocodataset.org/#detection-2020) com [mean Average Precision (mAP)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

Binários pré-criados dessas ferramentas (arquitetura ARM de 64 bits do Android), além de documentação, podem ser encontrados aqui:

- [Classificação de imagens ImageNet](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification) ([mais detalhes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification))
- [Detecção de objetos COCO](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection) ([mais detalhes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection))

O exemplo abaixo demonstra a [avaliação da classificação de imagens](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification) com NNAPI utilizando o Edge-TPU do Google em um Pixel 4:

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

A saída esperada é uma lista de métricas top-K de 1 a 10:

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### Avaliação agnóstica à tarefa

Para as tarefas em que não há uma ferramenta de avaliação no dispositivo estabelecida, ou se você estiver testando modelos personalizados, o TensorFlow Lite tem a ferramenta [Diff da inferência](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) (binário da arquitetura ARM de 64 bits do Android [aqui](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff)).

A Diff da inferência compara a execução do TensorFlow Lite (em termos de desvio do valor de saída e latência) em duas configurações:

- Inferência de CPU de thread único
- Inferência definida pelo usuário: definida por [estes parâmetros](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)

Para fazer isso, a ferramenta gera dados gaussianos aleatórios, que são passados por dois interpretadores do TFLite: um que executa kernels de CPU de thread único e outro que é parametrizado pelos argumentos do usuário.

Ela mede a latência de ambos, além da diferença absoluta entre os tensores de saída de cada interpretador, com base em cada elemento.

Para um modelo com um único tensor de saída, o resultado será algo assim:

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

Isso significa que, para o tensor de saída no índice `0`, os elementos da saída da CPU diferem da saída do delegado em uma média de `1.96e-05`.

Observe que a interpretação desses números exige conhecimento mais aprofundado sobre o modelo e o significado de cada tensor de saída. Se uma regressão simples determinar algum tipo de pontuação ou embedding, a diferença deve ser baixa (caso contrário, é um erro com o delegado). No entanto, saídas como a "classe de detecção" de modelos SSD são um pouco mais difíceis de interpretar. Por exemplo, ao usar essa ferramenta, pode indicar uma diferença, mas isso não significa necessariamente que há algo muito errado com o delegado: considere duas classes (falsas): "TV (ID: 10)" e "Monitor (ID: 20)" — se um delegado estiver um pouco fora da verdade absoluta e mostrar "monitor" em vez de "TV", a diferença da saída para esse tensor pode ser de até 20-10 = 10.
