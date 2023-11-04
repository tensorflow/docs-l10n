# Gere interfaces de modelos usando metadados

Utilizando os [metadados do TensorFlow Lite](../models/convert/metadata), os desenvolvedores podem gerar código encapsulador para permitir a integração com o Android. Para a maioria dos desenvolvedores, a interface gráfica do [Android Studio ML Model Binding](#mlbinding) (vinculação de modelos de ML do Android Studio) é a mais fácil de usar. Se você deseja mais personalização ou estiver usando uma ferramenta de linha de comando, o [TensorFlow Lite Codegen](#codegen) também está disponível.

## Use o Android Studio ML Model Binding {:#mlbinding}

Para modelos do TensorFlow Lite aprimorados com [metadados](../models/convert/metadata.md), os desenvolvedores podem usar o Android Studio ML Model Binding para definir automaticamente as configurações do projeto e gerar classes encapsuladoras com base nos metadados do modelo. O código encapsulador remove a necessidade de interagir diretamente com o `ByteBuffer`. Em vez disso, os desenvolvedores podem interagir com objetos tipados, como `Bitmap` e `Rect`.

Observação: é necessário ter o [Android Studio 4.1](https://developer.android.com/studio) ou superiores.

### Importe um modelo do TensorFlow Lite no Android Studio

1. Clique com o botão direito no módulo onde você quer usar o modelo do TF Lite ou clique em `File` &gt; `New` &gt; `Other` &gt; `TensorFlow Lite Model` (Arquivo &gt; Novo &gt; Outro &gt; Modelo do TensorFlow Lite). ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. Selecione o local do seu arquivo do TF Lite. Observe que a ferramenta configura a dependência do modelo com o ML Model Binding, e todas as dependências são incluídas automaticamente no arquivo `build.gradle` do seu módulo para Android.

    Opcional: marque a segunda caixa de seleção para importar a GPU do TensorFlow se você quiser usar a aceleração de GPU. ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. Clique em `Finish` (Finalizar).

4. A tela abaixo será exibida após a importação ser concluída com êxito. Para começar a usar o modelo, selecione Kotlin ou Java, e copie e cole o código abaixo da seção `Sample Code` (Código de exemplo). Para voltar a essa tela, basta clicar duas vezes no modelo do TF Lite sob o diretório `ml` no Android Studio. ![Model details page in Android Studio](../images/android/model_details.png)

### Como acelerar a inferência do modelo {:#acceleration}

O ML Model Binding conta com uma maneira de os desenvolvedores acelerarem o código por meio do uso de delegados e do número de threads.

Observação: o interpretador do TensorFlow Lite precisa ser criado no mesmo thread em que é executado. Caso contrário, pode ocorrer o seguinte erro: TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized (GpuDelegate precisa ser executado no mesmo thread em que foi inicializado).

Etapa 1 – Verifique se o arquivo `build.gradle` do modelo contém a seguinte dependência:

```java
    dependencies {
        ...
        // TFLite GPU delegate 2.3.0 or above is required.
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
    }
```

Etapa 2 – Detecte se a GPU sendo executada no dispositivo é compatível com o delegado GPU do TensorFlow. Caso não seja, execute o modelo usando diversos threads de CPU:

<div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = if(compatList.isDelegateSupportedOnThisDevice) {
        // if the device has a supported GPU, add the GPU delegate
        Model.Options.Builder().setDevice(Model.Device.GPU).build()
    } else {
        // if the GPU is not supported, run on 4 threads
        Model.Options.Builder().setNumThreads(4).build()
    }

    // Initialize the model as usual feeding in the options object
    val myModel = MyModel.newInstance(context, options)

    // Run inference per sample code
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">
    import org.tensorflow.lite.support.model.Model
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Model.Options options;
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        options = Model.Options.Builder().setDevice(Model.Device.GPU).build();
    } else {
        // if the GPU is not supported, run on 4 threads
        options = Model.Options.Builder().setNumThreads(4).build();
    }

    MyModel myModel = new MyModel.newInstance(context, options);

    // Run inference per sample code
      </pre>
    </section>
    </devsite-selector>
</div>

## Gere interfaces de modelos com o gerador de código do TensorFlow Lite {:#codegen}

Observação: o gerador de código do encapsulador para o TensorFlow Lite só tem suporte ao Android no momento.

Para o modelo do TensorFlow Lite aprimorado com [metadados](../models/convert/metadata.md), os desenvolvedores podem usar o gerador de código do encapsulador Android para o TensorFlow Lite para criar o código do encapsulador específico para a plataforma. O código do encapsulador remove a necessidade de interagir diretamente com o `ByteBuffer`. Em vez disso, os desenvolvedores podem interagir com o modelo do TensorFlow Lite usando objetos tipados, como `Bitmap` e `Rect`.

O nível de utilidade do gerador de código depende do nível de completude da entrada de metadados do modelo do TensorFlow Lite. Confira a seção `<Codegen usage>` abaixo dos campos relevantes em [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) para ver como a ferramenta codegen processa cada campo.

### Gere código encapsulador

Você precisará instalar a seguinte ferramenta em seu terminal:

```sh
pip install tflite-support
```

Após a conclusão, o gerador de código poderá ser utilizado por meio da seguinte sintaxe:

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

O código resultante ficará localizado no diretório de destino. Se você estiver usando o [Google Colab](https://colab.research.google.com/) ou outro ambiente remoto, pode ser mais fácil colocar o resultado em um arquivo ZIP e baixá-lo para seu projeto do Android Studio:

```python
# Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
```

### Uso do código gerado

#### Etapa 1 – Importe o código gerado

Se necessário, descompacte o código gerado em uma estrutura de diretórios. Pressupõe-se que a raiz do código gerado seja `SRC_ROOT`.

Abra o projeto do Android Studio no qual deseja usar o modelo do TensorFlow Lite e importe o módulo gerado em: File -&gt; New -&gt; Import Module (Arquivo -&gt; Novo -&gt; Importar módulo) e selecione `SRC_ROOT`

Usando o exemplo acima, o diretório e o módulo importado seriam chamados de `classify_wrapper`.

#### Etapa 2 – Atualize o arquivo `build.gradle` do aplicativo

No módulo do aplicativo que consumirá o módulo da biblioteca gerado:

Abaixo da seção "android", adicione o seguinte:

```build
aaptOptions {
   noCompress "tflite"
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

Abaixo da seção "dependencies" (dependências), adicione o seguinte:

```build
implementation project(":classify_wrapper")
```

#### Etapa 3 – Use o modelo

```java
// 1. Initialize the model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Set the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Run the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieve the result
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

### Como acelerar a inferência do modelo

O código gerado proporciona ao desenvolvedores uma forma de acelerar o código por meio do uso de [delegados](../performance/delegates.md) e do número de threads, que podem ser definidos ao inicializar o objeto do modelo, que aceita três parâmetros:

- **`Context`**: contexto de Atividade ou Serviço do Android.
- (Opcional) **`Device`**: delegado de aceleração do TF Lite, como GPUDelegate ou NNAPIDelegate, por exemplo.
- (Opcional) **`numThreads`**: número de threads usados para executar o modelo. O padrão é 1.

Por exemplo: para usar um delegado NNAPI e até três threads, você pode inicializar o modelo da seguinte forma:

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### Solução de problemas

Se for exibido o erro "java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed" (Este arquivo não pode ser aberto como descritor de arquivo; provavelmente está compactado), insira as linhas abaixo sob a seção "android" do módulo do aplicativo que usará o módulo da biblioteca:

```build
aaptOptions {
   noCompress "tflite"
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.
