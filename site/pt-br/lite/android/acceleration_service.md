# Acceleration Service para Android (Beta)

Beta: no momento, o Acceleration Service para Android está em Beta. Leia as seções [Ressalvas](#caveats) e [Termos e Privacidade] (#terms_privacy) desta página para mais detalhes.

O uso de processadores especializados, como GPUs, NPUs ou DSPs, para a aceleração de hardware pode melhorar drasticamente o desempenho da inferência (até 10x mais rápida em alguns casos) e a experiência do usuário no seu aplicativo Android com tecnologia de ML. No entanto, considerando a variedade de hardware e drivers dos usuários, escolher a configuração de aceleração de hardware ideal para o dispositivo de cada um pode ser desafiador. Além disso, habilitar a configuração errada em um dispositivo pode resultar em uma experiência ruim devido à alta latência ou, em alguns casos raros, erros de runtime e problemas de exatidão causados por incompatibilidades de hardware.

O Acceleration Service para Android é uma API que ajuda você a escolher a configuração de aceleração de hardware ideal para um determinado dispositivo de usuário e seu modelo `.tflite`, minimizando o risco de erros de runtime ou problemas de exatidão

O Acceleration Service avalia diferentes configurações de aceleração nos dispositivos dos usuários ao realizar benchmarks de inferência internos com seu modelo do TensorFlow Lite. Esses testes geralmente são concluídos em alguns segundos, dependendo do seu modelo. Você pode executar os benchmarks uma vez em cada dispositivo de usuário antes da inferência, armazenar o resultado em cache e usá-lo durante a inferência. Esses benchmarks são realizados fora do processo, o que minimiza o risco de falhas no seu app.

Forneça seu modelo, amostras de dados e resultados esperados (saídas e entradas "golden", ou excelentes) para o Acceleration Service realizar um benchmark de inferência do TFLite e oferecer recomendações de hardware.

![imagem](../images/acceleration/acceleration_service.png)

O Acceleration Service faz parte da pilha de ML personalizada do Android e funciona com o [TensorFlow Lite no Google Play Services](https://www.tensorflow.org/lite/android/play_services).

## Adicione as dependências ao seu projeto

Adicione as seguintes dependências ao arquivo build.gradle do seu aplicativo:

```
implementation  "com.google.android.gms:play-services-tflite-
acceleration-service:16.0.0-beta01"
```

A API Acceleration Service funciona com o [TensorFlow Lite no Google Play Services](https://www.tensorflow.org/lite/android/play_services). e você não estiver usando o runtime do TensorFlow Lite fornecido através do Play Services, precisará atualizar suas [dependências](https://www.tensorflow.org/lite/android/play_services#1_add_project_dependencies_2).

## Como usar a API Acceleration Service

Para usar o Acceleration Service, comece criando a configuração de aceleração que você quer avaliar para seu modelo (por exemplo, GPU com OpenGL). Em seguida, crie uma configuração de validação com seu modelo, alguns dados de amostra e a saída esperada do modelo. Por fim, chame `validateConfig()` ao passar ambas a configuração de aceleração e de validação.

![imagem](../images/acceleration/acceleration_service_steps.png)

### Crie configurações de aceleração

As configurações de aceleração são representações de configurações de hardware traduzidas em delegados durante o tempo de execução. O Acceleration Service usará essas configurações internamente para realizar inferências de teste.

No momento, o Acceleration Service permite avaliar as configurações de GPU (convertidas para delegados de GPU durante o tempo de execução) com a [GpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig) e a inferência de CPU (com [CpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig)). Estamos trabalhando no suporte a mais delegados para acessar outros hardwares no futuro.

#### Configuração de aceleração de GPU

Crie uma configuração de aceleração de GPU da seguinte maneira:

```
AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder()
  .setEnableQuantizedInference(false)
  .build();
```

Você precisa especificar se o modelo está usando ou não a quantização com [`setEnableQuantizedInference()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig.Builder#public-gpuaccelerationconfig.builder-setenablequantizedinference-boolean-value).

#### Configuração de aceleração de CPU

Crie a aceleração de CPU da seguinte maneira:

```
AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder()
  .setNumThreads(2)
  .build();
```

Use o método [`setNumThreads()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig.Builder#setNumThreads(int)) para definir o número de threads que você quer usar para avaliar a inferência de CPU.

### Crie configurações de validação

As configurações de validação permitem que você defina como quer que o Acceleration Service avalie as inferências. Você as usará para passar:

- entradas de amostra,
- saídas esperadas,
- lógica de validação da exatidão.

Forneça as entradas de amostra que você espera que tenham um bom desempenho no modelo (também conhecidas como amostras "golden").

Crie uma [`ValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidationConfig) com [`CustomValidationConfig.Builder`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder) da seguinte maneira:

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenOutputs(outputBuffer)
   .setAccuracyValidator(new MyCustomAccuracyValidator())
   .build();
```

Especifique o número de amostras golden com [`setBatchSize()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setBatchSize(int)). Passe as entradas das amostras golden usando [`setGoldenInputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldeninputs-object...-value). Forneça a saída esperada para a entrada passada com [`setGoldenOutputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldenoutputs-bytebuffer...-value).

Você pode definir um tempo de inferência máximo com [`setInferenceTimeoutMillis()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setinferencetimeoutmillis-long-value) (5000 ms por padrão). Se a inferência demorar mais do que o tempo definido, a configuração será rejeitada.

Como opção, você também pode criar um [`AccuracyValidator`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.AccuracyValidator) personalizado da seguinte maneira:

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

Defina uma lógica de validação adequada para seu caso de uso.

Observe que, se os dados de validação já estiverem incorporados no seu modelo, você pode usar [`EmbeddedValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/EmbeddedValidationConfig).

##### Gere saídas de validação

As saídas golden são opcionais e, desde que você forneça entradas golden, o Acceleration Service pode gerar internamente as saídas golden. Você também pode definir a configuração de aceleração usada para gerar essas saídas golden ao chamar [`setGoldenConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setGoldenConfig(com.google.android.gms.tflite.acceleration.AccelerationConfig)):

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenConfig(customCpuAccelerationConfig)
   [...]
   .build();
```

### Valide a configuração de aceleração

Depois de criar uma configuração de aceleração e de validação, você pode avaliá-las para seu modelo.

Confira se o runtime do TensorFlow Lite com o Play Services foi inicializado de maneira adequada e se o delegado de GPU está disponível para o dispositivo ao executar:

```
TfLiteGpu.isGpuDelegateAvailable(context)
   .onSuccessTask(gpuAvailable -> TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(gpuAvailable)
        .build()
      )
   );
```

Instancie o [`AccelerationService`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService) ao chamar [`AccelerationService.create()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#create(android.content.Context)).

Em seguida, você pode validar sua configuração de aceleração para seu modelo ao chamar [`validateConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfig(com.google.android.gms.tflite.acceleration.Model,%20com.google.android.gms.tflite.acceleration.AccelerationConfig,%20com.google.android.gms.tflite.acceleration.ValidationConfig)):

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

Você também pode validar várias configurações ao chamar [`validateConfigs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfigs(com.google.android.gms.tflite.acceleration.Model,%20java.lang.Iterable%3Ccom.google.android.gms.tflite.acceleration.AccelerationConfig%3E,%20com.google.android.gms.tflite.acceleration.ValidationConfig)) e passar um objeto `Iterable<AccelerationConfig>` como parâmetro.

`validateConfig()` retornará uma `Task<`[`ValidatedAccelerationConfigResult`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidatedAccelerationConfigResult)`>` da [API Task](https://developers.google.com/android/guides/tasks) do Google Play Services que permite tarefas assíncronas. <br> Para obter o resultado da chamada de validação, adicione uma callback [`addOnSuccessListener()`](https://developers.google.com/android/reference/com/google/android/gms/tasks/OnSuccessListener).

#### Use a configuração validada no seu interpretador

Depois de conferir se `ValidatedAccelerationConfigResult` retornou que a callback é válida, você pode definir a configuração validada como uma configuração de aceleração para seu interpretador chamando `interpreterOptions.setAccelerationConfig()`.

#### Armazene a configuração em cache

É improvável que a configuração de aceleração ideal para seu modelo mude no dispositivo. Então, depois de receber uma configuração de aceleração satisfatória, armazene no dispositivo e deixe seu aplicativo recuperá-la e usá-la para criar `InterpreterOptions` durante as próximas sessões em vez de realizar outra validação. Os métodos `serialize()` e `deserialize()` em `ValidatedAccelerationConfigResult` facilitam o processo de armazenamento e recuperação.

### Aplicativo de amostra

Para consultar uma integração in situ do Acceleration Service, confira o [app de amostra](https://github.com/tensorflow/examples/tree/master/lite/examples/acceleration_service/android_play_services).

## Limitações

O Acceleration Service tem as seguintes limitações:

- Somente as configurações de CPU e GPU são compatíveis no momento,
- Ele só é compatível com o TensorFlow Lite no Google Play Services e você não pode usá-lo se estiver usando a versão de bundle do TensorFlow Lite,
- Ele não é compatível com a [Biblioteca Task](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview) do TensorFlow, já que não é possível inicializar [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) diretamente com o objeto `ValidatedAccelerationConfigResult`.
- O SDK do Acceleration Service só é compatível com o nível da API 22 ou superior.

## Ressalvas {:#caveats}

Revise as seguintes ressalvas com cuidado, especialmente se você planeja usar esse SDK em produção:

- Antes de encerrar o Beta e lançar a versão estável da API Acceleration Service, vamos publicar um novo SDK que pode ser um pouco diferente do Beta atual. Para continuar usando o Acceleration Service, você precisará migrar para esse novo SDK e enviar prontamente uma atualização para seu app. Se não fizer isso, poderá haver quebras, já que o SDK do Beta não será mais compatível com o Google Play Services após um tempo.

- Não há garantia de que um recurso específico na API Acceleration Service ou a API como um todo ficará algum dia disponível para todos. Ela poderá permanecer na versão Beta indefinidamente, ser suspensa ou combinada com outros recursos em pacotes criados para públicos de desenvolvedores específicos. Alguns recursos com a API Acceleration Service ou a API inteira podem ser disponibilizados para o público, mas não há um cronograma estipulado para isso.

## Termos e privacidade {:#terms_privacy}

#### Termos de Serviço

O uso das APIs Acceleration Service está sujeito aos [Termos de Serviço das APIs do Google](https://developers.google.com/terms/).<br> Além disso, as APIs do Acceleration Service estão atualmente na versão Beta e, por isso, ao usá-las, você reconhece os possíveis problemas descritos na seção Ressalvas acima e confirma que o Acceleration Service pode nem sempre ter o desempenho especificado.

#### Privacidade

Ao usar as APIs Acceleration Service, o processamento dos dados de entrada (por exemplo, imagens, vídeo e texto) acontecem totalmente no dispositivo, e **o Acceleration Service não envia esses dados aos servidores do Google**. Como resultado, você pode usar nossas APIs para processamento de dados de entrada que não devem deixar o dispositivo.<br> As APIs Acceleration Service podem entrar em contato com os servidores do Google eventualmente para receber, por exemplo, correções de bug, modelos atualizados e informações sobre a compatibilidade de aceleradores de hardware. As APIs Acceleration Service também podem enviar métricas sobre o desempenho e a utilização de APIs no seu app para o Google. O Google usa esses dados de métricas para medir o desempenho, depurar, manter e melhorar as APIs e detectar uso indevido ou abuso, conforme detalhado na nossa [Política de Privacidade](https://policies.google.com/privacy).<br> **Você é responsável por informar aos usuários do seu app sobre o processamento que o Google faz dos dados de métricas do Acceleration Service conforme exigido pela legislação aplicável.**<br> Os dados que coletamos incluem os seguintes:

- Informações do dispositivo (como fabricante, modelo, versão de SO e build) e aceleradores de hardware de ML disponíveis (GPU e DSP). Usadas para diagnóstico e análise de uso.
- Informações do app (nome do pacote/id do bundle, versão do app). Usadas para diagnóstico e análise de uso.
- Configuração da API (como formato e resolução da imagem). Usada para diagnóstico e análise de uso.
- Tipo de evento (como inicializar, baixar modelo, atualizar, executar, detectar). Usado para diagnóstico e análise de uso.
- Códigos de erro. Usados para diagnóstico.
- Métricas de desempenho. Usadas para diagnóstico.
- Identificadores por instalação que não identifiquem exclusivamente um usuário ou dispositivo físico. Usados para operação de configuração remota e análise de uso.
- Endereços IP do remetente de solicitação da rede. Usados para diagnóstico de configuração remota. Os endereços IP coletados são retidos temporariamente.

## Suporte e feedback

Você pode fornecer feedback e receber suporte pelo Issue Tracker do TensorFlow. Informe problemas e solicitações de suporte usando o [modelo de issue](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) para o TensorFlow Lite no Google Play Services.
