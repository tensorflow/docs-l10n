# O componente de pipeline InfraValidator TFX

O InfraValidator é um componente TFX usado como uma camada de alerta antecipado antes de enviar um modelo para produção. O nome validador “infra” vem do fato de estar validando o modelo no próprio modelo que atende a “infraestrutura”. Se o [Evaluator](evaluator.md) precisa garantir o desempenho do modelo, o InfraValidator precisa garantir que o modelo esteja mecanicamente correto e evitar que modelos ruins sejam enviados.

## Como funciona?

O InfraValidator pega o modelo, inicia um servidor de modelos em sandbox com o modelo e verifica se ele pode ser carregado com sucesso e, opcionalmente, pesquisado. O resultado da infra-validação será gerado na saída de `blessing` da mesma forma que o [Evaluator](evaluator.md).

O InfraValidator foca na compatibilidade entre o binário do servidor do modelo (por exemplo, [TensorFlow Serving](serving.md)) e o modelo a ser implantado. Apesar do nome validador "infra", é **responsabilidade do usuário** configurar o ambiente corretamente, e o infravalidador apenas interage com o servidor de modelos no ambiente configurado pelo usuário para verificar se funciona bem. Configurar este ambiente corretamente garantirá que a aprovação ou falha na infravalidação será um indicativo de se o modelo seria utilizável no ambiente do serviço em produção. Isto implica, mas não se limita a, algumas das seguintes questões:

1. O InfraValidator está usando o mesmo modelo binário de servidor que será usado em produção. Este é o nível mínimo para o qual o ambiente de infravalidação deve convergir.
2. O InfraValidator está usando os mesmos recursos (por exemplo, quantidade de alocação e tipo de CPU, memória e aceleradores) que serão usados ​​em produção.
3. O InfraValidator está usando o mesmo modelo de configuração de servidor que será usado em produção.

Dependendo da situação, os usuários poderão escolher até que ponto o InfraValidator deve ser idêntico ao ambiente em produção. Tecnicamente, um modelo pode ser infravalidado num ambiente Docker local e depois servido num ambiente completamente diferente (por exemplo, num cluster Kubernetes) sem problemas. No entanto, o InfraValidator não terá verificado esta divergência.

### Modo de operação

Dependendo da configuração, a infravalidação é feita num dos seguintes modos:

- Modo `LOAD_ONLY`: verifica se o modelo foi carregado com sucesso na infraestrutura de serviço ou não, **OU**
- Modo `LOAD_AND_QUERY`: modo `LOAD_ONLY` mais o envio de algumas solicitações de amostra para verificar se o modelo é capaz de servir inferências. O InfraValidator não se importa se a previsão estava correta ou não, apenas se a solicitação foi bem-sucedida ou não.

## Como usar?

Geralmente o InfraValidator é definido junto com um componente Evaluator, e sua saída é alimentada a um Pusher. Se o InfraValidator falhar, o modelo não será enviado.

```python
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=tfx.proto.EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

### Configurando um componente InfraValidator.

Existem três tipos de protos para configurar o InfraValidator.

#### `ServingSpec`

`ServingSpec` é a configuração mais importante para o InfraValidator. Ele define:

- <u>que</u> tipo de servidor modelo executar
- <u>onde</u> executá-lo

Para tipos de servidores de modelo (chamados de binários de serviço), oferecemos suporte a

- [TensorFlow Serving](serving.md)

Observação: O InfraValidator permite especificar diversas versões do mesmo tipo de servidor de modelos para atualizar a versão do servidor de modelos sem afetar a compatibilidade dos modelos. Por exemplo, o usuário pode testar a imagem `tensorflow/serving` com as versões `2.1.0` e `latest`, para garantir que o modelo também será compatível com a versão mais recente do `tensorflow/serving`.

As seguintes plataformas de serviço são atualmente suportadas:

- Docker local (o Docker deve ser instalado com antecedência)
- Kubernetes (suporte limitado apenas para KubeflowDagRunner)

A escolha do serviço binário e da plataforma de serviço é feita especificando um bloco [`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof) do `ServingSpec`. Por exemplo, para usar o binário TensorFlow Serving em execução no cluster Kubernetes, os campos `tensorflow_serving` e `kubernetes` devem ser definidos.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(
        tensorflow_serving=tfx.proto.TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=tfx.proto.KubernetesConfig()
    )
)
```

Para configurar ainda mais o `ServingSpec`, veja a [Definição do protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `ValidationSpec`

Configuração opcional para ajustar os critérios de infravalidação ou workflow.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...),
    validation_spec=tfx.proto.ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

Todos os campos ValidationSpec possuem um valor padrão sólido. Confira mais detalhes na [definição do protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `RequestSpec`

Configuração opcional para especificar como criar solicitações de exemplo ao executar a infra-validação no modo `LOAD_AND_QUERY`. Para usar o modo `LOAD_AND_QUERY`, é necessário especificar as propriedades de execução `request_spec`, bem como o canal de entrada `examples` na definição do componente.

```python
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=tfx.proto.ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
        local_docker=tfx.proto.LocalDockerConfig(),
    ),
    request_spec=tfx.proto.RequestSpec(
        # InfraValidator will look at how "classification" signature is defined
        # in the model, and automatically convert some samples from `examples`
        # artifact to prediction RPC requests.
        tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

### Produzindo um SavedModel com warmup (aquecimento)

(Da versão 0.30.0)

Já que o InfraValidator valida o modelo com solicitações reais, ele pode facilmente reutilizar essas solicitações de validação como [solicitações de warmup (aquecimento)](https://www.tensorflow.org/tfx/serving/saved_model_warmup) de um SavedModel. O InfraValidator fornece uma opção (`RequestSpec.make_warmup`) para exportar um SavedModel com warmup.

```python
infra_validator = InfraValidator(
    ...,
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)
```

Em seguida, o artefato de saída `InfraBlessing` conterá um SavedModel com warmup e também poderá ser enviado pelo [Pusher](pusher.md), assim como o artefato `Model`.

## Limitações

O InfraValidator atual ainda não está pronto e tem algumas limitações.

- Somente o formato de modelo do TensorFlow [SavedModel](/guide/saved_model) pode ser validado.

- Ao executar o TFX no Kubernetes, o pipeline deve ser executado pelo `KubeflowDagRunner` dentro do Kubeflow Pipelines. O servidor de modelos será iniciado no mesmo cluster Kubernetes e no mesmo namespace que o Kubeflow está usando.

- O InfraValidator foca principalmente em implantações no [TensorFlow Serving](serving.md) e, embora ainda seja útil, é menos exato para implantações no [TensorFlow Lite](/lite) e [TensorFlow.js](/js) ou em outros frameworks de inferência.

- Há suporte limitado no modo `LOAD_AND_QUERY` para a assinatura do método [Predict](/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def) (que é o único método exportável no TensorFlow 2). O InfraValidator requer que a assinatura do Predict consuma um[`tf.Example`](/tutorials/load_data/tfrecord#tfexample) serializado como a única entrada.

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    - Veja o exemplo de código do [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local_infraval.py) para ver como essa assinatura interage com outros componentes no TFX.
