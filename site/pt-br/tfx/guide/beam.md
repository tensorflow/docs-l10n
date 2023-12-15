# Apache Beam e TFX

O [Apache Beam](https://beam.apache.org/) fornece um framework para executar tarefas de processamento de dados em lote e via streaming que são executadas em vários motores de execução. Diversas bibliotecas do TFX usam o Beam para executar tarefas, o que permite um alto grau de escalabilidade em clusters de computação. O Beam inclui suporte para uma variedade de motores de execução ou "executores", incluindo um executor direto que é executado num único nó de computação e é muito útil para desenvolvimento, testes ou pequenas implantações. O Beam fornece uma camada de abstração que permite que o TFX seja executado em qualquer executor compatível sem modificações no código. O TFX usa a API Beam em Python, portanto, é limitado aos executores compatíveis com a API Python.

## Implantação e escalabilidade

À medida que os requisitos de carga de trabalho aumentam, o Beam pode ser escalonado para implantações muito grandes em grandes clusters de computação. Isto é limitado apenas pela escalabilidade do executor subjacente. Os executores em grandes implantações normalmente serão implantados num sistema de orquestração de containers, como Kubernetes ou Apache Mesos, para automatizar a implantação, o escalonamento e o gerenciamento de aplicativos.

Veja a documentação do [Apache Beam](https://beam.apache.org/) para mais informações sobre o Apache Beam.

Para usuários do Google Cloud, o [Dataflow](https://cloud.google.com/dataflow) é o executor recomendado, que fornece uma plataforma econômica e sem servidor por meio de escalonamento automático de recursos, rebalanceamento dinâmico de trabalho, integração profunda com outros serviços do Google Cloud, segurança integrada e monitoramento.

## Código Python personalizado e dependências

Uma complexidade notável do uso do Beamnum pipeline TFX é lidar com código personalizado e/ou as dependências necessárias de módulos Python adicionais. Aqui estão alguns exemplos de quando isso pode ser um problema:

- preprocessing_fn precisa se referir ao próprio módulo Python do usuário
- um extrator personalizado para o componente Evaluator
- módulos personalizados que herdam (como subclasses) de um componente TFX

O TFX conta com o suporte do Beam para [gerenciar dependências de pipeline do Python](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) para lidar com dependências do Python. Atualmente existem duas maneiras de gerenciar isso:

1. Fornecendo código Python e dependências como um pacote fonte
2. [Somente Dataflow] Usando uma imagem de container como worker

Essas soluções são discutidas a seguir.

### Fornecendo código Python e dependências como um pacote fonte

Isto é recomendado para usuários que:

1. Estão familiarizados com empacotamento Python e
2. Usem apenas código-fonte Python (ou seja, sem módulos C ou bibliotecas compartilhadas).

Siga um dos caminhos em [Gerenciando dependências de pipeline do Python](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) para fornecer isto usando um dos seguintes argumentos beam_pipeline_args:

- --setup_file
- --extra_package
- --requirements_file

Aviso: Em qualquer um dos casos acima, certifique-se de que a mesma versão do `tfx` esteja listada como uma dependência.

### [Somente Dataflow] Usando uma imagem de container para um worker

O TFX 0.26.0 e versões posteriores têm suporte experimental para o uso de [imagem de container personalizada](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images) para workers Dataflow.

Para usar isto, você precisa:

- Criar uma imagem Docker que tenha `tfx` e o código personalizado e as dependências dos usuários pré-instalados.
    - Para usuários que (1) usam `tfx>=0.26` e (2) usam python 3.7 para desenvolver seus pipelines, a maneira mais fácil de fazer isso é estendendo a versão correspondente da imagem oficial `tensorflow/tfx`:

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

- Enviar a imagem criada para um registro de imagens em container que é acessível pelo projeto usado pelo Dataflow.
    - Os usuários do Google Cloud podem considerar o uso do [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build), que automatiza perfeitamente as etapas acima.
- Fornecer os seguintes `beam_pipeline_args`:

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562): remover use_runner_v2 quando ele for padrão para o Dataflow.**

**TODO(b/179738639): criar documentação sobre como testar o container personalizado localmente após https://issues.apache.org/jira/browse/BEAM-5440.**

## Argumentos do pipeline do Beam

Diversos componentes do TFX dependem do Beam para processamento distribuído de dados. Eles são configurados com `beam_pipeline_args`, que é especificado durante a criação do pipeline:

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

O TFX 0.30 ou superior adiciona uma interface, `with_beam_pipeline_args`, para estender os argumentos do Beam no nível do pipeline por componente:

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```
