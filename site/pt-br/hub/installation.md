# Instalação

## Instalação da biblioteca tensorflow_hub

A biblioteca `tensorflow_hub` pode ser instalada junto com o TensorFlow 1 e TensorFlow 2. Recomendamos que novos usuários comecem usando o TensorFlow 2 imediatamente e que usuários existentes façam o upgrade.

### Uso com o TensorFlow 2

Use o [pip](https://pip.pypa.io/) para [instalar o TensorFlow 2](https://www.tensorflow.org/install) como sempre (confira mais instruções sobre suporte a GPUs). Em seguida, instale uma versão atual de [`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/) (precisa ser a versão 0.5.0 ou mais recente).

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

A API ao estilo do TF1 do TensorFlow Hub funciona com o modo de compatibilidade com a v1 do TensorFlow 2.

### Uso legado com o TensorFlow 1

O TensorFlow 1.15 é a única versão do TensorFlow 1.x que ainda tem suporte à biblioteca `tensorflow_hub` (a partir da versão 0.11.0). O padrão do TensorFlow 1.15 é o comportamento compatível com o TF1, mas contém diversos recursos do TF2 por baixo dos panos que permitem o uso das APIs ao estilo do TF2 do TensorFlow Hub.

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### Uso de versões pré-lançamentoa

Os pacotes pip `tf-nightly` e `tf-hub-nightly` são criados automaticamente a partir do código fonte no GitHub, sem testes de versão. Dessa forma, os desenvolvedores podem testar o código mais recente sem [compilar a partir do fonte](build_from_source.md).

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## Próximos passos

- [Visão geral da biblioteca](lib_overview.md)
- Tutoriais:
    - [Classificação de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb)
    - [Classificação de imagens](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb)
    - Exemplos adicionais [no GitHub](https://github.com/tensorflow/hub/blob/master/examples/README.md)
- Encontre modelos em [tfhub.dev](https://tfhub.dev).
