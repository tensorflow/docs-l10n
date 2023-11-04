# Guia de início rápido para dispositivos Linux com Python

Usar o TensorFlow Lite com o Python é excelente para dispositivos Linux embarcados, como [Raspberry Pi](https://www.raspberrypi.org/){:.external} e [dispositivos Coral com Edge TPU](https://coral.withgoogle.com/){:.external}, entre muitos outros

Esta página mostra como começar a executar modelos do TensorFlow Lite com o Python em questão de minutos. Você só precisa de um modelo do TensorFlow [convertido para o TensorFlow Lite](../models/convert/) (caso você ainda não tenha um modelo convertido, pode usar o modelo fornecido no exemplo indicado abaixo).

## Sobre o pacote de runtime do TensorFlow Lite

Para começar rapidamente a executar modelos do TensorFlow Lite com o Python, você pode instalar apenas o interpretador do TensorFlow Lite, em vez de todos os pacotes do TensorFlow. Esse pacote Python simplificado é chamado de `tflite_runtime`.

O pacote `tflite_runtime` tem uma fração do tamanho do pacote completo `tensorflow` e inclui o código mínimo necessário para executar inferências com o TensorFlow Lite, principalmente a classe do Python [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter). Esse pacote pequeno é ideal quando você só quer executar modelos `.tflite` e evitar o desperdício de espaço em disco com a biblioteca grande do TensorFlow.

Observação: se você precisar de acesso a outras APIs do Python, como [TensorFlow Lite Converter](../models/convert/) (conversor do TF Lite), precisa instalar o [pacote completo do TensorFlow](https://www.tensorflow.org/install/). Por exemplo: as operações específicas do TF (https://www.tensorflow.org/lite/guide/ops_select) não estão incluídas no pacote `tflite_runtime`. Se os seus modelos tiverem dependências para alguma operação específica do TF, você precisará usar o pacote completo do TensorFlow.

## Instale o TensorFlow Lite para Python

É possível instalar no Linux pelo pip:

<pre class="devsite-terminal devsite-click-to-copy">
python3 -m pip install tflite-runtime
</pre>

## Plataformas com suporte

Os Wheels do Python `tflite-runtime` são pré-compilados e fornecidos para estas plataformas:

- Linux armv7l (por exemplo: Raspberry Pi 2, 3, 4 e Zero 2 com Raspberry Pi OS de 32 bits)
- Linux aarch64 (por exemplo: Raspberry Pi 3, com Debian ARM64)
- Linux x86_64

Se você quiser executar modelos do TensorFlow Lite em outras plataformas, deve usar o [pacote completo do TensorFlow](https://www.tensorflow.org/install/) ou [compilar o pacote tflite-runtime a partir do código-fonte](build_cmake_pip.md).

Se você estiver usando o TensorFlow com Coral Edge TPU, deve seguir a [documentação de configuração do Coral](https://coral.ai/docs/setup).

Observação: não atualizamos mais o pacote Debian `python3-tflite-runtime`. O último pacote Debian é para o TF versão 2.5, que você pode instalar seguindo [estas instruções antigas](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python).

Observação: não lançamos mais Wheels `tflite-runtime` pré-compilados para Windows e macOS. Para essas plataformas, você deve usar o [pacote completo do TensorFlow](https://www.tensorflow.org/install/) ou [compilar o pacote tflite-runtime a partir do código-fonte](build_cmake_pip.md).

## Execute uma inferência usando o tflite_runtime

Em vez de importar `Interpreter` do módulo `tensorflow`, agora você precisa importá-lo do `tflite_runtime`.

Por exemplo: após instalar o pacote acima, copie e execute o arquivo [`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/). Provavelmente, haverá uma falha, pois a biblioteca `tensorflow` não está instalada. Para corrigir esse problema, edite esta linha do arquivo:

```python
import tensorflow as tf
```

Para:

```python
import tflite_runtime.interpreter as tflite
```

E altere esta linha

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

Para:

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

Agora, execute `label_image.py` novamente. Pronto! Agora, você está executando modelos do TensorFlow Lite.

## Saiba mais

- Para saber mais sobre a API `Interpreter`, leia [Carregue e execute um modelo no Python](inference.md#load-and-run-a-model-in-python).

- Se você tiver um Raspberry Pi, confira uma [série de vídeos](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe) sobre como executar detecção de objetos no Raspberry Pi usando o TensorFlow Lite.

- Se você estiver usando um acelerador Coral ML, confira os [exemplos do Coral no GitHub](https://github.com/google-coral/tflite/tree/master/python/examples).

- Para converter outros modelos do TensorFlow para TensorFlow Lite, leia mais sobre o [TensorFlow Lite Converter](../models/convert/) (conversor do TF Lite).

- Se você quiser compilar o Wheel `tflite_runtime`, leia [Compile o pacote Wheel do Python para TensorFlow Lite](build_cmake_pip.md)
