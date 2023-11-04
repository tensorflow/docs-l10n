# Problemas comuns de implementação

Esta página descreve os problemas comuns de implementação ao implementar um novo dataset.

## O `SplitGenerator` legado deve ser evitado

A antiga API `tfds.core.SplitGenerator` está obsoleta.

```python
def _split_generator(...):
  return [
      tfds.core.SplitGenerator(name='train', gen_kwargs={'path': train_path}),
      tfds.core.SplitGenerator(name='test', gen_kwargs={'path': test_path}),
  ]
```

Deve ser substituído por:

```python
def _split_generator(...):
  return {
      'train': self._generate_examples(path=train_path),
      'test': self._generate_examples(path=test_path),
  }
```

**Justificativa**: a nova API é menos detalhada e mais explícita. A API antiga será removida na versão futura.

## Novos datasets devem armazenados em  uma pasta própria

Ao adicionar um dataset dentro do repositório `tensorflow_datasets/`, certifique-se de seguir a estrutura "dataset como pasta" (todas as checksums, dados de teste, código de implementação, etc. armazenados numa pasta).

- Datasets antigos (ruim): `<category>/<ds_name>.py`
- Novos datasets (bom): `<category>/<ds_name>/<ds_name>.py`

Use a [CLI do TFDS](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset) (`tfds new` ou `gtfds new` para googlers) para gerar o modelo.

**Justificativa**: A estrutura antiga exigia caminhos absolutos para checksums, dados falsos e distribuía os arquivos do datasets em muitos lugares. Dificultava a implementação de datasets fora do repositório TFDS. Para manter a consistência, a nova estrutura deve ser usada em todos os lugares.

## Listas de descrição devem ser formatadas como markdown

A `str` `DatasetInfo.description` é formatada como markdown. As listas em markdown requerem uma linha vazia antes do primeiro item:

```python
_DESCRIPTION = """
Some text.
                      # << Empty line here !!!
1. Item 1
2. Item 1
3. Item 1
                      # << Empty line here !!!
Some other text.
"""
```

**Justificativa**: uma descrição mal formatada cria artefatos visuais em nossa documentação de catálogo. Sem as linhas vazias, o texto acima seria renderizado como:

Some text. 1. Item 1 2. Item 1 3. Item 1 Some other text

## Nomes ClassLabel

Ao usar `tfds.features.ClassLabel`, tente fornecer rótulos `str` legíveis por humanos com `names=` ou `names_file=` (em vez de `num_classes=10`).

```python
features = {
    'label': tfds.features.ClassLabel(names=['dog', 'cat', ...]),
}
```

**Justificativa**: rótulos legíveis por humanos são usados ​​em muitos lugares:

- Permitem o uso de `str` diretamente em `_generate_examples`: `yield {'label': 'dog'}`
- Expostos nos usuários como `info.features['label'].names` (método de conversão `.str2int('dog')`,... também disponível)
- Usados nos [utilitários de visualização](https://www.tensorflow.org/datasets/overview#tfdsas_dataframe) `tfds.show_examples`, `tfds.as_dataframe`

## Formato de imagens

Ao usar `tfds.features.Image`, `tfds.features.Video`, se as imagens tiverem formato estático, elas deverão ser especificadas explicitamente:

```python
features = {
    'image': tfds.features.Image(shape=(256, 256, 3)),
}
```

**Justificativa**: permite inferência de forma estática (por exemplo, `ds.element_spec['image'].shape`), que é necessário para a criação de lotes (um lote de imagens de formato desconhecido exigiria primeiro redimensioná-las).

## Prefira um tipo mais específico em vez de `tfds.features.Tensor`

Quando possível, dê preferência aos tipos mais específicos `tfds.features.ClassLabel`, `tfds.features.BBoxFeatures`,... em vez do tipo genérico `tfds.features.Tensor`.

**Justificativa**: além de serem mais semanticamente corretos, recursos específicos fornecem metadados adicionais aos usuários e são detectados por ferramentas.

## Importações lazy no espaço global

As importações lazy não devem ser retiradas do espaço global. Por exemplo, fazer o seguinte é errado:

```python
tfds.lazy_imports.apache_beam # << Error: Import beam in the global scope

def f() -> beam.Map:
  ...
```

**Justificativa**: usar importações lazy no escopo global importaria o módulo para todos os usuários do tfds, anulando o propósito das importações lazy.

## Computação dinâmica de divisões de treinamento/teste

Se o dataset não fornecer divisões oficiais, o TFDS também não deve fazer o mesmo. O seguinte deve ser evitado:

```python
_TRAIN_TEST_RATIO = 0.7

def _split_generator():
  ids = list(range(num_examples))
  np.random.RandomState(seed).shuffle(ids)

  # Split train/test
  train_ids = ids[_TRAIN_TEST_RATIO * num_examples:]
  test_ids = ids[:_TRAIN_TEST_RATIO * num_examples]
  return {
      'train': self._generate_examples(train_ids),
      'test': self._generate_examples(test_ids),
  }
```

**Justificativa**: o TFDS tenta fornecer datasets tão próximos quanto os dados originais. A [API subsplit (subdivisão)](https://www.tensorflow.org/datasets/splits) deve ser usada para permitir que os usuários criem dinamicamente as subdivisões que desejarem:

```python
ds_train, ds_test = tfds.load(..., split=['train[:80%]', 'train[80%:]'])
```

## Guia de estilo Python

### Dê preferência ao uso da API pathlib

Em vez da API `tf.io.gfile`, é preferível usar a [API pathlib](https://docs.python.org/3/library/pathlib.html). Todos os métodos `dl_manager` retornam objetos do tipo pathlib compatíveis com GCS, S3,...

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

**Justificativa**: a API pathlib é uma API de arquivo orientada a objetos moderna que remove código boilerplate. Usar `.read_text()` / `.read_bytes()` também garante que os arquivos sejam fechados corretamente.

### Se o método não estiver usando `self`, ele deve ser uma função

Se um método de classe não estiver usando `self`, ele deveria ser uma função simples (definida fora da classe).

**Justificativa**: deixa explícito ao leitor que a função não tem efeitos colaterais, nem entrada/saída oculta:

```python
x = f(y)  # Clear inputs/outputs

x = self.f(y)  # Does f depend on additional hidden variables ? Is it stateful ?
```

## Importações lazy em Python

Importamos de forma lazy grandes módulos como o TensorFlow. As importações lazy adiam a importação real do módulo para o primeiro uso do módulo. Portanto, os usuários que não precisam deste grande módulo jamais irão importá-lo.

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
# After this statement, TensorFlow is not imported yet

...

features = tfds.features.Image(dtype=tf.uint8)
# After using it (`tf.uint8`), TensorFlow is now imported
```

Nos bastidores, a [classe `LazyModule`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/lazy_imports_utils.py) atua como uma fábrica, que só importará o módulo quando um atributo for acessado (`__getattr__`).

Você também pode usá-lo convenientemente com um gerenciador de contexto:

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import lazy_imports

with lazy_imports(error_callback=..., success_callback=...):
  import some_big_module
```
