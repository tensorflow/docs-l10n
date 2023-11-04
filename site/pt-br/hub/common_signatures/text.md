# Assinaturas comuns para texto

Esta página descreve assinaturas comuns que devem ser implementadas por módulos no [formato TF1 Hub](../tf1_hub_module.md) para tarefas que recebem texto (para o [formato SavedModel do TF2](../tf2_saved_model.md), confira a [API SavedModel](../common_saved_model_apis/text.md) análoga).

## Vetor de características de texto

Um módulo de **vetor de características de texto** cria uma representação densa de vetor a partir de características de texto. Ele recebe um lote de strings de formato `[batch_size]` (tamanho do lote) e as mapeia para um tensor `float32` de formato `[batch_size, N]`. Geralmente, isso é chamado de **embedding de texto** na dimensão `N`.

### Uso básico

```python
  embed = hub.Module("path/to/module")
  representations = embed([
      "A long sentence.",
      "single-word",
      "http://example.com"])
```

### Uso da coluna de características

```python
    feature_columns = [
      hub.text_embedding_column("comment", "path/to/module", trainable=False),
    ]
    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
    estimator.train(input_fn, max_steps=100)
```

## Observações

Os módulos foram pré-treinados em diferentes domínios e/ou tarefas e, portanto, nem todo módulo de vetor de características de texto será adequado para seu problema. Por exemplo, alguns módulos foram treinados em um único idioma.

Essa interface não permite ajustes finos da representação de texto em TPUs, pois isso requer que o módulo instancie tanto o processamento de strings quanto as variáveis treináveis ao mesmo tempo.
