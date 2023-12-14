# Firmas comunes para texto

En esta página se describen las firmas comunes que se deberían implementar por módulos en el [formato TF1 Hub](../tf1_hub_module.md) para tareas que aceptan entradas de texto. (Para el [formato TF2 SavedModel](../tf2_saved_model.md), consulte la [API SavedModel](../common_saved_model_apis/text.md) análoga).

## Vector de características de texto

Un módulo de **vector de características de texto** crea una representación de un vector denso a partir de las características del texto. Acepta un lote de <em>strings</em> de forma `[batch_size]` y lo mapea al tensor `float32` de forma `[batch_size, N]`. Por lo general, se le denomina **incorporaciones de texto** en dimensión `N`.

### Uso básico

```python
  embed = hub.Module("path/to/module")
  representations = embed([
      "A long sentence.",
      "single-word",
      "http://example.com"])
```

### Uso de la columna de características

```python
    feature_columns = [
      hub.text_embedding_column("comment", "path/to/module", trainable=False),
    ]
    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
    estimator.train(input_fn, max_steps=100)
```

## Notas

Los módulos han sido previamente entrenados con diferentes dominios o tareas y, por lo tanto, no todos los módulos de vectores de características de texto serían adecuados para el mismo problema. P. ej., algunos módulos podrían haber sido entrenados en un solo idioma.

Esta interfaz no permite el ajuste fino de la representación de textos en las TPU, porque exige que el módulo instancie el procesamiento de las <em>strings</em> y las variables entrenables al mismo tiempo.
