# Descripción general de la biblioteca de TensorFlow Hub

La biblioteca [`tensorflow_hub`](https://github.com/tensorflow/hub) le permite descargar y reutilizar modelos entrenados en su programa de TensorFlow con una cantidad mínima de código. La forma principal de cargar un modelo entrenado es mediante la API `hub.KerasLayer`.

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

**Nota:** En esta documentación se usan identificadores de URL de TFhub.dev en los ejemplos. Puede ver más información sobre otros tipos de identificadores válidos [aquí](tf2_saved_model.md#model_handles).

## Establecer la ubicación del caché para descargas

De forma predeterminada, `tensorflow_hub` usa un directorio temporal en todo el sistema para almacenar en caché los modelos descargados y sin comprimir. Consulte [Almacenamiento en caché](caching.md) para conocer las opciones para usar otras ubicaciones, posiblemente más persistentes.

## Estabilidad de la API

Si bien nos gustaría evitar cambios importantes, este proyecto aún se está desarrollando de forma activa y todavía no tenemos la garantía de que tenga una API o formato de modelo estable.

## Equidad

Como todo lo que sea respecto al aprendizaje automático, [la equidad](http://ml-fairness.com) es una consideración [importante](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html). Muchos modelos preentrenados se entrenan en conjuntos de datos grandes. Al reutilizar cualquier modelo, es importante tener en cuenta qué datos se usaron para entrenarlo (y si existen sesgos) y cómo podrían afectar su uso.

## Seguridad

Dado que contienen gráficos arbitrarios de TensorFlow, los modelos pueden considerarse programas. En [Usar TensorFlow de forma segura](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) se describen las implicaciones de seguridad cuando se hacer referencia a un modelo desde una fuente no confiable.

## Próximos pasos

- [Usar la biblioteca](tf2_saved_model.md)
- [SavedModel reutilizables](reusable_saved_models.md)
