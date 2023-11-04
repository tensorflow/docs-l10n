# Personalizar funciones de decodificación

La API `tfds.decode` le permite invalidar la función de decodificación predeterminada. El caso de uso principal es omitir la decodificación de imágenes para obtener un mejor rendimiento.

Nota: Esta API le brinda acceso al formato `tf.train.Example` de bajo nivel en el disco (según lo define el `FeatureConnector`). Esta API es para los usuarios avanzados que desean un mejor rendimiento de lectura con imágenes.

## Ejemplos de uso

### Omitir la decodificación de imágenes

Para mantener un control total sobre el proceso de decodificación o para aplicar un filtro antes de que se decodifiquen las imágenes (para un mejor rendimiento), puede omitir la decodificación de la imagen en general. Esto funciona tanto con `tfds.features.Image` como con `tfds.features.Video`.

```python
ds = tfds.load('imagenet2012', split='train', decoders={
    'image': tfds.decode.SkipDecoding(),
})

for example in ds.take(1):
  assert example['image'].dtype == tf.string  # Images are not decoded
```

### Filtrar/aleatorizar conjuntos de datos antes de decodificar las imágenes

De manera similar al ejemplo anterior, se puede usar `tfds.decode.SkipDecoding()` para insertar una personalización adicional de la canalización `tf.data` antes de decodificar la imagen. De esta manera, las imágenes filtradas no se decodificarán y se podrá usar un búfer aleatorio más grande.

```python
# Load the base dataset without decoding
ds, ds_info = tfds.load(
    'imagenet2012',
    split='train',
    decoders={
        'image': tfds.decode.SkipDecoding(),  # Image won't be decoded here
    },
    as_supervised=True,
    with_info=True,
)
# Apply filter and shuffle
ds = ds.filter(lambda image, label: label != 10)
ds = ds.shuffle(10000)
# Then decode with ds_info.features['image']
ds = ds.map(
    lambda image, label: ds_info.features['image'].decode_example(image), label)

```

### Recortar y decodificar al mismo tiempo

Para invalidar la operación `tf.io.decode_image` predeterminada, puede crear un objeto `tfds.decode.Decoder` nuevo con el decorador `tfds.decode.make_decoder()`.

```python
@tfds.decode.make_decoder()
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.feature.shape[-1],
  )

ds = tfds.load('imagenet2012', split='train', decoders={
    # With video, decoders are applied to individual frames
    'image': decode_example(),
})
```

Lo que equivale a:

```python
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.shape[-1],
  )

ds, ds_info = tfds.load(
    'imagenet2012',
    split='train',
    with_info=True,
    decoders={
        'image': tfds.decode.SkipDecoding(),  # Skip frame decoding
    },
)
ds = ds.map(functools.partial(decode_example, feature=ds_info.features['image']))
```

### Personalizar la decodificación de vídeo

Los vídeos son `Sequence(Image())`. Al aplicar decodificadores personalizados, estos se aplicarán a fotogramas individuales. Esto significa que los decodificadores de imágenes son automáticamente compatibles con el vídeo.

```python
@tfds.decode.make_decoder()
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.feature.shape[-1],
  )

ds = tfds.load('ucf101', split='train', decoders={
    # With video, decoders are applied to individual frames
    'video': decode_example(),
})
```

Lo que equivale a:

```python
def decode_frame(serialized_image):
  """Decodes a single frame."""
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=ds_info.features['video'].shape[-1],
  )


def decode_video(example):
  """Decodes all individual frames of the video."""
  video = example['video']
  video = tf.map_fn(
      decode_frame,
      video,
      dtype=ds_info.features['video'].dtype,
      parallel_iterations=10,
  )
  example['video'] = video
  return example


ds, ds_info = tfds.load('ucf101', split='train', with_info=True, decoders={
    'video': tfds.decode.SkipDecoding(),  # Skip frame decoding
})
ds = ds.map(decode_video)  # Decode the video
```

### Decodificar solo un subconjunto de las funciones

También es posible omitir totalmente algunas funciones y especificar solo las que se necesitan. Se ignorarán/omitirán todas las demás funciones.

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': True,
    'metadata': {'num_objects', 'scene_name'},
    'objects': {'label'},
})
```

TFDS seleccionará el subconjunto de `builder.info.features` que coincida con la estructura `tfds.decode.PartialDecoding` dada.

En el código anterior, los elementos destacados se extraen implícitamente para que coincidan con `builder.info.features`. También es posible definir explícitamente las características. El código anterior es equivalente a:

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': tfds.features.Image(),
    'metadata': {
        'num_objects': tf.int64,
        'scene_name': tfds.features.Text(),
    },
    'objects': tfds.features.Sequence({
        'label': tfds.features.ClassLabel(names=[]),
    }),
})
```

Los metadatos originales (nombres de etiquetas, forma de imagen,...) se reusan automáticamente por lo que no es necesario proporcionarlos.

`tfds.decode.SkipDecoding` se puede pasar a `tfds.decode.PartialDecoding`, a través de los kwargs `PartialDecoding(..., decoders={})`.
