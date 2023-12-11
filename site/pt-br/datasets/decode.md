# Personalizando a decodificação de características

A API `tfds.decode` permite substituir a decodificação padrão de características. O principal caso de uso é pular a decodificação da imagem para melhor desempenho.

Observação: esta API fornece acesso ao formato de baixo nível `tf.train.Example` no disco (conforme definido pelo `FeatureConnector`). Esta API é voltada para usuários avançados que buscam melhor desempenho na leitura de imagens.

## Exemplos de uso

### Pulando a decodificação da imagem

Para manter controle total sobre o pipeline de decodificação ou para aplicar um filtro antes que as imagens sejam decodificadas (para melhor desempenho), você pode pular totalmente a decodificação da imagem. Isto funciona com `tfds.features.Image` e `tfds.features.Video`.

```python
ds = tfds.load('imagenet2012', split='train', decoders={
    'image': tfds.decode.SkipDecoding(),
})

for example in ds.take(1):
  assert example['image'].dtype == tf.string  # Images are not decoded
```

### Filtre/embaralhe o dataset antes que as imagens sejam decodificadas

Da mesma forma que no exemplo anterior, você pode usar `tfds.decode.SkipDecoding()` para inserir personalização adicional no pipeline `tf.data` antes de decodificar a imagem. Dessa forma, as imagens filtradas não serão decodificadas e você poderá usar um buffer aleatório maior.

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

### Recortando e decodificando ao mesmo tempo

Para substituir a operação padrão `tf.io.decode_image`, você pode criar um novo objeto `tfds.decode.Decoder` usando o decorador `tfds.decode.make_decoder()`.

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

O que equivale a:

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

### Personalizando a decodificação de vídeo

Vídeos são `Sequence(Image())`. Ao aplicar decodificadores personalizados, eles serão aplicados a quadros individuais. Isto significa que os decodificadores de imagens são automaticamente compatíveis com vídeo.

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

O que equivale a:

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

### Decodifique apenas um subconjunto das características.

Também é possível ignorar completamente algumas características especificando apenas as características necessárias. Todas as outras características serão ignoradas/puladas.

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': True,
    'metadata': {'num_objects', 'scene_name'},
    'objects': {'label'},
})
```

O TFDS selecionará o subconjunto de `builder.info.features` que corresponde à estrutura `tfds.decode.PartialDecoding` fornecida.

No código acima, os recursos são extraídos implicitamente para corresponder com `builder.info.features`. Também é possível definir as características explicitamente. O código acima é equivalente a:

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

Os metadados originais (nomes dos rótulos, formato da imagem,...) são reutilizados automaticamente, portanto não é necessário fornecê-los.

`tfds.decode.SkipDecoding` pode ser passado para `tfds.decode.PartialDecoding`, através dos kwargs `PartialDecoding(..., decoders={})`.
