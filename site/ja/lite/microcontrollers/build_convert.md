# モデルの構築と変換

マイクロコントローラの RAM とストレージは限られているため、機械学習モデルのサイズが制限されます。さらに、マイクロコントローラ向け TensorFlow Lite で現在サポートされている演算は限定されているため、すべてのモデルアーキテクチャは可能ではありません。

このドキュメントでは、TensorFlow モデルをマイクロコントローラで実行するように変換するプロセスについて説明します。また、サポートされている演算の概要と、限られたメモリに収まるようにモデルを設計およびトレーニングするためのガイダンスも提供します。

For an end-to-end, runnable example of building and converting a model, see the following Colab which is part of the *Hello World* example:

<a class="button button-primary" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb">train_hello_world_model.ipynb</a>

## モデル変換

トレーニング済みの TensorFlow モデルをマイクロコントローラで実行するように変換するには、[ TensorFlow Lite コンバータ Python API ](https://www.tensorflow.org/lite/convert/)を使用する必要があります。これにより、モデルが[ `FlatBuffer`](https://google.github.io/flatbuffers/)に変換され、モデルのサイズが小さくなり、TensorFlow Lite 演算を使用するようにモデルが変更されます。

To obtain the smallest possible model size, you should consider using [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization).

### C 配列に変換

Many microcontroller platforms do not have native filesystem support. The easiest way to use a model from your program is to include it as a C array and compile it into your program.

The following unix command will generate a C source file that contains the TensorFlow Lite model as a `char` array:

```bash
xxd -i converted_model.tflite > model_data.cc
```

出力は以下のようになります。

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

Once you have generated the file, you can include it in your program. It is important to change the array declaration to `const` for better memory efficiency on embedded platforms.

For an example of how to include and use a model in your program, see [`model.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/model.cc) in the *Hello World* example.

## Model architecture and training

When designing a model for use on microcontrollers, it is important to consider the model size, workload, and the operations that are used.

### Model size

A model must be small enough to fit within your target device's memory alongside the rest of your program, both as a binary and at runtime.

To create a smaller model, you can use fewer and smaller layers in your architecture. However, small models are more likely to suffer from underfitting. This means for many problems, it makes sense to try and use the largest model that will fit in memory. However, using larger models will also lead to increased processor workload.

Note: The core runtime for TensorFlow Lite for Microcontrollers fits in 16KB on a Cortex M3.

### ワークロード

モデルのサイズと複雑さは、ワークロードに影響を与えます。大規模で複雑なモデルでは、デューティサイクルが高くなる可能性があります。これは、デバイスのプロセッサがより多くの時間を費やしてアイドル時間が短縮されることを意味します。これにより、消費電力と熱出力が増加するためアプリによっては問題となる可能性があるます。

### 演算のサポート

TensorFlow Lite for Microcontrollers currently supports a limited subset of TensorFlow operations, which impacts the model architectures that it is possible to run. We are working on expanding operation support, both in terms of reference implementations and optimizations for specific architectures.

The supported operations can be seen in the file [`all_ops_resolver.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/all_ops_resolver.cc)
