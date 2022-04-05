# モバイル用 TFX

## はじめに

このガイドでは、Tensorflow Extended（TFX）を利用してオンデバイスでデプロイする機械学習モデルを作成および評価する方法を紹介します。TFX では、[TFLite](https://www.tensorflow.org/lite) のネイティブサポートが提供されているので、モバイルデバイスで非常に効率的に推論を実行できます。

このガイドでは、TFLite モデルを生成および評価するためにパイプラインに追加する変更について説明します。完全な例は[こちら](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)を参照してください。この例では、TFX が [MNIST](http://yann.lecun.com/exdb/mnist/) データセットからトレーニングされた TFLite モデルをトレーニングおよび評価する方法を示します。また、同じパイプラインを使用して、標準の Keras ベースの [SavedModel](https://www.tensorflow.org/guide/saved_model) と TFLite の両方を同時にエクスポートし、2つの品質を比較する方法を示します。

本チュートリアルは、ユーザーが TFX、コンポーネント、パイプラインに精通していることを前提としています。そうでない場合は、この[チュートリアル](https://www.tensorflow.org/tfx/tutorials/tfx/components)をご覧ください。

## 手順

TFX で TFLite モデルを作成および評価するのに必要な手順は 2 つだけです。まず、[TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) のコンテキスト内で TFLite リライターを呼び出して、トレーニング済みの TensorFlow モデルを TFLite モデルに変換します。次に、TFLite モデルを評価するように Evaluator を構成します。では、それぞれの手順を見ていきましょう。

### トレーナー内で TFLite リライターを呼び出す

TFX トレーナーは、ユーザー定義の `run_fn` がモジュールファイルで指定されていることを想定しています。この `run_fn` は、トレーニングするモデルを定義し、指定された反復回数でモデルをトレーニングし、トレーニングされたモデルをエクスポートします。

このセクションの残りの部分では、TFLite リライターを呼び出して TFLite モデルをエクスポートするために必要な変更を示すコードスニペットを提供します。このコードはすべて、[MNIST TFLite モジュール](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py)の `run_fn` にあります。

以下のコードが示すように、最初に、すべての特徴の `Tensor` を入力として受け取るシグネチャを作成する必要があります。これは、シリアル化された [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) プロトを入力として受け取る TFX のほとんどの既存のモデルと異なるので注意してください。

```python
 signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 784],
                      dtype=tf.float32,
                      name='image_floats'))
  }
```

次に、Keras モデルは通常どおりに SavedModel として保存します。

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

最後に、TFLite リライターのインスタンス (`tfrw`) を作成し、それを SavedModel で呼び出して、TFLite モデルを取得します。この TFLite モデルは、`run_fn` の呼び出し元によって提供された `serving_model_dir` に格納されます。このようにして、TFLite モデルは、すべてのダウンストリーム TFX コンポーネントがモデルを見つけることを期待する場所に格納されます。

```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```

### TFLite モデルを評価する

[TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) は、トレーニングされたモデルを分析して、さまざまな指標の品質を理解する機能を提供します。SavedModels の分析の他、TFX Evaluator は TFLite モデルも分析できるようになりました。

次のコードスニペット ([MNIST パイプライン](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)から複製) は、TFLite モデルを分析する Evaluator を構成する方法を示しています。

```python
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  ...

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  model_analyzer_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite,
  ).with_id('mnist_lite')
```

上に示したように、唯一の変更は、`model_type` フィールドを `tf_lite` に設定することだけです。TFLite モデルを分析するために、他の構成を変更する必要はありません。TFLite モデルまたは SavedModel のどちらを分析する場合でも `Evaluator` の出力はまったく同じ構造になります。

ただし、Evaluator は TFLite モデルが `tflite` というファイルの trainer_lite.outputs['model'] に保存されていることを前提とすることに注意してください。
