# パフォーマンス測定

## ベンチマークツール

現在、TensorFlow Lite ベンチマークツールは次の重要なパフォーマンス指標の統計を測定および計算します。

- 初期化時間
- ウォームアップ状態の推論時間
- 定常状態の推論時間
- 初期化時のメモリ使用量
- 全体的なメモリ使用量

ベンチマークツールは、Android と iOS のベンチマークアプリ、および、ネイティブコマンドラインバイナリとして利用できます。すべて同一のコアパフォーマンス測定ロジックを共有します。ランタイム環境の違いにより、利用可能なオプションと出力形式がわずかに異なることに注意してください。

### Android ベンチマークアプリ

Android でベンチマークツールを使用するには、2つのオプションがあります。1つ目は[ネイティブベンチマークバイナリ](#native-benchmark-binary)で、2つ目は Android ベンチマークアプリです。これは、モデルがアプリでどのように機能するかをより正確に把握するためのものです。いずれの場合も、ベンチマークツールの数値は、実際のアプリでモデルを使用して推論を実行した場合とは少し異なります。

この Android ベンチマークアプリには UI がありません。`adb`コマンドを使用してインストールおよび実行し、`adb logcat`コマンドを使用して結果を取得します。

#### アプリをダウンロードまたはビルドする

以下のリンクを使用して、ナイトリービルドの Android ベンチマークアプリをダウンロードします。

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk)

[Flex デリゲート](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)を介して [TF 演算](https://www.tensorflow.org/lite/guide/ops_select)をサポートする Android ベンチマークアプリは、以下のリンクをクリックしてください。

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex.apk)

また、これらの[手順](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)に従って、ソースからアプリをビルドすることもできます。

注：x86 CPU または Hexagon デリゲートで Android ベンチマーク apk を実行する場合、またはモデルに [Select TF 演算子](../guide/ops_select)または[カスタム演算子](../guide/ops_custom)が含まれている場合は、ソースからアプリをビルドする必要があります。

#### ベンチマークを準備する

ベンチマークアプリを実行する前に、アプリをインストールし、次のようにモデルファイルをデバイスにプッシュします。

```shell
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```

#### ベンチマークを実行する

```shell
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/your_model.tflite \
              --num_threads=4"'
```

`graph`は必須パラメータです。

- `graph`: `string` <br> TFLite モデルファイルへのパス。

ベンチマークを実行するためのオプションのパラメータをさらに指定できます。

- `num_threads`: `int` (デフォルト=1) <br> TFLite インタープリタの実行に使用するスレッドの数。
- `use_gpu`: `bool` (デフォルト=false) <br>[GPU デレゲート](gpu)を使用する。
- `use_nnapi`: `bool` (デフォルト=false) <br>[NNAPI デレゲート](nnapi)を使用する。
- `use_xnnpack`: `bool` (デフォルト=`false`) <br>[XNNPACK デレゲート](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack)を使用する。
- `use_hexagon`: `bool` (デフォルト=`false`) <br>[Hexagon デレゲート](hexagon_delegate)を使用する。

使用しているデバイスによっては、これらのオプションの一部が使用できない場合や使用しても効果がない場合があります。ベンチマークアプリで実行できるその他のパフォーマンスパラメータについては、[パラメータ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters)を参照してください。

`logcat`コマンドを使用して結果を表示します。

```shell
adb logcat | grep "Average inference"
```

ベンチマーク結果は次のように報告されます。

```
... tflite  : Average inference timings in us: Warmup: 91471, Init: 4108, Inference: 80660.1
```

### ネイティブベンチマークバイナリ

ベンチマークツールは、ネイティブバイナリ`Benchmark_model`としても提供されます。このツールは、Linux、Mac、組み込みデバイス、および Android デバイスのシェルコマンドラインから実行できます。

#### バイナリをダウンロードまたはビルドする

以下のリンクを使用して、ナイトリービルドのネイティブコマンドラインバイナリをダウンロードします。

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model)

[Flex デリゲート](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)を介して [TF 演算](https://www.tensorflow.org/lite/guide/ops_select)をサポートする ナイトリ―ビルドのバイナリは、以下のリンクをクリックしてください。

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_plus_flex)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)

コンピュータの[ソース](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)からネイティブベンチマークバイナリをビルドすることもできます。

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Android NDK ツールチェーンを使用してビルドするには、まずこの[ガイド](../guide/build_android#set_up_build_environment_without_docker)に従ってビルド環境をセットアップするか、この[ガイド](../guide/build_android#set_up_build_environment_using_docker)で説明されているように Docker イメージを使用する必要があります。

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

注意：ベンチマークのために Android デバイスでバイナリを直接プッシュして実行することも可能ですが、実際の Android アプリ内での実行と比較してパフォーマンスにわずかな（観察可能な）違いが生じる可能性があります。特に、Android のスケジューラーは、スレッドとプロセスの優先度に基づいて動作を調整します。これは、フォアグラウンドアクティビティ/アプリケーションと`adb shell ...`を介して実行される通常のバックグラウンドバイナリとの間で異なります。この調整された動作は、TensorFlow Lite でマルチスレッド CPU 実行を有効にする場合に最も顕著になります。したがって、パフォーマンス測定には Android ベンチマークアプリが推奨されます。

#### ベンチマークを実行する

コンピュータでベンチマークを実行するには、シェルからバイナリを実行します。

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

上記と同じ[パラメータのセット](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters)をネイティブコマンドラインバイナリで使用できます。

#### モデル演算のプロファイリング

ベンチマークモデルバイナリを使用すると、モデル演算のプロファイルを作成し、各演算子の実行時間を取得することもできます。これを実行するには、呼び出し中にフラグ`--enable_op_profiling=true`を`benchmark_model`に渡します。詳細は[こちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators)を参照してください。

### 1回の実行で複数のパフォーマンスオプションを実行するためのネイティブベンチマークバイナリ

また、1回の実行で[複数のパフォーマンスオプション](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#benchmark-multiple-performance-options-in-a-single-run)をベンチマークするために便利でシンプルな C++ バイナリが提供されています。このバイナリは、一度に1つのパフォーマンスオプションしかベンチマークできない前述のベンチマークツールに基づいて構築されています。これらは同一のビルド/インストール/実行プロセスを共有しますが、このバイナリの BUILD ターゲット名は`Benchmark_model_performance_options`で、複数の追加パラメータを取ります。このバイナリの重要なパラメータは次のとおりです。

`perf_options_list`: `string` (デフォルト='all') <br>ベンチマークする TFLite パフォーマンスオプションのコンマ区切りリスト。

以下のリストから、このツール用にナイトリービルドのバイナリを取得できます。

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_performance_options)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_performance_options)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_performance_options)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_performance_options)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)

### iOS ベンチマークアプリ

iOS デバイスでベンチマークを実行するには、[ソース](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)からアプリをビルドする必要があります。TensorFlow Lite モデルファイルをソースツリーの [benchmark_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/benchmark_data) ディレクトリに配置し、`Benchmark_params.json`ファイルを変更します。これらのファイルはアプリにパッケージ化され、アプリはディレクトリからデータを読み取ります。詳細については、[iOS ベンチマークアプリ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)を参照してください。

## 既知のモデルのパフォーマンスベンチマーク

このセクションは、一部の Android および iOS デバイスで既知のモデルを実行した場合の TensorFlow Lite のパフォーマンスベンチマークについて説明します。

### Android パフォーマンスベンチマーク

これらのパフォーマンスベンチマーク数は、[ネイティブベンチマークバイナリ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)を使用して生成されています。

Android ベンチマークの場合、CPU アフィニティは、デバイスで多いコア数を使用して差異を減らすように設定されています（[詳細はこちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android)を参照してください）。

モデルがダウンロードされ、`/data/local/tmp/tflite_models`ディレクトリに解凍されたことが想定されます。ベンチマークバイナリは、[これらの指示](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android)を使用して構築され、`/data/local/tmp`ディレクトリにあると想定されます。

ベンチマークの実行：

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

nnapi デリゲートで実行するには、`-use_nnapi = true`を設定します。GPU デリゲートで実行するには、`-use_gpu = true`を設定します。

以下のパフォーマンス値は Android 10 で測定されています。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>デバイス</th>
      <th>CPU、4 スレッド</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>     </td>
    <td>Pixel 3</td>
    <td>23.9 ms</td>
    <td>6.45 ms</td>
    <td>13.8 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>14.0 ms</td>
    <td>9.0 ms</td>
    <td>14.8 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>     </td>
    <td>Pixel 3</td>
    <td>13.4 ms</td>
    <td>---</td>
    <td>6.0 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>5.0 ms</td>
    <td>---</td>
    <td>3.2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>     </td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>---</td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>34.5 ms</td>
    <td>---</td>
    <td>99.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>     </td>
    <td>Pixel 3</td>
    <td>35.8 ms</td>
    <td>9.5 ms</td>
    <td>18.5 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>23.9 ms</td>
    <td>11.1 ms</td>
    <td>19.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>     </td>
    <td>Pixel 3</td>
    <td>422 ms</td>
    <td>99.8 ms</td>
    <td>201 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>272.6 ms</td>
    <td>87.2 ms</td>
    <td>171.1 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>     </td>
    <td>Pixel 3</td>
    <td>486 ms</td>
    <td>93 ms</td>
    <td>292 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>324.1 ms</td>
    <td>97.6 ms</td>
    <td>186.9 ms</td>
  </tr>
 </table>

### iOS パフォーマンスベンチマーク

これらのパフォーマンスベンチマークの数値は、[iOS ベンチマークアプリ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)を使用して生成されています。

iOS ベンチマークを実行するために、適切なモデルを含めるためにベンチマークアプリが変更され、`num_threads`を2に設定するために`benchmark_params.json`が変更されました。また、GPU デリゲートを使用するために、`"use_gpu" : "1"`および`"gpu_wait_type" : "aggressive"`オプションも`benchmark_params.json`に追加されました 。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>デバイス</th>
      <th>CPU、2 スレッド</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>     </td>
    <td>iPhone XS</td>
    <td>14.8 ms</td>
    <td>3.4 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>     </td>
    <td>iPhone XS</td>
    <td>11 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>     </td>
    <td>iPhone XS</td>
    <td>30.4 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>     </td>
    <td>iPhone XS</td>
    <td>21.1 ms</td>
    <td>15.5 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>     </td>
    <td>iPhone XS</td>
    <td>261.1 ms</td>
    <td>45.7 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>     </td>
    <td>iPhone XS</td>
    <td>309 ms</td>
    <td>54.4 ms</td>
  </tr>
 </table>

## Android で TensorFlow Lite の内部をトレースする

注意：この機能は実験的なものであり、Android アプリがナイトリ―リリースの Tensorflow Lite ライブラリでビルドされている場合にのみ利用できます。v2.3 以前の安定版ライブラリはこれをサポートしていません。

Android アプリの TensorFlow Lite インタープリタからの内部イベントは、[Android トレースツール](https://developer.android.com/topic/performance/tracing)でキャプチャできます。これは Android [Trace](https://developer.android.com/reference/android/os/Trace) API と同じイベントであるため、Java/Kotlin コードからキャプチャされたイベントは TensorFlow Lite 内部イベントと共に表示されます。

イベントの例は次のとおりです。

- 演算子の呼び出し
- デリゲートによるグラフの変更
- テンソルの割り当て

トレースをキャプチャするオプションは複数ありますが、本ガイドでは Android Studio CPU Profiler とシステムトレースアプリについて説明します。その他のオプションについては、[Perfetto コマンドラインツール](https://developer.android.com/studio/command-line/perfetto)または [Systrace コマンドラインツール](https://developer.android.com/topic/performance/tracing/command-line)を参照してください。

### Java コードにトレースイベントを追加する

これは、[画像分類](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)サンプルアプリのコードスニペットです。TensorFlow Lite インタープリタは、`recognizeImage/runInference`セクションで実行されます。この手順はオプションですが、推論呼び出しが行われた場所を確認するのに役立ちます。

```java
  Trace.beginSection("recognizeImage");
  ...
  // Runs the inference call.
  Trace.beginSection("runInference");
  tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
  Trace.endSection();
  ...
  Trace.endSection();

```

### TensorFlow Lite トレースを有効にする

TensorFlow Lite トレースを有効にするには、Android アプリを起動する前に、Android システムプロパティ`debug.tflite.tracing`を1に設定します。

```shell
adb shell setprop debug.tflite.trace 1
```

TensorFlow Lite インタープリタの初期化時にこのプロパティが設定されている場合、インタープリタからの主要なイベント（演算子の呼び出しなど）がトレースされます。

すべてのトレースをキャプチャしたら、プロパティ値を0に設定してトレースを無効にします。

```shell
adb shell setprop debug.tflite.trace 0
```

### Android Studio CPU Profiler

以下の手順に従って、[Android Studio CPU Profiler](https://developer.android.com/studio/profile/cpu-profiler) でトレースをキャプチャします。

1. トップメニューから**実行&gt;プロファイル「アプリ」**を選択します。

2. プロファイラーウィンドウが表示されたら、CPU タイムラインの任意の場所をクリックします。

3. CPU プロファイリングモードから「システムコールのトレース」を選択します。

    ![[システムコールのトレース]を選択します](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/performance/images/as_select_profiling_mode.png?raw=true)

4. 「記録」ボタンを押します。

5. 「停止」ボタンを押します。

6. トレース結果を調査します。

    ![Android Studio トレース](images/as_traces.png)

この例では、スレッド内のイベントの階層と各演算子の時間の統計、および、スレッド間のアプリ全体のデータフローを確認できます。

### システムトレースアプリ

Android Studio を使用せずにトレースをキャプチャするには[システムトレースアプリ](https://developer.android.com/topic/performance/tracing/on-device)で詳しく説明されている手順に従います。

この例では、同じ TFLite イベントがキャプチャされ、Android デバイスのバージョンに応じて、Perfetto または Systrace 形式で保存されました。キャプチャされたトレースファイルは、[Perfetto UI](https://ui.perfetto.dev/#!/) で開くことができます。

![Perfetto トレース](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/performance/images/perfetto_traces.png?raw=true)

### トレースデータの使用

トレースデータを使用すると、パフォーマンスのボトルネックを特定できます。

以下はプロファイラーから得られる洞察とパフォーマンスを向上させるための潜在的なソリューションの例です。

- 使用可能な CPU コアの数が推論スレッドの数よりも少ない場合、CPU スケジューリングのオーバーヘッドがパフォーマンスを低下させる可能性があります。アプリで他の CPU を集中的に使用するタスクを再スケジュールし、モデルの推論との重複を回避したり、インタープリタースレッドの数を微調整したりできます。
- 演算子が完全にデレゲートされていない場合、モデルグラフの一部は、期待されるハードウェアアクセラレータではなく、CPU で実行されます。サポートされていない演算子は、同様のサポートされている演算子に置き換えます。
