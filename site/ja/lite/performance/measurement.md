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

Android でベンチマークツールを使用するには、2つのオプションがあります。1つ目はネイティブベンチマークバイナリで、2つ目は Android ベンチマークアプリです。これは、モデルがアプリでどのように機能するかをより正確に把握するためのものです。いずれの場合も、ベンチマークツールの数値は、実際のアプリでモデルを使用して推論を実行した場合とは少し異なります。

この Android ベンチマークアプリには UI がありません。adbコマンドを使用してインストールおよび実行し、adb logcatコマンドを使用して結果を取得します。

#### アプリをダウンロードまたはビルドする

以下のリンクを使用して、ナイトリービルドの Android ベンチマークアプリをダウンロードします。

- android_aarch64
- android_arm

Flex デリゲートを介して TF 演算をサポートする Android ベンチマークアプリは、以下のリンクをクリックしてください。

- android_aarch64
- android_arm

また、これらの説明に従って、ソースからアプリをビルドすることもできます。

注：x86 CPU または Hexagon デリゲートで Android ベンチマーク apk を実行する場合、またはモデルに Select TF 演算子またはカスタム演算子が含まれている場合は、ソースからアプリをビルドする必要があります。

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

graphは必須パラメータです。

- graph: string  TFLite モデルファイルへのパス。

ベンチマークを実行するためのオプションのパラメータをさらに指定できます。

- num_threads: int (デフォルト=1)  TFLite インタープリタの実行に使用するスレッドの数。
- use_gpu: bool (デフォルト=false) GPU デレゲートを使用する。
- use_nnapi: bool (デフォルト=false) NNAPI デレゲートを使用する。
- use_xnnpack: bool (デフォルト=false) XNNPACK デレゲートを使用する。
- use_hexagon: bool (デフォルト=false) Hexagon デレゲートを使用する。

使用しているデバイスによっては、これらのオプションの一部が使用できない場合や使用しても効果がない場合があります。ベンチマークアプリで実行できるその他のパフォーマンスパラメータについては、パラメータを参照してください。

logcatコマンドを使用して結果を表示します。

```shell
adb logcat | grep "Average inference"
```

ベンチマーク結果は次のように報告されます。

```
... tflite  : Average inference timings in us: Warmup: 91471, Init: 4108, Inference: 80660.1
```

### ネイティブベンチマークバイナリ

ベンチマークツールは、ネイティブバイナリBenchmark_modelとしても提供されます。このツールは、Linux、Mac、組み込みデバイス、および Android デバイスのシェルコマンドラインから実行できます。

#### バイナリをダウンロードまたはビルドする

以下のリンクを使用して、ナイトリービルドのネイティブコマンドラインバイナリをダウンロードします。

- linux_x86-64
- linux_aarch64
- linux_arm
- android_aarch64
- android_arm

Flex デリゲートを介して TF 演算をサポートする ナイトリ―ビルドのバイナリは、以下のリンクをクリックしてください。

- linux_x86-64
- linux_aarch64
- linux_arm
- android_aarch64
- android_arm

To benchmark with [TensorFlow Lite Hexagon delegate](https://www.tensorflow.org/lite/performance/hexagon_delegate), we have also pre-built the required `libhexagon_interface.so` files (see [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md) for details about this file). After downloading the file of the corresponding platform from the links below, please rename the file to `libhexagon_interface.so`.

- linux_x86-64
- linux_aarch64

コンピュータのソースからネイティブベンチマークバイナリをビルドすることもできます。

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Android NDK ツールチェーンを使用してビルドするには、まずこのガイドに従ってビルド環境をセットアップするか、このガイドで説明されているように Docker イメージを使用する必要があります。

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

注意：ベンチマークのために Android デバイスでバイナリを直接プッシュして実行することも可能ですが、実際の Android アプリ内での実行と比較してパフォーマンスにわずかな（観察可能な）違いが生じる可能性があります。特に、Android のスケジューラーは、スレッドとプロセスの優先度に基づいて動作を調整します。これは、フォアグラウンドアクティビティ/アプリケーションとadb shell ...を介して実行される通常のバックグラウンドバイナリとの間で異なります。この調整された動作は、TensorFlow Lite でマルチスレッド CPU 実行を有効にする場合に最も顕著になります。したがって、パフォーマンス測定には Android ベンチマークアプリが推奨されます。

#### ベンチマークを実行する

コンピュータでベンチマークを実行するには、シェルからバイナリを実行します。

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

上記と同じパラメータのセットをネイティブコマンドラインバイナリで使用できます。

#### モデル演算のプロファイリング

ベンチマークモデルバイナリを使用すると、モデル演算のプロファイルを作成し、各演算子の実行時間を取得することもできます。これを実行するには、呼び出し中にフラグ--enable_op_profiling=trueをbenchmark_modelに渡します。詳細はこちらを参照してください。

### 1回の実行で複数のパフォーマンスオプションを実行するためのネイティブベンチマークバイナリ

また、1回の実行で複数のパフォーマンスオプションをベンチマークするために便利でシンプルな C++ バイナリが提供されています。このバイナリは、一度に1つのパフォーマンスオプションしかベンチマークできない前述のベンチマークツールに基づいて構築されています。これらは同一のビルド/インストール/実行プロセスを共有しますが、このバイナリの BUILD ターゲット名はBenchmark_model_performance_optionsで、複数の追加パラメータを取ります。このバイナリの重要なパラメータは次のとおりです。

perf_options_list: string (デフォルト='all') ベンチマークする TFLite パフォーマンスオプションのコンマ区切りリスト。

以下のリストから、このツール用にナイトリービルドのバイナリを取得できます。

- linux_x86-64
- linux_aarch64
- linux_arm
- android_aarch64
- android_arm

### iOS ベンチマークアプリ

iOS デバイスでベンチマークを実行するには、ソースからアプリをビルドする必要があります。TensorFlow Lite モデルファイルをソースツリーの benchmark_data ディレクトリに配置し、Benchmark_params.jsonファイルを変更します。これらのファイルはアプリにパッケージ化され、アプリはディレクトリからデータを読み取ります。詳細については、iOS ベンチマークアプリを参照してください。

## 既知のモデルのパフォーマンスベンチマーク

このセクションは、一部の Android および iOS デバイスで既知のモデルを実行した場合の TensorFlow Lite のパフォーマンスベンチマークについて説明します。

### Android パフォーマンスベンチマーク

これらのパフォーマンスベンチマーク数は、ネイティブベンチマークバイナリを使用して生成されています。

Android ベンチマークの場合、CPU アフィニティは、デバイスで多いコア数を使用して差異を減らすように設定されています（詳細はこちらを参照してください）。

モデルがダウンロードされ、/data/local/tmp/tflite_modelsディレクトリに解凍されたことが想定されます。ベンチマークバイナリは、これらの指示を使用して構築され、/data/local/tmpディレクトリにあると想定されます。

ベンチマークの実行：

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

nnapi デリゲートで実行するには、-use_nnapi = trueを設定します。GPU デリゲートで実行するには、-use_gpu = trueを設定します。

以下のパフォーマンス値は Android 10 で測定されています。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>Device </th>
      <th>CPU、4 スレッド</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       Mobilenet_1.0_224(float)     </td>
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
    <td rowspan="2">       Mobilenet_1.0_224 (quant)     </td>
    <td>Pixel 3</td>
    <td>13.4 ms</td>
    <td>--- </td>
    <td>6.0 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>5.0 ms</td>
    <td>--- </td>
    <td>3.2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       NASNet mobile     </td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>--- </td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>34.5 ms</td>
    <td>--- </td>
    <td>99.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       SqueezeNet     </td>
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
    <td rowspan="2">       Inception_ResNet_V2     </td>
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
    <td rowspan="2">       Inception_V4     </td>
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

これらのパフォーマンスベンチマークの数値は、iOS ベンチマークアプリを使用して生成されています。

iOS ベンチマークを実行するために、適切なモデルを含めるためにベンチマークアプリが変更され、num_threadsを2に設定するためにbenchmark_params.jsonが変更されました。また、GPU デリゲートを使用するために、"use_gpu" : "1"および"gpu_wait_type" : "aggressive"オプションもbenchmark_params.jsonに追加されました 。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>Device </th>
      <th>CPU、2 スレッド</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       Mobilenet_1.0_224(float)     </td>
    <td>iPhone XS</td>
    <td>14.8 ms</td>
    <td>3.4 ms</td>
  </tr>
  <tr>
    <td>       Mobilenet_1.0_224 (quant)     </td>
    <td>iPhone XS</td>
    <td>11 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       NASNet mobile     </td>
    <td>iPhone XS</td>
    <td>30.4 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       SqueezeNet     </td>
    <td>iPhone XS</td>
    <td>21.1 ms</td>
    <td>15.5 ms</td>
  </tr>
  <tr>
    <td>       Inception_ResNet_V2     </td>
    <td>iPhone XS</td>
    <td>261.1 ms</td>
    <td>45.7 ms</td>
  </tr>
  <tr>
    <td>       Inception_V4     </td>
    <td>iPhone XS</td>
    <td>309 ms</td>
    <td>54.4 ms</td>
  </tr>
 </table>

## Trace TensorFlow Lite internals

### Android で TensorFlow Lite の内部をトレースする

Note: This feature is available from Tensorflow Lite v2.4.

Internal events from the TensorFlow Lite interpreter of an Android app can be captured by [Android tracing tools](https://developer.android.com/topic/performance/tracing). They are the same events with Android [Trace](https://developer.android.com/reference/android/os/Trace) API, so the captured events from Java/Kotlin code are seen together with TensorFlow Lite internal events.

イベントの例は次のとおりです。

- 演算子の呼び出し
- デリゲートによるグラフの変更
- テンソルの割り当て

トレースをキャプチャするオプションは複数ありますが、本ガイドでは Android Studio CPU Profiler とシステムトレースアプリについて説明します。その他のオプションについては、Perfetto コマンドラインツールまたは Systrace コマンドラインツールを参照してください。

#### Java コードにトレースイベントを追加する

これは、画像分類サンプルアプリのコードスニペットです。TensorFlow Lite インタープリタは、recognizeImage/runInferenceセクションで実行されます。この手順はオプションですが、推論呼び出しが行われた場所を確認するのに役立ちます。

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

#### TensorFlow Lite トレースを有効にする

To enable TensorFlow Lite tracing, set the Android system property `debug.tflite.trace` to 1 before starting the Android app.

```shell
adb shell setprop debug.tflite.trace 1
```

TensorFlow Lite インタープリタの初期化時にこのプロパティが設定されている場合、インタープリタからの主要なイベント（演算子の呼び出しなど）がトレースされます。

すべてのトレースをキャプチャしたら、プロパティ値を0に設定してトレースを無効にします。

```shell
adb shell setprop debug.tflite.trace 0
```

#### Android Studio CPU Profiler

以下の手順に従って、Android Studio CPU Profiler でトレースをキャプチャします。

1. トップメニューから実行&gt;プロファイル「アプリ」を選択します。

2. プロファイラーウィンドウが表示されたら、CPU タイムラインの任意の場所をクリックします。

3. CPU プロファイリングモードから「システムコールのトレース」を選択します。

    ![Select 'Trace System Calls'](images/as_select_profiling_mode.png)

4. 「記録」ボタンを押します。

5. 「停止」ボタンを押します。

6. トレース結果を調査します。

    ![Android Studio trace](images/as_traces.png)

この例では、スレッド内のイベントの階層と各演算子の時間の統計、および、スレッド間のアプリ全体のデータフローを確認できます。

#### システムトレースアプリ

Android Studio を使用せずにトレースをキャプチャするにはシステムトレースアプリで詳しく説明されている手順に従います。

この例では、同じ TFLite イベントがキャプチャされ、Android デバイスのバージョンに応じて、Perfetto または Systrace 形式で保存されました。キャプチャされたトレースファイルは、Perfetto UI で開くことができます。

![Perfetto trace](images/perfetto_traces.png)

### Trace TensorFlow Lite internals in iOS

Note: This feature is available from Tensorflow Lite v2.5.

Internal events from the TensorFlow Lite interpreter of an iOS app can be captured by [Instruments](https://developer.apple.com/library/archive/documentation/ToolsLanguages/Conceptual/Xcode_Overview/MeasuringPerformance.html#//apple_ref/doc/uid/TP40010215-CH60-SW1) tool included with Xcode. They are the iOS [signpost](https://developer.apple.com/documentation/os/logging/recording_performance_data) events, so the captured events from Swift/Objective-C code are seen together with TensorFlow Lite internal events.

イベントの例は次のとおりです。

- 演算子の呼び出し
- デリゲートによるグラフの変更
- テンソルの割り当て

#### TensorFlow Lite トレースを有効にする

Set the environment variable `debug.tflite.trace` by following the steps below:

1. Select **Product &gt; Scheme &gt; Edit Scheme...** from the top menus of Xcode.

2. Click 'Profile' in the left pane.

3. Deselect 'Use the Run action's arguments and environment variables' checkbox.

4. Add `debug.tflite.trace` under 'Environment Variables' section.

    ![Set environment variable](images/xcode_profile_environment.png)

If you want to exclude TensorFlow Lite events when profiling the iOS app, disable tracing by removing the environment variable.

#### XCode Instruments

Capture traces by following the steps below:

1. Select **Product &gt; Profile** from the top menus of Xcode.

2. Click **Logging** among profiling templates when Instruments tool launches.

3. Press 'Start' button.

4. 「停止」ボタンを押します。

5. Click 'os_signpost' to expand OS Logging subsystem items.

6. Click 'org.tensorflow.lite' OS Logging subsystem.

7. トレース結果を調査します。

    ![Xcode Instruments trace](images/xcode_traces.png)

In this example, you can see the hierarchy of events and statistics for each operator time.

### トレースデータの使用

トレースデータを使用すると、パフォーマンスのボトルネックを特定できます。

以下はプロファイラーから得られる洞察とパフォーマンスを向上させるための潜在的なソリューションの例です。

- 使用可能な CPU コアの数が推論スレッドの数よりも少ない場合、CPU スケジューリングのオーバーヘッドがパフォーマンスを低下させる可能性があります。アプリで他の CPU を集中的に使用するタスクを再スケジュールし、モデルの推論との重複を回避したり、インタープリタースレッドの数を微調整したりできます。
- 演算子が完全にデレゲートされていない場合、モデルグラフの一部は、期待されるハードウェアアクセラレータではなく、CPU で実行されます。サポートされていない演算子は、同様のサポートされている演算子に置き換えます。
