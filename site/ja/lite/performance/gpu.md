# TensorFlow Lite GPU デリゲート

[TensorFlow Lite](https://www.tensorflow.org/lite) では、複数のハードウェアアクセラレータがサポートされています。このドキュメントでは、Android および iOS で TensorFlow Lite デリゲート API を使用して GPU バックエンドを使用する方法について説明します。

GPUs are designed to have high throughput for massively parallelizable workloads. Thus, they are well-suited for deep neural nets, which consist of a huge number of operators, each working on some input tensor(s) that can be easily divided into smaller workloads and carried out in parallel, typically resulting in lower latency. In the best scenario, inference on the GPU may now run fast enough for previously not available real-time applications.

CPU とは異なり、GPU は 16 ビットまたは 32 ビットの浮動小数点数で計算し、最適なパフォーマンスを得るために量子化を必要としません。 デリゲートは 8 ビットの量子化モデルを受け入れますが、計算は浮動小数点数で実行されます。詳細については、[高度なドキュメント](gpu_advanced.md)をご覧ください。

GPU の推論のもう 1 つの利点は、電力効率です。GPU は非常に効率的かつ最適化された方法で計算を実行するため、同じタスクを CPU で実行する場合よりも消費電力と発熱が少なくなります。

## デモアプリのチュートリアル

The easiest way to try out the GPU delegate is to follow the below tutorials, which go through building our classification demo applications with GPU support. The GPU code is only binary for now; it will be open-sourced soon. Once you understand how to get our demos working, you can try this out on your own custom models.

### Android（Android Studio を使用）

For a step-by-step tutorial, watch the [GPU Delegate for Android](https://youtu.be/Xkhgre8r5G0) video.

注意: これには、OpenCL または OpenGL ES （3.1 以降）が必要です。

#### ステップ 1. TensorFlow ソースコードを複製して Android Studio で開く

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### ステップ 2. `app/build.gradle`を編集して、ナイトリーの GPU AAR を使用する

Add the `tensorflow-lite-gpu` package alongside the existing `tensorflow-lite` package in the existing `dependencies` block.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### ステップ 3. 構築して実行する

Run → Run ‘app’. When you run the application you will see a button for enabling the GPU. Change from quantized to a float model and then click GPU to run on the GPU.

![running android gpu demo and switch to gpu](images/android_gpu_demo.gif)

### iOS（XCode を使用）

For a step-by-step tutorial, watch the [GPU Delegate for iOS](https://youtu.be/a5H4Zwjp49c) video.

注意: これには XCode v10.1 以降が必要です。

#### ステップ 1. デモのソースコードを入手して、コンパイルできることを確認する。

iOS デモアプリの[チュートリアル](https://www.tensorflow.org/lite/demo_ios)に従います。これにより、変更されていない iOS カメラのデモがモバイルデバイスで機能するようになります。

#### ステップ 2. TensorFlow Lite GPU CocoaPod を使用するように Podfile を変更する

2.3.0 リリース以降では、バイナリサイズを減らすために、デフォルトで GPU デリゲートがポッドから除外されていますが、サブスペックを指定してそれらを含めることができます。`TensorFlowLiteSwift`ポッドの場合は、次のようになります。

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

または

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Objective-C（2.4.0リリース以降）または C API を使用する場合は、`TensorFlowLiteObjC` または `TensorFlowLitC` に対して同様に行うことができます。

<div>
  <devsite-expandable>
    <h4 class="showalways">2.3.0 リリースより前のバージョン</h4>
    <h4>TensorFlow Lite 2.0.0 以前</h4>
    <p>GPU デリゲートを含むバイナリ CocoaPod を構築しました。これを使用するようにプロジェクトを切り替えるには、tensorflow/tensorflow/lite/examples/ios/camera/Podfile ファイルを変更して、TensorFlowLite の代わりに TensorFlowLiteGpuExperimental ポッドを使用します。</p>
    <pre class="prettyprint lang-ruby notranslate" translate="no"><code>
    target 'YourProjectName'
      # pod 'TensorFlowLite', '1.12.0'
      pod 'TensorFlowLiteGpuExperimental'
    </code></pre>
    <h4>TensorFlow Lite 2.2.0 以前</h4>
    <p>TensorFlow Lite 2.1.0 から 2.2.0 まででは、GPU デリゲートはTensorFlowLiteC ポッドに含まれています。言語に応じて、TensorFlowLiteC または TensorFlowLiteSwift を選択できます。</p>
  </devsite-expandable>
</div>

#### ステップ 3. GPU デリゲートを有効化する

To enable the code that will use the GPU delegate, you will need to change `TFLITE_USE_GPU_DELEGATE` from 0 to 1 in `CameraExampleViewController.h`.

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### ステップ 4. デモアプリを構築して実行する

前の手順を実行すると、アプリを実行できるようになります。

#### ステップ 5. リリースモード

While in Step 4 you ran in debug mode, to get better performance, you should change to a release build with the appropriate optimal Metal settings. In particular, To edit these settings go to the `Product > Scheme > Edit Scheme...`. Select `Run`. On the `Info` tab, change `Build Configuration`, from `Debug` to `Release`, uncheck `Debug executable`.

![setting up release](images/iosdebug.png)

次に、`Options`タブをクリックし、`GPU Frame Capture`を`Disabled`に、`Metal API Validation`をc`Disabled`に変更します。

![setting up metal options](images/iosmetal.png)

最後に、必ず 64 ビットアーキテクチャでリリースのみのビルドを選択してください。`Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings`で、`Build Active Architecture Only> Release`を「Yes」に設定します。

![setting up release options](images/iosrelease.png)

## 独自のモデルで GPU デリゲートを試す

### Android

Note: The TensorFlow Lite Interpreter must be created on the same thread as where it is run. Otherwise, `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.` may occur.

モデルアクセラレーションを呼び出す方法は 2 つありますが、[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration) または TensorFlow Lite Interpreter を使用しているかによって、方法は異なります。

#### TensorFlow Lite Interpreter

Look at the demo to see how to add the delegate. In your application, add the AAR as above, import `org.tensorflow.lite.gpu.GpuDelegate` module, and use the`addDelegate` function to register the GPU delegate to the interpreter:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_32&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">iOS</h3>
<p data-md-type="paragraph">注：GPU デリゲートは、Objective-C コードで CAPI を使用することもできます。 TensorFlow Lite 2.4.0 リリース以前は、これが唯一のオプションでした。</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    import TensorFlowLite
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_33&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">サポートされているモデルと演算</h2>
<p data-md-type="paragraph">GPU デリゲートのリリースでは、バックエンドで実行できるいくつかのモデルが含まれています。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite" data-md-type="link" class="">[ダウンロード]</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite" data-md-type="link" class="_active_edit_href">[download]</a> <br><i data-md-type="raw_html">(モバイルおよび組み込みベースのビジョンアプリケーション向けに設計された画像分類モデル)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html" data-md-type="link">DeepLab セグメンテーション (257x257)</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite" data-md-type="link">[ダウンロード]</a> <br><i data-md-type="raw_html">(入力画像のすべてのピクセルにセマンティックラベル（犬、猫、車など）を割り当てる画像セグメンテーションモデル)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html" data-md-type="link">MobileNet SSD 物体検出</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite" data-md-type="link">[ダウンロード]</a> <br><i data-md-type="raw_html">(バウンディングボックスで複数のオブジェクトを検出する画像分類モデル)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet" data-md-type="link">ポーズ推定のための PoseNet</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite" data-md-type="link">[ダウンロード]</a> <br><i data-md-type="raw_html">(画像または動画内の人物のポーズを推定するビジョンモデル)</i>
</li>
</ul>
<p data-md-type="paragraph">サポートされている演算の完全なリストは、<a href="gpu_advanced.md" data-md-type="link">アドバンストドキュメント</a>をご覧ください。</p>
<h2 data-md-type="header" data-md-header-level="2">サポートされていないモデルと演算</h2>
<p data-md-type="paragraph">一部の演算が GPU デリゲートでサポートされていない場合、フレームワークは GPU でグラフの一部のみを実行し、残りの部分を CPU で実行します。CPU と GPU 同期のコストは高いため、このような分割実行モードでは、ネットワーク全体が CPU のみで実行されている場合よりもパフォーマンスが遅くなることがよくあります。この場合、ユーザーには次のような警告が表示されます。</p>
<pre data-md-type="block_code" data-md-language="none"><code class="language-none">WARNING: op code #42 cannot be handled by this delegate.
</code></pre>
<p data-md-type="paragraph">これは実行時エラーではないため、このエラーに対するコールバックは提供されませんが、開発者はデリゲートでネットワークを実行する際に確認できます。</p>
<h2 data-md-type="header" data-md-header-level="2">最適化のヒント</h2>
<p data-md-type="paragraph">演算によっては、CPU では簡単で GPU ではコストが高くなる可能性があります。このような演算の 1 つのクラスは、<code data-md-type="codespan">BATCH_TO_SPACE</code>、<code data-md-type="codespan">SPACE_TO_BATCH</code>、<code data-md-type="codespan">SPACE_TO_DEPTH</code>など、さまざまな形の変形演算です。ネットワークアーキテクトの論理的思考のためだけにこれらの演算がネットワークに挿入されている場合、パフォーマンスのためにそれらを削除することをお勧めします。</p>
<p data-md-type="paragraph">GPU では、テンソルデータは 4 チャネルにスライスされます。したがって、形状 <code data-md-type="codespan">[B,H,W,5]</code>のテンソルに対する計算は、形状<code data-md-type="codespan">[B,H,W,8]</code> のテンソルに対しする計算とほぼ同じように実行されますが、パフォーマンスは<code data-md-type="codespan">[B,H,W,4]</code>と比べて大幅に低下します。</p>
<p data-md-type="paragraph">そのため、カメラハードウェアが RGBA の画像フレームをサポートしている場合、メモリコピー (3 チャネル RGB から 4 チャネル RGBX へ) を回避できるため、4 チャネル入力のフィードは大幅に速くなります。</p>
<p data-md-type="paragraph">最高のパフォーマンスを得るには、モバイル向けに最適化されたネットワークアーキテクチャで分類器を再トレーニングします。これは、デバイス上の推論の最適化の重要な部分です。</p>
</div>
