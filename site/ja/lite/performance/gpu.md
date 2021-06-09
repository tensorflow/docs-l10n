# TensorFlow Lite GPU デリゲート

[TensorFlow Lite](https://www.tensorflow.org/lite) では、複数のハードウェアアクセラレータがサポートされています。このドキュメントでは、Android および iOS で TensorFlow Lite デリゲート API を使用して GPU バックエンドを使用する方法について説明します。

GPU は、大規模に実行する並列化可能なワークロードで高い処理能力を実現するように設計されています。そのため、これは多数の演算で構成されるディープニューラルネットに適しています。各演算は、より小さなワークロードに簡単に分割でき、並列に実行する入力テンソルで機能するため、通常レイテンシが低くなります。現在、最良のシナリオでは、GPU での推論は以前は利用できなかったリアルタイムアプリケーションで十分に速く実行できます。

CPU とは異なり、GPU は 16 ビットまたは 32 ビットの浮動小数点数で計算し、最適なパフォーマンスを得るために量子化を必要としません。デリゲートは 8 ビットの量子化モデルを受け入れますが、計算は浮動小数点数で実行されます。詳細については、[高度なドキュメント](gpu_advanced.md)をご覧ください。

GPU の推論のもう 1 つの利点は、電力効率です。GPU は非常に効率的かつ最適化された方法で計算を実行するため、同じタスクを CPU で実行する場合よりも消費電力と発熱が少なくなります。

## デモアプリのチュートリアル

以下のチュートリアルに従って GPU をサポートする分類デモアプリケーションを構築し、GPU デリゲートを簡単に試すことができます。現時点では、GPU コードはバイナリのみですが、近日中にオープンソースになります。デモが機能することを確認したら、独自のカスタムモデルでお試しください。

### Android（Android Studio を使用）

ステップバイステップのチュートリアルは、[Android 向け GPU デリゲート](https://youtu.be/Xkhgre8r5G0)の動画をご覧ください。

注意: これには、OpenCL または OpenGL ES （3.1 以降）が必要です。

#### ステップ 1. TensorFlow ソースコードを複製して Android Studio で開く

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### ステップ 2. `app/build.gradle`を編集して、ナイトリーの GPU AAR を使用する

`tensorflow-lite-gpu`パッケージを既存の`tensorflow-lite`パッケージと共に、既存の`dependencies`ブロックに追加します。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### ステップ 3. 構築して実行する

Run → Run ‘app’ (実行 → 「アプリ」を実行)。アプリを実行すると、GPU を有効にするためのボタンが表示されます。量子化モデルから浮動小数点数モデルに変更し、GPU をクリックして GPU で実行します。

![running android gpu demo and switch to gpu](images/android_gpu_demo.gif)

### iOS（XCode を使用）

ステップバイステップのチュートリアルは、[iOS 向け GPU デリゲート](https://youtu.be/a5H4Zwjp49c)の動画をご覧ください。

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

Objective-C（2.4.0 リリース以降）または C API を使用する場合は、`TensorFlowLiteObjC`または`TensorFlowLitC`に対して同様に行うことができます。

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

GPU デリゲートを使用するコードを有効にするには、`CameraExampleViewController.h`で`TFLITE_USE_GPU_DELEGATE`を 0 から 1 に変更する必要があります。

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### ステップ 4. デモアプリを構築して実行する

前述の手順に従うと、アプリを実行できるようになります。

#### ステップ 5. リリースモード

パフォーマンスを向上させるには、ステップ 4 でデバッグモードで実行している間、適切な最適な Metal 設定でリリースビルドに変更する必要があります。特に、これらの設定を編集するには、`Product > Scheme > Edit Scheme...`に移動します。 `Run`を選択します。`Info`タブで、`Build Configuration `を`Debug` から`Release`に変更し、`Debug executable`のチェックを外します。

![リリースの設定](images/iosdebug.png)

次に、`Options`タブをクリックし、`GPU Frame Capture`を`Disabled`に、`Metal API Validation`をc`Disabled`に変更します。

![メタルオプションの設定](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/performance/images/iosmetal.png?raw=true)

最後に、必ず 64 ビットアーキテクチャでリリースのみのビルドを選択してください。`Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings`で、`Build Active Architecture Only> Release`を「Yes」に設定します。

![setting up release options](images/iosrelease.png)

## 独自のモデルで GPU デリゲートを試す

### Android

注意: TensorFlow Lite インタープリタは、実行時と同じスレッドで作成する必要があります。そうでないと、「`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`」が表示される可能性があります。

モデルアクセラレーションを呼び出す方法は 2 つありますが、[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration) または TensorFlow Lite インタープリタを使用しているかによって、方法は異なります。

#### TensorFlow Lite インタープリタ

デモを見て、デリゲートを追加する方法を確認してください。アプリで、上記のように AAR を追加し、`org.tensorflow.lite.gpu.GpuDelegate`モジュールをインポートし、`addDelegate`関数を使用して GPU デリゲートをインタープリタに登録します。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate
&amp;amp;lt;/div&amp;amp;gt;
&amp;amp;lt;pre data-md-type=&amp;quot;block_code&amp;quot; data-md-language=&amp;quot;&amp;quot;&amp;amp;gt;&amp;amp;lt;code&amp;amp;gt;GL_CODE_37&amp;amp;lt;/code&amp;amp;gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">iOS</h3>
<p data-md-type="paragraph">注意：GPU デリゲートは、Objective-C コードで CAPI を使用することもできます。TensorFlow Lite 2.4.0 リリース以前は、これが唯一のオプションでした。</p>
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
<p data-md-type="paragraph">GPU デリゲートのリリースには、バックエンドで実行できるいくつかのモデルが含まれています。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite" data-md-type="link">MobileNet v1 (224x224) 画像分類</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite" data-md-type="link" class="_active_edit_href">[ダウンロード]</a> <br><i data-md-type="raw_html">(モバイルおよび組み込みベースのビジョンアプリケーション向けに設計された画像分類モデル)</i>
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
<p data-md-type="paragraph">サポートされている演算の完全なリストは、<a href="gpu_advanced.md" data-md-type="link">アドバンストドキュメント</a>を参照してください。</p>
<h2 data-md-type="header" data-md-header-level="2">サポートされていないモデルと演算</h2>
<p data-md-type="paragraph">一部の演算が GPU デリゲートでサポートされていない場合、フレームワークは GPU でグラフの一部のみを実行し、残りの部分を CPU で実行します。CPU と GPU 同期のコストは高いため、このような分割実行モードでは、ネットワーク全体が CPU のみで実行されている場合よりもパフォーマンスが遅くなることがよくあります。この場合、ユーザーには次のような警告が表示されます。</p>
<pre data-md-type="block_code" data-md-language="none"><code class="language-none">WARNING: op code #42 cannot be handled by this delegate.
</code></pre>
<p data-md-type="paragraph">これは実行時エラーではないため、このエラーに対するコールバックは提供されませんが、開発者はデリゲートでネットワークを実行する際に確認できます。</p>
<h2 data-md-type="header" data-md-header-level="2">最適化のヒント</h2>
<p data-md-type="paragraph">演算によっては、CPU では簡単で GPU ではコストが高くなる可能性があります。このような演算の 1 つのクラスは、<code data-md-type="codespan">BATCH_TO_SPACE</code>、<code data-md-type="codespan">SPACE_TO_BATCH</code>、<code data-md-type="codespan">SPACE_TO_DEPTH</code>など、さまざまな形の変形演算です。ネットワークアーキテクトの論理的思考のためだけにこれらの演算がネットワークに挿入されている場合、パフォーマンスのためにそれらを削除することをお勧めします。</p>
<p data-md-type="paragraph">GPU では、テンソルデータは 4 チャネルにスライスされます。したがって、形状 <code data-md-type="codespan">[B,H,W,5]</code>のテンソルに対する計算は、形状<code data-md-type="codespan">[B,H,W,8]</code>のテンソルに対しする計算とほぼ同じように実行されますが、パフォーマンスは<code data-md-type="codespan">[B,H,W,4]</code>と比べて大幅に低下します。</p>
<p data-md-type="paragraph">そのため、カメラハードウェアが RGBA の画像フレームをサポートしている場合、メモリコピー (3 チャネル RGB から 4 チャネル RGBX へ) を回避できるため、4 チャネル入力のフィードは大幅に速くなります。</p>
<p data-md-type="paragraph">最高のパフォーマンスを得るには、モバイル向けに最適化されたネットワークアーキテクチャで分類器を再トレーニングします。これは、デバイス上の推論の最適化の重要な部分です。</p>
</div>
