# メタデータを使用してモデルインターフェイスを生成する

[TensorFlow Lite メタデータ](../convert/metadata)を使用すると、開発者はラッパーコードを生成して、Android での統合を有効にできます。ほとんどの開発者にとって最も使いやすいのは、[Android Studio ML モデルバインディング](#mlbinding)のグラフィカルインターフェイスです。さらにカスタマイズが必要な場合、またはコマンドラインツールを使用している場合は、[TensorFlow Lite Codegen](#codegen) も利用できます。

## Android Studio ML モデルバインディングを使用する {:#mlbinding}

[メタデータ](../convert/metadata.md)で拡張された TensorFlow Lite モデルの場合、開発者は Android Studio ML モデルバインディングを使用して、プロジェクトの設定を自動的に構成し、モデルメタデータに基づいてラッパークラスを生成できます。ラッパーコードを使用すると、`ByteBuffer`と直接対話する必要がなくなり、開発者は`Bitmap`や`Rect`などの型付きオブジェクトを使用して TensorFlow Lite モデルと対話できます。

注意: [Android Studio 4.1](https://developer.android.com/studio) 以上が必要です。

### TensorFlow Lite モデルを Android Studio にインポートする

1. TF Lite モデルを使用するモジュールを右クリックするか、`File` をクリックして、`New` &gt; `Other` &gt; `TensorFlow Lite Model` に移動します。![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. TFLite ファイルの場所を選択します。ユーザーに代わってツールが、ML Model バインディングとのモジュールの依存関係と Android モジュールの `build.gradle` ファイルに自動的に挿入されたすべての依存関係を構成します。

    オプション: <a>GPU アクセラレーション</a>を使用する場合は、TensorFlow GPU をインポートするための 2 番目のチェックをオンにしてください。<img>

3. `Finish` をクリックします。

4. インポートが正常に完了すると、次の画面が表示されます。モデルを使用し始めるには、Kotlin または Java を選択し、`Sample Code` セクションにあるコードをコピーして貼り付けます。Android Studio の `ml` ディレクトリにある TFLite モデルをダブルクリックすると、この画面に戻ることができます。![Model details page in Android Studio](../images/android/model_details.png)

### モデル推論の加速

ML Model Binding を利用すると、開発者はデリゲートや様々なスレッドを使用してコードを高速化できます。

Note: The TensorFlow Lite Interpreter must be created on the same thread as when is is run. Otherwise, TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized. may occur.

ステップ1.　モジュール`build.gradle`ファイルに、次の依存関係が含まれていることを確認します。

```java
    dependencies {
        ...
        // TFLite GPU delegate 2.3.0 or above is required.
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
    }
```

ステップ 2.デバイスで実行されている GPU が TensorFlow GPU デリゲートと互換性があるかどうかを検出します。互換性がない場合は、複数の CPU スレッドを使用してモデルを実行します。

<div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_13&lt;/code&gt;</pre>
<div data-md-type="block_html"></div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">TensorFlow Lite コードジェネレータを使用してモデルインターフェイスを生成する{:#codegen}</h2>
<p data-md-type="paragraph">注意：TensorFlow Lite ラッパーコードジェネレータは現在、Android のみをサポートしています。</p>
<p data-md-type="paragraph"><a href="../convert/metadata.md" data-md-type="link">メタデータ</a>で拡張された TensorFlow Lite モデルの場合、開発者は TensorFlow Lite Android ラッパーコードジェネレータを使用してプラットフォーム固有のラッパーコードを作成できます。ラッパーコードにより、<code data-md-type="codespan">ByteBuffer</code>と直接対話する必要がなくなり、開発者は<code data-md-type="codespan">Bitmap</code>や<code data-md-type="codespan">Rect</code>などの型付きオブジェクトを使用して TensorFlow Lite モデルと対話できます。</p>
<p data-md-type="paragraph">コードジェネレータの有用性は、TensorFlow Lite モデルのメタデータエントリの完全性に依存します。codegen ツールが各フィールドを解析する方法については、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs" data-md-type="link">metadata_schema.fbs</a> の関連フィールドの下にある<code>&lt;Codegen usage&gt;</code>セクションを参照してください。</p>
<h3 data-md-type="header" data-md-header-level="3">ラッパーコードを生成する</h3>
<p data-md-type="paragraph">端末に以下のツールをインストールする必要があります。</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">pip install tflite-support
</code></pre>
<p data-md-type="paragraph">完了すると、次の構文を使用してコードジェネレータを使用できます。</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
</code></pre>
<p data-md-type="paragraph">結果のコードは、宛先ディレクトリに配置されます。<a href="https://colab.research.google.com/" data-md-type="link">Google Colab</a> またはその他のリモート環境を使用している場合は、結果を zip アーカイブに圧縮して Android Studio プロジェクトにダウンロードする方が簡単な場合があります。</p>
<pre data-md-type="block_code" data-md-language="python"><code class="language-python"># Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
</code></pre>
<h3 data-md-type="header" data-md-header-level="3">生成されたコードを使用する</h3>
<h4 data-md-type="header" data-md-header-level="4">ステップ 1：生成されたコードをインポートする</h4>
<p data-md-type="paragraph">必要に応じて、生成されたコードをディレクトリ構造に解凍します。生成されたコードのルートは、<code data-md-type="codespan">SRC_ROOT</code>であると想定されます。</p>
<p data-md-type="paragraph">TensorFlow Lite モデルを使用する Android Studio プロジェクトを開き、生成されたモジュールを次の方法でインポートします。And File-&gt; New-&gt; Import Module-&gt; <code data-md-type="codespan">SRC_ROOT</code>を選択します。</p>
<p data-md-type="paragraph">上記の例を使用すると、インポートされたディレクトリとモジュールは<code data-md-type="codespan">classify_wrapper</code>と呼ばれます。</p>
<h4 data-md-type="header" data-md-header-level="4">ステップ 2：アプリの<code data-md-type="codespan">build.gradle</code>ファイルを更新します</h4>
<p data-md-type="paragraph">生成されたライブラリモジュールを使用するアプリモジュールでは、次のようになります。</p>
<p data-md-type="paragraph">Android セクションの下に、以下を追加します。</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">aaptOptions {
   noCompress "tflite"
}
</code></pre>
<p data-md-type="paragraph">依存関係セクションの下に、以下を追加します。</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">implementation project(":classify_wrapper")
</code></pre>
<h4 data-md-type="header" data-md-header-level="4">ステップ 3：モデルの使用</h4>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">// 1. Initialize the model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Set the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Run the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieve the result
    Map&lt;String, Float&gt; labeledProbability = outputs.getProbability();
}
</code></pre>
<h3 data-md-type="header" data-md-header-level="3">モデル推論の高速化</h3>
<p data-md-type="paragraph">生成されたコードを使用すると、開発者は<a href="../performance/delegates.md" data-md-type="link">デリゲート</a>とスレッド数を使用してコードを高速化できます。これは、次の 3 つのパラメータを使用し、モデルオブジェクトを開始するときに設定できます。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<strong data-md-type="double_emphasis"><code data-md-type="codespan">Context</code></strong>：Android アクティビティまたはサービスからのコンテキスト</li>
<li data-md-type="list_item" data-md-list-type="unordered">（オプション）<strong data-md-type="double_emphasis"><code data-md-type="codespan">Device</code></strong>：GPUDelegate や NNAPIDelegate などの TFLite アクセラレーションデリゲート</li>
<li data-md-type="list_item" data-md-list-type="unordered">（オプション）<strong data-md-type="double_emphasis"><code data-md-type="codespan">numThreads</code></strong>：モデルの実行に使用されるスレッドの数。デフォルトは 1 です。</li>
</ul>
<p data-md-type="paragraph">例えば、NNAPI デリゲートと最大 3 つのスレッドを使用するには、次のようにモデルを初期化します。</p>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
</code></pre>
<h3 data-md-type="header" data-md-header-level="3">トラブルシューティング</h3>
<p data-md-type="paragraph">「java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed」エラーが発生する場合、ライブラリモジュールを使用するアプリモジュールの Android セクションの下に次の行を挿入します。</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">aaptOptions {
   noCompress "tflite"
}
</code></pre>
