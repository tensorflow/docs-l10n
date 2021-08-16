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

注意: TensorFlow Lite インタープリタは、実行時と同じスレッドで作成する必要があります。そうでないと、「TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.」が表示される可能性があります。

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

    val compatList = CompatibilityList()

    val options = if(compatList.isDelegateSupportedOnThisDevice) {
        // if the device has a supported GPU, add the GPU delegate
        Model.Options.Builder().setDevice(Model.Device.GPU).build()
    } else {
        // if the GPU is not supported, run on 4 threads
        Model.Options.Builder().setNumThreads(4).build()
    }

    // Initialize the model as usual feeding in the options object
    val myModel = MyModel.newInstance(context, options)

    // Run inference per sample code
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.support.model.Model
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Model.Options options;
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        options = Model.Options.Builder().setDevice(Model.Device.GPU).build();
    } else {
        // if the GPU is not supported, run on 4 threads
        options = Model.Options.Builder().setNumThreads(4).build();
    }

    MyModel myModel = new MyModel.newInstance(context, options);

    // Run inference per sample code
      </pre>
    </section>
    </devsite-selector>
</div>

## TensorFlow Lite コードジェネレータを使用してモデルインターフェイスを生成する{:#codegen}

注意：TensorFlow Lite ラッパーコードジェネレータは現在、Android のみをサポートしています。

[メタデータ](../convert/metadata.md)で拡張された TensorFlow Lite モデルの場合、開発者は TensorFlow Lite Android ラッパーコードジェネレータを使用してプラットフォーム固有のラッパーコードを作成できます。ラッパーコードにより、`ByteBuffer`と直接対話する必要がなくなり、開発者は`Bitmap`や`Rect`などの型付きオブジェクトを使用して TensorFlow Lite モデルと対話できます。

コードジェネレータの有用性は、TensorFlow Lite モデルのメタデータエントリの完全性に依存します。codegen ツールが各フィールドを解析する方法については、[metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) の関連フィールドの下にある<code>&lt;Codegen usage&gt;</code>セクションを参照してください。

### ラッパーコードを生成する

端末に以下のツールをインストールする必要があります。

```sh
pip install tflite-support
```

完了すると、次の構文を使用してコードジェネレータを使用できます。

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

結果のコードは、宛先ディレクトリに配置されます。[Google Colab](https://colab.research.google.com/) またはその他のリモート環境を使用している場合は、結果を zip アーカイブに圧縮して Android Studio プロジェクトにダウンロードする方が簡単な場合があります。

```python
# Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
```

### 生成されたコードを使用する

#### ステップ 1：生成されたコードをインポートする

必要に応じて、生成されたコードをディレクトリ構造に解凍します。生成されたコードのルートは、`SRC_ROOT`であると想定されます。

TensorFlow Lite モデルを使用する Android Studio プロジェクトを開き、生成されたモジュールを次の方法でインポートします。And File-&gt; New-&gt; Import Module-&gt; `SRC_ROOT`を選択します。

上記の例を使用すると、インポートされたディレクトリとモジュールは`classify_wrapper`と呼ばれます。

#### ステップ 2：アプリの`build.gradle`ファイルを更新します

生成されたライブラリモジュールを使用するアプリモジュールでは、次のようになります。

Android セクションの下に、以下を追加します。

```build
aaptOptions {
   noCompress "tflite"
}
```

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

依存関係セクションの下に、以下を追加します。

```build
implementation project(":classify_wrapper")
```

#### ステップ 3：モデルの使用

```java
// 1. Initialize the model
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
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

### モデル推論の高速化

生成されたコードを使用すると、開発者は[デリゲート](../performance/delegates.md)とスレッド数を使用してコードを高速化できます。これは、次の 3 つのパラメータを使用し、モデルオブジェクトを開始するときに設定できます。

-  **`Context`**：Android アクティビティまたはサービスからのコンテキスト
- （オプション）**`Device`**：GPUDelegate や NNAPIDelegate などの TFLite アクセラレーションデリゲート
- （オプション）**`numThreads`**：モデルの実行に使用されるスレッドの数。デフォルトは 1 です。

例えば、NNAPI デリゲートと最大 3 つのスレッドを使用するには、次のようにモデルを初期化します。

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### トラブルシューティング

「java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed」エラーが発生する場合、ライブラリモジュールを使用するアプリモジュールの Android セクションの下に次の行を挿入します。

```build
aaptOptions {
   noCompress "tflite"
}
```

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。
