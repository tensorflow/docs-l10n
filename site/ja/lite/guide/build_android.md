# Android 用の TensorFlow Lite を構築する

このドキュメントでは、TensorFlow Lite Android ライブラリを独自に構築する方法について説明します。通常、TensorFlow Lite Android ライブラリをローカルで構築する必要はありません。使用するだけの場合、最も簡単な方法は、[MavenCentral でホストされている TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite) を使用することです。Android プロジェクトでこれらを使用する方法の詳細については、[Android クイックスタート](../guide/android.md)をご覧ください。

## Nightly スナップショットを使用する

Nightly スナップショットを使用するには、以下のリポジトリをルートの Gradle ビルド構成に追加します。

```build
allprojects {
    repositories {      // should be already there
        mavenCentral()  // should be already there
        maven {         // add this repo to use snapshots
          name 'ossrh-snapshot'
          url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
    }
}
```

## ローカルで TensorFlow Lite を構築する

場合によっては、TensorFlow Lite のローカルビルドの使用も可能です。例えば、[TensorFlow から選択した演算](https://www.tensorflow.org/lite/guide/ops_select) を含むカスタムバイナリを構築する場合や、TensorFlow Lite にローカルの変更を加える場合などがあります。

### Docker を使用して構築環境をセットアップする

- Docker ファイルをダウンロードします。Docker ファイルをダウンロードすることで、以下の利用規約に同意したことになります。

*同意をクリックすることで、すべての Android Studio および Android Native Development Kit の使用は、https://developer.android.com/studio/terms から入手可能な Android Software Development Kit License Agreement に準拠することに同意したことになります（このような URL は Google により随時更新または変更される場合があります）。*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} Docker ファイルは<a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/tflite-android.Dockerfile">こちら</a>からダウンロードできます。{% dynamic else %}ファイルをダウンロードするには、利用規約に同意する必要があります。<a class="button button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">同意する</a> {% dynamic endif %}

<!-- mdformat on -->

- オプションで Android の SDK や NDK のバージョンを変更することができます。ダウンロードした Docker ファイルを空のフォルダに入れ、実行して Docker イメージを構築します。

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- 現在のフォルダをコンテナ内の /host_dir にマウントして、Docker コンテナをインタラクティブに起動します（/tensorflow_src はコンテナ内の TensorFlow リポジトリであることに注意してください）。

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

Windows で PowerShell を使用する場合は、"$PWD" を "pwd" に置き換えます。

ホスト上で TensorFlow リポジトリを使用する場合は、代わりにそのホストディレクトリ (-v hostDir:/host_dir) をマウントします。

- コンテナの中に入ると、以下を実行して追加の Android ツールやライブラリをダウンロードすることができます（ライセンスに同意する必要があるかもしれないので注意してください）。

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

次に、[WORKSPACE と .bazelrc の構成](#configure_workspace_and_bazelrc) セクションに進み、ビルド設定を構成します。

ライブラリの構築が終了したら、コンテナ内の /host_dir にコピーして、ホスト上でアクセスできるようにします。

### Docker を使用せずに構築環境をセットアップする

#### Bazel と Android の前提条件をインストールする

Bazel は TensorFlow の主なビルドシステムです。これを使用して構築する場合には、Bazel および Android の NDK と SDK をシステムにインストールする必要がありあます。

1. [Bazel ビルドシステム](https://bazel.build/versions/master/docs/install.html)の最新バージョンをインストールします。
2. TensorFlow Lite のネイティブコード (C/C++) を構築するには、Android NDK が必要です。現在の推奨バージョンは 19c で、[こちら](https://developer.android.com/ndk/downloads/older_releases.html#ndk-19c-downloads)から入手できます。
3. Android SDK とビルドツールは[こちら](https://developer.android.com/tools/revisions/build-tools.html)から、または [Android Studio](https://developer.android.com/studio/index.html) の一部として入手できます。TensorFlow Lite の構築に推奨されるビルドツールの API バージョンは 23 かそれ以降です。

### WORKSPACE と .bazelrc の構成

これは、TF Lite ライブラリを構築するために必要となる一度限りの構成ステップです。ルートの TensorFlow チェックアウトディレクトリにある `./configure` スクリプトを実行して、Android ビルド用の `./WORKSPACE` をインタラクティブに設定するかを尋ねられたら「Yes」と答えます。スクリプトは以下の環境変数を使用して設定を試みます。

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

これらの変数が設定されていない場合は、スクリプトのプロンプトでインタラクティブに提供する必要があります。構成が正しく完了する、ルートフォルダ内の `.tf_configure.bazelrc` ファイルに以下に類似したエントリが生成されます。

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r19c"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### 構築してインストールする

Bazel が正しく設定されたら、以下のようにしてルートチェックアウトディレクトリから TensorFlow Lite AAR を構築します。

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  //tensorflow/lite/java:tensorflow-lite
```

これにより、`bazel-bin/tensorflow/lite/java/` に AAR ファイルが生成されます。これは、複数の異なるアーキテクチャの「大きな」AAR を構築するので注意してください。その全部が必要ではない場合は、デプロイ環境に適切なサブセットを使用します。

以下のようにすると、1 つのモデルセットのみをターゲットするより小型の AAR ファイルを構築できます。

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上記のスクリプトは、`tensorflow-lite.aar` ファイルを生成し、モデルのいずれかが Tensorflow 演算子を使用している場合には、オプションで `tensorflow-lite-select-tf-ops.aar` ファイルを生成します。詳細については、[TensorFlow Lite バイナリサイズの削減](../guide/reduce_binary_size.md)の項目をご覧ください。

#### プロジェクトに AAR を直接追加する

プロジェクト内の `tensorflow-lite.aar` ファイルを `libs` というディレクトリに移動します。アプリの `build.gradle` ファイルを新しいディレクトリを参照するように変更し、既存の TensorFlow Lite の依存関係を新しいローカルライブラリに置き換えます。例を示します。

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    compile(name:'tensorflow-lite', ext:'aar')
}
```

#### ローカルの Maven リポジトリに AAR をインストールする

ルートチェックアウト ディレクトリから以下のコマンドを実行します。

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

アプリの `build.gradle` で、`mavenLocal()` 依存関係があることを確認し、標準の TensorFlow Lite 依存関係を、セレクト TensorFlow 演算子をサポートする依存関係に置き換えます。

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
}
```

ここで示している `0.1.100` バージョンは、純粋にテストと開発目的のバージョンなので注意してください。ローカルに AAR がインストールされていると、アプリコード内で標準の [TensorFlow Lite Java 推論 API](../guide/inference.md) を使用することができます。
