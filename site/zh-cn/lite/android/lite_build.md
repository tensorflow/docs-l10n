# 为 Android 构建 TensorFlow Lite 库

This document describes how to build TensorFlow Lite Android library on your own. Normally, you do not need to locally build TensorFlow Lite Android library. If you just want to use it, the easiest way is using the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite). See [Android quickstart](../guide/android.md) for more details on how to use them in your Android projects.

## Use Nightly Snapshots

To use nightly snapshots, add the following repo to your root Gradle build config.

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

## 在本地构建 TensorFlow Lite 库

在某些情况下，您可能希望使用本地构建的 TensorFlow Lite。例如，您可能在构建包含 [Select TensorFlow 算子](https://www.tensorflow.org/lite/guide/ops_select)的自定义二进制文件，或者您可能希望在本地对 TensorFlow Lite 进行更改。

### 使用 Docker 设置构建环境

- 下载 Docker 文件。下载 Docker 文件，即表示您同意以下监管您的使用行为的服务条款：

*点击以接受，即表示您同意对 Android Studio 和 Android Native Development Kit 的所有使用行为将受到 Android Software Development Kit 许可协议的约束。该许可协议位于以下网址：https://developer.android.com/studio/terms（Google 可能随时更新或更改此网址）。*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} 您可以在<a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/tflite-android.Dockerfile">此处</a>下载 Docker 文件 {% dynamic else %} 您必须确认服务条款才能下载此文件。<a class="button button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">确认</a> {% dynamic endif %}

<!-- mdformat on -->

- 您可以选择更改 Android SDK 或 NDK 版本。将下载的 Docker 文件放在一个空文件夹中，然后运行以下代码即可构建 Docker 镜像：

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- 通过将当前文件夹挂载到容器内的 /tmp，以交互方式启动 Docker 容器（请注意，/tensorflow_src 是容器内部的 TensorFlow 仓库）：

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

如果是在 Windows 上使用 PowerShell，请将“$PWD”替换为“pwd”。

如果您希望在主机上使用 TensorFlow 仓库，请挂载该主机目录 (-v hostDir:/tmp)。

- 进入容器后，您可以运行以下代码，下载其他 Android 工具和库（请注意，您可能需要接受许可协议）：

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

Now you should proceed to the [Configure WORKSPACE and .bazelrc](#configure_workspace_and_bazelrc) section to configure the build settings.

After you finish building the libraries, you can copy them to /host_dir inside the container so that you can access them on the host.

### 不使用 Docker 设置构建环境

#### 安装 Bazel 和 Android 前提条件

Bazel 是适用于 TensorFlow 的主要构建系统。要使用 Bazel 构建，您必须在系统上安装此工具以及 Android NDK 与 SDK。

1. 安装最新版本的 [Bazel 构建系统](https://bazel.build/versions/master/docs/install.html)。
2. The Android NDK is required to build the native (C/C++) TensorFlow Lite code. The current recommended version is 19c, which may be found [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-19c-downloads).
3. 在[此处](https://developer.android.com/tools/revisions/build-tools.html)可以获取 Android SDK 和构建工具，或者，您也可以通过 [Android Studio](https://developer.android.com/studio/index.html) 获取。对于 TensorFlow Lite 模型构建，推荐的构建工具 API 版本是 23 或更高版本。

### 配置工作区和 .bazelrc

This is a one-time configuration step that is required to build the TF Lite libraries. Run the `./configure` script in the root TensorFlow checkout directory, and answer "Yes" when the script asks to interactively configure the `./WORKSPACE` for Android builds. The script will attempt to configure settings using the following environment variables:

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

如果不设置这些变量，则必须在脚本提示中以交互方式提供。如果配置成功，则会在根文件夹的 `.tf_configure.bazelrc` 文件中产生类似以下代码的条目：

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r19c"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### 构建和安装

正确配置 Bazel 后，您可以从根签出目录构建 TensorFlow Lite AAR，具体代码如下：

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  //tensorflow/lite/java:tensorflow-lite
```

这会在 `bazel-bin/tensorflow/lite/java/` 中产生 AAR 文件。请注意，这会构建具有多个不同架构的“胖”AAR 文件；如果您不需要所有架构，请使用适用于您的部署环境的子集。

You can build smaller AAR files targeting only a set of models as follows:

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上面的脚本会生成 `tensorflow-lite.aar` 文件，如果有模型使用 TensorFlow 算子，还可以选择生成 `tensorflow-lite-select-tf-ops.aar` 文件。有关更多详细信息，请参阅[缩减 TensorFlow Lite 二进制文件大小](../guide/reduce_binary_size.md)部分。

#### 将 AAR 直接添加到项目

将 `tensorflow-lite.aar` 文件移到项目中名为 `libs` 的目录中。修改应用的 `build.gradle` 文件以引用新目录，并使用新本地库替换现有 TensorFlow Lite 依赖项，例如：

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

#### 将 AAR 安装到本地 Maven 存储库

从根签出目录执行以下命令：

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

在应用的 `build.gradle` 中，确保添加 `mavenLocal()` 依赖项，并将标准 TensorFlow Lite 依赖项替换为支持 Select TensorFlow 算子的依赖项：

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

请注意，这里的 `0.1.100` 版本纯粹是为了进行测试/开发。安装本地 AAR 后，您可以在应用代码中使用标准 [TensorFlow Lite Java 推断 API](../guide/inference.md)。
