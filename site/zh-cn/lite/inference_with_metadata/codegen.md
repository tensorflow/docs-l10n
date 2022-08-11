# 使用元数据生成模型接口

开发者可以使用 [TensorFlow Lite 元数据](../models/convert/metadata)生成封装容器代码，以实现在 Android 上的集成。对于大多数开发者来说，[Android Studio 机器学习模型绑定](#mlbinding)的图形界面最易于使用。如果您需要更多的自定义或正在使用命令行工具，也可以使用 [TensorFlow Lite Codegen](#codegen)。

## 使用 Android Studio 机器学习模型绑定 {:#mlbinding}

对于使用[元数据](../models/convert/metadata.md)增强的 TensorFlow Lite 模型，开发者可以使用 Android Studio 机器学习模型绑定来自动配置项目设置，并基于模型元数据生成封装容器类。封装容器代码消除了直接与 `ByteBuffer` 交互的需要。相反，开发者可以使用 `Bitmap` 和 `Rect` 等类型化对象与 TensorFlow Lite 模型进行交互。

注：需要 [Android Studio 4.1](https://developer.android.com/studio) 或以上版本

### 在 Android Studio 中导入 TensorFlow Lite 模型

1. 右键点击要使用 TFLite 模型的模块，或者点击 `File`，然后依次点击 `New`&gt;`Other`&gt;`TensorFlow Lite Model` ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. 选择 TFLite 文件的位置。请注意，该工具将使用机器学习绑定代您配置模块的依赖关系，且所有依赖关系会自动插入 Android 模块的 `build.gradle` 文件。

    可选：如果要使用 GPU 加速，请选择导入 TensorFlow GPU 的第二个复选框。![Import dialog for TFLite model](../images/android/import_dialog.png)

3. 点击 `Finish`。

4. 导入成功后，会出现以下界面。要开始使用该模型，请选择 Kotlin 或 Java，复制并粘贴 `Sample Code` 部分的代码。在 Android Studio 中双击 `ml` 目录下的 TFLite 模型，可以返回此界面。![Model details page in Android Studio](../images/android/model_details.png)

### 加速模型推断 {:#acceleration}

机器学习模型绑定为开发者提供了一种通过使用委托和线程数量来加速代码的方式。

注：TensorFlow Lite 解释器必须创建在其运行时的同一个线程上。不然，TfLiteGpuDelegate Invoke: GpuDelegate 必须在初始化它的同一线程上运行。否则可能会发生错误。

步骤 1. 检查模块 `build.gradle` 文件是否包含以下依赖关系：

```java
    dependencies {
        ...
        // TFLite GPU delegate 2.3.0 or above is required.
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
    }
```

步骤 2. 检测设备上运行的 GPU 是否兼容 TensorFlow GPU 委托，如不兼容，则使用多个 CPU 线程运行模型：

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

## 用 TensorFlow Lite 代码生成器生成模型接口 {:#codegen}

注：TensorFlow Lite 封装容器代码生成器目前只支持 Android。

对于使用[元数据](../models/convert/metadata.md)增强的 TensorFlow Lite 模型，开发者可以使用 TensorFlow Lite Android 封装容器代码生成器来创建特定平台的封装容器代码。封装容器代码消除了直接与 `ByteBuffer` 交互的需要。相反，开发者可以使用 `Bitmap` 和 `Rect` 等类型化对象与 TensorFlow Lite 模型进行交互。

代码生成器是否有用取决于 TensorFlow Lite 模型的元数据条目是否完整。请参考 [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) 中相关字段下的 `<Codegen usage>` 部分，查看代码生成器工具如何解析每个字段。

### 生成封装容器代码

您需要在终端中安装以下工具:

```sh
pip install tflite-support
```

完成后，可以使用以下句法来使用代码生成器：

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

生成的代码将位于目标目录中。如果您使用的是 [Google Colab](https://colab.research.google.com/) 或其他远程环境，将结果压缩成 zip 归档并将其下载到您的 Android Studio 项目中可能会更加容易：

```python
# Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
```

### 使用生成的代码

#### 第 1 步：导入生成的代码

如有必要，将生成的代码解压缩到目录结构中。假定生成的代码的根目录为 `SRC_ROOT`。

打开要使用 TensorFlow lite 模型的 Android Studio 项目，然后通过以下步骤导入生成的模块：File -&gt; New -&gt; Import Module -&gt; 选择 `SRC_ROOT`

使用上面的示例，导入的目录和模块将称为 `classify_wrapper `。

#### 第 2 步：更新应用的 `build.gradle` 文件

在将使用生成的库模块的应用模块中：

在 android 部分下，添加以下内容：

```build
aaptOptions {
   noCompress "tflite"
}
```

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。

在 android 部分添加以下内容：

```build
implementation project(":classify_wrapper")
```

#### 第 3 步：使用模型

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

### 加速模型推断

生成的代码为开发者提供了一种通过使用[委托](../performance/delegates.md)和线程数来加速代码的方式。这些可以在初始化模型对象时设置，因为它需要三个参数：

-  **`Context`**：Android 操作组件或服务的上下文
- （可选）**`Device`**：TFLite 加速委托，例如 GPUDelegate 或 NNAPIDelegate
- （可选） **`numThreads`**：用于运行模型的线程数（默认为 1）。

例如，要使用 NNAPI 委托和最多三个线程，您可以像下面这样初始化模型：

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### 问题排查

如果您遇到 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed' 错误，请在将使用库模块的应用模块的 android 部分插入以下各行：

```build
aaptOptions {
   noCompress "tflite"
}
```

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。
