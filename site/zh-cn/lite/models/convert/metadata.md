# 向 TensorFlow Lite 模型添加元数据

TensorFlow Lite metadata provides a standard for model descriptions. The metadata is an important source of knowledge about what the model does and its input / output information. The metadata consists of both

- 人员可读部分，用于传达使用模型时的最佳做法，以及
- machine readable parts that can be leveraged by code generators, such as the [TensorFlow Lite Android code generator](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator) and the [Android Studio ML Binding feature](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding).

All image models published on [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) have been populated with metadata.

## Model with metadata format

<center><img src="../../images/convert/model_with_metadata.png" alt="model_with_metadata" width="70%"></center>
<center>图 1. 带有元数据和相关文件的 TFLite 模型。</center>

Model metadata is defined in [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs), a [FlatBuffer](https://google.github.io/flatbuffers/index.html#flatbuffers_overview) file. As shown in Figure 1, it is stored in the [metadata](https://github.com/tensorflow/tensorflow/blob/bd73701871af75539dd2f6d7fdba5660a8298caf/tensorflow/lite/schema/schema.fbs#L1208) field of the [TFLite model schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs), under the name, `"TFLITE_METADATA"`. Some models may come with associated files, such as [classification label files](https://github.com/tensorflow/examples/blob/dd98bc2b595157c03ac9fa47ac8659bb20aa8bbd/lite/examples/image_classification/android/models/src/main/assets/labels.txt#L1). These files are concatenated to the end of the original model file as a ZIP using the ZipFile ["append" mode](https://pymotw.com/2/zipfile/#appending-to-files) (`'a'` mode). TFLite Interpreter can consume the new file format in the same way as before. See [Pack the associated files](#pack-the-associated-files) for more information.

See the instruction below about how to populate, visualize, and read metadata.

## 设置元数据工具

Before adding metadata to your model, you will need to a Python programming environment setup for running TensorFlow. There is a detailed guide on how to set this up [here](https://www.tensorflow.org/install).

After setup the Python programming environment, you will need to install additional tooling:

```sh
pip install tflite-support
```

TensorFlow Lite metadata tooling supports Python 3.

## Adding metadata using Flatbuffers Python API

Note: to create metadata for the popular ML tasks supported in [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/overview), use the high-level API in the [TensorFlow Lite Metadata Writer Library](metadata_writer_tutorial.ipynb).

模型元数据[模式](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)中包含三个部分：

1. **Model information** - Overall description of the model as well as items such as license terms. See [ModelMetadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L640).
2. **输入信息** - 输入以及诸如归一化等所需预处理的描述。请参阅 [SubGraphMetadata.input_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L590)。
3. **输出信息** - 输出以及诸如标签映射等所需后处理的描述，请参阅 [SubGraphMetadata.output_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L599)。

Since TensorFlow Lite only supports single subgraph at this point, the [TensorFlow Lite code generator](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator) and the [Android Studio ML Binding feature](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding) will use `ModelMetadata.name` and `ModelMetadata.description`, instead of `SubGraphMetadata.name` and `SubGraphMetadata.description`, when displaying metadata and generating code.

### 支持的输入/输出类型

TensorFlow Lite metadata for input and output are not designed with specific model types in mind but rather input and output types. It does not matter what the model functionally does, as long as the input and output types consists of the following or a combination of the following, it is supported by TensorFlow Lite metadata:

- 特征 - 无符号整数或 float32 类型的数字。
- 图像 - 元数据当前支持 RGB 和灰度图像。
- 边界框 - 矩形边界框。模式支持[多种编号方案](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L214)。

### 打包关联文件

TensorFlow Lite 模型可能随附不同的关联文件。例如，自然语言模型通常具有可将单词片段映射到单词 ID 的 vocab 文件；分类模型可能具有指示对象类别的标签文件。不使用关联文件（如有），模型将无法正常运行。

The associated files can now be bundled with the model through the metadata Python library. The new TensorFlow Lite model becomes a zip file that contains both the model and the associated files. It can be unpacked with common zip tools. This new model format keeps using the same file extension, `.tflite`. It is compatible with existing TFLite framework and Interpreter. See [Pack metadata and associated files into the model](#pack-metadata-and-associated-files-into-the-model) for more details.

The associated file information can be recorded in the metadata. Depending on the file type and where the file is attached to (i.e. `ModelMetadata`, `SubGraphMetadata`, and `TensorMetadata`), [the TensorFlow Lite Android code generator](../../inference_with_metadata/codegen) may apply corresponding pre/post processing automatically to the object. See [the &lt;Codegen usage&gt; section of each associate file type](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L77-L127) in the schema for more details.

### 归一化和量化参数

归一化是机器学习中的常见数据预处理技术。归一化的目标是将值更改为通用的标量，而不会扭曲值范围内的差异。

[模型量化](https://www.tensorflow.org/lite/performance/model_optimization#model_quantization)技术支持对权重的低精度表示，并可以选择激活存储和计算。

就预处理和后处理而言，归一化和量化是两个独立的步骤。详情如下。

 | 归一化 | 量化
:-: | --- | ---
\ | **浮点模型**: \ | **浮点模型**: \
: MobileNet 中分别       : - mean：127.5 \        : - zeroPoint：0 \        : |  |
: 针对浮点模型和 : - std：127.5 \         : - scale：1.0 \          : |  |
: 量化模型的          : **量化模型**: \     : **量化模型**: \      : |  |
: 输入图像 : - mean：127.5 \        : - zeroPoint：128.0 \    : |  |
: 的参数值          : - std：127.5           : - scale：0.0078125f \    : |  |
: 示例。          :                         :                          : |  |
\ | \ | **浮点模型**
: \                       : \                       : 不需要量化。\ : |  |
: \                       : **输入**：如果在训练中   : **量化模型**在预处理/  : |  |
: \                       : 对输入数据进行了   : 后处理中          : |  |
: 何时调用？         : 归一化，则需要对     : 可能需要量化，: |  |
:                         : 推理的输入数据 : 也可能不需要量化。具体取决于   : |  |
:                         : 执行相应的        : 输入/输出张量的       : |  |
:                         : 归一化。\          : 数据类型。\  : |  |
:                         : **输出**：输出    : - 浮点张量：预处理/     : |  |
:                         : 数据通常        : 后处理中不需要 : |  |
:                         : 不进行归一化。  : 进行量化。量化 : |  |
:                         :                         : 运算和去量化运算    : |  |
:                         :                         : 被烘焙到模型     : |  |
:                         :                         : 计算图中。\                 : |  |
:                         :                         : - int8/uint8 张量：  : |  |
:                         :                         : 需要在预处理/后处理    : |  |
:                         :                         : 中进行量化。     : |  |
\ | \ | **对输入进行量化**：
: \                       : \                       : \                        : |  |
: 公式                 : normalized_input =      : q = f / scale +          : |  |
:                         : (输入 - 平均) / std    : zeroPoint \              : |  |
:                         :                         : **对输出进行         : |  |
:                         :                         : 去量化**：\            : |  |
:                         :                         : f = (q - zeroPoint) *    : |  |
:                         :                         : scale                    : |  |
\ | 由模型创建者填充 | 由 TFLite 转换器
: 参数位于           : 并存储在模型     : 自动填充，并    : |  |
: 什么位置              : 元数据中，作为            : 存储在 tflite 模型   : |  |
:                         : `NormalizationOptions`  : 文件中。                  : |  |
如何获得 | 通过 | 通过 TFLite
: 参数？            : `MetadataExtractor` API : `Tensor` API [1] 或      : |  |
:                         : [2]                     : 通过              : |  |
:                         :                         : `MetadataExtractor` API  : |  |
:                         :                         : [2]                      : |  |
浮点和量化 | Yes, float and quant | 否，浮点模型
: 模型是否共享相同的   : 模型使用相同的   : 不需要量化。   : |  |
: 值？                  : 归一化           :                          : |  |
:                         : 参数              :                          : |  |
TFLite 代码 | \ | \
: 生成器或 Android    : 是                     : 是                      : |  |
: Studio 机器学习绑定       :                         :                          : |  |
: 是否会在数据处理过程中  :                         :                          : |  |
: 自动生成参数？  :                         :                          : |  |

[1] The [TensorFlow Lite Java API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Tensor.java#L73) and the [TensorFlow Lite C++ API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/c/common.h#L391).
 [2] The [metadata extractor library](#read-the-metadata-from-models)

针对 uint8 模型处理图像数据时，有时可以跳过归一化和量化步骤。当像素值在 [0, 255] 区间内时可以跳过。但通常来说，如果适用，应始终根据归一化和量化参数处理数据。

[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/overview) can handle normalization for you if you set up `NormalizationOptions` in metadata. Quantization and dequantization processing is always encapsulated.

### 示例

注：在运行脚本之前，指定的导出目录必须已存在；过程中不会创建目录。

您可以在此处找到有关如何针对不同类型的模型填充元数据的示例：

#### 图像分类

在[此处](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py)下载用于将元数据填充至 [mobilenet_v1_0.75_160_quantized.tflite](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/default/1) 的脚本。通过如下方法运行脚本：

```sh
python ./metadata_writer_for_image_classifier.py \
    --model_file=./model_without_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --label_file=./model_without_metadata/labels.txt \
    --export_directory=model_with_metadata
```

要针对其他图像分类模型填充元数据，请将与[此](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py#L63-L74)类似的模型规范添加到脚本中。本指南其余部分将重点介绍图像分类示例中的一些关键部分，以说明关键要素。

### 深入了解图像分类示例

#### 模型信息

元数据首先需要创建新的模型信息：

```python
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

""" ... """
"""Creates the metadata for an image classifier."""

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "MobileNetV1 image classifier"
model_meta.description = ("Identify the most prominent object in the "
                          "image from a set of 1,001 categories such as "
                          "trees, animals, food, vehicles, person etc.")
model_meta.version = "v1"
model_meta.author = "TensorFlow"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")
```

#### 输入/输出信息

本部分介绍如何描述模型的输入和输出签名。自动代码生成器可以使用此元数据创建预处理和后处理代码。要创建有关张量的输入或输出信息，请运行以下代码：

```python
# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
```

#### 图像输入

图像是机器学习的常见输入类型。TensorFlow Lite 元数据支持色彩空间等信息以及归一化等预处理信息。不需要手动指定图像尺寸，因为输入张量的形状已提供了该信息，并且可以自动推断。

```python
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats
```

#### 标签输出

可以使用 `TENSOR_AXIS_LABELS` 通过关联文件将标签映射到输出张量。

```python
# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 1001 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename("your_path_to_label_file")
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]
```

#### Create the metadata Flatbuffers

以下代码可将模型信息与输入和输出信息组合在一起：

```python
# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()
```

#### 将元数据和关联文件打包到模型中

Once the metadata Flatbuffers is created, the metadata and the label file are written into the TFLite file via the `populate` method:

```python
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["your_path_to_label_file"])
populator.populate()
```

You can pack as many associated files as you want into the model through `load_associated_files`. However, it is required to pack at least those files documented in the metadata. In this example, packing the label file is mandatory.

## 可视化元数据

您可以使用 [Netron](https://github.com/lutzroeder/netron) 可视化您的元数据，也可以使用 `MetadataDisplayer` 将 TensorFlow Lite 模型中的元数据读取为 json 格式：

```python
displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
export_json_file = os.path.join(FLAGS.export_directory,
                    os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)
```

Android Studio also supports displaying metadata through the [Android Studio ML Binding feature](https://developer.android.com/studio/preview/features#tensor-flow-lite-models).

## 元数据版本控制

The [metadata schema](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) is versioned both by the Semantic versioning number, which tracks the changes of the schema file, and by the Flatbuffers file identification, which indicates the true version compatibility.

### 语义化版本控制编号

元数据模式可以通过诸如 MAJOR.MINOR.PATCH 等[语义化版本控制编号](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L53)实现版本控制。它可以依据[此处](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L32-L44)所述规则跟踪模式变更。请参阅 `1.0.0` 版本之后添加的[字段历史记录](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L63)。

### The Flatbuffers file identification

Semantic versioning guarantees the compatibility if following the rules, but it does not imply the true incompatibility. When bumping up the MAJOR number, it does not necessarily mean the backward compatibility is broken. Therefore, we use the [Flatbuffers file identification](https://google.github.io/flatbuffers/md__schemas.html), [file_identifier](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L61), to denote the true compatibility of the metadata schema. The file identifier is exactly 4 characters long. It is fixed to a certain metadata schema and not subject to change by users. If the backward compatibility of the metadata schema has to be broken for some reason, the file_identifier will bump up, for example, from “M001” to “M002”. File_identifier is expected to be changed much less frequently than the metadata_version.

### 所需元数据解析器最低版本

The [minimum necessary metadata parser version](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L681) is the minimum version of metadata parser (the Flatbuffers generated code) that can read the metadata Flatbuffers in full. The version is effectively the largest version number among the versions of all the fields populated and the smallest compatible version indicated by the file identifier. The minimum necessary metadata parser version is automatically populated by the `MetadataPopulator` when the metadata is populated into a TFLite model. See the [metadata extractor](#read-the-metadata-from-models) for more information on how the minimum necessary metadata parser version is used.

## Read the metadata from models

The Metadata Extractor library is convenient tool to read the metadata and associated files from a models across different platforms (see the [Java version](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/java) and the [C++ version](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/cc)). You can build your own metadata extractor tool in other languages using the Flatbuffers library.

### Read the metadata in Java

To use the Metadata Extractor library in your Android app, we recommend using the [TensorFlow Lite Metadata AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-metadata). It contains the `MetadataExtractor` class, as well as the FlatBuffers Java bindings for the [metadata schema](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) and the [model schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs).

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.1.0'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](https://www.tensorflow.org/lite/android/lite_build#use_nightly_snapshots).

You can initialize a `MetadataExtractor` object with a `ByteBuffer` that points to the model:

```java
public MetadataExtractor(ByteBuffer buffer);
```

The `ByteBuffer` must remain unchanged for the entire lifetime of the `MetadataExtractor` object. The initialization may fail if the Flatbuffers file identifier of the model metadata does not match that of the metadata parser. See [metadata versioning](#metadata-versioning) for more information.

With matching file identifiers, the metadata extractor will successfully read metadata generated from all past and future schema due to the Flatbuffers' forwards and backward compatibility mechanism. However, fields from future schemas cannot be extracted by older metadata extractors. The [minimum necessary parser version](#the-minimum-necessary-metadata-parser-version) of the metadata indicates the minimum version of metadata parser that can read the metadata Flatbuffers in full. You can use the following method to verify if the minimum necessary parser version condition is met:

```java
public final boolean isMinimumParserVersionSatisfied();
```

Passing in a model without metadata is allowed. However, invoking methods that read from the metadata will cause runtime errors. You can check if a model has metadata by invoking the `hasMetadata` method:

```java
public boolean hasMetadata();
```

`MetadataExtractor` provides convenient functions for you to get the input/output tensors' metadata. For example,

```java
public int getInputTensorCount();
public TensorMetadata getInputTensorMetadata(int inputIndex);
public QuantizationParams getInputTensorQuantizationParams(int inputIndex);
public int[] getInputTensorShape(int inputIndex);
public int getoutputTensorCount();
public TensorMetadata getoutputTensorMetadata(int inputIndex);
public QuantizationParams getoutputTensorQuantizationParams(int inputIndex);
public int[] getoutputTensorShape(int inputIndex);
```

Though the [TensorFlow Lite model schema](https://github.com/tensorflow/tensorflow/blob/aa7ff6aa28977826e7acae379e82da22482b2bf2/tensorflow/lite/schema/schema.fbs#L1075) supports multiple subgraphs, the TFLite Interpreter currently only supports a single subgraph. Therefore, `MetadataExtractor` omits subgraph index as an input argument in its methods.

## Read the associated files from models

The TensorFlow Lite model with metadata and associated files is essentially a zip file that can be unpacked with common zip tools to get the associated files. For example, you can unzip [mobilenet_v1_0.75_160_quantized](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/metadata/1) and extract the label file in the model as follows:

```sh
$ unzip mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
Archive:  mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
 extracting: labels.txt
```

You can also read associated files through the Metadata Extractor library.

In Java, pass the file name into the `MetadataExtractor.getAssociatedFile` method:

```java
public InputStream getAssociatedFile(String fileName);
```

Similarly, in C++, this can be done with the method, `ModelMetadataExtractor::GetAssociatedFile`:

```c++
tflite::support::StatusOr<absl::string_view> GetAssociatedFile(
      const std::string& filename) const;
```
