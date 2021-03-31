# TensorFlow Lite Task Library

TensorFlow Lite Task Library 包含了一套功能强大且易于使用的任务专用库，供应用开发者使用 TFLite 创建机器学习体验。它为热门的机器学习任务（如图像分类、问答等）提供了经过优化的开箱即用的模型接口。模型接口专为每个任务而设计，以实现最佳性能和可用性。Task Library 可跨平台工作，支持 Java、C++ 和 Swift。

## Task Library 可以提供的内容

- **非机器学习专家也能使用的干净且定义明确的 API** <br>只需 5 行代码就可以完成推断。使用 Task Library 中强大且易用的 API 作为构建模块，帮助您在移动设备上使用 TFLite 轻松进行机器学习开发。

- **复杂但通用的数据处理** <br>支持通用的视觉和自然语言处理逻辑，可在您的数据和模型所需的数据格式之间进行转换。为训练和推断提供相同的、可共享的处理逻辑。

- **高性能增益** <br>数据处理时间不会超过几毫秒，保证了使用 TensorFlow Lite 的快速推断体验。

- **可扩展性和自定义** <br>您可以利用 Task Library 基础架构提供的所有优势，轻松构建您自己的 Android/iOS 推断 API。

## 支持的任务

以下是支持的任务类型的列表。随着我们继续提供越来越多的用例，该列表预计还会增加。

- **视觉 API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)

- **自然语言 (NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLCLassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)

- **自定义 API**

    - 扩展任务 API 基础架构并构建[自定义 API](customized_task_api.md)。
