# 智能回复


<img src="https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/g3doc/models/images/smart_reply.png" class="attempt-right">

## 开始

我们的智能回复模型可以基于聊天消息生成回复建议。这些建议是与上下文相关的一键式响应，可以帮助用户轻松回复收到的消息。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">Download starter model</a>

### 示例应用

有一个 TensorFlow Lite 示例应用可以在 Android 上演示这个智能回复模型。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">View Android example</a>

阅读 [GitHub 页面](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/)以了解应用的工作原理。在此项目中，您还会学习如何使用 C++ 运算构建应用。

## 工作原理

模型可为对话聊天消息生成回复建议。

该设备端模型具备以下优势：

<ul>
  <li>运行快速：该模型在设备上运行并且无需网络连接。因此，推断速度非常快，平均延迟只有几毫秒。</li>
  <li>资源高效：该模型在设备中占用的内存很小。</li>
  <li>保护隐私：用户数据从不离开设备。</li>
</ul>

## 示例输出


<img alt="Animation showing smart reply" src="https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/g3doc/models/smart_reply/images/smart_reply.gif">

## 了解更多

<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">Research paper</a></li>
  <li><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Source code</a></li>
</ul>

## 用户

<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Inbox</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Smart Replies on Android Wear</a></li>
</ul>
