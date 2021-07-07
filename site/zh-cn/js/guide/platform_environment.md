# 平台和环境

TensorFlow.js 可以在浏览器和 Node.js 中运行，并且在两个平台中都具有许多不同的可用配置。每个平台都有一组影响应用开发方式的独特注意事项。

在浏览器中，TensorFlow.js 支持移动设备以及桌面设备。每种设备都有一组特定的约束（例如可用 WebGL API），系统会自动为您确定和配置这些约束。

在 Node.js 中，TensorFlow.js 支持直接绑定到 TensorFlow API 或搭配较慢的普通 CPU 实现运行。

## 环境

执行 TensorFlow.js 程序时，特定配置称为环境。环境由单个全局后端以及一组控制 TensorFlow.js 细粒度功能的标志构成。

### 后端

TensorFlow.js 支持可实现张量存储和数学运算的多种不同后端。在任何给定时间内，均只有一个后端处于活动状态。在大多数情况下，TensorFlow.js 会根据当前环境自动为您选择最佳后端。但是，有时必须要知道正在使用哪个后端以及如何进行切换。

要确定您使用的后端，请运行以下代码：

```js
console.log(tf.getBackend());
```

如果要手动更改后端，请运行以下代码：

```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```

#### WebGL 后端

WebGL 后端 'webgl' 是当前适用于浏览器的功能最强大的后端。此后端的速度比普通 CPU 后端快 100 倍。张量将存储为 WebGL 纹理，而数学运算将在 WebGL 着色器中实现。以下为使用此后端时需要了解的一些实用信息：

##### 避免阻塞界面线程

当调用诸如 tf.matMul(a, b) 等运算时，生成的 tf.Tensor 会被同步返回，但是矩阵乘法计算实际上可能还未准备就绪。这意味着返回的 tf.Tensor 只是计算的句柄。当您调用 `x.data()` 或 `x.array()` 时，这些值将在计算实际完成时解析。这样，就必须对同步对应项 `x.dataSync()` 和 `x.arraySync()` 使用异步 `x.data()` 和 `x.array()` 方法，以避免在计算完成时阻塞界面线程。

##### 内存管理

请注意，使用 WebGL 后端时需要显式内存管理。浏览器不会自动回收 WebGLTexture（最终存储张量数据的位置）的垃圾。

要销毁 `tf.Tensor` 的内存，您可以使用 `dispose()` 方法：

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

在应用中将多个运算链接在一起十分常见。保持对用于处置这些运算的所有中间变量的引用会降低代码的可读性。为了解决这个问题，TensorFlow.js 提供了 `tf.tidy()` 方法，可清理执行函数后未被该函数返回的所有 `tf.Tensor`，类似于执行函数时清理局部变量的方式：

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> 注：在具有自动垃圾回收功能的非 WebGL 环境（例如 Node.js 或 CPU 后端）中使用 `dispose()` 或 `tidy()` 没有弊端。实际上，与自然发生垃圾回收相比，释放张量内存的性能可能会更胜一筹。

##### 精度

在移动设备上，WebGL 可能仅支持 16 位浮点纹理。但是，大多数机器学习模型都使用 32 位浮点权重和激活进行训练。这可能会导致为移动设备移植模型时出现精度问题，因为 16 位浮点数只能表示 `[0.000000059605, 65504]` 范围内的数字。这意味着您应注意模型中的权重和激活不超出此范围。要检查设备是否支持 32 位纹理，请检查 `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')` 的值，如果为 false，则设备仅支持 16 位浮点纹理。您可以使用 `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')` 来检查 TensorFlow.js 当前是否使用 32 位纹理。

##### 着色器编译和纹理上传

TensorFlow.js 通过运行 WebGL 着色器程序在 GPU 上执行运算。当用户要求执行运算时，这些着色器会迟缓地进行汇编和编译。着色器的编译在 CPU 主线程上进行，可能十分缓慢。TensorFlow.js 将自动缓存已编译的着色器，从而大幅加快第二次调用具有相同形状输入和输出张量的同一运算的速度。通常，TensorFlow.js 应用在应用生命周期内会多次使用同一运算，因此第二次通过机器学习模型的速度会大幅提高。

TensorFlow.js 还会将 tf.Tensor 数据存储为 WebGLTextures。创建 `tf.Tensor` 时，我们不会立即将数据上传到 GPU，而是将数据保留在 CPU 上，直到在运算中使用 `tf.Tensor` 为止。第二次使用 `tf.Tensor` 时，数据已位于 GPU 上，因此不存在上传成本。在典型的机器学习模型中，这意味着在模型第一次预测期间会上传权重，而第二次通过模型则会快得多。

如果您在意通过模型或 TensorFlow.js 代码执行首次预测的性能，我们建议您在使用实际数据之前先通过传递相同形状的输入张量来预热模型。

例如：

```js
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);
```

#### Node.js TensorFlow 后端

在 TensorFlow Node.js 后端 'node' 中，使用 TensorFlow C API 加速运算。这将在可用情况下使用计算机的可用硬件加速（例如 CUDA）。

在这个后端中，就像 WebGL 后端一样，运算会同步返回 `tf.Tensor`。但与 WebGL 后端不同的是，运算在返回张量之前就已完成。这意味着调用 `tf.matMul(a, b)` 将阻塞界面线程。

因此，如果打算在生产应用中使用，则应在工作线程中运行 TensorFlow.js 以免阻塞主线程。

有关 Node.js 的更多信息，请参阅本指南。

#### WASM 后端

TensorFlow.js 提供了 [WebAssembly 后端](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md) (`wasm`)，可实现 CPU 加速，并且可以替代普通的 JavaScript CPU (`cpu`) 和 WebGL 加速 (`webgl`) 后端。用法如下：

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

如果您的服务器在不同的路径上或以不同的名称提供 `.wasm` 文件，则在初始化后端前请使用 `setWasmPath`。有关更多信息，请参阅自述文件中的“[使用 Bundler](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers)”部分：

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> 注：TensorFlow.js 会为每个后端定义优先级并为给定环境自动选择支持程度最高的后端。要显式使用 WASM 后端，我们需要调用 `tf.setBackend('wasm')`。

##### 为何使用 WASM？

[WASM](https://webassembly.org/) 于 2015 年作为一种基于 Web 的新型二进制格式面世，提供以 JavaScript、C、C++ 等语言编写的程序。WASM 自 2017 年起受到 Chrome、Safari、Firefox 和 Edge [支持](https://webassembly.org/roadmap/)，并获得全球 [90% 设备](https://caniuse.com/#feat=wasm)的支持。

**性能**

WASM 后端利用 [XNNPACK 库](https://github.com/google/XNNPACK)来优化神经网络算子的实现。

*对比 JavaScript*：浏览器加载、解析和执行 WASM 二进制文件通常比 JavaScript 软件包要快得多。JavaScript 的动态键入和垃圾回收功能可能会导致运行时速度缓慢。

*对比 WebGL*：WebGL 对于大多数模型而言速度均快于 WASM，但 WASM 针对小型模型的性能则会比 WebGL 更胜一筹，原因是执行 WebGL 着色器存在固定的开销成本。下文中的“应在何时使用 WASM”部分讨论了做此决定的启发法。

**可移植性和稳定性**

WASM 具有可移植的 32 位浮点运算，可在所有设备之间提供精度奇偶校验。另一方面，WebGL 特定于硬件，不同的设备可能具有不同的精度（例如，在 iOS 设备上回退到 16 位浮点）。

与 WebGL 一样，WASM 也受到所有主流浏览器的官方支持。与 WebGL 的不同之处为，WASM 可以在 Node.js 中运行，并且无需编译原生库即可在服务器端使用。

##### 应在何时使用 WASM？

**模型大小和计算需求**

通常，当模型较小或您在意不具备 WebGL 支持（`OES_texture_float` 扩展）或 GPU 性能较弱的的低端设备时，WASM 是一种不错的选择。下表显示了在 2018 款 MacBook Pro 上使用 Chrome 基于 WebGL、WASM 和 CPU 后端针对官方支持的 5 种[模型](https://github.com/tensorflow/tfjs-models)的推断时间（自 TensorFlow.js 1.5.2 起）：

**较小的模型**

模型 | WebGL | WASM | CPU | 内存
--- | --- | --- | --- | ---
BlazeFace | 22.5 ms | 15.6 ms | 315.2 ms | 0.4 MB
FaceMesh | 19.3 ms | 19.2 ms | 335 ms | 2.8 MB

**较大的模型**

模型 | WebGL | WASM | CPU | 内存
--- | --- | --- | --- | ---
PoseNet | 42.5 ms | 173.9 ms | 1514.7 ms | 4.5 MB
BodyPix | 77 ms | 188.4 ms | 2683 ms | 4.6 MB
MobileNet v2 | 37 ms | 94 ms | 923.6 ms | 13 MB

上表显示，针对这些模型，WASM 比普通的 JS CPU 后端快 10-30 倍；并且针对 [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface)（轻量化 (400KB) 但运算数量尚可 (~140)）之类的较小模型，则可与 WebGL 抗衡。考虑到 WebGL 程序每执行一次运算的固定开销成本，这就解释了像 BlazeFace 这样的模型在 WASM 上速度更快的原因。

**这些结果将因您的具体设备而异。确定 WASM 是否适合您的应用的最佳方式是在我们不同的后端上对其进行测试。**

##### 推断与训练

为解决部署预训练模型的主要用例，WASM 后端的开发工作在*推断*方面的支持将优先于*训练*。请参见 WASM 所支持运算的[最新列表](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts)，如果您的模型具有不受支持的运算，请[告诉我们](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)。对于训练模型，建议使用 Node (TensorFlow C++) 后端或 WebGL 后端。

#### CPU 后端

CPU 后端 'cpu' 是性能最低且最简单的后端。所有运算均在普通的 JavaScript 中实现，这使它们的可并行性较差。这些运算还会阻塞界面线程。

此后端对于测试或在 WebGL 不可用的设备上非常有用。

### 标志

TensorFlow.js 具有一组可自动评估的环境标志，这些标志可以确定当前平台中的最佳配置。大部分标志为内部标志，但有一些可以使用公共 API 控制的全局标志。

- `tf.enableProdMode()`：启用生产模式，在此模式下将移除模型验证、NaN 检查和其他有利于性能的正确性检查。
- `tf.enableDebugMode()`：启用调试模式，在此模式下会将执行的每项运算以及运行时性能信息（如内存占用量和总内核执行时间）记录到控制台。请注意，这将大幅降低您应用的速度，请勿在生产中使用。

> 注：这两个方法应在使用任何 TensorFlow.js 代码之前使用，因为它们会影响将缓存的其他标志的值。出于相同的原因，没有“disable”模拟函数。

> 注：您可以通过将 `tf.ENV.features` 记录到控制台来查看所有已评估的标志。尽管它们**不是公共 API 的一部分**（因此不能保证版本之间的稳定性），但它们对于跨平台和设备进行调试或微调行为而言非常实用。您可以使用 `tf.ENV.set` 重写标志的值。
