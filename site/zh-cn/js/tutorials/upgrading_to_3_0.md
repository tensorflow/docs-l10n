# 升级到 TensorFlow.js 3.0

## TensorFlow.js 3.0 有何变化

版本说明可在[此处查看](https://github.com/tensorflow/tfjs/releases)。几个值得注意的面向用户的功能包括：

### 自定义模块

我们提供了对创建自定义 tfjs 模块的支持，以支持生成大小经过优化的浏览器软件包，从而向您的用户传送更少的 JavaScript。要了解更多相关信息，[请参阅此教程](size_optimized_bundles.md)。

此功能面向浏览器中的部署，但是启用此功能会产生如下所述的一些变化。

### ES2017 代码

除了某些预编译软件包，**目前我们将代码传送到 NPM 的主要方式是采用 [ES2017 语法](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)的 [ES 模块](https://2ality.com/2016/02/ecmascript-2017.html)**。这使得开发人员可以利用[现代 JavaScript 功能](https://web.dev/publish-modern-javascript/)并更好地控制他们向最终用户传送的内容。

我们的 package.json `module`入口指向 ES2017 格式的单个库文件（即，不是软件包）。这实现了摇树优化，并且开发人员可以更好地控制下游转译。

我们提供了一些替代格式的预编译软件包，以支持旧版浏览器和其他模块系统。它们遵循下表中描述的命名约定，您可以从 JsDelivr 和 Unpkg 等流行的 CDN 加载它们。

<table>
  <tr>
   <td>文件名</td>
   <td>模块格式</td>
   <td>语言版本</td>
  </tr>
  <tr>
   <td>tf[-package].[min].js*</td>
   <td>UMD</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.[min].js</td>
   <td>UMD</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>tf[-package].node.js**</td>
   <td>CommonJS</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.fesm.[min].js</td>
   <td>ESM（单个平面文件）</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>index.js***</td>
   <td>ESM</td>
   <td>ES2017</td>
  </tr>
</table>

* [package] 指的是主 tf.js 包的子包的内核/转换器/层等名称。[min] 说明除了未缩小的文件外，我们还提供缩小的文件。

** 我们的 package.json `main` 入口指向此文件。

*** 我们的 package.json `module` 入口指向此文件。

如果您通过 npm 使用 tensorflow.js 并且将使用打包工具，您可能需要调整打包工具配置，以确保其可以使用 ES2017 模块或将其指向输出 package.json 中的其他某个条目。

### @tensorflow/tfjs-core 默认更精简

为了实现更好的[摇树优化](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)，我们默认不再将有关张量的链式/流式 api 包含在 @tensorflow/tfjs-core 中。我们建议直接使用运算 (ops) 来获得最小的程序包。我们提供导入 `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';` 来恢复链式 api。

我们也默认不再为内核注册梯度。如果您需要梯度/训练支持，您可以 `import '@tensorflow/tfjs-core/dist/register_all_gradients';`

> 注意：如果您使用 @tensorflow/tfjs 或 @tensorflow/tfjs-layers 或任何其他更高级别的包，这将自动为您完成。

### 代码重组，内核和梯度注册

我们重新组织了代码，以便更容易贡献运算和内核，以及实现自定义运算、内核和梯度。[有关详细信息，请参阅本指南](custom_ops_kernels_gradients.md)。

### 重大更改

完整的重大更改列表可以在[此处](https://github.com/tensorflow/tfjs/releases)找到，但其中包括删除所有 *Strict 运算，如 mulStrict 或 addStrict。

## 从 2.x 升级代码

### @tensorflow/tfjs 的用户

了解此处列出的所有重大更改 (https://github.com/tensorflow/tfjs/releases)

### @tensorflow/tfjs-core 的用户

了解此处列出的所有重大更改 (https://github.com/tensorflow/tfjs/releases)，然后执行以下操作：

#### 添加链式运算增强器或直接使用运算

而不是

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

您需要执行

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

您还可以使用以下 import 来导入所有链式/流式 api

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

或者，您可以直接使用运算（您也可以在此处使用已命名导入）

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### 导入初始化代码

如果您仅使用已命名导入（而不是 `import * as ...`），那么在某些情况下，您可能需要在程序顶部执行

```
import @tensorflow/tfjs-core
```

这可以防止激进的摇树优化器放弃任何必要的初始化。

## 从 1.x 升级代码

### @tensorflow/tfjs 的用户

了解[此处](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0)列出的所有重大更改。然后按照说明从 2.x 升级

### @tensorflow/tfjs-core 的用户

了解[此处](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0)列出的所有重大更改，按下文所述选择后端，然后按照从 2.x 升级的步骤进行操作

#### 选择后端

在 TensorFlow.js 2.0 中，我们将 cpu 和 webgl 后端移到它们自己的软件包中。有关如何包含这些后端的说明，请参阅 [@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu)、[@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl)、[@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm)、[@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu)。
