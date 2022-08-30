# 使用 TensorFlow.js 生成大小经过优化的浏览器软件包

## 概述

TensorFlow.js 3.0 支持构建*大小经过优化、面向生产的浏览器软件包*。换句话说，我们希望让您更容易向浏览器发送更少的 JavaScript。

此功能面向具有生产用例并且尤其能够从有效负载的缩减中受益的用户（因此他们愿意付出努力来实现这一点）。要使用此功能，您应该熟悉 [ES 模块](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)、JavaScript 打包工具（例如 [webpack](https://webpack.js.org/) 或 [rollup](https://rollupjs.org/guide/en/)）以及[摇树优化/死码消除](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)等概念。

本教程将演示如何创建可以与打包工具配合使用的自定义 tensorflow.js 模块，以便使用 tensorflow.js 生成大小经过优化的程序版本。

### 术语

在本文档的上下文中，我们将使用一些关键术语：

**[ES 模块](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)** - **标准的 JavaScript 模块系统**。在 ES6/ES2015 中引入。可通过 **import** 和 **export** 语句来标识。

**打包** - 获取一组 JavaScript 资产并将它们分组/打包成一个或多个可在浏览器中使用的 JavaScript 资产。此步骤通常会生成提供给浏览器的最终资产。***应用程序通常会直接从转移的库源自行打包*。**常见的<em>打包工具</em>包括 *rollup* 和 *webpack*。打包的最终结果称为**软件包**（有时会分成多个部分，称为**区块**）

**[摇树优化/死码消除](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)** - 删除最终写好的应用程序未使用的代码。这*通常*在打包过程的压缩步骤中完成。

**运算 (Ops)** - 对一个或多个张量进行的数学运算，产生一个或多个张量作为输出。运算是“高级”代码，可以使用其他运算来定义其逻辑。

**内核** - 与特定硬件功能关联的运算的特定实现。内核相对“低级”并且特定于后端。一些运算具有从运算到内核的一对一映射，而另一些运算使用多个内核。

## 范围和用例

### 仅推断计算图模型

我们从相关用户那里听说并且受此版本支持的主要用例是**使用 *TensorFlow.js 计算图模型*进行推断**。如果您正在使用 *TensorFlow.js 层模型*，可以使用 [tfjs-converter](https://www.npmjs.com/package/@tensorflow/tfjs-converter) 将其转换为计算图模型格式。计算图模型格式对于推断用例更高效。

### 使用 tfjs-core 操纵低级张量

我们支持的另一个用例是直接使用 @tensorflow/tjfs-core 包操纵低级张量的程序。

## 我们自定义版本的方法

设计此功能时，我们的核心原则包括以下内容：

- 最大限度地利用 JavaScript 模块系统 (ESM) 并允许 TensorFlow.js 的用户也这样做。
- *通过现有打包工具*（例如 webpack、rollup 等）使 TensorFlow.js 尽可能可以摇树优化。这样用户能够利用这些打包工具的所有功能，包括代码拆分等功能。
- *对于对软件包大小不是很敏感的用户，尽可能保持易用性*。这意味着生产版本将需要更多工作，因为我们库中的许多默认版本支持易用性，而大小没有经过优化。

我们的工作流程的主要目标是为 TensorFlow.js 生成自定义 *JavaScript 模块*，其中仅包含我们尝试优化的程序所需的功能。我们依靠现有的打包工具来进行实际优化。

虽然我们主要依赖 JavaScript 模块系统，但我们还提供了一个*自定义* *CLI 工具*，来处理面向用户的代码中不易通过模块系统指定的部分。这方面的两个示例是：

- `model.json` 文件中存储的模型规范
- 对我们使用的后端特定内核分派系统的操作。

这使得生成自定义 tfjs 版本比只是将打包工具指向常规的 @tensorflow/tfjs 包更复杂一些。

## 如何创建大小经过优化的自定义软件包

### 第 1 步. 确定您的程序使用哪些内核

**此步骤让我们确定您运行的任何模型所使用的所有内核，以及鉴于您选择的后端而执行的预处理/后处理代码所使用的所有内核。**

使用 tf.profile 运行您的应用程序中使用 tensorflow.js 的部分并获得内核。如下所示

```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```

将该内核列表复制到剪贴板以进行下一步。

> 您需要使用您要在自定义软件包中使用的同一后端来剖析代码。

> 如果您的模型或预处理/后处理代码发生变化，您将需要重复此步骤。

### 第 2 步. 为自定义 tfjs 模块编写配置文件

以下是一个示例配置文件。

如下所示：

```
{
  "kernels": ["Reshape", "_FusedMatMul", "Identity"],
  "backends": [
      "cpu"
  ],
  "models": [
      "./model/model.json"
  ],
  "outputPath": "./custom_tfjs",
  "forwardModeOnly": true
}
```

- kernels：要包含在软件包中的内核列表。从第 1 步的输出中复制此列表。
- backends：要包含的后端列表。有效选项包括“cpu”、“webgl”和“wasm”。
- models：您在应用程序中加载的模型的 model.json 文件列表。如果您的程序不使用 tfjs_converter 加载计算图模型，则可以为空。
- outputPath：放置生成的模块的文件夹路径。
- forwardModeOnly：如果您希望包含前面列出的内核的梯度，则将此项设置为 false。

### 第 3 步. 生成自定义 tfjs 模块

运行自定义构建工具，将配置文件作为参数。您需要安装 **@tensorflow/tfjs** 包才能访问此工具。

```
npx tfjs-custom-module  --config custom_tfjs_config.json
```

这将在 `outputPath` 创建一个文件夹，并放入一些新文件。

### 第 4 步. 配置打包工具，为 tfjs 设置别名以指向新的自定义模块。

在 webpack 和 rollup 等打包工具中，我们可以为 tfjs 模块的现有引用设置别名，以指向新生成的自定义 tfjs 模块。为了最大程度地减小软件包大小，需要对三个模块设置别名。

以下是 webpack 中的一个代码段（[完整示例在这里](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js)）：

```
...

config.resolve = {
  alias: {
    '@tensorflow/tfjs$':
        path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    '@tensorflow/tfjs-core$': path.resolve(
        __dirname, './custom_tfjs/custom_tfjs_core.js'),
    '@tensorflow/tfjs-core/dist/ops/ops_for_converter': path.resolve(
        __dirname, './custom_tfjs/custom_ops_for_converter.js'),
  }
}

...
```

以下是 rollup 中的等效代码段（[完整示例在这里](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js)）：

```
import alias from '@rollup/plugin-alias';

...

alias({
  entries: [
    {
      find: /@tensorflow\/tfjs$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    },
    {
      find: /@tensorflow\/tfjs-core$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs_core.js'),
    },
    {
      find: '@tensorflow/tfjs-core/dist/ops/ops_for_converter',
      replacement: path.resolve(__dirname, './custom_tfjs/custom_ops_for_converter.js'),
    },
  ],
}));

...
```

> 如果打包工具不支持模块别名，您需要更改 `import` 语句以从第 3 步创建的 `custom_tfjs.js` 中导入 tensorflow.js。运算定义不会被摇树优化，但内核仍将被摇树优化。一般来说，摇树优化的内核可以最大程度地减小最终软件包的大小。

> 如果您仅使用 @tensoflow/tfjs-core 包，那么仅需要为该包设置别名。

### 第 5 步. 创建软件包

运行打包工具（例如 `webpack` 或 `rollup`）来生成软件包。软件包的大小应该小于在没有模块别名的情况下运行打包工具生成的软件包。您还可以使用类似[这种](https://www.npmjs.com/package/rollup-plugin-visualizer)的可视化工具来查看最终软件包中的内容。

### 第 6 步. 测试您的应用程序

确保测试您的应用程序是否按预期工作！
