# Generating size-optimized browser bundles with TensorFlow.js

## Overview

TensorFlow.js 3.0 brings support for building _size-optimized, production oriented browser bundles_. To put it another way we want to make it easier for you to ship less JavaScript to the browser.

This feature is geared towards users with production use cases who would particularly benefit from shaving bytes off their payload (and are thus willing to put in the effort to achieve this). To use this feature you should be familiar with [ES Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules), JavaScript bundling tools such as [webpack](https://webpack.js.org/) or [rollup](https://rollupjs.org/guide/en/), and concepts such as [tree-shaking/dead-code elimination](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking).

This tutorial demonstrates how to create a custom tensorflow.js module that can be used with a bundler to generate a size optimized build for a program using tensorflow.js.


### Terminology

In the context of this document there are a few key terms we will be using:

**[ES Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)** - **The standard JavaScript module system**. Introduced in ES6/ES2015. Identifiable by use of **import** and **export** statements.

**Bundling** - Taking a set of JavaScript assets and grouping/bundling them into one or more JavaScript assets that are usable in a browser. This is the step that usually produces the final assets that are served to the browser. **_Applications will generally do their own bundling directly from transpiled library sources_.** Common **bundlers** include _rollup_ and _webpack_. The end result of bundling is a known as a **bundle** (or sometimes as a **chunk** if it is split into multiple parts)

**[Tree-Shaking / Dead Code Elimination](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)** - Removal of code that is not used by the final written application. This is done during bundling, _typically_ in the minification step.

**Operations (Ops)** - A mathematical operation on one or more tensors that produces one or more tensors as output. Ops are ‘high level’ code and can use other ops to define their logic.

**Kernel** - A specific implementation of an op tied to specific hardware capabilities. Kernels are ‘low level’  and backend specific. Some ops have a one-to-one mapping from op to kernel while other ops use multiple kernels.


## Scope and Use Cases

### Inference only graph-models

The primary use case we heard about from users related to this, and are supporting in this release is that of doing **inference with _TensorFlow.js graph models_**. If you are using a _TensorFlow.js layers model_, you can convert this to the graph-model format using the [tfjs-converter](https://www.npmjs.com/package/@tensorflow/tfjs-converter). The graph model format is more efficient for the inference use case.

### Low level Tensor manipulation with tfjs-core

The other use case we support is programs that directly use the @tensorflow/tjfs-core package for lower level tensor manipulation.


## Our approach to custom builds

Our core principles when designing this functionality includes the following:

*   Make maximal use of the JavaScript module system (ESM) and allow users of TensorFlow.js to do the same.
*   Make TensorFlow.js as tree-shakeable as possible _by existing bundlers_ (e.g. webpack, rollup, etc). This enables users to take advantage of all the capabilities of those bundlers including features like code splitting.
*   As much as possible maintain _ease of use for users who are not as sensitive to bundle size_. This does mean that production builds will require more effort as many of the defaults in our libraries support ease of use over size optimized builds.

The primary goal of our workflow is to produce a custom _JavaScript module_ for TensorFlow.js that contains only the functionality required for the program we are trying to optimize. We rely on existing bundlers to do the actual optimization.

While we primarily rely on the JavaScript module system, we also provide a _custom_ _CLI tool_ to handle parts that aren’t easy to specify through the module system in user facing code. Two examples of this are:



*   Model specifications stored in `model.json` files
*   The op to backend-specific-kernel dispatching system we use.

This makes generating a custom tfjs build a bit more involved than just pointing a bundler to the regular @tensorflow/tfjs package.


## How to create size optimized custom bundles


### Step 1: Determine which kernels your program is using

**This step lets us determine all the kernels used by any models you run or pre/post-processing code given the backend you have selected.**

Use tf.profile to run the parts of your application that use tensorflow.js and get the kernels. It will look something like this


```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```


Copy that list of kernels to your clipboard for the next step.

> You need to profile the code using the same backend(s) that you want to use in your custom bundle.

> You will need to repeat this step if your model changes or your pre/post-processing code changes.


### Step 2. Write a configuration file for the custom tfjs module

Here is an example config file.

It looks like this:


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




*   kernels: The list of kernels to include in the bundle. Copy this from the output of Step 1.
*   backends: The list of backend(s) you want to include. Valid options include "cpu", "webgl" and “wasm”.
*   models: A list of model.json files for models you load in your application. Can be empty if your program does not use tfjs\_converter to load a graph model.
*   outputPath: A path to a folder to put the generated modules in.
*   forwardModeOnly: Set this to false if you want to include gradients for the kernels listed prior.


### Step 3. Generate the custom tfjs module

Run the custom build tool with the config file as an argument. You need to have the **@tensorflow/tfjs** package installed in order to have access to this tool.


```
npx tfjs-custom-module  --config custom_tfjs_config.json
```


This will create a folder at `outputPath` with some new files.


### Step 4. Configure your bundler to alias tfjs to the new custom module.

In bundlers like webpack and rollup we can alias the existing references to tfjs modules to point to our newly generated custom tfjs modules. There are three modules that need to be aliased for maximum savings in bundle size.

Here is an snippet of what that looks like in webpack ([full example here](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js)):


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


And here is the equivalent code snippet for rollup ([full example here](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js)):


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


> If your bundler does not support module aliasing, you will need to change your `import` statements to import tensorflow.js from generated `custom_tfjs.js` that was created in Step 3. Op definitions will not be tree-shaken out, but kernels still will be tree-shaken. Generally tree-shaking kernels is what provides the biggest savings in final bundle size.

> If you are only using the @tensoflow/tfjs-core package, then you only need to alias that one package.


### Step 5. Create your bundle

Run your bundler (e.g. `webpack` or `rollup`) to produce your bundle. The size of the bundle should be smaller than if you run the bundler without module aliasing. You can also use visualizers like [this one](https://www.npmjs.com/package/rollup-plugin-visualizer) to see what made it into your final bundle.


### Step 6. Test your app

Make sure to test that your app is working as expected!
