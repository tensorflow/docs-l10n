# Upgrading to TensorFlow.js 3.0

## What’s changed in TensorFlow.js 3.0

Release notes are [available here](https://github.com/tensorflow/tfjs/releases). A few notable user facing features include:

### Custom Modules

We provide support for creating custom tfjs modules to support producing size optimized browser bundles. Ship less JavaScript to your users. To learn more about this, [see this tutorial](size_optimized_bundles.md).

This feature is geared towards deployment in the browser, however enabling this capability motivates some of the changes described below.

### ES2017 Code

In addition to some pre-compile bundles, **the main way that we now ship our code to NPM is as [ES Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) with [ES2017 syntax](https://2ality.com/2016/02/ecmascript-2017.html)**. This allows developers to take advantage of [modern JavaScript features](https://web.dev/publish-modern-javascript/) and have greater control over what they ship to their end users.

Our package.json `module` entry point to individual library files in ES2017 format (i.e. not a bundle). This enables tree shaking and greater developer control over downstream transpilation.

We do provide a few alternate formats as pre-compiled bundles to support legacy browsers and other module systems. They follow the naming convention described in the table below and you can load them from popular CDNs such as JsDelivr and Unpkg.

<table>
  <tr>
   <td>File Name
   </td>
   <td>Module Format
   </td>
   <td>Language Version
   </td>
  </tr>
  <tr>
   <td>tf[-package].[min].js*
   </td>
   <td>UMD
   </td>
   <td>ES5
   </td>
  </tr>
  <tr>
   <td>tf[-package].es2017.[min].js
   </td>
   <td>UMD
   </td>
   <td>ES2017
   </td>
  </tr>
  <tr>
   <td>tf[-package].node.js**
   </td>
   <td>CommonJS
   </td>
   <td>ES5
   </td>
  </tr>
  <tr>
   <td>tf[-package].es2017.fesm.[min].js
   </td>
   <td>ESM (Single flat file)
   </td>
   <td>ES2017
   </td>
  </tr>
  <tr>
   <td>index.js***
   </td>
   <td>ESM
   </td>
   <td>ES2017
   </td>
  </tr>
</table>

\* [package] refers to names like core/converter/layers for subpackages of the main tf.js package. [min] describes where we provide minified files in addition to unminified files.

\*\* Our package.json `main` entry points to this file.

\*\*\* Our package.json `module` entry points to this file.

If you are using tensorflow.js via npm and you are using bundler, you may need to adjust your bundler configuration to make sure it can either consume the ES2017 modules or point it to another one of the entries in out package.json.

### @tensorflow/tfjs-core is slimmer by default

To enable better [tree-shaking](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking) we no longer include the chaining/fluent api on tensors by default in @tensorflow/tfjs-core. We recommend using operations (ops) directly to get the smallest bundle. We provide an import `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';` that restores the chaining api.

We also no longer register gradients for kernels by default. If you want gradient/training support you can `import '@tensorflow/tfjs-core/dist/register_all_gradients';`

> Note: If you are using @tensorflow/tfjs or @tensorflow/tfjs-layers or any of the other higher level packages, this is done for you automatically.

### Code Reorganization, kernel & gradient registries

We have re-organized our code to make it easier to both contribute ops and kernels as well as implement custom ops, kernels and gradients. [See this guide for more information](custom_ops_kernels_gradients.md).

### Breaking Changes

A full list of breaking changes can be found [here](https://github.com/tensorflow/tfjs/releases), but they include removal of all \*Strict ops like mulStrict or addStrict.

## Upgrading Code from 2.x

### Users of @tensorflow/tfjs

Address any breaking changes listed here (https://github.com/tensorflow/tfjs/releases)

### Users of @tensorflow/tfjs-core

Address any breaking changes listed here (https://github.com/tensorflow/tfjs/releases), then do the following:

#### Add chained op augmentors or use ops directly

Rather than

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

You need to do

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

You can also import all the chaining/fluent api with the following import

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

Alternatively you can use the op directly (you could use named imports here too)

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### Import initialization code

If you are exclusively using named imports (instead of `import * as ...`) then in some cases you may need to do

```
import @tensorflow/tfjs-core
```

near the top of your program, this prevents aggressive tree-shakers from dropping any necessary initialization.

## Upgrading Code from 1.x

### Users of @tensorflow/tfjs

Address any breaking changes listed [here](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0). Then follow the instructions for upgrading from 2.x

### Users of @tensorflow/tfjs-core

Address any breaking changes listed [here](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0), select a backend as described below and then follow the steps for upgrading from 2.x

#### Selecting a backend(s)

In TensorFlow.js 2.0 we removed the cpu and webgl backends into their own packages. See [@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu), [@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl), [@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm), [@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu) for instructions on how to include those backends.
