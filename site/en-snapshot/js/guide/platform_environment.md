# Platform and environment

TensorFlow.js works in the browser and Node.js, and in both platforms there are many different available configurations. Each platform has a unique set of considerations that will affect the way applications are developed.

In the browser, TensorFlow.js supports mobile devices as well as desktop devices. Each device has a specific set of constraints, like available WebGL APIs, which are automatically determined and configured for you.

In Node.js, TensorFlow.js supports binding directly to the TensorFlow API or running with the slower vanilla CPU implementations.


## Environments

When a TensorFlow.js program is executed, the specific configuration is called the environment. The environment is comprised of a single global backend as well as a set of flags that control fine-grained features of TensorFlow.js.


### Backends

TensorFlow.js support multiple different backends that implement tensor storage and mathematical operations. At any given time, only one backend is active. Most of the time, TensorFlow.js will automatically choose the best backend for you given the current environment. However, sometimes it's important to know which backend is being used and how to switch it.

To find which backend you are using:


```js
console.log(tf.getBackend());
```


If you want to manually change the backend:


```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```



#### WebGL backend

The WebGL backend, 'webgl', is currently the most powerful backend for the browser. This backend is up to 100x faster than the vanilla CPU backend. Tensors are stored as WebGL textures and mathematical operations are implemented in WebGL shaders. Here are a few useful things to know when using this backend:  \



##### Avoid blocking the UI thread

When an operation is called, like tf.matMul(a, b), the resulting tf.Tensor is synchronously returned, however the computation of the matrix multiplication may not actually be ready yet. This means the tf.Tensor returned is just a handle to the computation. When you call `x.data()` or `x.array()`, the values will resolve when the computation has actually completed. This makes it important to use the asynchronous `x.data()` and `x.array()` methods over their synchronous counterparts `x.dataSync()` and `x.arraySync()` to avoid blocking the UI thread while the computation completes.


##### Memory management

One caveat when using the WebGL backend is the need for explicit memory management. WebGLTextures, which is where Tensor data is ultimately stored, are not automatically garbage collected by the browser.

To destroy the memory of a `tf.Tensor`, you can use the `dispose()` method:


```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```


It is very common to chain multiple operations together in an application. Holding a reference to all of the intermediate variables to dispose them can reduce code readability. To solve this problem, TensorFlow.js provides a `tf.tidy()` method which cleans up all `tf.Tensor`s that are not returned by a function after executing it, similar to the way local variables are cleaned up when a function is executed:


```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```


> Note: there is no downside of using `dispose()` or `tidy()` in non-webgl environments (like Node.js or a CPU backend) that have automatic garbage collection. In fact, it often can be a performance win to free tensor memory faster than would naturally happen with garbage collection.


##### Precision

On mobile devices, WebGL might only support 16 bit floating point textures. However, most machine learning models are trained with 32 bit floating point weights and activations. This can cause precision issues when porting a model for a mobile device as 16 bit floating numbers can only represent numbers in the range `[0.000000059605, 65504]`. This means that you should be careful that weights and activations in your model do not exceed this range. To check whether the device supports 32 bit textures, check the value of `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')`, if this is false then the device only supports 16 bit floating point textures. You can use `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')` to check if TensorFlow.js is currently using 32 bit textures.


##### Shader compilation & texture uploads

TensorFlow.js executes operations on the GPU by running WebGL shader programs. These shaders are assembled and compiled lazily when the user asks to execute an operation. The compilation of a shader happens on the CPU on the main thread and can be slow. TensorFlow.js will cache the compiled shaders automatically, making the second call to the same operation with input and output tensors of the same shape much faster. Typically, TensorFlow.js applications will use the same operations multiple times in the lifetime of the application, so the second pass through a machine learning model is much faster.

TensorFlow.js also stores tf.Tensor data as WebGLTextures. When a `tf.Tensor` is created, we do not immediately upload data to the GPU, rather we keep the data on the CPU until the `tf.Tensor` is used in an operation. If the `tf.Tensor` is used a second time, the data is already on the GPU so there is no upload cost. In a typical machine learning model, this means weights are uploaded during the first prediction through the model and the second pass through the model will be much faster.

If you care about the performance of the first prediction through your model or TensorFlow.js code, we recommend warming the model up by passing an input Tensor of the same shape before real data is used.

For example:


```js
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);
```



#### Node.js TensorFlow backend

In the TensorFlow Node.js backend, 'node', the TensorFlow C API is used to accelerate operations. This will use the machine's available hardware acceleration, like CUDA, if available.

In this backend, just like the WebGL backend, operations return `tf.Tensor`s synchronously. However, unlike the WebGL backend, the operation is completed before you get the tensor back. This means that a call to `tf.matMul(a, b)` will block the UI thread.

For this reason, if you intend to use this in a production application, you should run TensorFlow.js in worker threads to not block the main thread.

For more information on Node.js, see this guide.


#### WASM backend

TensorFlow.js provides a [WebAssembly backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md) (`wasm`), which offers CPU acceleration and can be used as an alternative to the vanilla JavaScript CPU (`cpu`) and WebGL accelerated (`webgl`) backends.  To use it:

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

If your server is serving the `.wasm` file on a different path or a different name, use `setWasmPath` before you initialize the backend. See the ["Using Bundlers"](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers) section in the README for more info:

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> Note: TensorFlow.js defines a priority for each backend and will automatically choose the best supported backend for a given environment. To explicitly use the WASM backend, we need to call `tf.setBackend('wasm')`.

##### Why WASM?

[WASM](https://webassembly.org/) was introduced in 2015 as a new web-based binary format, providing programs written
in JavaScript, C, C++, etc. a compilation target for running on the web. WASM has been [supported](https://webassembly.org/roadmap/)
by Chrome, Safari, Firefox, and Edge since 2017, and is supported by [90% of devices](https://caniuse.com/#feat=wasm)
worldwide.

**Performance**

WASM backend leverages the [XNNPACK library](https://github.com/google/XNNPACK) for optimized implementation of neural
network operators.

*Versus JavaScript*: WASM binaries are generally much faster than JavaScript bundles for browsers to load, parse, and
execute. JavaScript is dynamically typed and garbage collected, which can cause slowdowns at runtime.

*Versus WebGL*: WebGL is faster than WASM for most models, but for tiny models WASM can outperform WebGL due to the
fixed overhead costs of executing WebGL shaders. The “When should I use WASM” section below discusses heuristics for
making this decision.

**Portability and Stability**

WASM has portable 32-bit float arithmetic, offering precision parity across all devices. WebGL, on the other hand, is
hardware-specific and different devices can have varying precision (e.g. fallback to 16-bit floats on iOS devices).

Like WebGL, WASM is officially supported by all major browsers. Unlike WebGL, WASM can run in Node.js, and be used
server-side without any need to compile native libraries.

##### When should I use WASM?

**Model size and computational demand**

In general, WASM is a good choice when models are smaller or you care about lower-end devices that lack WebGL
support (`OES_texture_float` extension) or have less powerful GPUs. The chart below shows inference times (as of TensorFlow.js 1.5.2) in Chrome
on a 2018 MacBook Pro for 5 of our officially supported [models](https://github.com/tensorflow/tfjs-models) across the
WebGL, WASM, and CPU backends:

**Smaller models**

| Model     | WebGL | WASM | CPU   | Memory  |
|-----------|-------|------|-------|---------|
| BlazeFace | 22.5 ms | 15.6 ms | 315.2 ms | .4 MB  |
| FaceMesh  | 19.3 ms | 19.2 ms | 335 ms   | 2.8 MB |

**Larger models**

| Model        | WebGL | WASM  | CPU    | Memory  |
|--------------|-------|-------|--------|-----|
| PoseNet      | 42.5 ms | 173.9 ms | 1514.7 ms | 4.5 MB |
| BodyPix      | 77 ms   | 188.4 ms | 2683 ms   | 4.6 MB |
| MobileNet v2 | 37 ms   | 94 ms    | 923.6 ms  | 13 MB  |

The table above shows that WASM is 10-30x faster than the plain JS CPU backend across models, and competitive with
WebGL for smaller models like [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface), which is lightweight (400KB), yet has a decent number of ops (~140). Given
that WebGL programs have a fixed overhead cost per op execution, this explains why models like BlazeFace are faster
on WASM.

**These results will vary depending on your device. The best way to determine whether WASM is right for your application is to test it on our different backends.**

##### Inference vs Training

To address the primary use-case for deployment of pre-trained models, the WASM backend development will prioritize
*inference* over *training* support. See an [up-to-date list](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts)
of supported ops in WASM and [let us know](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)
if your model has an unsupported op. For training models, we recommend using the Node (TensorFlow C++) backend or the
WebGL backend.

#### CPU backend

The CPU backend, 'cpu', is the least performant backend, however it is the simplest. Operations are all implemented in vanilla JavaScript, which makes them less parallelizable. They also block the UI thread.

This backend can be very useful for testing, or on devices where WebGL is unavailable.


### Flags

TensorFlow.js has a set of environment flags that are automatically evaluated and determine the best configuration in the current platform. These flags are mostly internal, but a few global flags can be controlled with public API.

*   `tf.enableProdMode():` enables production mode, which will remove model validation, NaN checks, and other correctness checks in favor of performance.
*   `tf.enableDebugMode()`: enables debug mode, which will log to the console every operation that is executed, as well as runtime performance information like memory footprint and total kernel execution time. Note that this will greatly slow down your application, do not use this in production.

> Note: These two methods should be used before using any TensorFlow.js code as they affect the values of other flags which will get cached. For the same reason, there is no "disable" analog function.

> Note: You can see all of the flags that have been evaluated by logging `tf.ENV.features` to the console. While these are **not part of the public API **(and thus have no guarantee of stability between versions), they can be useful for debugging or fine tuning behavior across platforms & devices. You can use `tf.ENV.set` to override the value of a flag.
