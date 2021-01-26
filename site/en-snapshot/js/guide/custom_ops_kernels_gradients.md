# Writing custom ops, kernels and gradients in TensorFlow.js


## Overview

This guide outlines the mechanisms for defining custom operations (ops), kernels and gradients in TensorFlow.js. It aims to provide an overview of the main concepts and pointers to code that demonstrate the concepts in action.


### Who is this guide for?

This is a fairly advanced guide that touches on some internals of TensorFlow.js, it may be particularly useful for the following groups of people:



*   Advanced users of TensorFlow.js interested in customizing behaviour of various mathematical operations (e.g. researchers overriding existing gradient implementations or users who need to patch missing functionality in the library)
*   Users building libraries that extend TensorFlow.js (e.g. a general linear algebra library built on top of TensorFlow.js primitives or a new TensorFlow.js backend).
*   Users interested in contributing new ops to tensorflow.js who want to get a general overview of how these mechanisms work.

This **is not** a guide to general use of TensorFlow.js as it goes into internal implementation mechanisms. You do not need to understand these mechanisms to use TensorFlow.js

You do need to be comfortable with (or willing to try) reading TensorFlow.js source code to make the most use of this guide.


## Terminology

For this guide a few key terms are useful to describe upfront.

**Operations (Ops)** — A mathematical operation on one or more tensors that produces one or more tensors as output. Ops are ‘high level’ code and can use other ops to define their logic.

**Kernel** — A specific implementation of an op tied to specific hardware/platform capabilities. Kernels are ‘low level’  and backend specific. Some ops have a one-to-one mapping from op to kernel while other ops use multiple kernels.

**Gradient** **/ GradFunc** — The ‘backward mode’ definition of an **op/kernel** that computes the derivative of that function with regards to some input. Gradients are ‘high level’ code (not backend specific) and can call other ops or kernels.

**Kernel Registry** - A map from a **(kernel name, backend name)** tuple to a kernel implementation.

**Gradient Registry** — A map from a **kernel name to a gradient implementation**.


## Code organization

[Operations](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops) and [Gradients](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients) are defined in [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core).

Kernels are backend specific and are defined in their respective backend folders (e.g. [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels)).

Custom ops, kernels and gradients do not need to be defined inside these packages. But will often use similar symbols in their implementation.


## Implementing Custom Ops

One way to think of a custom op is just as a JavaScript function that returns some tensor output, often with tensors as input.



*   Some ops can be completely defined in terms of existing ops, and should just import and call these functions directly. [Here is an example](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts).
*   The implementation of an op can also dispatch to backend specific kernels. This is done via `Engine.runKernel` and will be described further in the “implementing custom kernels” section. [Here is an example](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts).


## Implementing Custom Kernels

Backend specific kernel implementations allow for optimized implementation of the logic for a given operation. Kernels are invoked by ops calling [`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F). A kernel implementations is defined by four things



*   A kernel name.
*   The backend the kernel is implemented in.
*   Inputs: Tensor arguments to the kernel function.
*   Attributes: Non-tensor arguments to the kernel function.

Here is an example of [a kernel implementation](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts). The conventions used to implement are backend specific and are best understood from looking at each particular backend’s implementation and documentation.

Generally kernels operate at a level lower than tensors and instead directly read and write to memory that will be eventually wrapped into tensors by tfjs-core.

Once a kernel is implemented it can be registered with TensorFlow.js by using [`registerKernel` function](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F) from tfjs-core. You can register a kernel for every backend you want that kernel to work in. Once registered the kernel can be invoked with `tf.engine().runKernel(...)` and TensorFlow.js will make sure to dispatch to the implementation in the current active backend.




## Implementing Custom Gradients

Gradients are generally defined for a given kernel (identified by the same kernel name used in a call to `tf.engine().runKernel(...)`). This allows tfjs-core to use a registry to look up gradient definitions for any kernel at runtime.

 Implementing custom gradients are useful for:



*   Adding a gradient definition that may not be present in the library
*   Overriding an existing gradient definition to customize the gradient computation for a given kernel.

You can see examples of [gradient implementations here](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients).

Once you have implemented a gradient for a given call it can be registered with TensorFlow.js by using [`registerGradient` function](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F) from tfjs-core.

The other approach to implementing custom gradients that by-passes the gradient registry (and thus allows for computing gradients for arbitrary functions in arbitrary ways is using [tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad).

Here is an [example of an op within the library](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64) of using customGrad
