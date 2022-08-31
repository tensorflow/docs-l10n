# TensorFlow.js에서 사용자 정의 ops, 커널 및 그래디언트 작성하기

## 개요

이 가이드는 TensorFlow.js에서 사용자 정의 연산(ops), 커널 및 그래디언트를 정의하는 메커니즘을 간단히 설명합니다. 주요 개념의 개요와 실행 중인 개념을 보여주는 코드에 대한 조언을 제공하는 것이 이 가이드의 목표입니다.

### 이 가이드의 대상은?

이 가이드는 TensorFlow.js의 일부 내적인 내용에 대해 다루는 상당한 고급 가이드이며, 특히 다음과 같은 그룹의 사람들에게 유용할 수 있습니다.

- 다양한 수학적 연산의 동작을 사용자 정의하는 데 관심 있는 TensorFlow.js의 고급 사용자(예: 기존의 그래디언트 구현을 재정의하는 연구자 또는 라이브러리에서 누락된 기능을 패치해야 하는 사용자)
- TensorFlow.js를 확장하는 라이브러리를 구축하는 사용자(예: TensorFlow.js 기본 형식 위에 구축된 일반적인 선형 대수학 라이브러리 또는 새로운 TensorFlow.js 백엔드).
- 이러한 메커니즘이 작동하는 방법의 일반적인 개요를 얻고자 하는 Tensorflow.js에 새로운 ops를 제공하는 데 관심이 있는 사용자.

이 문서는 내부 구현 메커니즘을 다루기 때문에 TensorFlow.js의 일반적인 이용에 대한 가이드가 **아닙니다.** TensorFlow.js를 사용하기 위해 이러한 메커니즘을 이해할 필요는 없습니다.

이 가이드를 최대한으로 활용하려면 TensorFlow.js 소스 코드를 쉽게 읽을 수 있어야 합니다.

## Terminology

이 가이드의 경우 몇몇 주요 용어가 선행 설명에 유용합니다.

**연산(Ops)** — 하나 이상의 텐서를 출력으로 생산하는 하나 이상의 텐서에 대한 수학적 연산입니다. Ops는 '고수준' 코드이며 다른 연산을 사용해 로직을 정의할 수 있습니다.

**커널** — 특정 하드웨어/플랫폼 역량과 관련 있는 연산의 특정 구현입니다. 커널은 '저수준'이며 백엔드에 따라 다릅니다. 일부 연산은 연산에서 커널로 1 대 1 매핑을 하는 반면 다른 연산은 여러 커널을 사용합니다.

**Gradient** **/ GradFunc** — The ‘backward mode’ definition of an **op/kernel** that computes the derivative of that function with regards to some input. Gradients are ‘high level’ code (not backend specific) and can call other ops or kernels.

**커널 레지스트리** - **(커널명, 백엔드명)** 튜플에서 커널 구현까지의 맵.

**그래디언트 레지스트리** — **커널명부터 그래디언트 구현까지의** 맵.

## 코드 구성

[연산](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops) 및 [그래디언트](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients)는 [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core)로 정의됩니다.

커널은 백엔드마다 다르며 각각의 백엔드 폴더에서 정의됩니다(예: [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels)).

사용자 정의 ops, 커널 및 그래디언트는 이러한 패키지 내에 정의될 필요가 없습니다. 하지만 구현에 종종 유사한 기호를 사용합니다.

## 사용자 정의 Ops 구현

사용자 정의 op를 사고하는 한 가지 방식은 일부 텐서 출력을 반환하는 JavaScript 함수로서, 종종 출력으로 텐서를 사용하는 것입니다.

- Some ops can be completely defined in terms of existing ops, and should just import and call these functions directly. [Here is an example](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts).
- The implementation of an op can also dispatch to backend specific kernels. This is done via `Engine.runKernel` and will be described further in the “implementing custom kernels” section. [Here is an example](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts).

## 사용자 정의 커널 구현

백엔드별 커널 구현은 주어진 연산에 대한 최적화된 논리 구현을 허용합니다. 커널은 [`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F)를 호출하는 ops에 의해 호출됩니다. 커널 구현은 네 가지로 정의됩니다.

- 커널 이름.
- 커널이 구현된 백엔드.
- Inputs: Tensor arguments to the kernel function.
- Attributes: Non-tensor arguments to the kernel function.

여기 [커널 구현](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts) 예시가 있습니다. 구현을 위해 사용된 규칙은 벡엔드에 따라 다르며 각 특정 벡엔드의 구현 및 설명서를 보면 가장 잘 이해할 수 있습니다.

일반적으로 커널은 텐서보다 저수준에서 작동하며 대신 메모리에 직접 읽고 쓰며 tfjs-core로 인해 텐서로 결국 래핑될 것입니다.

커널이 구현되면 tfjs-core의 [`registerKernel` 함수](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F)를 사용하여 TensorFlow.js와 등록될 수 있습니다. 해당 커널이 작동하도록 원하는 모든 백엔드에 커널을 등록할 수 있습니다. 등록되면 커널은 `tf.engine().runKernel(...)`과 호출될 수 있으며 TensorFlow.js는 현재 활성 백엔드의 구현으로 반드시 디스패치합니다.

## 사용자 정의 그래디언트 구현

그래디언트는 일반적으로 지정된 커널을 위해 정의됩니다(`tf.engine().runKernel(...)`에 대한 호출에 사용된 동일한 커널 이름으로 식별됨). 이를 통해 tfjs-core는 런타임에 모든 커널에 대한 그래디언트 정의를 찾기 위해 레지스트리를 사용할 수 있습니다.

사용자 정의 그래디언트 구현은 다음 작업에 유용합니다.

- 라이브러리에 나타나지 않을 수 있는 그래디언트 정의 추가
- Overriding an existing gradient definition to customize the gradient computation for a given kernel.

[여기에서 그래디언트 구현](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients) 예시를 볼 수 있습니다.

지정된 호출에 대한 그래디언트를 구현하면 tfjs-core에서 <br>[`registerGradient` 함수](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F)를 사용하여 TensorFlow.js와 등록될 수 있습니다.

The other approach to implementing custom gradients that by-passes the gradient registry (and thus allows for computing gradients for arbitrary functions in arbitrary ways is using [tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad).

Here is an [example of an op within the library](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64) of using customGrad
