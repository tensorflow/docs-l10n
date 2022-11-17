# TensorFlow.js 3.0으로 업그레이드

## TensorFlow.js 3.0 변경 사항

릴리즈 노트는 [여기에서 사용 가능](https://github.com/tensorflow/tfjs/releases)합니다. 몇몇 주목할 만한 사용자 대상 기능은 다음을 포함합니다.

### 사용자 정의 모듈

크기가 최적화된 브라우저 번들 생산을 지원하기 위해 사용자 정의 tfjs 모듈을 생성하는 데 지원을 제공합니다. 사용자에게 JavaScript를 보다 적게 제공하십시오. 이에 대해 더 자세히 알아보시려면, [이 튜토리얼을 확인합니다](size_optimized_bundles.md).

이 기능은 브라우저 내 배포에 맞춰졌지만, 이 기능을 활성화하면 아래에 설명된 변경 사항의 일부가 적용됩니다.

### ES2017 코드

일부 사전 컴파일 번들에 더해, **NPM에 코드를 현재 제공하는 주요 방식은 [ES2017 syntax](https://2ality.com/2016/02/ecmascript-2017.html)를 포함한 [ES 모듈](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)로 제공하는 것입니다**. 이를 통해 개발자는 [최신 JavaScript 기능](https://web.dev/publish-modern-javascript/)의 이점을 취할 수 있으며 최종 사용자에 제공되는 기능을 보다 더 제어를 할 수 있습니다.

package.json `모듈` 엔트리는 ES2017 형식의 개별 라이브러리 파일을 가리킵니다(즉 번들 아님). 이를 통해 트리 쉐이킹을 할 수 있고 다운스트림 트랜스파일화에 대한 개발자 제어가 향상됩니다.

사전 컴파일된 번들로 일부 대체 형식을 제공하여 레거시 브라우저와 기타 모듈 시스템을 지원합니다. 이는 아래 있는 표에 설명된 명명 규칙을 따르며 JsDelivr 및 Unpkg과 같은 인기 있는 CDN에서 로드할 수 있습니다.

<table>
  <tr>
   <td>파일 이름</td>
   <td>모듈 형식</td>
   <td>언어 버전</td>
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
   <td>ESM (Single flat file)</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>index.js***</td>
   <td>ESM</td>
   <td>ES2017</td>
  </tr>
</table>

* [package]는 주요 tf.js 패키지의 하위 패키지에 대한 코어/컨버터/레이어와 같은 이름을 나타냅니다. [min]은 최소화되지 않은 파일에 더해 최소화된 파일을 제공하는 위치를 설명합니다.

** package.json `main` 엔트리는 이 파일을 가리킵니다.

** package.json `module` 엔트리는 이 파일을 가리킵니다.

npm을 통해 tensorflow.js를 사용하고 번들을 사용하는 경우, ES2017 모듈을 사용하거나 package.json의 엔트리 중 다른 하나를 가리킬 수 있도록 번들러 구성을 조정해야 할 수 있습니다.

### 기본적으로 @tensorflow/tfjs-core는 더욱 빈약합니다.

보다 더 나은 [트리 쉐이킹](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)을 활성화하려면 @tensorflow/tfjs-core의 기본으로 텐서에 대해 Chaining/Fluent API를 더 이상 포함하지 않습니다. 연산(ops)을 직접 사용하여 가장 작은 번들을 확보하는 것이 좋습니다. Chaining API를 복구하는 가져오기 `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';`를 제공합니다.

또한 기본적으로 커널에 대해 더 이상 그래디언트를 등록하지 않습니다. 그래디언트/훈련 지원을 원한다면 `import '@tensorflow/tfjs-core/dist/register_all_gradients';`가 가능합니다.

> 참고: @tensorflow/tfjs 또는 @tensorflow/tfjs-layers 또는 기타 모든 고수준 패키지를 사용하는 경우, 이는 자동으로 이루어집니다.

### 코드 재구성, 커널 및 그래디언트 레지스트리

사용자 정의 ops, 커널 및 그래디언트를 구현하고 ops 및 커널을 기여하는 것이 쉽도록 코드를 재구성했습니다. [자세한 정보는 이 설명서를 참조하십시오.](custom_ops_kernels_gradients.md)

### 주요 변경 사항

주요 변경 사항의 전체 목록은 [여기](https://github.com/tensorflow/tfjs/releases)에서 확인할 수 있지만, 모든 *mulStrict 또는 addStrict 같은 엄격한 ops 제거도 포함되어 있습니다.

## 2.x부터 코드 업그레이드

### @tensorflow/tfjs 사용자

여기에 나열된 모든 주요 변경 사항 해결(https://github.com/tensorflow/tfjs/releases)

### @tensorflow/tfjs-core 사용자

여기에 나열된 모든 주요 변경 사항을 해결한 다음 (https://github.com/tensorflow/tfjs/releases), 다음을 수행하십시오.

#### 연결된 op 증강자를 추가하거나 연산을 직접 사용합니다

다음을 수행하지 않습니다

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

다음을 수행해야 합니다

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

다음 import로 모든 Chaining/Fluent API를 가져올 수 있씁니다.

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

그 대신 op를 직접 사용할 수 있습니다(여기에서도 명명된 가져오기를 사용할 수 있습니다)

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### 초기화 코드 가져오기

전적으로 명명된 가져오기(`import * as ...` 대신)를 사용한다면 몇몇 사례의 경우 다음을 수행해야 합니다.

```
import @tensorflow/tfjs-core
```

프로그램의 상단 근처에서, 이를 통해 공격적인 트리 쉐이커가 모든 필요한 초기화를 삭제하는 것을 방지합니다.

## 1.x부터 코드 업그레이드

### @tensorflow/tfjs 사용자

[여기](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0) 나열된 모든 주요 변경 사항을 해결합니다. 그런 다음 2.x부터 업그레이드에 대한 지침을 따릅니다.

### @tensorflow/tfjs-core 사용자

[여기](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0) 나열된 모든 주요 변경 사항을 해결하고, 아래 설명된 대로 백엔드를 선택한 다음 2.x부터 업그레이드에 대한 단계를 따릅니다.

#### 백엔드 선택

TensorFlow.js 2.0에서 CPU와 webgl 백엔드를 자체 패키지로 삭제했습니다. 이러한 백엔드를 포함하는 방법에 대한 지침은 [@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu), [@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl), [@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm), [@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu)를 참조하십시오.
