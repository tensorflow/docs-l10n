# TensorFlow.js로 크기 최적화 브라우저 번들 생성

## 개요

*TensorFlow.js 3.0은 크기에 최적화된 프로덕션 지향 브라우저 번들* 빌드를 지원합니다. 다른 말로하면 브라우저에 더 적은 JavaScript를 더 쉽게 제공할 수 있도록 하려는 것입니다.

이 기능은 특히 페이로드에서 바이트를 줄이는 데 도움이 되는 프로덕션 사용 사례를 가진 사용자를 대상으로 합니다. 이 기능을 사용하려면[ES Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) [, webpack](https://webpack.js.org/) 또는 [rollup](https://rollupjs.org/guide/en/) 과 같은 JavaScript 번들링 도구 [, tree-shaking/dead-code 제거](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking) 와 같은 개념에 익숙해야 합니다.

이 튜토리얼은 tensorflow.js를 사용하여 프로그램에 최적화된 크기의 빌드를 생성하기 위해 번들러와 함께 사용할 수 있는 사용자 지정 tensorflow.js 모듈을 만드는 방법을 보여줍니다.

### 술어

이 문서의 맥락에서 우리가 사용할 몇 가지 핵심 용어가 있습니다.

**[ES 모듈](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)** - **표준 JavaScript 모듈 시스템** 입니다. ES6/ES2015에 도입되었습니다. **import** 및 **export** 문을 사용하여 식별할 수 있습니다.

**번들링** - JavaScript 자산 세트를 가져와 브라우저에서 사용할 수 있는 하나 이상의 JavaScript 자산으로 그룹화/번들링합니다. 이것은 일반적으로 브라우저에 제공되는 최종 자산을 생성하는 단계입니다. ***응용 프로그램은 일반적으로 트랜스파일된 라이브러리 소스에서 직접 번들링을 수행합니다* .** 일반적인 **번** *들러에는 롤업* 및 *웹팩이* 포함됩니다. (이것은 다수의 부분으로 분할 인 경우 **청크** 또는 때때로) 묶음의 최종 결과는 **다발로** 알려져

**[Tree-Shaking / Dead Code 제거](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)** - 최종 작성된 애플리케이션에서 사용하지 않는 코드 제거. *이는 일반적으로* 축소 단계에서 번들링 중에 수행됩니다.

**연산(Ops)** - 하나 이상의 텐서를 출력으로 생성하는 하나 이상의 텐서에 대한 수학 연산입니다. 작업은 '고수준' 코드이며 다른 작업을 사용하여 논리를 정의할 수 있습니다.

**커널** - 특정 하드웨어 기능에 연결된 작업의 특정 구현입니다. 커널은 '낮은 수준'이며 백엔드에 따라 다릅니다. 일부 작업에는 작업에서 커널로 일대일 매핑이 있는 반면 다른 작업에서는 여러 커널을 사용합니다.

## 범위 및 사용 사례

### 추론 전용 그래프 모델

이와 관련하여 사용자로부터 들었고 이번 릴리스에서 지원하는 주요 사용 사례는 ***TensorFlow.js 그래프 모델로* 추론하는 것** 입니다. *TensorFlow.js 레이어 모델을* [사용하는 경우 tfjs-converter를](https://www.npmjs.com/package/@tensorflow/tfjs-converter) 사용하여 이를 그래프 모델 형식으로 변환할 수 있습니다. 그래프 모델 형식은 추론 사용 사례에 더 효율적입니다.

### tfjs-core를 사용한 저수준 Tensor 조작

우리가 지원하는 다른 사용 사례는 저수준 텐서 조작을 위해 @tensorflow/tjfs-core 패키지를 직접 사용하는 프로그램입니다.

## 맞춤형 빌드에 대한 우리의 접근 방식

이 기능을 설계할 때의 핵심 원칙에는 다음이 포함됩니다.

- JavaScript 모듈 시스템(ESM)을 최대한 활용하고 TensorFlow.js 사용자가 동일한 작업을 수행할 수 있도록 합니다.
- *TensorFlow.js를 기존 번* 들러(예: webpack, 롤업 등)에서 가능한 한 tree-shakeable로 만듭니다. 이를 통해 사용자는 코드 분할과 같은 기능을 포함하여 이러한 번들러의 모든 기능을 활용할 수 있습니다.
- *번들 크기에 민감하지 않은 사용자를 위해* 최대한 사용 편의성을 유지합니다. 이는 우리 라이브러리의 많은 기본값이 크기 최적화 빌드보다 사용 용이성을 지원하기 때문에 프로덕션 빌드에 더 많은 노력이 필요하다는 것을 의미합니다.

워크플로의 주요 목표는 최적화하려는 프로그램에 필요한 기능만 포함하는 TensorFlow.js용 *맞춤형 JavaScript 모듈을 생성하는 것입니다.* 우리는 실제 최적화를 수행하기 위해 기존 번들러에 의존합니다.

우리는 주로 JavaScript 모듈 시스템에 의존하지만 사용자 대면 코드에서 모듈 시스템을 통해 지정하기 쉽지 않은 부분을 처리하기 위해 *사용자 지정* *CLI 도구도 제공합니다.* 이에 대한 두 가지 예는 다음과 같습니다.

- `model.json` 파일에 저장된 모델 사양
- 우리가 사용하는 백엔드 특정 커널 디스패치 시스템에 대한 op.

이것은 사용자 정의 tfjs 빌드 생성을 일반 @tensorflow/tfjs 패키지에 번들러를 지정하는 것보다 조금 더 복잡하게 만듭니다.

## 크기에 최적화된 사용자 지정 번들을 만드는 방법

### 1단계: 프로그램에서 사용 중인 커널 확인

**이 단계를 통해 실행하는 모든 모델에서 사용하는 모든 커널 또는 선택한 백엔드가 제공된 코드를 사전/사후 처리하는지 확인할 수 있습니다.**

tf.profile을 사용하여 tensorflow.js를 사용하는 애플리케이션 부분을 실행하고 커널을 가져옵니다. 다음과 같이 보일 것입니다.

```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```

다음 단계를 위해 해당 커널 목록을 클립보드에 복사합니다.

> 사용자 지정 번들에서 사용하려는 것과 동일한 백엔드를 사용하여 코드를 프로파일링해야 합니다.

> 모델이 변경되거나 사전/사후 처리 코드가 변경되면 이 단계를 반복해야 합니다.

### 2단계. 사용자 정의 tfjs 모듈에 대한 구성 파일 작성

다음은 구성 파일의 예입니다.

다음과 같습니다.

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

- kernels: 번들에 포함할 커널 목록입니다. 1단계의 출력에서 이것을 복사합니다.
- backends: 포함하려는 백엔드 목록입니다. 유효한 옵션은 "cpu", "webgl" 및 "wasm"입니다.
- 모델: 애플리케이션에서 로드하는 모델의 model.json 파일 목록입니다. 프로그램이 tfjs_converter를 사용하여 그래프 모델을 로드하지 않는 경우 비어 있을 수 있습니다.
- outputPath: 생성된 모듈을 넣을 폴더의 경로입니다.
- forwardModeOnly: 이전에 나열된 커널에 대한 그라디언트를 포함하려면 false로 설정하십시오.

### 3단계. 사용자 지정 tfjs 모듈 생성

구성 파일을 인수로 사용하여 사용자 정의 빌드 도구를 실행하십시오. 이 도구에 액세스하려면 **@tensorflow/tfjs** 패키지가 설치되어 있어야 합니다.

```
npx tfjs-custom-module  --config custom_tfjs_config.json
```

그러면 새 파일이 `outputPath` 폴더가 생성됩니다.

### 4단계. 번들러를 구성하여 tfjs의 별칭을 새 사용자 지정 모듈로 지정합니다.

webpack 및 롤업과 같은 번들러에서 tfjs 모듈에 대한 기존 참조를 별칭으로 지정하여 새로 생성된 사용자 지정 tfjs 모듈을 가리킬 수 있습니다. 번들 크기를 최대한 줄이려면 별칭을 지정해야 하는 세 가지 모듈이 있습니다.

다음은 webpack에서 보이는 것의 스니펫입니다( [전체 예제는 여기](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js) ).

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

다음은 롤업에 해당하는 코드 조각입니다( [전체 예제는 여기](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js) ).

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

> 번들러가 모듈 앨리어싱을 지원하지 않는 경우 3단계에서 생성된 생성된 `custom_tfjs.js` `import` 문을 변경해야 합니다. Op 정의는 트리 쉐이크 아웃되지 않지만 커널은 여전히 트리입니다. -쉐이킹 식. 일반적으로 트리 쉐이킹 커널은 최종 번들 크기에서 가장 큰 절감 효과를 제공합니다.

> @tensoflow/tfjs-core 패키지만 사용하는 경우 해당 패키지의 별칭만 지정하면 됩니다.

### 5단계. 번들 생성

`webpack` (예: webpack 또는 `rollup` )를 실행하여 번들을 생성합니다. 번들의 크기는 모듈 앨리어싱 없이 번들러를 실행할 때보다 작아야 합니다. 당신은 또한 같은 비주얼을 사용하여 [이 일](https://www.npmjs.com/package/rollup-plugin-visualizer) 최종 다발로 만들 었는지 확인합니다.

### 6단계. 앱 테스트

앱이 예상대로 작동하는지 테스트하십시오!
