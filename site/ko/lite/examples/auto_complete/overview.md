# 자동 완성

<table class="tfo-notebook-buttons" align="left">
  <td><a target="_blank" href="https://www.tensorflow.org/lite/examples/auto_complete/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">TensorFlow.org에서 보기</a></td>
  <td><a target="_blank" href="https://colab.sandbox.google.com/github/tensorflow/codelabs/blob/main/KerasNLP/io2023_workshop.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Google Colab에서 실행하기</a></td>
</table>

## 소개

대규모 언어 모델(LLM)은 대규모 데이터세트를 활용해 텍스트를 생성하도록 학습된 머신러닝 모델의 한 종류입니다. 텍스트 생성, 질문 답변, 기계 번역 등 자연어 처리(NLP) 작업에 LLM을 사용할 수 있습니다. LLM은 트랜스포머 아키텍처를 기반으로 하며 수십억 개의 단어가 포함된 방대한 양의 텍스트 데이터 등으로 학습됩니다. GPT-2와 같이 규모가 작은 LLM도 인상적인 성능을 보일 수 있습니다. 더 가볍고 빠르며 전력 소모가 적은 TensorFlow 모델로 전환하면 온디바이스로 생성형 AI 모델을 실행할 수 있으며, 데이터가 기기를 떠나지 않기 때문에 사용자 보안이 강화되는 이점이 있습니다.

이 작업 가이드(runbook)는 TensorFlow Lite로 Android 앱을 빌드하여 Keras LLM을 실행하는 방법을 보여주고, 훨씬 더 많은 메모리와 더 큰 연산 능력이 필요한 정량화 기법을 사용하여 모델을 최적화하는 방법을 제안합니다.

우리는 호환되는 모든 TFLite LLM이 연결할 수 있는 [Android 앱 프레임워크](https://github.com/tensorflow/examples/tree/master/lite/examples/generative_ai/)를 오픈소스로 공개했습니다. 이 중 두가지 데모는 다음과 같습니다.

- 그림 1에서는 Keras GPT-2 모델을 사용하여 기기에서 텍스트 완성 작업을 수행했습니다.
- 그림 2에서는 지침을 튜닝한 [PaLM 모델](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 버전(15억 개의 매개변수)을 TFLite로 변환한 후 TFLite 런타임을 통해 실행했습니다.

<center>
<center> ![PaLM으로 자동 완성](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/autocomplete_fig1.gif){: width="400px"}<figcaption> <b>그림 1:</b> Pixel 7에서 텍스트 완성을 수행하기 위해 기기에서 Keras GPT-2 모델(이 [Codelab](https://codelabs.developers.google.com/kerasnlp-tflite)에서 변환됨)을 실행하는 예입니다. 데모는 속도 향상이 없는 실제 대기 시간.</figcaption> </center> </center>

<center>
<center> ![PaLM으로 자동 완성](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/autocomplete_fig2.gif){: width="400px"}<figcaption> <b>그림 2:</b> 15억 개의 매개변수가 있는 [PaLM 모델](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 버전 실행의 예. 데모는 재생 속도 향상 없이 Pixel 7 Pro에서 녹화됩니다.</figcaption> </center> </center>

## 가이드

### 모델 제작(Model authoring)

이 데모에서는 GPT-2 모델을 가져오기 위해 KerasNLP를 사용합니다. KerasNLP는 자연어 처리 작업을 수행하는 사전 학습된 최신 모델이 포함된 라이브러리로, 전체 개발 주기에 걸쳐 사용자를 지원할 수 있습니다. [KerasNLP 리포지토리](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models)에서 사용할 수 있는 모델 목록을 확인할 수 있습니다. 워크플로는 바로 사용할 수 있는 최신 사전 설정 가중치와 아키텍처로 구성된 모듈식 구성 요소로 구축되어 있으며 더 많은 제어가 필요한 경우 쉽게 사용자 정의할 수 있습니다. GPT-2 모델은 다음 단계에 따라 생성할 수 있습니다.

```python
gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")

gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=256,
    add_end_token=True,
)

gpt2_lm =
keras_nlp.models.GPT2CausalLM.from_preset(
"gpt2_base_en",
preprocessor=gpt2_preprocessor
)
```

이 세 줄의 코드의 한 가지 공통점은 `from_preset()` 메서드로, 이는 사전 설정된 아키텍처 및/또는 가중치에서 Keras API의 일부를 인스턴스화하고 사전 학습된 모델을 로드합니다. 이 코드 스니펫에서 세 가지 모듈식 구성 요소를 확인할 수 있습니다.

1. **토큰화**: 원시 문자열 입력을 Keras 임베딩 레이어에 적합한 정수 토큰 ID로 변환합니다. 구체적으로 GPT-2는 바이트 페어 인코딩(BPE) 토크나이저를 사용합니다.

2. **전처리기**: Keras 모델에 공급할 입력을 토큰화하고 패킹하기 위한 레이어입니다. 여기서 전처리기는 토큰화 후 토큰 ID의 텐서를 지정된 길이(256)로 추가합니다.

3. **백본**: SoTA 트랜스포머 백본 아키텍처를 따르고 사전 설정된 가중치를 갖는 Keras 모델입니다.

추가로, [GitHub](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models/gpt2)에서 전체 GPT-2 모델 구현을 확인할 수 있습니다.

### 모델 변환

TensorFlow Lite는 모바일, 마이크로컨트롤러 및 기타 에지 기기에 메서드를 배포하기 위한 모바일 라이브러리입니다. 첫 번째 단계는 TensorFlow Lite **변환기**를 사용하여 Keras 모델을 보다 간결한 TensorFlow Lite 형식으로 변환한 다음, 모바일 기기에 고도로 최적화된 TensorFlow Lite **인터프리터**를 사용하여 변환된 모델을 실행하는 것입니다.


<img src="https://www.tensorflow.org/lite/examples/auto_complete/images/tflite_workflow.png" class="attempt-right"> Start with the `generate()` function from `GPT2CausalLM` that performs the conversion. Wrap the `generate()` function to create a concrete TensorFlow function:

```python
@tf.function
def generate(prompt, max_length):
    """
    Args:
        prompt: input prompt to the LLM in string format
        max_length: the max length of the generated tokens
    """
    return gpt2_lm.generate(prompt, max_length)

concrete_func = generate.get_concrete_function(tf.TensorSpec([], tf.string), 100)
```

변환을 수행하기 위해 [`TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model)에서 `from_keras_model()`을 사용할 수도 있습니다.

이제 입력과 TFLite 모델로 추론을 실행할 헬퍼 함수를 정의합니다. TensorFlow 텍스트 연산은 TFLite 런타임에 속한 기본 제공 연산이 아니므로 인터프리터가 이 모델에서 추론을 수행하려면 이러한 사용자 정의 연산을 추가해야 합니다. 이 헬퍼 함수는 입력과 변환을 수행하는 함수, 즉 위에 정의된 `generator()` 함수를 받습니다.

```python
def run_inference(input, generate_tflite):
    interp = interpreter.InterpreterWithCustomOps(
        model_content=generate_tflite,
        custom_op_registerers=
            tf_text.tflite_registrar.SELECT_TFTEXT_OPS
    )

    interp.get_signature_list()

    generator = interp.get_signature_runner('serving_default')
    output = generator(prompt=np.array([input]))
```

이제 모델을 변환할 수 있습니다.

```python
gpt2_lm.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    gpt2_lm)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TF ops
]
converter.allow_custom_ops = True
converter.target_spec.experimental_select_user_tf_ops = [
    "UnsortedSegmentJoin",
    "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
generate_tflite = converter.convert()
run_inference("I'm enjoying a", generate_tflite)
```

### 양자화

TensorFlow Lite는 모델 크기를 줄이고 추론을 가속화할 수 있는 **양자화**라는 최적화 기법을 구현했습니다. 양자화 프로세스를 통해 32비트 부동 수는 더 작은 8비트 정수로 매핑되며, 이로 인해 모델 크기가 4배 줄어들기에 최신 하드웨어에서 보다 효율적으로 실행할 수 있습니다. TensorFlow에서 양자화를 수행하는 방법에는 여러 가지가 있습니다. 자세한 내용은 [TFLite 모델 최적화](https://www.tensorflow.org/lite/performance/model_optimization) 및 [TensorFlow 모델 최적화 툴킷](https://www.tensorflow.org/model_optimization) 페이지를 참조하세요. 양자화 유형은 아래에 간략하게 설명되어 있습니다.

여기서는 변환기 최적화 플래그를 `tf.lite.Optimize.DEFAULT`로 설정한 GPT-2 모델에서 [학습 후 동적 범위 양자화](https://www.tensorflow.org/lite/performance/post_training_quant)를 사용하며, 나머지 변환 프로세스는 앞서 설명한 것과 동일합니다. 이 양자화 기법을 사용하면 최대 출력 길이를 100으로 설정한 Pixel 7에서 지연 시간이 약 6.7초인 것으로 테스트되었습니다.

```python
gpt2_lm.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    gpt2_lm)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TF ops
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.experimental_select_user_tf_ops = [
    "UnsortedSegmentJoin",
    "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
quant_generate_tflite = converter.convert()
run_inference("I'm enjoying a", quant_generate_tflite)
```

**동적 범위**

동적 범위 양자화는 온디바이스 모델 최적화를 위한 권장 시작 지점입니다. 모델 크기를 약 4배 줄일 수 있으며, 캘리브레이션을 위한 대표 데이터세트를 제공하지 않아도 메모리 사용량을 줄여주고 더 빠른 컴퓨팅을 할 수 있도록 해주므로 시작 지점으로 권장됩니다. 이러한 유형의 양자화는 변환 시 부동 소수점부터 8비트 정수까지의 가중치로만 정적으로 양자화합니다.

**FP16**

부동 소수점 모델도 가중치를 float16 유형으로 정량화하여 최적화할 수 있습니다. [Float16 양자화](https://www.tensorflow.org/lite/performance/post_training_float16_quant)의 장점은 모델 크기를 최대 절반으로 줄여(모든 가중치가 절반 크기로 되므로) 정확성 손실을 최소화하고, float16 데이터에서 직접 작동할 수 있는 GPU 대리자를 지원한다는 것입니다(따라서 float32 데이터보다 계산 속도가 빨라짐). Float16 가중치로 변환된 모델은 추가 수정 없이 CPU에서 계속 실행할 수 있습니다. Float16 가중치는 첫 번째 추론 전에 float32로 업샘플링되므로 지연과 정확성에 미치는 영향을 최소화하는 대신 모델 크기를 줄일 수 있습니다.

**전체 정수 양자화**

[전체 정수 양자화](https://www.tensorflow.org/lite/performance/post_training_integer_quant)는 가중치와 활성화 등 32비트 부동 소수점 숫자를 모두 가장 가까운 8비트 정수로 변환합니다. 이러한 유형의 양자화는 추론 속도가 향상된 더 작은 모델을 생성하므로 마이크로컨트롤러를 사용할 때 매우 유용합니다. 이 모드는 활성화가 양자화에 대해 민감한 경우에 권장됩니다.

### Android 앱 통합

이 [안드로이드 예제](https://github.com/tensorflow/examples/tree/master/lite/examples/generative_ai)를 따라 TFLite 모델을 Android 앱에 통합할 수 있습니다.

### 전제 조건

아직 설치하지 않았다면 웹사이트의 지침에 따라 [Android Studio](https://developer.android.com/studio/index.html)를 설치하세요.

- Android Studio 2022.2.1 이상
- 4G 이상의 메모리를 갖춘 Android 기기 또는 Android 에뮬레이터

### Android Studio로 빌드 및 실행하기

- Android Studio를 열고 시작 화면에서 **기존 Android Studio 프로젝트 열기**를 선택합니다.
- 파일 열기 또는 프로젝트 창이 나타나면, TensorFlow Lite 샘플 GitHub 리포지토리를 복제한 위치에서 [`lite/examples/generative_ai/android`](https://github.com/tensorflow/examples/tree/master/lite/examples/generative_ai/android) 디렉터리로 이동하여 선택합니다.
- 오류 메시지에 따라 다양한 플랫폼과 도구를 설치해야 할 수도 있습니다.
- 변환된 .tflite 모델의 이름을 `autocomplete.tflite`로 변경하고 `app/src/main/assets/` 폴더에 복사합니다.
- **빌드 -&gt; 프로젝트 만들기** 메뉴를 선택하여 앱을 빌드합니다(버전에 따라 Ctrl+F9).
- 메뉴 **실행 -&gt; '앱' 실행**을 클릭합니다(버전에 따라 Shift+F10).

또는 [gradle 래퍼](https://docs.gradle.org/current/userguide/gradle_wrapper.html#gradle_wrapper)를 사용하여 명령줄에서 빌드할 수도 있습니다. 자세한 정보는 [Gradle 문서](https://docs.gradle.org/current/userguide/command_line_interface.html)를 참조하세요.

### (선택 사항).aar 파일 빌드하기

기본적으로 앱은 필요한 `.aar` 파일을 자동으로 다운로드합니다. 직접 빌드하려면 `app/libs/build_aar/` 폴더로 전환한 뒤 `./build_aar.sh`를 실행하면 됩니다. 이 스크립트는 TensorFlow Text에서 필요한 연산을 가져와서 Select TF 연산자를 위한 aar를 빌드합니다.

컴파일이 완료되면 `tftext_tflite_flex.aar` 파일이 새로 생성됩니다. `app/libs/` 폴더에서 .aar 파일을 교체하고 앱을 다시 빌드합니다.

이때, 표준 `tensorflow-lite` aar를 gradle 파일에 포함해야 한다는 점에 유의해야 합니다.

### 컨텍스트 창 크기

<img src="https://www.tensorflow.org/lite/examples/auto_complete/images/context_window.png" class="attempt-right">

앱에는 변경 가능한 매개변수 '컨텍스트 창 크기'가 있습니다. 이는 현재 LLM이 일반적으로 모델에 '프롬프트'로 공급될 수 있는 단어/토큰의 수를 제한하는 고정된 컨텍스트 크기를 갖기 때문에 필요합니다('단어'와 '토큰'은 토큰화 방법이 다르기 때문에 이 경우 컨텍스트 크기가 반드시 동일하지는 않습니다). 이 숫자가 중요한 이유는 다음과 같습니다.

- 너무 작게 설정하면 모델에 의미 있는 출력을 생성하기에 충분한 컨텍스트가 없게 됩니다.
- 너무 크게 설정하면 모델에 작업할 공간이 충분하지 않게 됩니다(출력 시퀀스에 프롬프트가 포함되므로).

적합한 크기를 실험해 볼 수 있지만 출력 시퀀스 길이의 50% 이하로 설정하는 것이 좋습니다.

## 안전과 책임감 있는 AI

원래 [OpenAI GPT-2 발표문](https://openai.com/research/better-language-models)에서 언급했듯이, GPT-2 모델에는 [주목할 만한 주의 사항과 제한 사항이 있습니다](https://github.com/openai/gpt-2#some-caveats). 실제로 현재 LLM에는 일반적으로 환각, 공정성, 편향성 등으로 잘 알려진 몇 가지 문제가 있으며, 이는 LLM이 실제 데이터를 기반으로 학습되어 현실 세계의 문제를 반영하기 때문에 발생합니다.

이 코드랩은 TensorFlow 도구를 사용하여 LLM으로 구동되는 앱을 만드는 방법을 보여주기 위한 목적으로만 제작되었습니다. 이 코드랩에서 생성된 모델은 교육 목적으로 사용되며 LLM 운영 용도로 사용할 수 없습니다.

LLM 운영에 사용하려면 신중하게 훈련 데이터세트를 선택하고 포괄적인 안전 조치를 수립해야 합니다. 이 안드로이드 앱에서 제공하는 기능 중 하나는 욕설 필터로, 부적절한 사용자 입력이나 모델 출력을 거부합니다. 부적절한 언어가 감지되면 앱은 해당 작업을 거부합니다. LLM의 맥락에서 책임감 있는 AI에 대해 자세히 알아보려면 Google I/O 2023의 '생성형 언어 모델로 안전하고 책임감 있게 개발하기' 기술 세션을 시청하고 [책임감 있는 AI 툴킷](https://www.tensorflow.org/responsible_ai)을 확인해 보세요.
