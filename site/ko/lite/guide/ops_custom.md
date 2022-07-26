# 사용자 정의 연산자

TensorFlow Lite 내장 연산자 라이브러리는 제한된 수의 TensorFlow 연산자만 지원하므로 모든 모델을 변환할 수 있는 것은 아닙니다. 자세한 내용은 [연산자 호환성](ops_compatibility.md)을 참조하세요.

변환이 기능하도록 하기 위해 사용자는 사용자 정의 연산자라고 하는 TensorFlow Lite에서 지원되지 않는 TensorFlow 연산자를 자체적으로 사용자 정의하여 구현할 수 있습니다. *그렇지 않고, 지원되지 않는(또는 지원되는) 일련의 TensorFlow 연산자를 하나의 융합되고 최적화된 사용자 정의 연산자로 결합하려면 [연산자 융합](https://www.tensorflow.org/lite/convert/operation_fusion)을 참조하세요.*

사용자 정의 연산자 사용은 다음 4개 단계로 구성됩니다.

- [TensorFlow 모델을 만듭니다.](#create-a-tensorflow-model) 저장된 모델(또는 Graph Def)이 올바르게 명명된 TensorFlow Lite 연산자를 참조하는지 확인하세요.

- [TensorFlow Lite 모델로 변환합니다.](#convert-to-a-tensorflow-lite-model) 모델을 성공적으로 변환하려면 올바른 TensorFlow Lite 변환기 속성을 설정해야 합니다.

- [연산자를 생성하고 등록합니다.](#create-and-register-the-operator) 이는 TensorFlow Lite 런타임이 그래프의 연산자와 매개변수를 실행 가능한 C/C++ 코드에 매핑하는 방법을 알 수 있도록 하기 위한 것입니다.

- [연산자를 테스트하고 프로파일링합니다.](#test-and-profile-your-operator) 사용자 정의 연산자만 테스트하려면 사용자 정의 연산자만 사용하여 모델을 만들고 [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc) 프로그램을 사용하는 것이 가장 좋습니다.

TensorFlow에서는 지원되지만 TensorFlow Lite에서는 지원되지 않는 사용자 정의 연산자 `tf.sin`(`Sin`으로 명명됨, #create-a-tensorflow-model 참조)을 사용하여 모델을 실행하는 엔드 투 엔드 예제를 살펴보겠습니다.

참고: 실제로 `tf.sin`은 사용자 정의 연산자가 **아닙니다**. TensorFlow와 TensorFlow Lite에서 모두 지원하는 정규 연산자입니다. 그러나 간단한 워크플로를 보여주기 위해 다음 예제에서는 이를 사용자 정의 연산자라고 **가정**합니다.

## 예: 사용자 정의 `Sin` 연산자

TensorFlow Lite에 없는 TensorFlow 연산자를 지원하는 예를 살펴보겠습니다. `Sin` 연산자를 사용하고 `offset`의 훈련 가능한 함수 `y = sin(x + offset)`에 대한 매우 간단한 모델을 빌드한다고 가정합니다.

### TensorFlow 모델 만들기

다음 코드 조각은 간단한 TensorFlow 모델을 훈련합니다. 이 모델에는 함수 `y = sin(x + offset)`인 사용자 정의 연산자 `Sin`만 포함되어 있으며, 여기서 `offset`을 훈련할 수 있습니다.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-0.6569866 ,  0.99749499,  0.14112001, -0.05837414,  0.80641841]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Sin`
@tf.function
def sin(x):
  return tf.sin(x + offset, name="Sin")

  # Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = sin(x)
      loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))

for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())
```

```python
The actual offset is: 1.0
The predicted offset is: 1.0000001
```

이 시점에서 기본 변환기 플래그를 사용하여 TensorFlow Lite 모델을 생성하려고 하면 다음 오류 메시지가 표시됩니다.

```none
Error:
Some of the operators in the model are not supported by the standard TensorFlow
Lite runtime...... Here is
a list of operators for which you will need custom implementations: Sin.
```

### TensorFlow Lite 모델로 변환하기

아래와 같이 변환기 특성 `allow_custom_ops`를 설정하여 사용자 정의 연산자로 TensorFlow Lite 모델을 만듭니다.

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([sin.get_concrete_function(x)], sin)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

이 시점에서 기본 인터프리터로 실행하면 다음 오류 메시지가 표시됩니다.

```none
Error:
Didn't find custom operator for name 'Sin'
Registration failed.
```

### 연산자를 생성하고 등록합니다.

모든 TensorFlow Lite 연산자(사용자 정의 및 내장)는 네 가지 함수로 구성된 간단한 pure-C 인터페이스를 사용하여 정의됩니다.

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

`TfLiteContext` 및 `TfLiteNode`에 대한 자세한 내용은 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)를 참조하세요. 전자는 오류 보고 기능과 모든 텐서를 포함한 전역 객체에 대한 액세스를 제공합니다. 후자는 구현에서 입력과 출력에 액세스할 수 있도록 합니다.

인터프리터는 모델을 로드할 때 그래프의 각 노드에 대해 한 번씩 `init()`을 호출합니다. 그래프에서 연산자가 여러 번 사용되면 지정된 `init()`이 두 번 이상 호출됩니다. 사용자 정의 연산자의 경우, 매개변수 이름을 해당 값에 매핑하는 flexbuffer를 포함하는 구성 버퍼가 제공됩니다. 인터프리터가 이미 연산자 매개변수를 구문 분석했기 때문에 내장 연산자에 대한 버퍼는 비어 있습니다. 상태가 필요한 커널 구현은 여기에서 상태를 초기화하고 소유권을 호출자에게 전달해야 합니다. 각 `init()` 호출에서 해당하는 `free()` 호출이 있으므로 구현에서 `init()`에 할당했을 수 있는 버퍼를 삭제할 수 있습니다.

입력 텐서의 크기가 조정될 때마다 인터프리터는 그래프를 돌며 변경 구현을 알립니다. 그러면 그래프에서 내부 버퍼의 크기를 조정하고 입력 형상과 유형의 유효성을 확인하며 출력 형상을 다시 계산할 수 있습니다. 이 연산은 모두 `prepare()`를 통해 수행되며 구현은 `node->user_data`를 사용하여 해당 상태에 액세스할 수 있습니다.

마지막으로, 추론이 실행될 때마다 인터프리터는 그래프를 순회하며 `invoke()`를 호출하고, 여기에서도 상태를 `node->user_data`로 사용할 수 있습니다.

사용자 정의 연산자는 일반적으로 다음과 같은 네 가지 함수와 전역 등록 함수를 정의하여 내장 연산자와 완전히 동일한 방식으로 구현할 수 있습니다.

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

등록은 자동이 아니며 `Register_MY_CUSTOM_OP`에 대한 명시적인 호출이 이루어져야 합니다. 표준 `BuiltinOpResolver`(`:builtin_ops` 대상에서 사용 가능)가 내장 연산자의 등록을 처리하지만 사용자 정의 연산자는 별도의 사용자 정의 라이브러리에 수집해야 합니다.

### TensorFlow Lite 런타임에서 커널 정의하기

TensorFlow Lite에서 op를 사용하려면 두 가지 함수(`Prepare` 및 `Eval`)를 정의하고 `TfLiteRegistration`을 구성하기만 하면 됩니다.

```cpp
TfLiteStatus SinPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node,0);
  TfLiteTensor* output = GetOutput(context, node,0);

  float* input_data = input->data.f;
  float* output_data = output->data.f;

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = sin(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {nullptr, nullptr, SinPrepare, SinEval};
  return &r;
}
```

`OpResolver`를 초기화할 때 사용자 정의 op를 resolver에 추가합니다(아래 예 참조). 그러면 TensorFlow Lite가 새 구현을 사용할 수 있도록 연산자가 Tensorflow Lite에 등록됩니다. `TfLiteRegistration`의 마지막 두 인수는 사용자 정의 op에 대해 정의한 `SinPrepare` 및 `SinEval` 함수에 해당합니다. `SinInit` 및 `SinFree` 함수를 사용하여 op에 사용된 변수를 초기화하고 공간을 확보한 경우, 이들 함수는 `TfLiteRegistration`의 처음 두 인수에 추가됩니다. 이 예제에서 이들 인수는 `nullptr`로 설정됩니다.

### 커널 라이브러리에 연산자 등록하기

이제 커널 라이브러리에 연산자를 등록해야 합니다. 이를 위해 `OpResolver`를 사용합니다. 배후에서 인터프리터가 모델의 각 연산자를 실행하도록 할당될 커널 라이브러리를 로드합니다. 기본 라이브러리에는 내장 커널만 포함되어 있지만 사용자 정의 라이브러리 op 연산자로 대체/확대할 수 있습니다.

연산자 코드와 이름을 실제 코드로 변환하는 `OpResolver` 클래스가 다음과 같이 정의됩니다.

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

정규 사용을 위해서는 `BuiltinOpResolver`를 사용하고 다음을 작성해야 합니다.

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

위에서 만든 사용자 정의 op를 추가하려면 `AddOp`를 호출합니다(resolver를 `InterpreterBuilder`에 전달하기 전).

```c++
resolver.AddCustom("Sin", Register_SIN());
```

내장 연산자 세트가 너무 큰 것으로 여겨지면 주어진 연산자 하위 집합(보통은 주어진 모델에 포함된 연산자)을 바탕으로 새 `OpResolver`를 코드로 생성할 수 있습니다. 이는 TensorFlow의 선택적 등록과 동일하며 `tools` 디렉토리에서 간단한 버전을 사용할 수 있습니다.

Java에서 사용자 정의 연산자를 정의하려면, 고유한 사용자 정의 JNI 레이어를 빌드하고 [이 jni 코드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/builtin_ops_jni.cc)에서 자체 AAR을 컴파일해야 합니다. 마찬가지로, Python에서 사용할 수 있는 이러한 연산자를 정의하려면 [Python 래퍼 코드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc)에 등록할 수 있습니다.

단일 연산자 대신 일련의 연산자를 지원하는 경우에도 위와 유사한 프로세스를 따를 수 있습니다. 추가할 수 있는 `AddCustom` 연산자에 제한은 없습니다. 또한 `BuiltinOpResolver`를 사용하면 `AddBuiltin`을 통해 내장 구현을 재정의할 수도 있습니다.

### 연산자 테스트 및 프로파일링하기

TensorFlow Lite 벤치마크 도구로 op를 프로파일링하려면 TensorFlow Lite용 [벤치마크 모델 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool)를 사용할 수 있습니다. 테스트 목적으로 [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/register.cc)에 적절한 `AddCustom` 호출을 추가하여(위에 나타낸 바와 같이) TensorFlow Lite의 로컬 빌드가 사용자 정의 op를 인식하도록 할 수 있습니다.

## 모범 사례

1. 메모리 할당 및 할당 해제를 신중하게 최적화하세요. `Prepare`에서 메모리를 할당하는 것이 `Invoke`보다 효율적이며, 루프 전에 메모리를 할당하는 것이 매번 반복하는 것보다 낫습니다. Malloc을 직접 수행하지 말고 임시 텐서 데이터를 사용하세요(항목 2 참조). 복사하는 대신 되도록 포인터/참조를 사용하세요.

2. 전체 연산 중에 데이터 구조가 유지되는 경우, 임시 텐서를 사용하여 메모리를 미리 할당하는 것이 좋습니다. 다른 함수에서 텐서 인덱스를 참조하려면 OpData 구조체를 사용해야 할 수 있습니다. [컨볼루션 커널](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc)의 예를 참조하세요. 다음은 샘플 코드입니다.

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3. 낭비되는 메모리가 너무 많지 않은 경우, 실행을 반복할 때마다 동적으로 할당된 `std::vector`를 사용하는 것보다 고정된 정적 크기 배열(또는 `Resize`의 미리 할당된 `std::vector`)을 사용하는 것이 좋습니다.

4. 바이너리 크기에 영향을 미치므로 아직 존재하지 않는 표준 라이브러리 컨테이너 템플릿을 인스턴스화하지 마세요. 예를 들어, 다른 커널에 존재하지 않는 `std::map`이 연산에 필요한 경우, 직접 인덱싱 매핑과 함께 `std::vector`를 사용하면 바이너리 크기를 작게 유지하면서 동작할 수 있습니다. 정보를 얻거나 요청하기 위해 다른 커널이 무엇을 사용하는지 확인하세요.

5. `malloc`이 반환하는 메모리에 대한 포인터를 확인합니다. 이 포인터가 `nullptr`이면 해당 포인터를 사용하여 연산을 수행하지 않아야 합니다. 함수에서 `malloc`을 수행하고 오류 종료가 발생하면 종료하기 전에 메모리 할당을 해제하세요.

6. 특정 조건을 확인하려면 `TF_LITE_ENSURE(context, condition)`를 사용하세요. `TF_LITE_ENSURE`를 사용할 때 코드에서 메모리를 기다리게 하면 안 됩니다. 즉, 누출이 발생하는 리소스가 할당되기 전에 이러한 매크로를 사용해야 합니다.
