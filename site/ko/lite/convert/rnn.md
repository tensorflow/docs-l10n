# TensorFlow RNN을 TensorFlow Lite로 변환

## 개요

TensorFlow Lite는 TensorFlow RNN 모델을 TensorFlow Lite의 LSTM 융합 연산으로 변환하는 작업을 지원합니다. 융합 연산은 기본 커널 구현의 성능을 최대화할 뿐만 아니라 양자화와 같은 복잡한 변환을 정의하는 더 높은 수준의 인터페이스를 제공합니다.

TensorFlow에는 RNN API의 다양한 변형이 있으며, 두 가지 접근 방식을 소개합니다.

1. Keras LSTM과 같은 **표준 TensorFlow RNN API에 대한 기본 지원**을 제공합니다. 이것이 권장되는 옵션입니다.
2. **사용자 정의** **RNN 구현****에 대한 변환 인프라**에 **인터페이스**를 제공하여 TensorFlow Lite를 연결하고 변환합니다. 여기서는 lingvo의 [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) 및 [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) RNN 인터페이스를 사용하여 이러한 즉시 변환의 몇 가지 예를 제공합니다.

## 변환기 API

이 특성은 TensorFlow 2.3 릴리스의 일부입니다. [tf-nightly](https://pypi.org/project/tf-nightly/) pip 또는 head에서 사용할 수도 있습니다.

이 변환 기능은 SavedModel을 통해 또는 Keras 모델에서 직접 TensorFlow Lite로 변환할 때 사용할 수 있습니다. 사용 예를 참조하세요.

### SavedModel에서 사용

<a id="from_saved_model"></a>

```
# build a saved model. Here concrete_function is the exported function
# corresponding to the TensorFlow model containing one or more
# Keras LSTM layers.
saved_model, saved_model_dir = build_saved_model_lstm(...)
saved_model.save(saved_model_dir, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
```

### Keras 모델에서 사용

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## 예

Keras LSTM - TensorFlow Lite [Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb)은 TensorFlow Lite 인터프리터를 사용한 엔드 투 엔드 사용 예를 보여줍니다.

## TensorFlow RNN API 지원

<a id="rnn_apis"></a>

### Keras LSTM 변환(권장)

Keras LSTM에서 TensorFlow Lite로의 즉시 변환을 지원합니다. 동작 방식에 대한 자세한 내용은 [Keras LSTM 인터페이스](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;"> </span>및 [여기](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627)에서 변환 논리를 참조하세요.

또한 Keras 연산 정의와 관련하여 TensorFlow Lite의 LSTM 계약을 강조할 필요가 있습니다.

1. **입력** 텐서의 차원 0은 배치 크기입니다.
2. **recurrent_weight** 텐서의 차원 0은 출력 수입니다.
3. **weight** 및 **recurrent_kernel** 텐서는 전치됩니다.
4. 전치된 가중치, 전치된 recurrent_kernel 및 **바이어스** 텐서는 차원 0을 따라 동일한 크기의 4개 텐서로 분할됩니다. 이들 텐서는 <strong>input gate, forget gate, cell 및 output gate에 해당합니다.</strong>

#### 여러 형태의 Keras LSTM

##### Time major

사용자는 time-major 또는 비 time-major를 선택할 수 있습니다. Keras LSTM은 함수 def 속성에 time-major 속성을 추가합니다. 단방향 시퀀스 LSTM의 경우, unidirecional_sequence_lstm의 [time major 속성](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902)에 간단히 매핑할 수 있습니다.

##### 양방향 LSTM

양방향 LSTM은 두 개의 Keras LSTM 레이어(하나는 정방향, 다른 하나는 역방향)로 구현할 수 있습니다. [여기](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/wrappers.py#L382)의 예를 참조하세요. go_backward 속성을 보고 나면 이를 역방향 LSTM으로 인식한 다음 정방향 및 역방향 LSTM을 함께 그룹화합니다. **이 작업은 나중에 수행할 것입니다.** 현재는 TensorFlow Lite 모델에서 두 개의 UnidirectionalSequenceLSTM 연산을 생성합니다.

### 사용자 정의 LSTM 변환 예

TensorFlow Lite는 사용자 정의 LSTM 구현을 변환하는 방법도 제공합니다. 여기서는 Lingvo의 LSTM을 구현 방법을 보여주는 예로 사용합니다. 자세한 내용은 [lingvo.LSTMCellSimple 인터페이스](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228) 및 [여기](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130)서 변환 논리를 참조하세요. 또한 [lingvo.LayerNormalizedLSTMCellSimple 인터페이스](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173) 및 [여기](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137)의 변환 논리에서 Lingvo의 또 다른 LSTM 정의를 보여주는 예를 제공합니다.

## TensorFlow Lite에 "사용자 TensorFlow RNN 가져오기"

사용자의 RNN 인터페이스가 표준 지원 인터페이스와 다른 경우, 몇 가지 옵션이 있습니다.

**옵션 1:** TensorFlow python에서 어댑터 코드를 작성하여 Keras RNN 인터페이스에 맞게 RNN 인터페이스를 조정합니다. 즉, 생성된 RNN 인터페이스에서 [tf_implements 주석](https://github.com/tensorflow/community/pull/113)이 있는 tf.function는 Keras LSTM 레이어에서 생성된 함수와 동일한 함수입니다. 그 후에는 Keras LSTM에 사용된 동일한 변환 API가 제 기능을 수행합니다.

**옵션 2:** 위의 작업이 불가능한 경우(예: 레이어 정규화와 같이 현재 TensorFlow Lite의 LSTM 융합 연산에 의해 노출되는 일부 기능이 Keras LSTM에 없는 경우), 사용자 정의 변환 코드를 작성하여 TensorFlow Lite 변환기를 확장하고 [여기서](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115) prepare-composite-functions MLIR 전달에 연결합니다. 함수의 인터페이스는 API 계약처럼 취급해야 하며 TensorFlow Lite LSTM 융합 연산으로 변환하는 데 필요한 인수(예: 입력, 바이어스, 가중치, 투영, 레이어 정규화 등)를 포함해야 합니다. 이 함수에 인수로 전달되는 텐서가 알려진 순위(즉, MLIR의 RankedTensorType)를 갖는 것이 바람직합니다. 그러면 이러한 텐서를 RankedTensorType으로 가정할 수 있는 변환 코드를 훨씬 쉽게 작성할 수 있고 TensorFlow Lite 융합 연산자의 피연산자에 해당하는 순위 지정된 텐서로 이러한 텐서를 쉽게 변환할 수 있습니다.

이러한 변환 흐름의 전체 예는 Lingvo의 LSTMCellSimple - TensorFlow Lite 변환입니다.

Lingvo의 LSTMCellSimple은 [여기](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)에서 정의됩니다. 이 LSTM 셀로 훈련된 모델은 다음과 같이 TensorFlow Lite로 변환할 수 있습니다.

1. tf.function에서 LSTMCellSimple의 모든 사용을 해당 사용에 적절하게 레이블이 지정된 tf_implements 주석으로 래핑합니다(예: lingvo.LSTMCellSimple은 여기서 좋은 주석 이름임). 생성된 tf.function이 변환 코드에서 예상되는 함수의 인터페이스와 일치하도록 해야 합니다. 이것은 주석을 추가하는 모델 작성자와 변환 코드 간의 계약입니다.

2. prepare-composite-functions 전달을 확장하여 사용자 정의 복합 연산을 TensorFlow Lite 융합 LSTM 연산 변환에 연결합니다. [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) 변환 코드를 참조하세요.

    변환 계약:

3. **가중치** 및 **투영** 텐서는 전치됩니다.

4. **{input, recurrent}** - **{cell, input gate, forget gate, output gate}**는 전치된 가중치 텐서를 조각화하여 추출됩니다.

5. **{bias}** - **{cell, input gate, forget gate, output gate}**는 바이어스 텐서를 조각화하여 추출됩니다.

6. **투영**은 전치된 투영 텐서를 조각화하여 추출됩니다.

7. [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137)에 대해 유사한 변환이 작성됩니다.

8. 정의된 모든 [MLIR-pass](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57)와 TensorFlow Lite flatbuffer로의 최종 내보내기를 포함한 나머지 TensorFlow Lite 변환 인프라는 재사용할 수 있습니다.

## 알려진 문제/제한 사항

1. 현재, 상태 비저장 Keras LSTM(Keras의 기본 동작) 변환만 지원됩니다. 상태 저장 Keras LSTM 변환은 향후 제공될 예정입니다.
2. 기본 상태 비저장 Keras LSTM 레이어를 사용하고 사용자 프로그램에서 명시적으로 상태를 관리하는 상태 저장 Keras LSTM 레이어를 모델링하는 것은 여전히 가능합니다. TensorFlow 프로그램은 여기에 설명한 특성을 사용하여 TensorFlow Lite로 계속해서 변환할 수 있습니다.
3. 양방향 LSTM은 현재 TensorFlow Lite에서 두 개의 UnidirectionalSequenceLSTM 연산으로 모델링됩니다. 이들 연산은 단일 BidirectionalSequenceLSTM 연산으로 대체됩니다.
