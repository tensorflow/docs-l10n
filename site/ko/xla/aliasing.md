# XLA의 앨리어싱

이 설명서에서는 XLA용 앨리어싱 API를 설명합니다. XLA 프로그램을 빌드할 때 입력 및 출력 버퍼 간에 원하는 앨리어싱을 지정할 수 있습니다.

## 컴파일 시 앨리어싱 정의하기

예를 들어, 단순히 입력에 `1`을 더하는 간단한 HLO 모듈을 고려해보세요.

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

이 모듈은 2개의 4byte 버퍼를 할당합니다. 하나는 입력 `%p`용이고 다른 하나는 출력 `%out`용입니다.

그러나 종종 제자리에서 업데이트를 수행하는 것이 바람직합니다(예를 들어, 표현식을 생성하는 프런트 엔드에서 입력 변수가 증가 `p++`에서와 같이 계산 후 더 이상 살아 있지 않은 경우).

이러한 업데이트를 효율적으로 수행하기 위해 입력 앨리어싱을 지정할 수 있습니다.

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

이 형식은 전체 출력(`{}`로 표시됨)이 입력 매개변수 `0`으로 앨리어싱되도록 지정합니다.

프로그래밍 방식으로 앨리어싱을 지정하려면 [`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) API를 참조하세요.

## 런타임에 앨리어싱 정의하기

이전 단계에서 정의된 앨리어싱은 *컴파일* 중에 지정됩니다. 실행 중에는 [`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h) API를 사용하여 실제로 버퍼를 기부할지 여부를 선택할 수 있습니다.

프로그램에 대한 입력 버퍼는 [`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h)에 래핑되며, 이는 `MaybeOwningDeviceMemory` 트리를 포함합니다. 메모리가 *소유(owing)*로 지정되면(버퍼의 소유권이 XLA 런타임에 전달됨), 버퍼가 실제로 기부되고 컴파일 시 앨리어싱 API에서 요청한 대로 업데이트가 제자리에서 실행됩니다.

그러나 컴파일 시 앨리어싱된 버퍼가 런타임에 기부되지 *않으면* *복사 방지*가 시작됩니다. 추가 출력 버퍼 `O`가 할당되고 앨리어싱될 입력 버퍼 `P`의 콘텐츠가 `O`로 복사됩니다(따라서 사실상 프로그램은 마치 런타임에 버퍼 `O`가 기부된 것처럼 실행할 수 있습니다).

## 프런트 엔드 상호 운용성

### TF/XLA

XLA로 컴파일된 TensorFlow 프로그램의 클러스터에서, 모든 리소스 변수 업데이트는 컴파일 시 앨리어싱이 지정됩니다(런타임 시 앨리어싱 지정은 리소스 변수 텐서에 대한 참조가 다른 항목에 있는지 여부에 따라 다릅니다).
