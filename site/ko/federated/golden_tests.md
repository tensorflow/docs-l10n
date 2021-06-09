# 골든 테스트

TFF는 `golden`이라 부르는 작은 라이브러리를 포함하며, 이 라이브러리는 골든 테스트를 쉽게 작성하고 유지하는 데 사용됩니다.

## 골든 테스트란 무엇입니까? 언제 사용해야 합니까?

골든 테스트는 코드에서 함수의 출력을 변경했음을 개발자가 알기를 원할 때 사용됩니다. 명확하고 문서화된 특정 속성 세트를 테스트하기보다는 함수의 정확한 출력에 대해 약속한다는 점에서 훌륭한 단위 테스트의 많은 특성을 위반합니다. 골든 출력에 대한 변경이 언제 "예상"되는지 또는 골든 테스트에서 적용하기 위해 확보한 일부 속성을 위반하는지 여부가 명확하지 않은 경우가 있습니다. 따라서 신중하게 고려한 단위 테스트를 일반적으로 골든 테스트보다 선호합니다.

그러나 골든 테스트는 오류 메시지, 진단 또는 생성된 코드의 정확한 내용을 검증하는 데 매우 유용할 수 있습니다. 이러한 경우, 골든 테스트는 생성된 출력에 대한 모든 변경 사항이 "올바르게 보이는지"를 확인하는 데 도움이 될 수 있습니다.

## `golden`을 사용하여 테스트를 작성하려면 어떻게 해야 합니까?

`golden.check_string(filename, value)`는 `golden` 라이브러리의 기본 시작점입니다. 마지막 경로 요소가 `filename`인 파일의 내용에서 `value` 문자열을 확인합니다. `filename`의 전체 경로는 명령줄 `--golden <path_to_file>` 인수를 통해 제공되어야 합니다. 마찬가지로, 이들 파일은 `py_test` BUILD 규칙에 `data` 인수를 사용하는 테스트에 사용할 수 있어야 합니다. `location` 함수를 사용하여 올바른 상대 경로를 생성하세요.

```
py_string_test(
  ...
  args = [
    "--golden",
    "$(location path/to/first_test_output.expected)",
    ...
    "--golden",
    "$(location path/to/last_test_output.expected)",
  ],
  data = [
    "path/to/first_test_output.expected",
    ...
    "path/to/last_test_output.expected",
  ],
  ...
)
```

관례적으로 골든 파일은 테스트 대상과 이름이 같고 접미사가 `_goldens`인 형제 디렉터리에 배치되어야 합니다.

```
path/
  to/
    some_test.py
    some_test_goldens/
      test_case_one.expected
      ...
      test_case_last.expected
```

## `.expected` 파일을 어떻게 업데이트합니까?

`.expected` 파일은 인수 `--test_arg=--update_goldens --test_strategy=local`로 해당 테스트 대상을 실행하여 업데이트할 수 있습니다. 결과 차이에 예상치 못한 변경 사항이 있는지 확인해야 합니다.
