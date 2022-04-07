# TFDS 저장소에 기여하기

우리 라이브러리에 관심을 가져 주셔서 감사합니다! 우리는 의욕넘치는 커뮤니티와 함께하게 되어 기쁩니다.

## 시작하며

- 만약 당신이 TFDS가 처음이라면, 시작하는 가장 빠른 방법은 가장 많이 요청된 것에 집중해서 우리의[requested dataset](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22dataset+request%22+sort%3Areactions-%2B1-desc)중 하나를 구현해보는 것입니다. 지침은 [Follow our guide](https://www.tensorflow.org/datasets/add_dataset)에 있습니다.
- 이슈들, 기능요청들과 버그들,... 은 그것들이 전체 TFDS 커뮤니티에 이익을 주기 때문에 새로운 데이타셋을 추가하는 것보다 더욱 큰 영향이 있습니다.[potential contribution list](https://github.com/tensorflow/datasets/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+-label%3A%22dataset+request%22+)을 보십시오. Starts with the ones labeled with [contribution-welcome](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)의 이름을 가진 것으로 시작해보십시오. 그것은 시작하기에 쉽고 작은 이슈가 포함되어 있습니다.
- 이미 할당되었지만 한동안 업데이트되지 않은 버그를 주저하지 마십시오.
- 문제를 할당받을 필요가 없습니다. 작업을 시작할 때 문제에 대해 간단히 언급하십시오. :)
- 문제에 관심이 있지만 시작하는 방법을 모르는 경우 주저하지 말고 도움을 요청하십시오. 조기 피드백을 원하시면 PR 초안을 보내 주시기 바랍니다.
- 불필요한 작업 중복을 방지하려면 [보류중인 Pull Requests](https://github.com/tensorflow/datasets/pulls) 목록을 확인하고 작업중인 문제에 대해 의견을 말하세요.

## 설정

### 저장소 복제

시작하려면 [Tensorflow Datasets](https://github.com/tensorflow/datasets) 저장소를 복제하거나 다운로드하고 저장소를 로컬에 설치하세요.

```sh
git clone https://github.com/tensorflow/datasets.git
cd datasets/
```

개발 종속성을 설치하십시오.

```sh
pip install -e .  # Install minimal deps to use tensorflow_datasets
pip install -e ".[dev]"  # Install all deps required for testing and development
```

모든 dataset-specific deps를 설치하기 위해 `pip install -e ".[tests-all]"` 가 있음을 주의하십시오.

### Visual Studio 코드

[Visual Studio Code로](https://code.visualstudio.com/) 개발할 때 리포지토리에는 개발에 도움이되는 [미리 정의 된 설정](https://github.com/tensorflow/datasets/tree/master/.vscode/settings.json) (올바른 들여 쓰기, 파일 린트 등)이 제공됩니다.

참고 : VS Code에서 테스트 검색 활성화는 일부 VS Code 버그 [# 13301](https://github.com/microsoft/vscode-python/issues/13301) 및 [# 6594](https://github.com/microsoft/vscode-python/issues/6594) 로 인해 실패 할 수 있습니다. 문제를 해결하기 위해 테스트 검색 로그를 볼 수 있습니다.

- 몇가지 텐서플로우 경고가 있다면 [this fix](https://github.com/microsoft/vscode-python/issues/6594#issuecomment-555680813)를 확인하십시오.
- 설치해야하는 가져 오기 누락으로 인해 검색이 실패하면 PR을 보내 `dev` pip 설치를 업데이트하십시오.

## PR 체크리스트

### CLA에 서명

이 프로젝트에 기여하기 위해서는 반드시 CLA가 동반되어야 합니다. 당신의(또는 당신의 고용주) 기여에 대한 저작권을 유지하십시오; 이것은 말 그대로 우리에게 이것을 사용할 수 있게 해주고, 당신의 기여를 프로젝트의 일부분으로 재분배하게 합니다. 현재 파일에 대한 동의를 확인하거나 새로운 것에 대한 허가가 필요하다면 [https://cla.developers.google.com/](https://cla.developers.google.com/) 를 확인하십시오.

일반적으로 CLA의 확인은 한번만 필요하기 때문에, 만약 다른 프로젝트를 할 때도 이미 CLA가 확인되어있을 것이고 다시 할 필요는 없습니다.

### 모범 사례 따르기

- 가독성이 중요합니다. 코드는 최고의 프로그래밍 관행을 따라야합니다 (중복 방지, 작은 자체 포함 함수, 명시 적 변수 이름 등으로 분해).
- 단순할수록 좋습니다 (예 : 구현을 검토하기 쉬운 여러 개의 작은 독립형 PR로 분할).
- 필요한 경우 테스트를 추가합니다. 기존 테스트는 통과해야합니다.
- [입력 주석](https://docs.python.org/3/library/typing.html) 추가

### 스타일 가이드 확인

우리의 스타일은 [PEP 8 Python 스타일 가이드를](https://www.python.org/dev/peps/pep-0008) 기반으로하는 [Google Python 스타일 가이드](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) 를 기반으로합니다. 새 코드는 [블랙 코드 스타일](https://github.com/psf/black/blob/master/docs/the_black_code_style.md) 을 따라야하지만 다음을 사용합니다.

- 라인 길이 : 80
- 4 대신 2 공백 들여 쓰기
- 작은 따옴표 `'`

**중요 :** 코드의 형식이 올바른지 확인하려면 코드에서 `pylint` 를 실행해야합니다.

```sh
pip install pylint --upgrade
pylint tensorflow_datasets/core/some_file.py
```

`yapf` 를 사용하여 파일 형식을 자동으로 지정할 수 있지만 도구가 완벽하지 않으므로 나중에 수정 사항을 수동으로 적용해야 할 것입니다.

```sh
yapf tensorflow_datasets/core/some_file.py
```

`pylint` 와 `yapf` 둘다 `pip install -e ".[dev]"`와 함께 설치가 되어있어야 하지만, `pip install`을 통해서 수동으로 설치할 수도 있습니다. 만약 당신이 VS Code를 이용한다면, 이 도구들은 UI안에 통합되어야 합니다.

### 독 스트링 및 타이핑 주석

클래스와 함수는 독 스트링과 타이핑 주석으로 문서화되어야합니다. 독 스트링은 [Google 스타일을](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) 따라야합니다. 예를 들면 :

```python
def function(x: List[T]) -> T:
  """One line doc should end by a dot.

  * Use `backticks` for code and tripple backticks for multi-line.
  * Use full API name (`tfds.core.DatasetBuilder` instead of `DatasetBuilder`)
  * Use `Args:`, `Returns:`, `Yields:`, `Attributes:`, `Raises:`

  Args:
    x: description

  Returns:
    y: description
  """
```

### 단위 테스트 추가 및 실행

새로운 기능이 단위 테스트로 테스트되었는지 확인하십시오. VS Code 인터페이스 또는 명령 줄을 통해 테스트를 실행할 수 있습니다. 예를 들면 :

```sh
pytest -vv tensorflow_datasets/core/
```

`pytest` vs `unittest`: 전통적으로, 우리는 작성 테스트를 위해 `unittest` 모듈을 사용해왔습니다. 새로운 테스트 `pytest`는 더욱 간단하고, 융통성 있고, 현대적이며 많은 라이브러리(numpy, pandas, sklearn, matplotlib, scipy, six,...)에서 사용합니다. pytest가 익숙하지 않다면 [pytest guide](https://docs.pytest.org/en/stable/getting-started.html#getstarted)를 참고하십시오.

DatasetBuilder를 위한 테스트는 특별하며,  [guide to add a dataset](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md#test-your-dataset)안에 문서화되어 있습니다.

### 리뷰를 위해 PR을 보내십시오!

축하합니다! pull 요청 사용에 대한 자세한 내용은 [GitHub 도움말](https://help.github.com/articles/about-pull-requests/) 을 참조하십시오.
