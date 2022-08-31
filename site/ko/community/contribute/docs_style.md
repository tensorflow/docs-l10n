# TensorFlow 설명서 스타일 가이드

## 모범 사례

- 사용자 의도와 독자에게 초점을 맞춥니다.
- 일상적 단어를 사용하고 문장을 짧게 유지합니다.
- 일관된 문장 구성, 표현 및 대문자 사용 원칙을 지킵니다.
- 문서를 쉽게 둘러볼 수 있도록 제목과 목록을 사용합니다.
- [Google 개발자 설명서 스타일 가이드](https://developers.google.com/style/highlights)가 도움이 됩니다.

## Markdown

몇 가지 예외가 있지만 TensorFlow는 [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/)(GFM)과 유사한 Markdown 구문을 사용합니다. 이 섹션에서는 GFM Markdown 구문과 TensorFlow 설명서에 사용되는 Markdown의 차이점에 대해 설명합니다.

### 코드 작성

#### 인라인 코드 언급

텍스트에서 사용할 때 다음 기호 주위에 <code>`backticks`</code>를 놓습니다.

- 인수 이름: <code>`input`</code>, <code>`x`</code>, <code>`tensor`</code>
- 반환된 텐서 이름: <code>`output`</code>, <code>`idx`</code>, <code>`out`</code>
- 데이터 유형: <code>`int32`</code>, <code>`float`</code>, <code>`uint8`</code>
- 텍스트 내 기타 op 이름 참조: <code>`list_diff()`</code>, <code>`shuffle()`</code>
- 클래스 이름: <code>`tf.Tensor`</code>, <code>`Strategy`</code>
- 파일 이름: <code>`image_ops.py`</code>, <code>`/path_to_dir/file_name`</code>
- 산술 표현식 또는 조건: <code>`-1-input.dims() &lt;= dim &lt;=     input.dims()`</code>

#### 코드 블록

3개의 백틱을 사용하여 코드 블록을 열고 닫습니다. 선택적으로, 첫 번째 백틱 그룹 다음에 프로그래밍 언어를 지정합니다. 예를 들면 다음과 같습니다.

<pre><code>
```python
# some python code here
```
</code></pre>

### Markdown 및 노트북의 링크

#### 리포지토리의 파일 간 링크

단일 GitHub 리포지토리의 파일 사이에 상대 링크를 사용합니다. 파일 확장자를 포함합니다.

예를 들어 **읽고 있는 이 파일**은 [https://github.com/tensorflow/docs](https://github.com/tensorflow/docs) 리포지토리에서 가져온 것입니다. 따라서 다음과 같이 상대 경로를 사용하여 동일한 리포지토리의 다른 파일에 연결할 수 있습니다.

- <code>\[Basics\]\(../../guide/basics.ipynb\)</code>은 [기본](../../guide/basics.ipynb)을 생성합니다.

이 접근 방식은 [tensorflow.org](https://www.tensorflow.org), [GitHub](https://github.com/tensorflow/docs){:.external} 및 [Colab](https://github.com/tensorflow/docs/tree/master/site/en/guide/bazics.ipynb){:.external}에 대한 링크가 모두 작동하기 때문에 선호됩니다. 또한 사용자가 링크를 클릭할 경우 동일한 사이트에 머물게 됩니다.

참고: 상대 링크의 경우 `.ipynb` 혹은 `.md`와 같은 파일 확장자를 포함해야 합니다. 확장자가 없을 경우 `tensorflow.org`에서 렌더링됩니다.

#### 외부 링크

현재 리포지토리에 없는 파일에 대한 링크의 경우 전체 URI를 사용하는 표준 Markdown 링크를 사용해야 합니다. 가능한 경우 [tensorflow.org](https://www.tensorflow.org) URI에 연결하는 것이 좋습니다.

소스 코드에 연결하려면 <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>로 시작하고 GitHub 루트에서 시작하는 파일 이름이 붙은 링크를 사용합니다.

[tensorflow.org](https://www.tensorflow.org)에 대한 링크를 해제할 때에는 Markdown 링크에 `{:.external}`를 포함해야 "외부 링크" 기호가 표시됩니다.

- `[GitHub](https://github.com/tensorflow/docs){:.external}`은 [GitHub](https://github.com/tensorflow/docs){:.external}를 생성합니다.

링크에 URI 쿼리 매개변수는 포함하지 않습니다.

- 사용 가능: `https://www.tensorflow.org/guide/data`
- 사용 불가: `https://www.tensorflow.org/guide/data?hl=en`

#### 이미지

이전 섹션의 내용은 페이지 링크에 대한 것입니다. 이미지는 다르게 처리됩니다.

일반적으로 이미지는 체크 인하기보다는 PR에 [TensorFlow-Docs 팀](https://github.com/tensorflow/docs)을 추가하고 이들에게 [tensorflow.org](https://www.tensorflow.org)에서 이미지를 호스팅하도록 요청해야 합니다. 이렇게 하면 리포지토리의 크기를 줄이는 데 도움이 됩니다.

자신의 리포지토리에 이미지를 제출하는 경우 일부 시스템은 이미지에 대한 상대 경로를 처리하지 않습니다. [tensorflow.org](https://www.tensorflow.org)의 이미지의 최종 위치를 가리키는 전체 URL을 사용하는 것이 좋습니다.

#### API 설명서에 대한 링크

사이트를 게시하면 API 링크가 변환됩니다. 기호의 API 참조 페이지에 연결하려면 기호 경로를 백틱으로 감쌉니다.

- <code>`tf.data.Dataset`</code>은 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)를 생성합니다.

긴 경로를 제외하고는 전체 경로가 약간 더 선호됩니다. 선행 경로 구성 요소를 삭제하여 경로를 축약할 수 있습니다. 다음과 같은 경우 부분 경로가 링크로 변환됩니다.

- 경로에 `.`이 하나 이상 있음
- 부분 경로가 프로젝트 내에서 고유함

API 경로는 [tensorflow.org](https://www.tensorflow.org)에 게시된 Python API를 사용하는 **모든 프로젝트**와 연결되어 있습니다. API 이름을 백틱으로 래핑하여 단일 파일에서 여러 하위 프로젝트로 쉽게 연결할 수 있습니다. 예시:

- <code>`tf.metrics`</code>, <code>`tf_agents.metrics`</code>, <code>`text.metrics`</code>는 `tf.metrics`, `tf_agents.metrics`, `text.metrics`를 생성합니다.

여러 경로 별칭을 사용하는 기호의 경우 [tensorflow.org](https://www.tensorflow.org)의 API 페이지로 연결하는 경로에 대한 약간의 기본 설정이 있습니다. 모든 별칭은 올바른 페이지로 리디렉션됩니다.

### Markdown의 수학

Markdown 파일을 편집할 때 TensorFlow 내에서 MathJax를 사용할 수 있지만 다음 사항에 유의하세요.

- MathJax는 [tensorflow.org](https://www.tensorflow.org)에서 올바르게 렌더링됩니다.
- MathJax는 GitHub에서 올바르게 렌더링되지 않습니다.
- 이 표기법은 친숙하지 않은 개발자에게 적합하지 않습니다.
- 일관성을 위해 [tensorflow.org](https://www.tensorflow.org)는 Jupyter/Colab과 같은 규칙을 따릅니다.

MathJax 블록 주위에 <code>$$</code>를 사용합니다.

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

인라인 MathJax 표현식을 <code>$ ... $</code>로 래핑합니다.

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $
</code></pre>

다음은 인라인 MathJax 표현식의 예입니다. $ 2 \times 2 = 4 $

<code>\( ... \)</code> 구분 기호도 인라인 수학에 사용할 수 있지만 $ 형식이 더 읽기 쉽습니다.

참고: 텍스트 또는 MathJax 표현식에 달러 기호를 사용해야 하는 경우 앞에 슬래시를 넣어 이스케이프 처리합니다(`\$`). 코드 블록 내에서 달러 기호(예: Bash 변수 이름)는 이스케이프 처리할 필요가 없습니다.

## 문장 스타일

서술적 설명서의 상당 부분을 쓰거나 편집하려는 경우 [Google 개발자 설명서 스타일 가이드](https://developers.google.com/style/highlights)를 참조하세요.

### 좋은 스타일의 원리

- *기여하는 문서에서 철자와 문법을 검사합니다.* 대부분의 편집기는 맞춤법 검사기를 포함하거나 맞춤법 검사 플러그인을 사용할 수 있습니다. 보다 강력한 철자와 문법 검사를 위해 텍스트를 Google 문서나 다른 문서 소프트웨어에 붙여 넣을 수도 있습니다.
- *격의 없고 친근한 목소리를 사용합니다.* 다른 사람과 일대일로 대화하는 것처럼 TensorFlow 설명서를 대화체로 작성하세요. 힘을 주는 어투를 사용하세요.

참고: 격식을 차리지 않는다는 것이 전문성이 떨어진다는 의미는 아닙니다. 전문성이 아니라 문장을 단순화하세요.

- *고지 사항, 의견 및 가치 판단을 피합니다.* "쉽게", "단순히" 및 "간단한"과 같은 단어에는 가정의 의미가 담겨 있습니다. 자신에게는 쉬워 보이는 것이 다른 사람에게는 어려울 수 있습니다. 가능한 한 이러한 표현을 피하세요.
- *복잡한 전문 용어 없이 간단 명료한 문장을 사용합니다.* 복합 문장, 줄줄이 연결된 구절 및 특정 장소에 국한된 관용구는 텍스트를 이해하고 번역하기 어렵게 만들 수 있습니다. 문장을 두 개로 나눌 수 있다면 대체로 그렇게 하는 것이 좋습니다. 세미콜론을 피하세요. 해당하는 경우 글 머리표 목록을 사용하세요.
- *컨텍스트를 제공합니다.* 설명 없이 약어를 사용하지 마세요. 링크 없이 TensorFlow가 아닌 프로젝트를 언급하지 마세요. 코드가 왜 그렇게 작성되었는지 설명하세요.

## 사용 가이드

### Op

마크다운 파일에서 op가 반환하는 내용을 표시하려면 단일 등호 대신 `# ⇒`를 사용합니다.

```python
# 'input' is a tensor of shape [2, 3, 5]
tf.expand_dims(input, 0)  # ⇒ [1, 2, 3, 5]
```

노트북에서 주석을 추가하지 않고 결과를 표시합니다. (노트북 셀의 마지막 표현식이 변수에 할당되지 않은 경우 자동으로 표시됩니다.)

API 참조 문서에서는 [doctest](docs_ref.md#doctest)를 사용하여 결과를 표시하는 것이 좋습니다.

### 텐서

일반적으로 텐서에 대해 이야기할 때 *tensor*라는 단어에 대문자를 사용하지 않습니다. Op에 제공되거나 op에서 반환되는 특정 객체에 대해 이야기할 때는 단어 *Tensor*를 대문자로 처리하고 `Tensor` 객체에 대해 이야기하고 있으므로 그 주위에 백틱을 추가해야 합니다.

정말로 `Tensors` 객체에 대해 말하는 경우가 아니면 여러 `Tensor` 객체를 설명하기 위해 단어 *Tensors*(복수)를 사용하지 마세요. 대신 "`Tensor` 객체 목록(또는 모음)"이라는 표현을 사용하세요.

단어 *shape*을 사용하여 텐서의 축을 자세히 표시하고 형상을 백틱이 있는 대괄호 안에 표시합니다. 예를 들면 다음과 같습니다.

<pre><code>
If `input` is a three-axis `Tensor` with shape `[3, 4, 3]`, this operation
returns a three-axis `Tensor` with shape `[6, 8, 6]`.
</code></pre>

위와 같이 `Tensor`의 형상 요소에 대해 언급할 때 "차원"보다 "축" 또는 "색인"을 사용하는 것이 좋습니다. 그렇지 않으면 "차원"을 벡터 공간의 차원과 혼동하기 쉽습니다. "3차원 벡터"에는 길이가 3인 단일 축이 있습니다.
