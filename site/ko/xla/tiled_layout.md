# 바둑판식 레이아웃

주의: 바둑판식 레이아웃은 *시험판*이며, 여기서 동작 방식을 설명합니다. 오류가 있더라도 문제 삼지 않아도 됩니다.

<p align="center"><img src="images/xla_array_layout_figure1.png">그림 1</p>

Figure 1 shows how an array F32[3,5] is laid out in memory with 2x2 tiling. A shape with this layout is written as F32[3,5]{1,0:T(2,2)}, where 1,0 relates to the physical order of dimensions (minor_to_major field in Layout) while (2,2) after the colon indicates tiling of the physical dimensions by a 2x2 tile.

직관적으로, 타일은 형상을 덮도록 배치되고 각 타일 내에서는 요소가 바둑판식 배열 없이 배치됩니다. 이것을 나타낸 위의 예에서 오른쪽 부분은 메모리 내의 레이아웃을 나타내며, 여기에는 원래 배열 경계가 균등하지 않더라도 완전한 2x2 타일이 유지되도록 추가된 화이트 채우기 요소가 포함됩니다.

채우기의 추가 요소는 특정 값을 포함할 필요가 없습니다.

## 주어진 형상과 타일의 바둑판식 배열을 위한 선형 인덱스 공식

바둑판식 배열이 없으면, 배열 경계가 d=(d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)인 배열의 요소 e=(e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>)(d1이 가장 작은 차원)는 다음 위치에서 메이저-마이너 순서로 배치됩니다.

   linear_index(e, d) <br> = linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>), (d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) <br> = e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> + e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

이 문서에서는 표기를 단순화하기 위해 타일이 배열과 같은 수의 차원을 갖는다고 가정합니다. XLA의 바둑판식 배열 구현에서는 초기의 가장 큰 차원을 변경하지 않고 가장 작은 차원에만 바둑판식 배열을 적용하여 차원의 수가 더 적은 바둑판식 배열로 일반화되므로, 지정된 바둑판식 배열은 바둑판식으로 배열되는 형상의 물리적 차원에 추가되는 부분을 언급합니다.

크기 (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)의 바둑판식 배열을 사용하면, 인덱스 (e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>)를 가진 배열의 요소가 최종 레이아웃의 다음 위치로 매핑됩니다.

   linear_index_with_tile(e, d, t) <br> = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (산술 연산은 요소별이며 (a,b)는 연결임) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> + linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

레이아웃은 두 부분이 있는 것으로 생각할 수 있는데, 각각 (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉) 크기의 타일 배열에 있는 타일 인덱스에 해당하는 (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋) 및 타일 내 인덱스에 해당하는 (e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>)입니다. 타일이 더 큰 배열의 경계를 넘으면 그림 1과 같이 채우기가 삽입되기 때문에 ceil 함수가 ⌈d<sub>i</sub>/t<sub>i</sub>⌉에 나타납니다. 타일과 타일 내의 요소 모두 바둑판식 배열 없이 재귀적으로 배치됩니다.

그림 1의 예에서 요소 (2,3)에는 타일 인덱스 (1,1)과 타일 내 인덱스 (0,1)이 있어서 (1, 1, 0, 1)의 결합된 좌표 벡터를 구성합니다. 타일 인덱스에는 경계 (2, 3)이 있고 타일 자체는 (2, 2)여서 (2, 3, 2, 2)의 결합된 벡터를 구성합니다. 그러면, 논리적 형상의 인덱스 (2, 3)을 갖는 요소에 대한 타일에서 선형 인덱스는 다음과 같습니다.

   linear_index_with_tile((2,3), (3,5), (2,2)) <br> = linear_index((1,1,0,1), (2,3,2,2)) <br> = linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) <br> = (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) <br> = 17.

# 채우기-형상 변경-바꾸기로 바둑판식 배열하기

바둑판식 배열 기반 레이아웃은 다음과 같이 동작합니다.<br> 차원 (d<sub>n</sub>, d<sub>n-1</sub>, ... , d1)(d1이 가장 작은 차원)의 배열을 생각해 보겠습니다. 크기 (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)(t<sub>1</sub>이 가장 작은 차원)의 바둑판식 배열로 배치할 때, 이 바둑판식 배열은 다음과 같이 채우기-형상 변경-바꾸기 항으로 설명할 수 있습니다.

1. 배열은 (⌈d<sub>n</sub>/t<sub>n</sub>⌉∙t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉∙t<sub>1</sub>)로 채워집니다.
2. 각 차원 i는 (⌈d<sub>i</sub>/ti⌉, t<sub>i</sub>)로 나뉩니다. 즉, 배열은 <br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>1</sub>)로 형상이 변경됩니다.<br> 이 형상 변경 자체에는 물리적 레이아웃 변경이 없으므로 이 형상 변경은 비트캐스트(bitcast)입니다. 바둑판식 배열을 명시적으로 생각하지 않는 경우, 이 형상 변경은 채워진 형상과 같은 수의 요소를 가진 모든 형상을 표현할 수 있습니다. 여기서의 예는 타일을 이러한 방식으로 표현하는 방법과 관련됩니다.
3. 바꾸기는 t<sub>n</sub>, ... , t<sub>1</sub>을 상대적인 순서를 유지하면서 가장 작은 차원으로 이동하는 식으로 이루어지므로, 가장 큰 차원에서 가장 작은 차원의 순서는 다음과 같습니다.<br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, ... , t<sub>1</sub>)

최종 형상에는 각 차원에 있는 타일의 수를 설명하는 다음 부분이 앞에 붙습니다.<br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉<br> ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉). 배열 (e<sub>n</sub>, ... , e<sub>1</sub>)의 요소는 최종 형상에서 다음 요소에 매핑됩니다.<br> (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>0</sub>/t<sub>0</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>). 요소의 선형 인덱스가 예상대로 위의 공식을 따르는 것을 쉽게 알 수 있습니다.

# 반복 바둑판식 배열

XLA의 바둑판식 배열은 반복적으로 적용함으로써 더욱 유연해집니다.

<p align="center"><img src="images/xla_array_layout_figure2.png">그림 2</p>

그림 2는 크기가 4x8인 배열이 두 수준의 바둑판식 배열(처음은 2x4, 그 다음 2x1)에 의해 바둑판식으로 배열되는 방식을 보여줍니다. 이 반복 바둑판식 배열을 (2,4)(2,1)로 나타냅니다. 각 색상은 2x4 타일을 나타내고, 각 빨간색 테두리 상자는 2x1 타일입니다. 숫자는 타일 형식으로 해당 요소의 메모리에 있는 선형 인덱스를 나타냅니다. 이 형식은 첫 타일이 더 크다는 점을 제외하고 TPU의 BF16에 사용되는 형식과 일치합니다. 즉, 바둑판식 배열은 (8,128)(2,1)이고, 여기서 2x1에 의한 두 번째 바둑판식 배열의 목적은 두 개의 16bit 값을 수집하여 TPU의 아키텍처와 일치하는 방식으로 하나의 32bit 값을 만드는 것입니다.

두 번째 또는 그 이후의 타일은 (8,128)(2,1)을 사용하는 이 예에서와 같이 타일 내에서 단순히 데이터를 재배열하는 두 개의 마이너 타일 내 차원을 참조할 수 있지만, 이전 바둑판식 배열의 메이저 교차 타일 차원을 참조할 수도 있습니다.

# 타일을 사용하여 차원 결합하기

XLA의 바둑판식 배열은 차원의 결합도 지원합니다. 예를 들어, 우선 F32[2,7,8,11,10]{4,3,2,1,0}의 차원을 F32[112,110]{1,0}으로 결합한 다음 (2,3) 바둑판식으로 배열할 수 있습니다. 사용된 타일은 (∗,∗,2,∗,3)입니다. 여기서 타일의 별표는 해당 차원을 가져와 다음으로 더 작은 차원과 결합한다는 것을 의미합니다. 인접한 여러 차원이 하나의 차원으로 포함될 수 있습니다. 포함된 차원은 타일의 해당 차원에서 타일 값 -1로 표시됩니다. 이 경우가 아니라면 이 값은 타일에서 차원 크기로 유효하지 않습니다.

보다 정확하게 말하면, 타일의 별표를 통해 형상의 i 차원이 제거되면, 바둑판식 배열의 사전 정의가 적용되기 전에 바둑판식으로 배열되는 형상과 타일 벡터 모두에서 해당 차원이 제거되고, 형상의 i-1 차원이었던 부분의 배열 경계가 d<sub>i-1</sub>에서 d<sub>i</sub>d<sub>i-1</sub>로 증가합니다. 이 단계는 타일 벡터의 각 별표에 대해 반복됩니다.
