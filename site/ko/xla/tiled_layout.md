# 바둑판식 레이아웃

Caution: Tiled layout is *pre-release* and this describes how it's intended to work. Errors may be silently ignored.

<p align="center"><img src="images/xla_array_layout_figure1.png">그림 1</p>

Figure 1 shows how an array F32[3,5] is laid out in memory with 2x2 tiling. A shape with this layout is written as F32[3,5]{1,0:T(2,2)}, where 1,0 relates to the physical order of dimensions (minor_to_major field in Layout) while (2,2) after the colon indicates tiling of the physical dimensions by a 2x2 tile.

직관적으로, 타일은 형상을 덮도록 배치되고 각 타일 내에서는 요소가 바둑판식 배열 없이 배치됩니다. 이것을 나타낸 위의 예에서 오른쪽 부분은 메모리 내의 레이아웃을 나타내며, 여기에는 원래 배열 경계가 균등하지 않더라도 완전한 2x2 타일이 유지되도록 추가된 화이트 채우기 요소가 포함됩니다.

채우기의 추가 요소는 특정 값을 포함할 필요가 없습니다.

## 주어진 형상과 타일의 바둑판식 배열을 위한 선형 인덱스 공식

바둑판식 배열이 없으면, 배열 경계가 d=(d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)인 배열의 요소 e=(e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>)(d1이 가장 작은 차원)는 다음 위치에서 메이저-마이너 순서로 배치됩니다.

   linear_index(e, d) <br> = linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>), (d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) <br> = e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> + e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

For simplicity of notation in this document we assume a tile has the same number of dimensions as the array. In XLA's implementation of tiling, this is generalized to tilings with fewer dimensions by leaving the initial most-major dimensions unchanged and applying the tiling only to the most minor dimensions, so that the tiling that is specified mentions a suffix of the physical dimensions of the shape being tiled.

When tiling of size (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>) is used, an element in the array with indices (e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>) is mapped to this position in the final layout:

   linear_index_with_tile(e, d, t) <br> = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (산술 연산은 요소별이며 (a,b)는 연결임) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> + linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

The layout can be thought of as having two parts: (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), which corresponds to a tile index in an array of tiles of size (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉), and (e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), which corresponds to a within-tile index. The ceil function appears in ⌈d<sub>i</sub>/t<sub>i</sub>⌉ because if tiles overrun the bounds of the larger array, padding is inserted as in Figure 1. Both the tiles and elements within tiles are laid out recursively without tiling.

그림 1의 예에서 요소 (2,3)에는 타일 인덱스 (1,1)과 타일 내 인덱스 (0,1)이 있어서 (1, 1, 0, 1)의 결합된 좌표 벡터를 구성합니다. 타일 인덱스에는 경계 (2, 3)이 있고 타일 자체는 (2, 2)여서 (2, 3, 2, 2)의 결합된 벡터를 구성합니다. 그러면, 논리적 형상의 인덱스 (2, 3)을 갖는 요소에 대한 타일에서 선형 인덱스는 다음과 같습니다.

   linear_index_with_tile((2,3), (3,5), (2,2)) <br> = linear_index((1,1,0,1), (2,3,2,2)) <br> = linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) <br> = (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) <br> = 17.

# 채우기-형상 변경-바꾸기로 바둑판식 배열하기

바둑판식 배열 기반 레이아웃은 다음과 같이 동작합니다.<br> 차원 (d<sub>n</sub>, d<sub>n-1</sub>, ... , d1)(d1이 가장 작은 차원)의 배열을 생각해 보겠습니다. 크기 (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)(t<sub>1</sub>이 가장 작은 차원)의 바둑판식 배열로 배치할 때, 이 바둑판식 배열은 다음과 같이 채우기-형상 변경-바꾸기 항으로 설명할 수 있습니다.

1. 배열은 (⌈d<sub>n</sub>/t<sub>n</sub>⌉∙t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉∙t<sub>1</sub>)로 채워집니다.
2. 각 차원 i는 (⌈d<sub>i</sub>/ti⌉, t<sub>i</sub>)로 나뉩니다. 즉, 배열은 <br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>1</sub>)로 형상이 변경됩니다.<br> 이 형상 변경 자체에는 물리적 레이아웃 변경이 없으므로 이 형상 변경은 비트캐스트(bitcast)입니다. 바둑판식 배열을 명시적으로 생각하지 않는 경우, 이 형상 변경은 채워진 형상과 같은 수의 요소를 가진 모든 형상을 표현할 수 있습니다. 여기서의 예는 타일을 이러한 방식으로 표현하는 방법과 관련됩니다.
3. 바꾸기는 t<sub>n</sub>, ... , t<sub>1</sub>을 상대적인 순서를 유지하면서 가장 작은 차원으로 이동하는 식으로 이루어지므로, 가장 큰 차원에서 가장 작은 차원의 순서는 다음과 같습니다.<br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, ... , t<sub>1</sub>)

The final shape has the prefix
     (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉), which describes the number of tiles in each dimension. An element in the array (e<sub>n</sub>, ... , e<sub>1</sub>) is mapped to this element in the final shape:
     (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>0</sub>/t<sub>0</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>). It is easy to see that the linear index of the element follows the formula above as expected.

# 반복 바둑판식 배열

XLA의 바둑판식 배열은 반복적으로 적용함으로써 더욱 유연해집니다.

<p align="center"><img src="images/xla_array_layout_figure2.png">그림 2</p>

Figure 2 shows how an array of size 4x8 is tiled by two levels of tiling (first 2x4 then 2x1). We represent this repeated tiling as (2,4)(2,1). Each color indicates a 2x4 tile and each red border box is a 2x1 tile. The numbers indicates the linear index in memory of that element in the tiled format. This format matches the format used for BF16 on TPU, except that the initial tile is bigger, namely the tiling is (8,128)(2,1), where the purpose of the second tiling by 2x1 is to collect together two 16 bit values to form one 32 bit value in a way that aligns with the architecture of a TPU.

Note that a second or later tile can refer to both the minor within-tile dimensions, which just rearranges data within the tile, as in this example with (8,128)(2,1), but can also refer to the major cross-tile dimensions from the prior tiling.

# 타일을 사용하여 차원 결합하기

XLA의 바둑판식 배열은 차원의 결합도 지원합니다. 예를 들어, 우선 F32[2,7,8,11,10]{4,3,2,1,0}의 차원을 F32[112,110]{1,0}으로 결합한 다음 (2,3) 바둑판식으로 배열할 수 있습니다. 사용된 타일은 (∗,∗,2,∗,3)입니다. 여기서 타일의 별표는 해당 차원을 가져와 다음으로 더 작은 차원과 결합한다는 것을 의미합니다. 인접한 여러 차원이 하나의 차원으로 포함될 수 있습니다. 포함된 차원은 타일의 해당 차원에서 타일 값 -1로 표시됩니다. 이 경우가 아니라면 이 값은 타일에서 차원 크기로 유효하지 않습니다.

More precisely, if dimension i of the shape is eliminated via an asterisk in the tile, then before the prior definition of tiling is applied, that dimension is removed from both the shape being tiled and the tile vector, and what was dimension i-1 of the shape has its array bound increased from d<sub>i-1</sub> to d<sub>i</sub>d<sub>i-1</sub>. This step is repeated for each asterisk in the tile vector.
