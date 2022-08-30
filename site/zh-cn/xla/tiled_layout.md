# 平铺布局

小心：平铺布局目前处于*预发布*阶段，本文描述了它的预期工作方式。错误可能会被静默忽略。

<p align="center">   <img src="images/xla_array_layout_figure1.png">   Figure 1</p>

图 1 显示了如何使用 2x2 平铺在内存中布置数组 F32[3,5]。具有此布局的形状写为 F32[3,5]{1,0:(2,2)}，其中 1,0 与维度的物理顺序（布局中的 minor_to_major 字段）相关，而冒号后面的 (2,2) 表示通过 2x2 图块来平铺物理维度。

以直观方式布置图块来覆盖形状，随后在每个图块中以不平铺的方式布置各元素，如上面的示例中所示。此示例的右侧部分显示了内存中的布局，包括为获得完整的 2x2 图块而添加的白色填充元素（即使原始数组边界不平坦也是如此）。

填充中的额外元素不需要包含任何特定值。

## 给定形状和图块时平铺的线性索引公式

如果不平铺，则数组边界为 d=(d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)（d1 是最小维度）的数组中的元素 e=(e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>) 按位置以从大到小的顺序布置如下：

   linear_index(e, d) <br> = linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>), (d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) <br> = e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> + e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

为了简化本文档中的表示法，我们假设图块的维数与数组的相同。在 XLA 的平铺实现中，通过保持初始的最大维度不变，并将平铺仅应用于最小维度，将其泛化为维度较少的平铺，以便指定的平铺提及被平铺形状的物理维度的后缀。

使用大小为 (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>) 的平铺时，具有索引 (e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>) 的数组中的元素将被映射至最终布局中的以下位置：

   linear_index_with_tile(e, d, t) <br> = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (arithmetic is elementwise, (a,b) is concatenation) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> + linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

可将布局视为包含两个部分：(⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋)，它对应于大小为 (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉) 的图块数组中的图块索引；(e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>)，它对应于图块内索引。ceil 函数出现在 ⌈d<sub>i</sub>/t<sub>i</sub>⌉ 中，因为如果图块超出了较大数组的边界，则会按图 1 所示插入填充。将以递归方式布置图块和图块内元素，而不平铺。

在图 1 的示例中，对于组合坐标向量 (1, 1, 0, 1)，元素 (2,3) 的图块索引为 (1,1)，图块内索引为 (0,1)。对于组合向量 (2, 3, 2, 2)，图块索引的边界为 (2, 3)，图块本身为 (2, 2)。对于逻辑形状中索引为 (2, 3) 的元素，其图块线性索引为：

   linear_index_with_tile((2,3), (3,5), (2,2)) <br> = linear_index((1,1,0,1), (2,3,2,2)) <br> = linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) <br> = (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) <br> = 17。

# 以填充-变换-转置方式平铺

基于平铺的布局按如下方式运行：<br>考虑一个维度数组 (d<sub>n</sub>, d<sub>n-1</sub>, ... , d1)（d1 是最小维度）。使用大小为 (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)（t1 是最小维度）的平铺布置此数组时，可以按照以下方式以填充-变换-转置的形式描述该平铺。

1. 数组填充为 (⌈d<sub>n</sub>/t<sub>n</sub>⌉∙t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉∙t<sub>1</sub>)。
2. 将每个维度分解为 (⌈d<sub>i</sub>/ti⌉, t<sub>i</sub>)，即，将数组变换为 <br>     (⌈d<sub>n</sub>/t<sub>n</sub>⌉, t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>1</sub>)。<br>这种变换本身没有更改物理布局，因此它是一种 bitcast 运算。如果未明确考虑平铺，则此变换可以表示元素数量与填充形状相同的任何形状。此处的示例便介绍了如何以这种方式表示图块。
3. 通过将 t<sub>n</sub>, ... , t<sub>1</sub> 移至最小维度，同时保持它们的相对顺序来进行转置，这样，维度从最大到最小的顺序就会变为 <br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, ... , t<sub>1</sub>)。

最终形状的前缀为<br>     (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉)，它描述了每个维度中的图块数量。数组 (e<sub>n</sub>, ... , e<sub>1</sub>) 中的元素被映射至最终形状中的以下元素：<br>     (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>0</sub>/t<sub>0</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>)。不难看出，正如预期的那样，元素的线性索引符合上述公式。

# 重复平铺

通过重复应用，XLA 的平铺会变得更加灵活。

<p align="center">   <img src="images/xla_array_layout_figure2.png">   Figure 2</p>

图 2 显示了如何通过两级平铺（首先是 2x4，随后是 2x1）来平铺大小为 4x8 的数组。我们将此重复平铺表示为 (2,4)(2,1)。每种颜色表示一个 2x4 图块，每个红色边框表示一个 2x1 图块。数字以平铺格式表示该元素在内存中的线性索引。除了初始图块较大（即平铺为 (8,128)(2,1)）外，此格式与 TPU 上 BF16 所使用的格式匹配，其中第二次平铺采用 2x1 方式的目的是，按照符合 TPU 架构的方式将两个 16 位值收集到一起来构成一个 32 位值。

请注意，第二个或后面的图块不仅可以引用较小的图块内维度（只会重新排列图块内的数据，如本示例中的 (8,128)(2,1)），也可以引用前一个平铺中较大的交叉图块维度。

# 使用图块组合维度

XLA 的平铺也支持组合维度。例如，它可以先将 F32[2,7,8,11,10]{4,3,2,1,0} 中的维度组合成 F32[112,110]{1,0}，然后再使用 (2,3) 来平铺它。使用的图块为 (∗,∗,2,∗,3)。在这里，图块中的星号表示采用该维度并将其与下一个更小的维度组合。多个相邻维度可以合并到一个维度中。合并后的维度由该图块所对应维度中的图块值 -1 表示，否则它在图块中无法有效作为维度大小。

更确切地说，如果通过图块中的星号消除了形状的维度 i，则在应用平铺的先前定义之前，会将该维度从要平铺的形状和图块向量中移除，而形状的维度 i-1 的数组边界则从 d<sub>i-1</sub> 增加到 d<sub>i</sub>d<sub>i-1</sub>。对图块向量中的每个星号重复此步骤。
