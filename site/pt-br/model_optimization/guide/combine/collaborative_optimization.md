# Otimização colaborativa

<sub>Manutenção por Ferramentas de ML para Arm</sub>

Este documento oferece uma visão geral das APIs experimentais para combinar várias técnicas e otimizar os modelos de aprendizado de máquina para implantação.

## Visão geral

A otimização colaborativa é um processo abrangente que envolve várias técnicas para produzir um modelo que, na implantação, exibe o melhor equilíbrio de características alvo, como velocidade de inferência, tamanho do modelo e exatidão.

A ideia das otimizações colaborativas é aproveitar técnicas individuais com a aplicação de uma após a outra, alcançando um efeito de otimização cumulativo. As seguintes otimizações podem ser combinadas de várias formas:

- [Pruning de peso](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)

- [Clustering de peso](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

- Quantização

    - [Quantização pós-treinamento](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
    - [Treinamento consciente de quantização](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html) (QAT)

O problema que ocorre ao tentar usar essas técnicas em cadeia é que a aplicação de uma geralmente destrói os resultados da técnica anterior, acabando com o benefício geral da aplicação simultânea de todas elas. Por exemplo, o clustering não preserva a esparsidade introduzida pela API de pruning. Para resolver esse problema, apresentamos as seguintes técnicas experimentais de otimização colaborativa:

- [Clustering que preserva a esparsidade](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)
- [Treinamento consciente de quantização que preserva a esparsidade](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example) (PQAT)
- [Treinamento consciente de quantização que preserva os clusters](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example) (CQAT)
- [Treinamento consciente de quantização que preserva a esparsidade e os clusters](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example)

Essas técnicas oferecem vários caminhos de implantação que podem ser usados para comprimir um modelo de aprendizado de máquina e aproveitar a aceleração de hardware no momento de inferência. O diagrama abaixo demonstra vários caminhos de implantação que podem ser explorados em busca do modelo com as características de implantação desejadas, onde os nós das folhas são modelos prontos para implantação, ou seja, eles foram parcial ou totalmente quantizados e estão no formato tflite. O preenchimento verde indica onde é necessário treinar novamente/ajustar e a borda vermelha pontilhada destaca as etapas da otimização colaborativa. A técnica usada para obter um modelo em um determinado nó é indicada no rótulo correspondente.

![otimização colaborativa](images/collaborative_optimization.png "collaborative optimization")

O caminho de implantação direto com somente quantização (pós-treinamento ou QAT) é omitido na figura acima.

A ideia é alcançar o modelo completamente otimizado no terceiro nível da árvore de implantação acima. No entanto, qualquer um dos outros níveis de otimização podem se mostrar satisfatórios e alcançar o trade-off necessário de latência/exatidão da inferência, não precisando de otimização adicional. O processo de treinamento recomendado seria passar de maneira iterativa pelos níveis da árvore de implantação aplicáveis ao cenário de implantação alvo e ver se o modelo cumpre com os requisitos de latência da inferência. Caso contrário, use a técnica de otimização colaborativa correspondente para comprimir ainda mais o modelo e repita até que ele esteja totalmente otimizado (com pruning, clustering e quantização), se necessário.

A figura abaixo mostra as plotagens de densidade da amostra de kernel de peso passando pelo pipeline de otimização colaborativa.

![plotagem de densidade da otimização colaborativa](images/collaborative_optimization_dist.png "collaborative optimization density plot")

O resultado é um modelo de implantação quantizado com um número reduzido de valores únicos, além de um número significativo de pesos esparsos, dependendo da esparsidade alvo especificada durante o treinamento. Além das vantagens significativas de compressão do modelo, o suporte de hardware específico pode se beneficiar desses modelos esparsos e agrupados, reduzindo consideravelmente a latência da inferência.

## Resultados

Confira abaixo alguns resultados de exatidão e compressão obtidos ao experimentar com os caminhos de otimização colaborativa PQAT e CQAT.

### Treinamento consciente de quantização que preserva a esparsidade (PQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Pruned Model (50% sparsity)</th><th>QAT Model</th><th>PQAT Model</th></tr>
 <tr><td>DS-CNN-L</td><td>FP32 Top1 Accuracy</td><td><b>95.23%</b></td><td>94.80%</td><td>(Fake INT8) 94.721%</td><td>(Fake INT8) 94.128%</td></tr>
 <tr><td>{nbsp}</td><td>INT8 full integer quantization</td><td>94.48%</td><td><b>93.80%</b></td><td>94.72%</td><td><b>94.13%</b></td></tr>
 <tr><td>{nbsp}</td><td>Compression</td><td>528,128 → 434,879 (17.66%)</td><td>528,128 → 334,154 (36.73%)</td><td>512,224 → 403,261 (21.27%)</td><td>512,032 → 303,997 (40.63%)</td></tr>
 <tr><td>Mobilenet_v1-224</td><td>FP32 Top 1 Accuracy</td><td><b>70.99%</b></td><td>70.11%</td><td>(Fake INT8) 70.67%</td><td>(Fake INT8) 70.29%</td></tr>
 <tr><td>{nbsp}</td><td>INT8 full integer quantization</td><td>69.37%</td><td><b>67.82%</b></td><td>70.67%</td><td><b>70.29%</b></td></tr>
 <tr><td>{nbsp}</td><td>Compression</td><td>4,665,520 → 3,880,331 (16.83%)</td><td>4,665,520 → 2,939,734 (37.00%)</td><td>4,569,416 → 3,808,781 (16.65%)</td><td>4,569,416 → 2,869,600 (37.20%)</td></tr>
</table>
</figure>

### Treinamento consciente de quantização que preserva os clusters (CQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Clustered Model</th><th>QAT Model</th><th>CQAT Model</th></tr>
 <tr><td>Mobilenet_v1 on CIFAR-10</td><td>FP32 Top1 Accuracy</td><td><b>94.88%</b></td><td>94.48%</td><td>(Fake INT8) 94.80%</td><td>(Fake INT8) 94.60%</td></tr>
 <tr><td>{nbsp}</td><td>INT8 full integer quantization</td><td>94.65%</td><td><b>94.41%</b></td><td>94.77%</td><td><b>94.52%</b></td></tr>
 <tr><td>{nbsp}</td><td>Size</td><td>3.00 MB</td><td>2.00 MB</td><td>2.84 MB</td><td>1.94 MB</td></tr>
 <tr><td>Mobilenet_v1 on ImageNet</td><td>FP32 Top 1 Accuracy</td><td><b>71.07%</b></td><td>65.30%</td><td>(Fake INT8) 70.39%</td><td>(Fake INT8) 65.35%</td></tr>
 <tr><td>{nbsp}</td><td>INT8 full integer quantization</td><td>69.34%</td><td><b>60.60%</b></td><td>70.35%</td><td><b>65.42%</b></td></tr>
 <tr><td>{nbsp}</td><td>Compression</td><td>4,665,568 → 3,886,277 (16.7%)</td><td>4,665,568 → 3,035,752 (34.9%)</td><td>4,569,416 → 3,804,871 (16.7%)</td><td>4,569,472 → 2,912,655 (36.25%)</td></tr>
</table>
</figure>

### Resultados de CQAT e PCQAT para modelos agrupados por canal

Os resultados abaixo são obtidos com a técnica de [clustering por canal](https://www.tensorflow.org/model_optimization/guide/clustering). Eles mostram que, se as camadas convolucionais do modelo são agrupadas por canal, a exatidão do modelo é mais alta. Se o seu modelo tiver muitas camadas convolucionais, recomendamos o clustering por canal. A proporção de compressão permanece a mesma, mas a exatidão do modelo será mais alta. O pipeline de otimização do modelo é "clustering -&gt; QAT que preserva os clusters -&gt; quantização pós-treinamento, int8" em nossos experimentos.

<figure>
<table  class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Clustered -> CQAT, int8 quantized</th><th>Clustered per channel -> CQAT, int8 quantized</th>
 <tr><td>DS-CNN-L</td><td>95.949%</td><td> 96.44%</td></tr>
 <tr><td>MobileNet-V2</td><td>71.538%</td><td>72.638%</td></tr>
 <tr><td>MobileNet-V2 (pruned)</td><td>71.45%</td><td>71.901%</td></tr>
</table>
</figure>

## Exemplos

Para exemplos completos das técnicas de otimização colaborativa descritas aqui, consulte os notebooks de exemplo de [CQAT](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example), [PQAT](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example), [clustering que preserva a esparsidade](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example) e [PCQAT](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example).
