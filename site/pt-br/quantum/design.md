# Design do TensorFlow Quantum

O TensorFlow Quantum (TFQ) foi projetado para os problemas do aprendizado de máquina quântico da era NISQ (Quântica de Escala Intermediária Ruidosa). Ele traz primitivos computacionais quânticos, como o desenvolvimento de circuitos quânticos, para o ecossistema do TensorFlow. Os modelos e as operações criados com o TensorFlow usam esses primitivos para criar sistemas híbridos clássicos-quânticos avançados.

Usando o TFQ, os pesquisadores podem construir um grafo do TensorFlow usando um dataset quântico, um modelo quântico e parâmetros de controle clássicos. Todos eles são representados como tensores em um único grafo computacional. O resultado das medições quânticas — que levam a eventos probabilísticos clássicos — é obtido pelas ops do TensorFlow. O treinamento é realizado com a API [Keras](https://www.tensorflow.org/guide/keras/overview) padrão. O módulo `tfq.datasets` permite que os pesquisadores testem novos datasets quânticos interessantes.

## Cirq

<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> é um framework de programação quântica do Google. Ele fornece todas as operações básicas, como qubits, portas, circuitos e medição, para criar, modificar e invocar circuitos quânticos em um computador quântico ou simulado. O TensorFlow Quantum usa esses primitivos do Cirq para ampliar a computação de lotes, a criação de modelos e a computação de gradientes do TensorFlow. Para ser eficaz com o TensorFlow Quantum, é recomendável ser eficaz com o Cirq.

## Primitivos quânticos do TensorFlow

O TensorFlow Quantum implementa os componentes necessários para integrar o TensorFlow a hardware computacional quântico. Para isso, o TFQ apresenta dois tipos de dados primitivos:

- *Circuito quântico*: representa os circuitos quânticos definidos pelo <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> (`cirq.Circuit`) no TensorFlow. Crie lotes de circuitos de diversos tamanhos, semelhantes aos lotes de diferentes pontos de dados reais.
- *Soma de Pauli*: representa combinações lineares de produtos tensoriais de operadores de Pauli definidos no Cirq (`cirq.PauliSum`). Como os circuitos, crie lotes de operadores de diversos tamanhos.

### Ops fundamentais

Usando os primitivos de circuitos quânticos em um `tf.Tensor`, o TensorFlow Quantum implementa ops que processam esses circuitos e produzem saídas significativas.

As ops do TensorFlow são escritas em C++ otimizado. Essas ops obtêm amostras de circuitos, calculam valores esperados e geram o estado produzido pelos determinados circuitos. A escrita de ops flexíveis e de alto desempenho apresenta alguns desafios:

1. Os circuitos não são do mesmo tamanho. Para circuitos simulados, não é possível criar operações estáticas (como `tf.matmul` ou `tf.add`) e depois substituir números diferentes para circuitos de tamanhos diferentes. Essas ops precisam permitir tamanhos dinâmicos que o grafo computacional de tamanho estático do TensorFlow não permite.
2. Os dados quânticos podem induzir uma estrutura de circuitos completamente diferente. Esse é outro motivo para oferecer suporte a tamanhos dinâmicos nas ops do TFQ. Os dados quânticos podem representar uma mudança estrutural no estado quântico subjacente que é representada por modificações no circuito original. Conforme novos pontos de dados entram e saem no runtime, o grafo computacional do TensorFlow não pode ser modificado após a criação, então é necessário suporte a essas estruturas variáveis.
3. `cirq.Circuits` se assemelham aos grafos computacionais por serem uma série de operações, e alguns até podem conter símbolos/marcadores de posição. É importante tornar isso o mais compatível possível com o TensorFlow.

Por razões de desempenho, Eigen (a biblioteca C++ usada em várias ops do TensorFlow) não é adequada para a simulação de circuitos quânticos. Em vez disso, os simuladores de circuitos usados no <a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">experimento quântico além do clássico</a> servem de verificadores e são estendidos como a base de todas as ops do TFQ (todas escritas com instruções AVX2 e SSE). Foram criadas ops com assinaturas funcionais idênticas que usam um computador quântico físico. Alternar entre um computador quântico simulado e físico é tão fácil quanto mudar uma única linha de código. Essas ops ficam localizadas em <a href="https://github.com/tensorflow/quantum/blob/master/tensorflow_quantum/core/ops/circuit_execution_ops.py" class="external"><code>circuit_execution_ops.py</code></a>.

### Camadas

As camadas do TensorFlow Quantum expõem a amostragem, a expectativa e o cálculo do estado aos desenvolvedores usando a interface `tf.keras.layers.Layer`. É conveniente criar uma camada de circuito para parâmetros de controle clássicos ou para operações de leitura. Além disso, você pode criar uma camada com um alto nível de complexidade compatível com o circuito em lote, controlar o valor do parâmetro por lote e realizar operações de leitura em lote. Veja um exemplo em `tfq.layers.Sample`.

### Diferenciadores

Ao contrário de várias operações do TensorFlow, observáveis em circuitos quânticos não têm fórmulas para gradientes que são relativamente fáceis de calcular. Isso ocorre porque um computador clássico só consegue ler amostras dos circuitos executados em um computador quântico.

Para resolver esse problema, o módulo `tfq.differentiators` oferece várias técnicas de diferenciação padrão. Os usuários também podem definir seu próprio método para computar gradientes — tanto no cenário do "mundo real" de cálculo de expectativa baseado em amostra como no mundo exato analítico. Métodos como a diferença finita são geralmente os mais rápidos (tempo do relógio de parede) em um ambiente analítico/exato. Enquanto métodos mais práticos e lentos (tempo do relógio de parede), como os métodos de <a href="https://arxiv.org/abs/1811.11184" class="external">mudança de parâmetro</a> ou <a href="https://arxiv.org/abs/1901.05374" class="external">estocástico</a>, são geralmente mais eficazes. Um `tfq.differentiators.Differentiator` é instanciado e ligado a uma op existente com `generate_differentiable_op` ou passado ao construtor de `tfq.layers.Expectation` ou `tfq.layers.SampledExpectation`. Para implementar um diferenciador personalizado, herde da classe `tfq.differentiators.Differentiator`. Para definir uma operação de gradiente para amostragem ou cálculo de vetor de estado, use `tf.custom_gradient`.

### Datasets

Com o avanço do campo de computação quântica, mais dados quânticos e combinações de modelos surgirão, dificultando ainda mais a comparação estruturada. O módulo `tfq.datasets` é usado como a origem de dados para tarefas de aprendizado de máquina quântico. Isso garante comparações estruturadas para o modelo e desempenho.

Espera-se que, com as grandes contribuições da comunidade, o módulo `tfq.datasets` cresça para permitir pesquisas mais transparentes e reproduzíveis. Problemas cuidadosamente selecionados em: controle quântico, simulação fermiônica, classificação perto de transições de fase, sensor quântico etc. são todos ótimos candidatos para inclusão em `tfq.datasets`. Para propor um novo dataset, abra um <a href="https://github.com/tensorflow/quantum/issues">issue do GitHub</a>.
