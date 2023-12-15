# TensorFlow Quantum

O TensorFlow Quantum (TFQ) é um framework do Python para o [aprendizado de máquina quântico](concepts.md). Como um framework de aplicativos, o TFQ permite que pesquisadores de algoritmos quânticos e de aplicativos de ML aproveitem os frameworks computacionais quânticos do Google, tudo isso no TensorFlow.

O TensorFlow Quantum foca nos *dados quânticos* e na criação de *modelos clássicos-quânticos híbridos*. Ele oferece ferramentas para intercalar lógica e algoritmos quânticos criados no <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> com o TensorFlow. Para usar o TensorFlow Quantum de maneira eficaz, é necessário ter noções básicas sobre computação quântica.

Para começar a usar o TensorFlow Quantum, confira o [guia de instalação](install.md) e leia alguns dos [tutoriais em notebook](./tutorials/hello_many_worlds.ipynb) executáveis.

## Design

O TensorFlow Quantum implementa os componentes necessários para integrar o TensorFlow a hardware computacional quântico. Para isso, o TensorFlow Quantum apresenta dois tipos de dados primitivos:

- *Circuito quântico* — representa os circuitos quânticos definidos pelo Cirq no TensorFlow. Crie lotes de circuitos de diversos tamanhos, semelhantes aos lotes de diferentes pontos de dados reais.
- *Soma de Pauli* — representa combinações lineares de produtos tensoriais de operadores de Pauli definidos no Cirq. Como os circuitos, crie lotes de operadores de diversos tamanhos.

Usando esses primitivos para representar circuitos quânticos, o TensorFlow Quantum fornece as seguintes operações:

- Obtenha uma amostra das distribuições de saída dos lotes de circuitos.
- Calcule o valor esperado dos lotes de somas de Pauli em lotes de circuitos. O TFQ implementa o cálculo de gradientes compatível com a retropropagação.
- Simule lotes de circuitos e estados. Enquanto a inspeção de todas as amplitudes de estados quânticos diretamente em um circuito quântico é ineficaz em grande escala no mundo real, a simulação de estados pode ajudar os pesquisadores a entender como um circuito quântico mapeia os estados a um nível de precisão quase exato.

Leia mais sobre a implementação do TensorFlow Quantum no [guia de design](design.md).

## Relate problemas

Informe bugs ou solicite recursos usando o <a href="https://github.com/tensorflow/quantum/issues" class="external">issue tracker do TensorFlow Quantum</a>.
