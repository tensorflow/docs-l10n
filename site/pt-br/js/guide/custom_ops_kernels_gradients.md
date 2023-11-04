# Escrevendo ops, kernels e gradientes personalizados em TensorFlow.js

## Visão geral

Este guia descreve os mecanismos para definir operações (ops), kernels e gradientes personalizados em TensorFlow.js. O objetivo é apresentar uma visão geral dos principais conceitos e exemplos de código que demonstram esses conceitos.

### Este guia é destinado a quem?

Este guia é bem avançado e aborda detalhes internos de TensorFlow.js, o que pode ser útil para os seguintes grupos:

- Usuários avançados do TensorFlow.js interessados em personalizar o comportamento de diversas operações matemáticas (por exemplo, pesquisadores que sobrescrevem implementações de gradientes existentes ou usuários que precisam acrescentar funcionalidades ausentes na biblioteca).
- Usuários que criam bibliotecas para estender o TensorFlow.js (por exemplo, uma biblioteca de álgebra linear criada com primitivos do TensorFlow.js ou um novo backend do TensorFlow.js).
- Usuários interessados em contribuir com novas operações para o Tensorflow.js que desejam ter uma visão geral de como esses mecanismos funcionam.

Este **não é** um guia de uso geral do  TensorFlow.js, pois são abordados mecanismos de implementação internos. Você não precisa entendê-los para usar o TensorFlow.js.

Você precisa estar à vontade (ou ter disposição para tentar) ler código fonte do TensorFlow.js para aproveitar ao máximo este guia.

## Terminologia

Para este guia, vamos descrever antecipadamente termos essenciais:

**Operações (ops)** — Operação matemática feita em um ou mais tensores que produz um ou mais tensores como saída. As operações são código de "alto nível" e podem usar outras operações para definirem sua lógica.

**Kernel** — Implementação específica de uma operação ligada a capacidades de hardwares/plataformas específicos. Os kernels são de "baixo nível" e específicos de backends. Algumas operações têm um mapeamento de operação para kernel um para um, enquanto outras operações utilizam diversos kernels.

**Gradiente****/GradFunc** — A definição "modo para trás" de uma **op/kernel** que computa o derivativo dessa função em relação à entrada. Os gradientes são código de "alto nível (não são específicos de backends) e podem chamar outras operações ou kernels.

**Registro do kernel** — Um mapeamento de uma tupla **(kernel name, backend name)** (nome do kernel, nome do backend) para a implementação de um kernel.

**Registro do gradiente** — Um mapeamento de um **nome de kernel para a implementação de um gradiente**.

## Organização do código

As [operações](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops) e os [gradientes](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients) são definidos em [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core).

Os kernels são específicos de backends e são definidos nas pastas de seus respectivos backends (por exemplo, [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels)).

Operações, kernels e gradientes personalizados não precisam ser definidos dentro desses pacotes, mas costumam usar símbolos similares em sua implementação.

## Implementando operações personalizadas

Uma maneira de pensar em uma operação personalizada é uma função JavaScript que retorna um tensor como saída, geralmente com tensores como entrada.

- Algumas operações podem ser totalmente definidas em termos de operações existentes e devem apenas importar e chamar essas funções diretamente. [Veja um exemplo](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts).
- A implementação de uma operação também pode ser enviada para kernels de backends específicos. Isso é feito usando-se `Engine.runKernel` e será descrito mais adiante na seção “Implementando kernels personalizados”. [Veja um exemplo](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts).

## Implementando kernels personalizados

As implementações de kernels de backends específicos permite uma implementação otimizada da lógica de uma determinada operação. Os kernels são invocados pelas operações ao chamar [`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F). Uma implementação de kernel é definida por quatro aspectos:

- Nome do kernel.
- Backend no qual o kernel é implementado.
- Entradas: argumentos Tensores da função do kernel.
- Atributos: argumentos não Tensores da função do kernel.

Veja um exemplo da [implementação de um kernel](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts). As convenções usadas para implementar são específicas de backends e são melhor compreendidas verificando a implementação e a documentação de cada backend individual.

Geralmente, os kernels operam em um nível mais baixo do que os tensores e leem e gravam na memória diretamente, o que acabará sendo encapsulado em tensores por tfjs-core.

Após um kernel ser implementado, ele pode ser registrado no TensorFlow.js usando-se a [ função `registerKernel`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F) de tfjs-core. Você pode usar um kernel de cada backend no qual deseja que esse kernel funcione. Após registrado, o kernel pode ser invocado usando-se `tf.engine().runKernel(...)`, e TensorFlow.js vai enviar a implementação para o backend atual ativo.

## Implementando gradientes personalizados

Geralmente, os gradientes são definidos para um determinado kernel (identificado pelo mesmo nome de kernel usado em uma chamada a `tf.engine().runKernel(...)`). Dessa forma, tfjs-core pode usar um registro para buscar as definições de gradiente para qualquer kernel no runtime.

Implementar gradientes personalizados serve para:

- Adicionar uma definição de gradiente que pode não estar presente na biblioteca.
- Sobrescrever a definição de um gradiente existente para personalizar a computação do gradiente para um determinado kernel.

Confira exemplos de [implementações de gradientes aqui](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients).

Após implementar um gradiente para uma determinada chamada, ele pode ser registrado no TensorFlow.js usando-se a [função `registerGradient`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F) de tfjs-core.

Outra estratégia para implementar gradientes personalizados que faça o bypass do registro de gradiente (e, portanto, permite a computação de gradientes para funções arbitrárias de maneiras arbitrárias) é usar [tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad).

Veja um [exemplo de uma operação da biblioteca](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64) que usa customGrad.
