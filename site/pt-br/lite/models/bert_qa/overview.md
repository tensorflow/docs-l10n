# BERT — Perguntas e respostas

Use um modelo do TensorFlow Lite para responder a perguntas com base no conteúdo de um determinado trecho.

Observação: (1) para integrar um modelo existente, use a [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) (biblioteca de tarefas do TensorFlow Lite). (2) Para personalizar um modelo, use o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) (criador de modelos do TensorFlow Lite).

## Como começar

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou iOS, recomendamos conferir os exemplos de aplicativo abaixo que podem te ajudar a começar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Exemplo do Android</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">Exemplo do iOS</a>

Se você estiver usando outra plataforma que não o Android/iOS ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), pode baixar nosso modelo inicial de perguntas e respostas.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Baixar modelo inicial e vocabulário</a>

Para mais informações sobre os metadados e arquivos associados (por exemplo, `vocab.txt`), confira o artigo <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">Leia metadados de modelos</a>.

## Como funciona

O modelo pode ser usado para criar um sistema que responda a perguntas de usuários em linguagem natural. Ele foi criado utilizando-se um modelo BERT pré-treinado que passou por ajustes finos usando-se o dataset SQuAD 1.1.

[BERT](https://github.com/google-research/bert), sigla em inglês para Representações de Encoder Bidirecional de Transformadores, é um método de representações de linguagem pré-treinamento que obtém resultados avançados em uma ampla gama de tarefas de processamento de linguagem natural (NLP, na sigla em inglês).

Este aplicativo usa uma versão compactada do BERT, o MobileBERT, que é executado quatro vezes mais rápido, com tamanho do modelo quatro vezes menor.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), sigla em inglês para Stanford Question Answering Dataset (Dataset de respostas a perguntas da Stanford), é um dataset de compreensão de leitura que consiste de artigos da Wikipedia e um conjunto de pares pergunta/resposta para cada artigo

O modelo recebe um trecho e uma pergunta como entrada, depois retorna um segmento do trecho com a maior probabilidade de responder à pergunta. Esse modelo requer um pré-processamento semicomplexo, incluindo tokenização e passos de pré-processamento descritos no [artigo](https://arxiv.org/abs/1810.04805) do BERT e implementados no aplicativo de exemplo.

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Mobile Bert</a>
</td>
    <td rowspan="3">       100,5 MB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>123 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>74 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>257 ms**</td>
  </tr>
</table>

* 4 threads usados.

** 2 threads usados no iPhone para o resultado com maior desempenho.

## Exemplo de saída

### Trecho (entrada)

> O Google LLC é uma multinacional de tecnologia especializada em serviços e produtos de Internet, incluindo tecnologias de publicidade online, mecanismo de pesquisa, computação em nuvem, software e hardware. É considerado uma das quatro grandes empresas de tecnologia junto com Amazon, Apple e Facebook.
>
> O Google foi fundado em setembro de 1998 por Larry Page e Sergey Brin enquanto eram alunos de doutorado na Universidade de Stanford, na Califórnia. Juntos, detêm cerca de 14% das ações e controlam 56% dos votos de acionistas por meio das ações com direito a voto. Fizeram a incorporação do Google como uma empresa privada da Califórnia de 4 de setembro de 1998. Depois, foi feita uma nova incorporação do Google em Delaware, em 22 de outubro de 2002. Uma oferta pública inicial (IPO) de ações foi realizada em 19 de agosto de 2004, e o Google levou sua sede a Mountain View, Califórnia, apelidada de Googleplex. Em agosto de 2015, o Google anunciou planos de reorganizar seus diversos interesses como um conglomerado chamado Alphabet Inc. O Google é a principal subsidiária da Alphabet e empresa guarda-chuva para os interesses de Internet da Alphabet. Sundar Pichai foi nomeado CEO do Google, substituindo Larry Page, que se tornou o CEO da Alphabet.

### Pergunta (entrada)

> Quem é o CEO do Google?

### Resposta (saída)

> Sundar Pichai

## Saiba mais sobre o BERT

- Artigo acadêmico: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (BERT: Pré-treinamento de transformadores bidirecionais profundos para compreensão de linguagem)
- [Implementação do BERT em código aberto](https://github.com/google-research/bert)
