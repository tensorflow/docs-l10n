# Recomendação

<table class="tfo-notebook-buttons" align="left">   <td>     <a target="_blank" href="https://www.tensorflow.org/lite/examples/recommendation/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">Ver em TensorFlow.org</a>   </td>   {% dynamic if request.tld != 'cn' %}<td>     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Executar no Google Colab</a>   </td>{% dynamic endif %}   <td>     <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">Ver fonte no GitHub</a>   </td>
</table>

Recomendações personalizadas são amplamente utilizadas em diversos casos de uso em dispositivos móveis, como busca de conteúdo de mídia, sugestão de compra de produtos e recomendação de próximos aplicativos. Se você tiver interesse em fornecer recomendações personalizadas em seu aplicativo, mas sem deixar de respeitar a privacidade dos usuários, recomendamos conferir o exemplo e o kit de ferramentas abaixo.

Observação: para personalizar um modelo, experimente o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) (criador de modelos do TensorFlow Lite).

## Como começar

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Fornecemos um aplicativo de exemplo do TensorFlow que demonstra como recomendar itens relevantes para usuários do Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Exemplo do Android</a>

Se você estiver usando outra plataforma que não o Android ou se já conhecer bem as APIs do TensorFlow Lite, pode baixar nosso modelo inicial de recomendações.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">Baixar modelo inicial</a>

Também fornecemos o script de treinamento no GitHub para treinar seu próprio modelo de uma forma configurável.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">Código de treinamento</a>

## Compreendendo a arquitetura do modelo

Usamos uma arquitetura de modelo com dois encoders, em que o encoder de contexto codifica o histórico sequencial do usuário e o encoder de rótulo codifica o candidato previsto para recomendação. A semelhança entre as codificações de contexto e rótulo é usada para representar a probabilidade de o candidato previsto atender às necessidades do usuário.

São fornecidas três diferentes técnicas de codificação do histórico sequencial do usuário com este código base:

- Encoder saco-de-palavras (BOW, na sigla em inglês para Bag-of-words): média dos embeddings de atividades do usuário sem considerar a ordem do contexto.
- Encoder de rede neural convolucional (CNN): aplicação de várias camadas de redes neurais convolucionais para gerar codificação de contexto.
- Encoder de rede neural recorrente (RNN): aplicação de rede neural recorrente para codificar a sequência do contexto.

Para modelar cada atividade do usuário, podemos usar o ID do item de atividade (modelo baseado em IDs) ou diversas características do item (modelo baseado em características), ou uma combinação das duas estratégias. O modelo baseado em características utiliza diversas características para codificar coletivamente o comportamento do usuário. Com esse código base, você pode criar modelos baseados em IDs ou em características de uma maneira configurável.

Após o treinamento, um modelo do TensorFlow Lite será exportado, podendo fornecer diretamente top-K previsões dentre os candidatos para recomendação.

## Use seus dados de treinamento

Além do modelo treinado, oferecemos um [kit de ferramenta no GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) em código aberto para treinar modelos com seus próprios dados. Acompanhe este tutorial se quiser saber como usar o kit de ferramentas e implantar modelos treinados em seus próprios aplicativos móveis.

Acompanhe este [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) para aplicar a mesma técnica usada aqui para treinar um modelo de recomendações usando seus próprios datasets.

## Exemplos

Para exemplificar, treinamos modelos de recomendações baseados em IDs e baseados em características. O modelo baseado em IDs recebe somente os IDs dos filmes como entrada, enquanto o modelo baseado em características recebe tanto os IDs dos filmes quanto os IDs dos gêneros dos filmes como entrada. Confira abaixo os exemplos de entrada e saída.

Entradas

- IDs dos filmes do contexto:

    - O Rei Leão (ID: 362)
    - Toy Story (ID: 1)
    - (e muitos outros)

- IDs dos gêneros dos filmes do contexto:

    - Animação (ID: 15)
    - Infantil (ID: 9)
    - Musical (ID: 13)
    - Animação (ID: 15)
    - Infantil (ID: 9)
    - Comédia (ID: 2)
    - (e muitos outros)

Saídas:

- IDs dos filmes recomendados:
    - Toy Story 2 (ID: 3114)
    - (e muitos outros)

Observação: o modelo pré-treinado é criado baseado no dataset [MovieLens](https://grouplens.org/datasets/movielens/1m/) para fins de pesquisa.

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
  <tbody>
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">Recomendação (ID do filme como entrada)</a>
</td>
        <td rowspan="3">           0,52 MB</td>
        <td>Pixel 3</td>
        <td>0,09 ms*</td>
      </tr>
       <tr>
         <td>Pixel 4</td>
        <td>0,05 ms*</td>
      </tr>
    
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20210317/recommendation_cnn_i10i32o100.tflite">Recomendação (ID do filme e gênero do filme como entradas)</a>
</td>
        <td rowspan="3">           1,3 MB</td>
        <td>Pixel 3</td>
        <td>0.13 ms*</td>
      </tr>
       <tr>
         <td>Pixel 4</td>
        <td>0,06 ms*</td>
      </tr>
    
  </tbody>
</table>

* 4 threads usados.

## Use seus dados de treinamento

Além do modelo treinado, oferecemos um [kit de ferramenta no GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) em código aberto para treinar modelos com seus próprios dados. Acompanhe este tutorial se quiser saber como usar o kit de ferramentas e implantar modelos treinados em seus próprios aplicativos móveis.

Acompanhe este [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) para aplicar a mesma técnica usada aqui para treinar um modelo de recomendações usando seus próprios datasets.

## Dicas para personalizar o modelo com seus dados

O modelo pré-treinado integrado neste aplicativo de demonstração foi treinado com o dataset [MovieLens](https://grouplens.org/datasets/movielens/1m/). Talvez você queira modificar a configuração do modelo de acordo com seus próprios dados, como tamanho do vocabulário, dimensões dos embeddings e tamanho do contexto de entrada. Confira algumas dicas:

- Tamanho do contexto de entrada: o melhor tamanho varia de acordo com os datasets. Sugerimos selecionar um tamanho dependendo de quantos eventos de rótulo estão correlacionados aos interesses de longo prazo versus contexto de curto prazo.

- Seleção do tipo de encoder: sugerimos selecionar o tipo de encoder de acordo com o tamanho do contexto de entrada. O encoder saco-de-palavras funciona bem para tamanho pequeno (menos de 10); os encoders de CNN e RNN agregam mais capacidade de sumarização para um tamanho de contexto de entrada grande.

- O uso das características subjacentes para representar itens ou atividades do usuário pode aumentar o desempenho do modelo, acomodar melhor itens novos, possivelmente reduzir os espaços de embeddings, reduzindo assim o consumo de memória, além de ser mais otimizado para dispositivos.
