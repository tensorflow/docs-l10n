# Resposta inteligente

<img src="../images/smart_reply.png" class="attempt-right">

## Como começar

Nosso modelo de resposta inteligente gera sugestões de resposta com base em mensagens de bate-papo. O objetivo é que as respostas sejam relevantes para o contexto e ajudem o usuário a responder rapidamente a uma mensagem recebida.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">Baixar modelo inicial</a>

### Aplicativo de exemplo

Este é um aplicativo de exemplo do TensorFlow que demonstra o modelo de resposta inteligente no Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Ver exemplo do Android</a>

Leia a [página no GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/) para saber como o aplicativo funciona. Dentro do projeto, você também verá como compilar um aplicativo com operações personalizadas do C++.

## Como funciona

O modelo gera sugestões de respostas a mensagens de conversa no bate-papo.

O modelo para uso em dispositivos tem diversos benefícios. Ele:

<ul>
  <li>É rápido – o modelo fica no dispositivo e não requer uma conexão com a Internet. Portanto, a inferência é muito rápida, com uma latência média de apenas alguns milissegundos.</li>
  <li>É eficiente no uso de recursos – o modelo usa uma pequena quantidade de memória do dispositivo.</li>
  <li>Mantém a privacidade – os dados do usuário nunca saem do dispositivo.</li>
</ul>

## Exemplo de saída

<img src="images/smart_reply.gif" style="max-width: 300px" alt="Animation showing smart reply">

## Saiba mais

<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">Artigo de pesquisa</a></li>
  <li><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Código-fonte</a></li>
</ul>

## Usuários

<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Inbox</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Respostas inteligentes no Android Wear</a></li>
</ul>
