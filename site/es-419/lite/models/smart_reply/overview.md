# Respuesta inteligente

<img src="../images/smart_reply.png" class="attempt-right">

## Empecemos

Nuestro modelo de respuesta inteligente genera sugerencias de respuesta basadas en los mensajes de chat. Las sugerencias pretenden ser respuestas contextualmente relevantes, de un solo toque, que ayuden al usuario a responder fácilmente a un mensaje entrante.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">Descargue el modelo inicial</a>

### Aplicación de muestra

Existe una aplicación de muestra de TensorFlow Lite que demuestra el modelo de respuesta inteligente en Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Ver ejemplo en Android</a>

Lea la [página en GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/) para aprender cómo funciona la app. Dentro de este proyecto, también aprenderá a crear una app con ops personalizadas en C++.

## Cómo funciona

El modelo genera sugerencias de respuesta a los mensajes del chat conversacional.

El modelo en el dispositivo tiene varias prestaciones útiles. Es:

<ul>
  <li>Rápido: El modelo reside en el dispositivo y no requiere conectividad a Internet. Así, la inferencia es muy rápida y tiene una latencia media de sólo unos milisegundos.</li>
  <li>Uso eficiente de los recursos: El modelo ocupa poco espacio de memoria en el dispositivo.</li>
  <li>Respetuoso con la privacidad: los datos del usuario nunca salen del dispositivo.</li>
</ul>

## Salida de ejemplo

<img src="images/smart_reply.gif" style="max-width: 300px" alt="Animación que muestra la respuesta inteligente">

## Más información en este tema

<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">Artículo de investigación</a></li>
  <li><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Código fuente</a></li>
</ul>

## Usuarios

<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Bandeja de entrada</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Respuestas inteligentes en Android Wear</a></li>
</ul>
