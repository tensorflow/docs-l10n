# スマートリプライ


<img src="../images/smart_reply.png" class="attempt-right">

## はじめに

スマートリプライモデルは、チャットメッセージに基づいた返信提案を生成します。文脈的に適切な提案で、受信メッセージに対してユーザーが簡単に返信できるワンタッチ応答を目指しています。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">スターターモデルをダウンロードする</a>

### サンプルアプリ

Android 上でスマートリプライモデルを実演する TensorFlow Lite のサンプルアプリを提供しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Android の例を見る</a>

アプリの動作についての詳細は [GitHub ページ](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/) をご覧ください。このプロジェクト内では、カスタム C++ 演算を使用してアプリを構築する方法も学ぶことができます。

## 使い方

このモデルは、会話型チャットメッセージに対する返信提案を生成します。

オンデバイスモデルには、以下のようなメリットがあります。

<ul>
  <li>高速: モデルがデバイス上に存在するため、インターネットに接続する必要がありません。そのため推論は非常に高速で、平均レイテンシはわずか数ミリ秒です。</li>
  <li>リソース効率: モデルのデバイス上のフットプリントが小さくなります。</li>
  <li>プライバシーへの配慮: ユーザーデータがデバイスから収取されることはありません。</li>
</ul>

## 出力例


<img alt="Animation showing smart reply" src="images/smart_reply.gif" style="max-width: 300px">

## 詳細を読む

<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">研究論文</a></li>
  <li><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">ソースコード</a></li>
</ul>

## ユーザー

<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Inbox</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Android Wear のスマートリプライ</a></li>
</ul>
