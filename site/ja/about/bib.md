# TensorFlow を引用する

TensorFlow は Zenodo.org（[10.5281/zenodo.4724125](https://doi.org/10.5281/zenodo.4724125)） を使用してオープンソースコードの DOI を公開しています。

[このホワイトペーパーにアクセスする。](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

## 異種分散システムにおける大規模機械学習

[このホワイトペーパーにアクセスする。](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**抜粋:** TensorFlow は、機械学習アルゴリズムを表現するためのインターフェースであり、そのアルゴリズムを実行するための実装です。TensorFlow を用いて表現された計算は、携帯電話やタブレットなどのモバイルデバイスから、数百台のマシンやGPU カードなどの数千台の計算デバイスから成る大規模分散システムに至るまで、多種多様な異種システム上でほとんど変更することなく実行することができます。このシステムには柔軟性があり、ディープニューラルネットワークモデルのトレーニングアルゴリズムや推論アルゴリズムなど、多様なアルゴリズムの表現に使用することができます。また、音声認識、コンピュータビジョン、ロボット工学、情報検索、自然言語処理、地理情報抽出、計算創薬など、コンピュータサイエンスをはじめとする十数分野にわたる研究の実施や機械学習システムの本番展開に利用されています。本ペーパーでは、TensorFlow インターフェースと Google で構築したインターフェースの実装について説明します。TensorFlow API とリファレンス実装は、Apache 2.0 ライセンス下のオープンソースパッケージとして 2015 年 11 月にリリースされ、 www.tensorflow.org から入手可能です。

### BibTeX 形式

研究に TensorFlow を使用し、TensorFlow システムの引用が必要な場合には、このホワイトペーパーの引用をお勧めします。

<pre>@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
</pre>

[このホワイトペーパーにアクセスする。](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

<pre>
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
</pre>

## TensorFlow: 大規模機械学習のためのシステム

[このホワイトペーパーにアクセスする。](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**要約:** TensorFlow は、大規模かつ異種環境で動作する機械学習システムです。TensorFlow は、データフローグラフを使用して、計算、共有状態、およびその状態を変化させる操作を表現します。単一クラスター内では多数のマシン間で、また単一マシン内ではマルチコア CPU、汎用 GPU、Tensor Processing Unit (TPU) と呼ばれるカスタム設計の ASIC などの複数の計算デバイス間で、データフローグラフのノードをマッピングします。このアーキテクチャはアプリケーション開発者に柔軟性を与えます。以前の「パラメータサーバー」設計では共有状態の管理がシステムに組み込まれていましたが、TensorFlow では開発者が新しい最適化やトレーニングアルゴリズムを実験することが可能です。TensorFlow は特にディープニューラルネットワークのトレーニングと推論に焦点を当て、様々なアプリケーションをサポートしています。複数の Google のサービスが本番に TensorFlow を使用し、それをオープンソースプロジェクトとしてリリースしているため、機械学習の研究に幅広く利用されるようになりました。本ペーパーでは、TensorFlow データフローモデルについて解説し、複数の実際のアプリケーションにおける TensorFlow の優れたパフォーマンスを見ています。
