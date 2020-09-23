# リリースノート

## ナイトリー

### 機能

- シャーディング、シャッフル、およびサブスプリットの改善
- 任意のメタデータをデータセットとともに保存/復元される `tfds.core.DatasetInfo` に追加できるようになりました。`tfds.core.Metadata` をご覧ください。
- プロキシサポートの改善、証明書の追加
- デフォルトの特徴量デコードをオーバーライドする `decoders` kwargs の追加（[ガイド](https://github.com/tensorflow/datasets/tree/master/docs/decode.md)）
- [MimickNet 論文](https://arxiv.org/abs/1908.05782)の、超音波ファントムと生体内肝臓画像の `duke_ultrasound` データセットの追加
- [VTAB benchmark](https://arxiv.org/abs/1910.04867) の Dmlab データセットの追加
- 論文 [e-SNLI](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf) の e-SNLI データセットの追加
- [Opinosis データセット](https://www.aclweb.org/anthology/C10-1039.pdf)の追加
- [こちら](https://arxiv.org/pdf/1711.00350.pdf)で紹介された SCAN データセットの追加
- [Imagewang](https://github.com/fastai/imagenette) データセットの追加
- 論文 [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) の DIV2K データセットの追加
- [こちらの論文](https://openreview.net/pdf?id=SygcCnNKwr)の CFQ（Compositional Freebase Questions）データセットの追加
