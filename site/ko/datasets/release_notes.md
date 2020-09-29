# 릴리즈 노트

## 야간 버전

### 특징

- 더 나은 샤딩, 셔플링 및 하위 분할
- 이제 데이터세트로 저장/복원되는 `tfds.core.DatasetInfo`에 임의의 메타 데이터를 추가할 수 있습니다. `tfds.core.Metadata`를 참조하세요.
- 더 나은 프록시 지원, 인증서 추가 가능성
- `decoders` kwargs를 추가하여 기본 특성 디코딩 재정의([가이드](https://github.com/tensorflow/datasets/tree/master/docs/decode.md)).
- <a>MimickNet 논문</a>의 초음파 팬텀 및 생체 간 이미지의 <code>duke_ultrasound</code> 데이터세트 추가
- [VTAB 벤치마크](https://arxiv.org/abs/1910.04867)의 Dmlab 데이터세트 추가
- 논문 [e-SNLI](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf)의 e-SNLI 데이터세트 추가
- [Opinosis 데이터세트](https://www.aclweb.org/anthology/C10-1039.pdf) 추가
- [여기](https://arxiv.org/pdf/1711.00350.pdf)에 소개된 SCAN 데이터세트 추가
- [Imagewang](https://github.com/fastai/imagenette) 데이터세트 추가
- 논문 [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)의 DIV2K 데이터세트 추가
- [이 논문](https://openreview.net/pdf?id=SygcCnNKwr)의 CFQ(Compositional Freebase Questions) 데이터세트 추가
