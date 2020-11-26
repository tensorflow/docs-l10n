<!--* freshness: { owner: 'maringeo' } *-->

# 게시자 되기

## 서비스 약관

게시할 모델을 제출하면 [https://tfhub.dev/terms](https://tfhub.dev/terms)에 있는 TensorFlow Hub 서비스 약관에 동의하는 것입니다.

## 게시 프로세스 개요

게시의 전체 프로세스는 다음으로 구성됩니다.

1. 모델 만들기([모델 내보내기](exporting_tf2_saved_model.md) 방법 참조)
2. 설명서 작성하기([모델 설명서 작성하기](writing_model_documentation.md) 방법 참조)
3. 게시 요청 만들기([기여](contribute_a_model.md) 방법 참조)

## 게시자 페이지의 마크다운 형식

게시자 설명서는 [모델 설명서 작성하기](writing_model_documentation) 가이드에 설명된 것과 같은 종류의 마크다운 파일에서 선언하지만, 구문상 약간의 차이가 있습니다.

TensorFlow Hub 리포지토리에서 게시자 파일의 올바른 위치는 [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<publisher_name.md>입니다.

최소 게시자 설명서의 예를 참조하세요.

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

위의 예에서는 게시자 이름, 간단한 설명, 사용할 아이콘 경로 및 더 긴 자유 형식의 마크다운 설명서를 지정합니다.

### 게시자 이름 가이드라인

게시자 이름은 GitHub 사용자 이름 또는 관리하는 GitHub 조직의 이름이 될 수 있습니다.
