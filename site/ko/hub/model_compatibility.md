<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# TF1/TF2의 모델 호환성

## TF Hub 모델 형식

TF Hub는 TensorFlow 프로그램에서 다시 로드, 빌드 및 재훈련할 수 있는 재사용 가능한 모델 조각을 제공합니다. 두 가지 형식으로 제공됩니다.

- 사용자 정의 [TF1 Hub format](https://www.tensorflow.org/hub/tf1_hub_module): 주요 용도는 [hub.Module API](https://www.tensorflow.org/hub/api_docs/python/hub/Module)를 통한 TF1(또는 TF2의 TF1 호환성 모드)입니다. 전체 호환성 세부 정보는 [아래](#compatibility_of_hubmodule)에 나와 있습니다.
- 기본 [TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model) 형식: 주요 용도는 [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) 및 [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) API를 통해 TF2에 있습니다. 전체 호환성 세부 정보는 [아래](#compatibility_of_tf2_savedmodel)에 나와 있습니다.

모델 형식은 [tfhub.dev](https://tfhub.dev)의 모델 페이지에서 찾을 수 있습니다. 모델 **로딩/추론**, **미세 조정** 또는 **생성**은 모델 형식에 따라 TF1/2에서 지원되지 않을 수 있습니다.

## TF1 Hub 형식 {:#compatibility_of_hubmodule}의 호환성

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">작업</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2의 TF1/TF1 호환 모드<a href="#compatfootnote">[1]</a> </td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>로딩/추론</td>
    <td>전체 지원(<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">전체 TF1 Hub 형식 로딩 가이드</a>) <pre style="font-size: 12px;" lang="python">m = hub.Module(handle) outputs = m(inputs)</pre> </td>
    <td>hub.load 중 하나를 사용하는 것이 좋습니다. <pre style="font-size: 12px;" lang="python"> m = hub.load (handle) 출력 = m.signatures [ "sig"] (입력) </pre> 또는 hub.KerasLayer <pre style="font-size: 12px;" lang="python"> m = hub.KerasLayer (handle, signature = "sig") 출력 = m (입력) </pre> </td>
  </tr>
  <tr>
    <td>미세 조정</td>
    <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">전체 TF1 Hub 형식 미세 조정 가이드</a>) <pre style="font-size: 12px;" lang="python">m = hub.Module(handle, trainable=True, tags=["train"]*is_training) outputs = m(inputs)</pre> <div style="font-style: italic; font-size: 14px"> 참고: 별도의 훈련 그래프가 필요하지 않은 모듈에는 훈련 태그가 없습니다.</div> </td>
    <td style="text-align: center">지원되지 않음</td>
  </tr>
  <tr>
    <td>생성</td>
    <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">전체 TF1 Hub 형식 생성 가이드 참조</a>) <br><div style="font-style: italic; font-size: 14px"> 참고: TF1 Hub 형식은 TF1에 맞춰져 있으며 TF2에서는 부분적으로만 지원됩니다. TF2 SavedModel의 생성을 고려해 보세요.</div> </td>
    <td style="text-align: center">지원되지 않음</td>
  </tr>
</table>

## TF2 SavedModel {:#compatibility_of_tf2_savedmodel}의 호환성

TF1.15 이전에는 지원되지 않습니다.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">작업</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2의 TF1.15/TF1 호환 모드<a href="#compatfootnote">[1]</a> </td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>로딩/추론</td>
    <td>다음 중 하나를 사용하세요. hub.load <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre> 또는 hub.KerasLayer <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre> </td>
    <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">전체 TF2 SavedModel 로딩 가이드</a>). 다음 중 하나를 사용하세요. hub.load <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre> 또는 hub.KerasLayer <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre> </td>
  </tr>
  <tr>
    <td>미세 조정</td>
    <td>Model.fit()으로 훈련되거나 <a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">custom model_fn 가이드</a>에 따라 model_fn이 모델을 래핑하는 Estimator에서 훈련된 경우, tf.keras.Model에서 사용되는 hub.KerasLayer를 지원합니다.<br><div style="font-style: italic; font-size: 14px;">참고: hub.KerasLayer는 이전 tf.compat.v1.layers 또는 hub.Module API처럼 그래프 모음을 채우지 <span style="font-weight: bold;">않습니다</span>.</div> </td>
    <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">전체 TF2 저장된 모델 미세 조정 가이드</a>). 다음 중 하나를 사용하세요. hub.load: <pre style="font-size: 12px;" lang="python"> m = hub.load (handle) 출력 = m (inputs, training = is_training) </pre> 또는 hub.KerasLayer : <pre style="font-size: 12px;" lang="python"> m = hub.KerasLayer (handle, trainable = True) outputs = m (inputs) </pre> </td>
  </tr>
  <tr>
    <td>생성</td>
    <td>TF2 API <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">tf.saved_model.save()</a>는 호환 모드에서 호출할 수 있습니다.</td>
   <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">전체 TF2 SavedModel 생성 가이드</a> 참조)</td>
  </tr>
</table>

<p id="compatfootnote">[1] "TF2의 TF1 호환 모드"는 <a href="https://www.tensorflow.org/guide/migrate">TensorFlow 마이그레이션 가이드</a>의 설명과 같이 <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code>로 TF2를 가져오고 <code style="font-size: 12px;" lang="python">tf.disable_v2_behavior()</code>를 실행하는 결합된 효과를 말합니다.</p>
