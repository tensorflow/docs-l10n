<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

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
    <td style="text-align: center; background-color: #D0D0D0">TF2의 TF1/TF1 호환 모드<a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>로딩/추론</td>
    <td>       Fully supported (<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">complete TF1 Hub format loading guide</a>)       <pre style="font-size: 12px;" lang="python">m = hub.Module(handle) outputs = m(inputs)</pre>     </td>
    <td> It's recommended to use either hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m.signatures["sig"](inputs)</pre>       or hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle, signature="sig") outputs = m(inputs)</pre>     </td>
  </tr>
  <tr>
    <td>Fine-tuning</td>
    <td>       Fully supported (<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">complete TF1 Hub format fine-tuning guide</a>)     <pre style="font-size: 12px;" lang="python">m = hub.Module(handle,                trainable=True,                tags=["train"]*is_training) outputs = m(inputs)</pre>       <div style="font-style: italic; font-size: 14px">       Note: modules that don't need a separate train graph don't have a train         tag.       </div>     </td>
    <td style="text-align: center">       Not supported     </td>
  </tr>
  <tr>
    <td>생성</td>
    <td> Fully supported (see <a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">complete TF1 Hub format creation guide</a>) <br> <div style="font-style: italic; font-size: 14px">       Note: The TF1 Hub format is geared towards TF1 and is only partially supported in TF2. Consider creating a TF2 SavedModel.       </div> </td>
    <td style="text-align: center">Not supported</td>
  </tr>
</table>

## TF2 SavedModel {:#compatibility_of_tf2_savedmodel}의 호환성

Not supported before TF1.15.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">작업</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2의 TF1.15/TF1 호환 모드<a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>로딩/추론</td>
    <td>       Use either hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre>       or hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre>     </td>
    <td> Fully supported (<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">complete TF2 SavedModel loading guide</a>). Use either hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre>       or hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre>     </td>
  </tr>
  <tr>
    <td>Fine-tuning</td>
    <td>       Supported for a hub.KerasLayer used in  tf.keras.Model when trained with       Model.fit() or trained in an Estimator whose model_fn wraps the Model per the <a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">custom model_fn guide</a>.       <br><div style="font-style: italic; font-size: 14px;">         Note: hub.KerasLayer <span style="font-weight: bold;">does not</span>         fill in graph collections like the old tf.compat.v1.layers or hub.Module         APIs did.       </div>     </td>
    <td>       Fully supported (<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">complete TF2 SavedModel fine-tuning guide</a>).       Use either hub.load:       <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs, training=is_training)</pre>       or hub.KerasLayer:       <pre style="font-size: 12px;" lang="python">m =  hub.KerasLayer(handle, trainable=True) outputs = m(inputs)</pre>     </td>
  </tr>
  <tr>
    <td>생성</td>
    <td>      The TF2 API <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">       tf.saved_model.save()</a> can be called from within compat mode.    </td>
   <td>완전 지원(<a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">전체 TF2 SavedModel 생성 가이드</a> 참조)</td>
  </tr>
</table>

<p id="compatfootnote">[1] "TF1 compat mode in TF2" refers to the combined   effect of importing TF2 with   <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code>   and running   <code style="font-size: 12px;" lang="python">tf.disable_v2_behavior()</code>  as described in the   <a href="https://www.tensorflow.org/guide/migrate">TensorFlow Migration guide   </a>.</p>
