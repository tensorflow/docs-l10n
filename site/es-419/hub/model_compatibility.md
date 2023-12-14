# Compatibilidad de modelos para TF1/TF2

## Formatos de modelos TF Hub

TF Hub ofrece partes de modelo reutilizables que se pueden volver a cargar, construir y posiblemente volver a entrenar en un programa TensorFlow. Vienen en dos formatos diferentes:

- El [formato de TF1 Hub](https://www.tensorflow.org/hub/tf1_hub_module) personalizado. Su principal uso previsto es en TF1 (o en modo de compatibilidad TF1 en TF2) a través de su [API hub.Module](https://www.tensorflow.org/hub/api_docs/python/hub/Module). Puede ver todos los detalles de compatibilidad [a continuación](#compatibility_of_hubmodule).
- El formato nativo [TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model). Su principal uso previsto es en TF2 a través de las API [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) y [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer). Puede ver todos los detalles de compatibilidad [a continuación](#compatibility_of_tf2_savedmodel).

El formato del modelo se puede encontrar en la página del modelo en [tfhub.dev](https://tfhub.dev). Es posible que TF1/2 no admita **la carga/inferencia**, **el ajuste** o **la creación de modelos** según los formatos del modelo.

## Compatibilidad del formato TF1 Hub {:#compatibility_of_hubmodule}

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">Operación</td>
    <td style="text-align: center; background-color: #D0D0D0">Modo de compatibilidad TF1/TF1 en TF2 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>Carga / Inferencia</td>
    <td>Totalmente compatible (<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">guía completa de carga del formato TF1 Hub</a>)      <pre style="font-size: 12px;" lang="python">m = hub.Module(handle)
outputs = m(inputs)</pre>
</td>
    <td>Se recomienda usar hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m.signatures["sig"](inputs)</pre>       o hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle, signature="sig")
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Ajuste</td>
    <td>Totalmente compatible (<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">guía completa de ajuste del formato TF1 Hub</a>) <pre style="font-size: 12px;" lang="python">m = hub.Module(handle,
trainable=True,
tags=["train"]*is_training)
outputs = m(inputs)</pre> <div style="font-style: italic; font-size: 14px"> Nota: los módulos que no necesitan un gráfico de entrenamiento separado no tienen una etiqueta de train.</div>
</td>
    <td style="text-align: center">       No compatible</td>
  </tr>
  <tr>
    <td>Creación</td>
    <td> Totalmente compatible (consulte la <a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">guía completa de creación del formato TF1 Hub</a>) <br><div style="font-style: italic; font-size: 14px"> Nota: El formato TF1 Hub está orientado a TF1 y solo se admite parcialmente en TF2. Considere la posibilidad de crear un TF2 SavedModel.</div>
</td>
    <td style="text-align: center">No compatible</td>
  </tr>
</table>

## Compatibilidad de TF2 SavedModel {:#compatibility_of_tf2_savedmodel}

No compatible antes de TF1.15.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">Operación</td>
    <td style="text-align: center; background-color: #D0D0D0">Modo de compatibilidad TF1.15/TF1 en TF2 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>Carga / Inferencia</td>
    <td>Use hub.load <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs)</pre> o hub.KerasLayer <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle)
outputs = m(inputs)</pre>
</td>
    <td>Totalmente compatible (<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">guía completa de carga de TF2 SavedModel</a>). Use  hub.load <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs)</pre> o hub.KerasLayer <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle)
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Ajuste</td>
    <td>Compatible con hub.KerasLayer que se usa en tf.keras.Model cuando se entrena con Model.fit() o se entrena en un Estimador cuyo model_fn envuelve el modelo según la <a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">guía personalizada de model_fn</a>. <br><div style="font-style: italic; font-size: 14px;"> Nota: hub.KerasLayer <span style="font-weight: bold;">no</span> completa las colecciones de gráficos como lo hacían las antiguas API tf.compat.v1.layers o hub.Module.</div>
</td>
    <td>Totalmente compatible (<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">guía completa de ajuste de TF2 SavedModel</a>).       Use hub.load:       <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs, training=is_training)</pre>       o hub.KerasLayer:       <pre style="font-size: 12px;" lang="python">m =  hub.KerasLayer(handle, trainable=True)
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Creación</td>
    <td>La API TF2 <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">tf.saved_model.save()</a> se puede llamar desde el modo de compatibilidad.</td>
   <td>Totalmente compatible (consulte la <a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">guía completa de creación de TF2 SavedModel</a>)</td>
  </tr>
</table>

<p id="compatfootnote">[1] Modo de compatibilidad TF1 en TF2 se refiere al efecto combinado de importar TF2 con <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code> y ejecutar <code style="font-size: 12px;" lang="python">tf.disable_v2_behavior()</code> como se describe en la <a href="https://www.tensorflow.org/guide/migrate">guía de migración de TensorFlow</a>.</p>
