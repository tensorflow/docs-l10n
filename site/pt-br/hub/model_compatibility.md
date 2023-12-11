# Compatibilidade de modelo para TF1/TF2

## Formatos de modelo do TF Hub

O TF Hub oferece partes de modelo reutilizáveis que podem ser carregados, expandidos e possivelmente treinados novamente em um programa do TensorFlow. Há dois formatos diferentes:

- O [formato TF1 Hub](https://www.tensorflow.org/hub/tf1_hub_module) personalizado. Seu principal uso é no TF1 (ou no modo de compatibilidade com o TF1 no TF2) por meio de sua [API hub.Module](https://www.tensorflow.org/hub/api_docs/python/hub/Module). Confira os detalhes completos de compatibilidade [abaixo](#compatibility_of_hubmodule).
- O formato nativo [TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model). Seu principal uso é no TF2 por meio das APIs [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) e [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer). Confira os detalhes completos de compatibilidade [abaixo](#compatibility_of_tf2_savedmodel).

O formato do modelo fica disponível na página do modelo em [tfhub.dev](https://tfhub.dev). O **carregamento/inferência**, **ajustes finos** ou **criação** de um modelo podem não ter suporte do TF1/TF2 de acordo com os formatos do modelo.

## Compatibilidade do formato TF1 Hub {:#compatibility_of_hubmodule}

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">Operação</td>
    <td style="text-align: center; background-color: #D0D0D0">TF1/Modo de compatibilidade com o TF1 no TF2 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>Carregamento/Inferência</td>
    <td>       Suporte total (<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">guia completo de carregamento de formato TF1 Hub</a>)       <pre style="font-size: 12px;" lang="python">m = hub.Module(handle)
outputs = m(inputs)</pre>
</td>
    <td> Recomenda-se usar  hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m.signatures["sig"](inputs)</pre>       ou hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle, signature="sig")
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Ajustes finos</td>
    <td>       Suporte total (<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">guia completo de ajustes finos do formato TF1 Hub</a>)     <pre style="font-size: 12px;" lang="python">m = hub.Module(handle,
               trainable=True,
               tags=["train"]*is_training)
outputs = m(inputs)</pre>       <div style="font-style: italic; font-size: 14px">       Observação: os módulos que não precisem de um grafo de treinamento separado não têm uma etiqueta de treinamento.       </div>
</td>
    <td style="text-align: center">       Sem suporte</td>
  </tr>
  <tr>
    <td>Criação</td>
    <td>Suporte total (confira o <a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">guia completo de criação do formato TF1 Hub</a>) <br> <div style="font-style: italic; font-size: 14px">       Observação: o formato TF1 Hub é destinado ao TF1, e há um suporte parcial no TF2. Considere criar um SavedModel do TF2.       </div>
</td>
    <td style="text-align: center">Sem suporte</td>
  </tr>
</table>

## Compatibilidade do SavedModel do TF2{:#compatibility_of_tf2_savedmodel}

Não há suporte antes do TF1.15.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">Operação</td>
    <td style="text-align: center; background-color: #D0D0D0">TF1.15/Modo de compatibilidade com o TF1 no TF2 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>Carregamento/Inferência</td>
    <td>       Use hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs)</pre>       ou hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle)
outputs = m(inputs)</pre>
</td>
    <td>Suporte total (<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">guia completo de carregamento de SavedModel do TF2</a>). Use hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs)</pre>       ou hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle)
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Ajustes finos</td>
    <td>       Suporte a um hub.KerasLayer usado no tf.keras.Model quando treinado com Model.fit() ou treinado em um Estimator cujo model_fn encapsule o modelo de acordo com o  <a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">guia de model_fn personalizado</a>.       <br><div style="font-style: italic; font-size: 14px;">         Observação: hub.KerasLayer <span style="font-weight: bold;">não</span>         preenche coleções de grafos como as APIs antiags tf.compat.v1.layers ou hub.Module faziam.       </div>
</td>
    <td>       Suporte total (<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">guia completo de ajustes finos de SavedModel do TF2</a>).       Use hub.load:       <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs, training=is_training)</pre>       ou hub.KerasLayer:       <pre style="font-size: 12px;" lang="python">m =  hub.KerasLayer(handle, trainable=True)
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>Criação</td>
    <td>      A API <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">       tf.saved_model.save()</a> do TF2 pode ser chamada no modo de compatibilidade.</td>
   <td>Suporte total (confira <a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">o guia completo de criação de SavedModel do TF2</a>)</td>
  </tr>
</table>

<p id="compatfootnote">[1] "Modo de compatibilidade com o TF1 no TF2" refere-se ao efeito combinado de importar o TF2 com <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code> e executar <code style="font-size: 12px;" lang="python">tf.disable_v2_behavior()</code> conforme descrito no <a href="https://www.tensorflow.org/guide/migrate">guia de migração do TensorFlow</a>.</p>
