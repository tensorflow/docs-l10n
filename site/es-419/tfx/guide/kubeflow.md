# Cómo orquestar canalizaciones de TFX

## Kubeflow Pipelines

[Kubeflow](https://www.kubeflow.org/) es una plataforma de aprendizaje automático de código abierto dedicada a simplificar las implementaciones de flujos de trabajo de aprendizaje automático (ML) en Kubernetes, además de hacer que sean portátiles y escalables. [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-overview/) es parte de la plataforma Kubeflow que permite componer y ejecutar flujos de trabajo reproducibles en Kubeflow, integrados con experimentación y experiencias basadas en blocs de notas. Los servicios de Kubeflow Pipelines en Kubernetes incluyen el almacén de metadatos alojado, el motor de orquestación basado en contenedores, el servidor portátil y la interfaz de usuario para ayudar a los usuarios a desarrollar, ejecutar y administrar canalizaciones de aprendizaje automático complejas a escala. El SDK de Kubeflow Pipelines permite crear y compartir componentes, así como componer y canalizar de forma programática.

Consulte el [ejemplo de TFX en Kubeflow Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines) para obtener información sobre cómo ejecutar TFX a escala en la nube de Google.
