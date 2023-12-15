# Orquestração de pipelines TFX

## Kubeflow Pipelines

O [Kubeflow](https://www.kubeflow.org/) é uma plataforma de ML de código aberto dedicada a deixar as implantações de workflows de aprendizado de máquina (ML) no Kubernetes mas simples, portáteis e escalonáveis. O [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-overview/) é uma parte da plataforma Kubeflow que permite a composição e execução de workflows reproduzíveis no Kubeflow, integrados com experimentação e experiências baseadas em notebooks. Os serviços Kubeflow Pipelines no Kubernetes incluem a hospedagem do TF Metadata, mecanismo de orquestração baseado em container, servidor de notebooks e interface do usuário para ajudar os usuários a desenvolver, executar e gerenciar pipelines de ML complexos em escala. O SDK do Kubeflow Pipelines permite a criação e o compartilhamento de componentes, além da composição programática de pipelines.

Veja o [exemplo do TFX no Kubeflow Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines) para detalhes sobre como executar o TFX em escala na nuvem do Google.
