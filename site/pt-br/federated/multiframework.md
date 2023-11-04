# Suporte multiframework no TensorFlow Federated

O TensorFlow Federated (TFF) foi projetado para oferecer suporte a uma ampla variedade de computações federadas, expressas por meio de uma combinação de operadores federados do TFF que modelam a comunicação distribuída e a lógica de processamento local.

Atualmente, a lógica de processamento local pode ser expressa usando APIs do TensorFlow (via `@tff.tf_computation`) no front-end e é executada através do runtime do TensorFlow no back-end. No entanto, pretendemos oferecer suporte a vários outros frameworks de front-end e back-end (não TensorFlow) para computações locais, incluindo frameworks que não são de aprendizado de máquina (por exemplo, para lógica expressa em SQL ou linguagens de programação de uso geral).

Nesta seção, incluiremos informações sobre:

- Mecanismos que o TFF fornece para oferecer suporte a frameworks alternativos e como você pode incluir suporte para seu tipo preferido de front-end ou back-end ao TFF.

- Implementações experimentais de suporte para frameworks não TensorFlow, com exemplos.

- Roteiro futuro provisório para graduar essas capacidades além do status experimental.
