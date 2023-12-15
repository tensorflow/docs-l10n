<div align="center">   <img src="https://tensorflow.org/images/SIGAddons.png" width="60%"><br><br>
</div>

---

# TensorFlow Addons

O **TensorFlow Addons** é um repositório de contribuições que estão em conformidade com padrões bem estabelecidos de APIs e implementam novas funcionalidades não disponíveis no núcleo do TensorFlow. O TensorFlow nativamente oferece suporte a um grande número de operadores, camadas, métricas, funções de perda e otimizadores. Porém, em um campo de rápida evolução como o aprendizado de máquina, há vários novos avanços interessantes que não podem ser integrados ao núcleo do TensorFlow (porque a extensão da aplicabilidade deles ainda não está clara ou porque são usados por apenas um pequeno grupo da comunidade).

## Instalação

#### Compilações estáveis

Para instalar a versão mais recente, execute este código:

```
pip install tensorflow-addons
```

Como usar os complementos:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Builds noturnas

Também há builds noturnas do TensorFlow Addons no pacote pip `tfa-nightly`, que é baseado na última versão estável do TensorFlow. As builds noturnas incluem recursos mais recentes, mas podem ser menos estáveis do que as versões normais.

```
pip install tfa-nightly
```

#### Instalação a partir do código-fonte

Você também pode instalar a partir do código-fonte. Isso exige o sistema de build [Bazel](https://bazel.build/).

```
git clone https://github.com/tensorflow/addons.git
cd addons

# If building GPU Ops (Requires CUDA 10.0 and CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_TOOLKIT_PATH="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## Principais conceitos

#### API padronizada em subpacotes

A experiência do usuário e a capacidade de sustentar o projeto são conceitos centrais no TF-Addons. Para obtê-los, exigimos que nossos complementos estejam em conformidade com os padrões de API estabelecidos vistos no núcleo do TensorFlow.

#### Ops personalizadas de GPU/CPU

Um grande benefício do TensorFlow Addons é que ele inclui ops pré-compiladas. Se uma instalação do CUDA 10 não for encontrada, a op usará automaticamente a implementação de CPU.

#### Manutenção de proxy

O Addons foi projetado para compartimentar subpacotes e submódulos para que possam ser mantidos pelos usuários com experiência e interesse no componente.

A manutenção de subpacotes só será concedida após contribuições substanciais para limitar o número de usuários com permissão para escrever. As contribuições podem ser encerramentos de issues, correções de bugs, documentação, código novo ou otimização de código existente. A manutenção de submódulos pode ser concedida com menos barreiras de entrada, já que não inclui permissões para escrita no repositório.

Para mais informações, veja o [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) sobre esse assunto.

#### Avaliação periódica de subpacotes

Considerando a natureza desse repositório, subpacotes e submódulos podem se tornar cada vez menos úteis para a comunidade com o passar do tempo. Para a sustentabilidade do repositório, vamos fazer revisões bianuais do código, garantindo que seja adequado ao repositório. Os fatores que contribuem para essa revisão são:

1. Número de pessoas que fazem manutenção ativa
2. Quantidade de uso do OSS
3. Número de issues ou bugs atribuídos ao código
4. Disponibilidade de uma solução melhor

A funcionalidade no TensorFlow Addons pode ser categorizada em três grupos:

- **Recomendada**: API com boa manutenção; o uso é incentivado.
- **Desaconselhada**: está disponível uma alternativa melhor; a API é mantida por fins históricos; ou a API requer manutenção e está no período de espera para ser descontinuada.
- **Descontinuada**: use por sua conta e risco; sujeita a ser excluída.

A mudança de status entre esses três grupos é: Recomendada &lt; - &gt; Desaconselhada -&gt; Descontinuada.

O período entre a marcação da API como descontinuada e a exclusão dela será de 90 dias. A razão disso:

1. Caso o TensorFlow Addons faça lançamentos mensais, haverá 2 ou 3 versões antes que a API seja excluída. As notas da versão podem avisar o usuário a tempo.

2. 90 dias dá aos usuários responsáveis pela manutenção bastante tempo para corrigir o código.

## Contribuição

O TF-Addons é um projeto de código aberto liderado pela comunidade. Por isso, ele depende de contribuições públicas, correções de bugs e documentação. Veja as [diretrizes de contribuição](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md) para um guia sobre como contribuir. Esse projeto adere ao [código de conduta do TensorFlow](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md). Com sua participação, espera-se que você siga esse código.

## Comunidade

- [Lista de correspondência pública](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
- [Notas de encontros mensais do SIG](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    - Participe da nossa lista de correspondência e receba convites de agenda para os encontros

## Licença

[Licença Apache 2.0](LICENSE)
