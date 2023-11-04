# Agentes do TensorFlow

**Aprendizado por Reforço com o TensorFlow**

Os agentes facilitam a criação, a implementação e os testes de novos algoritmos de RL ao fornecer componentes modulares bem testados que podem ser modificados e estendidos. Isso permite a iteração de código rápida, com boa integração de teste e benchmarking.

Para começar, recomendamos conferir um dos nossos [tutoriais](/tutorials).

## Instalação

O TF-Agents publica builds noturnos e estáveis. Para uma lista de versões, leia a seção <a href="#Releases">Versões</a>. Os comandos abaixo abrangem a instalação de TF-Agents estável e noturna a partir de [pypi.org](https://pypi.org), assim como de um clone do GitHub.

> :aviso: Se estiver usando Reverb (buffer de replay), que é bastante comum, o TF-Agents só funcionará com o Linux.

> Observação: o Python 3.11 exige o pygame 2.1.3+.

### Estável

Execute os comandos abaixo para instalar a versão estável mais recente. A documentação da API para a versão está em [tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents).

```shell
$ pip install --user tf-agents[reverb]

# Use this tag get the matching examples and colabs.
$ git clone https://github.com/tensorflow/agents.git
$ cd agents
$ git checkout v0.17.0
```

Se você quiser instalar o TF-Agents com versões do TensorFlow ou [Reverb](https://github.com/deepmind/reverb) que estão sinalizadas como não compatíveis pela verificação de dependência pip, use o padrão abaixo por sua conta e risco.

```shell
$ pip install --user tensorflow
$ pip install --user dm-reverb
$ pip install --user tf-agents
```

Se você quiser usar o TF-Agents com o TensorFlow 1.15 ou 2.0, instale a versão 0.3.0:

```shell
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
$ pip install tensorflow-probability==0.8.0
$ pip install tf-agents==0.3.0
```

### Noturno

Os builds noturnos incluem recursos mais recentes, mas podem ser menos estáveis do que as versões versionadas. O build noturno é enviado como `tf-agents-nightly`. Sugerimos instalar versões noturnas do TensorFlow (`tf-nightly`) e TensorFlow Probability (`tfp-nightly`), já que são as versões testadas para o TF-Agents noturno.

Para instalar a versão de build noturno, execute o código a seguir:

```shell
# `--force-reinstall helps guarantee the right versions.
$ pip install --user --force-reinstall tf-nightly
$ pip install --user --force-reinstall tfp-nightly
$ pip install --user --force-reinstall dm-reverb-nightly

# Installing with the `--upgrade` flag ensures you'll get the latest version.
$ pip install --user --upgrade tf-agents-nightly
```

### A partir do GitHub

Depois de clonar o repositório, as dependências podem ser instaladas ao executar `pip install -e .[tests]`. O TensorFlow precisa ser instalado de maneira independente: `pip install --user tf-nightly`.

<a id="Contributing"></a>

## Contribuição

Estamos animados para colaborar com você! Veja em [`CONTRIBUTING.md`](https://github.com/tensorflow/agents/blob/master/CONTRIBUTING.md) um guia sobre como contribuir. Esse projeto adere ao [código de conduta](https://github.com/tensorflow/agents/blob/master/CODE_OF_CONDUCT.md) do TensorFlow. Ao participar, você deve seguir esse código.

<a id="Releases"></a>

## Versões

O TF Agents tem versões estáveis e noturnas. As versões noturnas são geralmente boas, mas podem ter problemas devido a bibliotecas upstream em fluxo. A tabela abaixo lista as versões do TensorFlow que se alinham a cada versão do TF Agents. Versões de interesse:

- 0.16.0 é a primeira versão compatível com o Python 3.11.
- 0.15.0 é a última versão compatível com o Python 3.7.
- Se estiver usando numpy &lt; 1.19, utilize o TF-Agents 0.15.0 ou mais recente.
- 0.9.0 é a última versão compatível com o Python 3.6.
- 0.3.0 é a última versão compatível com o Python 2.x.

Versão | Branch / Tag | Versão do TensorFlow | Versão dm-reverb
--- | --- | --- | ---
Noturna | [master](https://github.com/tensorflow/agents) | tf-nightly | dm-reverb-nightly
0.17.0 | [v0.17.0](https://github.com/tensorflow/agents/tree/v0.17.0) | 2.13.0 | 0.12.0
0.16.0 | [v0.16.0](https://github.com/tensorflow/agents/tree/v0.16.0) | 2.12.0 | 0.11.0
0.15.0 | [v0.15.0](https://github.com/tensorflow/agents/tree/v0.15.0) | 2.11.0 | 0.10.0
0.14.0 | [v0.14.0](https://github.com/tensorflow/agents/tree/v0.14.0) | 2.10.0 | 0.9.0
0.13.0 | [v0.13.0](https://github.com/tensorflow/agents/tree/v0.13.0) | 2.9.0 | 0.8.0
0.12.0 | [v0.12.0](https://github.com/tensorflow/agents/tree/v0.12.0) | 2.8.0 | 0.7.0
0.11.0 | [v0.11.0](https://github.com/tensorflow/agents/tree/v0.11.0) | 2.7.0 | 0.6.0
0.10.0 | [v0.10.0](https://github.com/tensorflow/agents/tree/v0.10.0) | 2.6.0 |
0.9.0 | [v0.9.0](https://github.com/tensorflow/agents/tree/v0.9.0) | 2.6.0 |
0.8.0 | [v0.8.0](https://github.com/tensorflow/agents/tree/v0.8.0) | 2.5.0 |
0.7.1 | [v0.7.1](https://github.com/tensorflow/agents/tree/v0.7.1) | 2.4.0 |
0.6.0 | [v0.6.0](https://github.com/tensorflow/agents/tree/v0.6.0) | 2.3.0 |
0.5.0 | [v0.5.0](https://github.com/tensorflow/agents/tree/v0.5.0) | 2.2.0 |
0.4.0 | [v0.4.0](https://github.com/tensorflow/agents/tree/v0.4.0) | 2.1.0 |
0.3.0 | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 e 2.0.0. |

<a id="Principles"></a>

## Princípios

Esse projeto adere aos [princípios de IA do Google](https://github.com/tensorflow/agents/blob/master/PRINCIPLES.md). Ao participar, usar ou contribuir com esse projeto, você deve seguir esses princípios.

<a id="Citation"></a>

## Citação

Se você usar esse código, cite desta maneira:

```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = {Sergio Guadarrama and Anoop Korattikara and Oscar Ramirez and
     Pablo Castro and Ethan Holly and Sam Fishman and Ke Wang and
     Ekaterina Gonina and Neal Wu and Efi Kokiopoulou and Luciano Sbaiz and
     Jamie Smith and Gábor Bartók and Jesse Berent and Chris Harris and
     Vincent Vanhoucke and Eugene Brevdo},
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```
