# Agentes de TensorFlow

**Aprendizaje de refuerzo con TensorFlow**

Agents facilita el diseño, la implementación y la prueba de nuevos algoritmos de RL, ya que ofrece componentes modulares bien verificados que pueden modificarse y ampliarse. Permite una rápida iteración del código, con una buena integración de pruebas y evaluación comparativa.

Para empezar, le recomendamos que consulte uno de nuestros [tutoriales](/tutorials).

## Instalación

TF-Agenst publica versiones nocturnas y estables. Para ver la lista de versiones, consulte la sección <a href="#Releases">Versiones</a>. Los siguientes comandos abordan la instalación de las versiones estable y nocturna de TF-Aegnts desde [pypi.org](https://pypi.org) y desde un clon de GitHub.

> Advertencia: Si usa Reverb (búfer de repetición), lo cual es muy común, TF-Agents solo funcionará con Linux.

> Nota: Python 3.11 requiere pygame 2.1.3+.

### Estable

Ejecute los siguientes comandos para instalar la versión estable más reciente. Puede encontrar la documentación de la API para esta versión en [tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents).

```shell
$ pip install --user tf-agents[reverb]

# Use this tag get the matching examples and colabs.
$ git clone https://github.com/tensorflow/agents.git
$ cd agents
$ git checkout v0.17.0
```

Si desea instalar TF-Agents con versiones de Tensorflow o [Reverb](https://github.com/deepmind/reverb) que la comprobación de dependencias de pip haya marcado como no compatibles, utilice el siguiente patrón bajo su propia responsabilidad.

```shell
$ pip install --user tensorflow
$ pip install --user dm-reverb
$ pip install --user tf-agents
```

SI desea usar TF-Agents con TensorFlow 1.15 o 2.0, instale la versión 0.3.0:

```shell
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
$ pip install tensorflow-probability==0.8.0
$ pip install tf-agents==0.3.0
```

### Nocturna

Las compilaciones nocturnas incluyen características más nuevas, pero podrían ser menos estables que las versiones publicadas. La compilación nocturna se envía como `tf-agents-nightly`. Sugerimos instalar las versiones nocturnas de TensorFlow (`tf-nightly`) y TensorFlow Probability (`tfp-nightly`) ya que son las versiones con las que se prueban las versiones nocturnas de TF-Agents.

Para instalar la versión de compilación nocturna, ejecute el siguiente comando:

```shell
# `--force-reinstall helps guarantee the right versions.
$ pip install --user --force-reinstall tf-nightly
$ pip install --user --force-reinstall tfp-nightly
$ pip install --user --force-reinstall dm-reverb-nightly

# Installing with the `--upgrade` flag ensures you'll get the latest version.
$ pip install --user --upgrade tf-agents-nightly
```

### Desde GitHub

Tras clonar el repositorio, puede instalar las dependencias mediante la ejecución de `pip install -e .[tests]`. TensorFlow se debe instalar de forma independiente: `pip install --user tf-nightly`.

<a id="Contributing"></a>

## Contribución

¡Estamos ansiosos por colaborar con usted! Consulte [`CONTRIBUTING.md`](https://github.com/tensorflow/agents/blob/master/CONTRIBUTING.md) para acceder a una guía sobre cómo contribuir. Este proyecto se rige por el [código de conducta](https://github.com/tensorflow/agents/blob/master/CODE_OF_CONDUCT.md) de TensorFlow. Se espera que, al participar, respete este código.

<a id="Releases"></a>

## Versiones

TF Agents tiene versiones estables y nocturnas. Las versiones nocturnas generalmente son buenas, pero pueden tener problemas debido al flujo de las bibliotecas ascendentes. La siguiente tabla enumera las versiones de TensorFlow que corresponden a las versiones de TF Agents. Versiones publicadas de interés:

- 0.16.0 es la primera versión compatible con Python 3.11.
- 0.15.0 es la última versión compatible con Python 3.7.
- Si usa numpy &lt; 1.19, entonces use TF-Agents 0.15.0 o versiones anteriores.
- 0.9.0 es la última versión compatible con Python 3.6.
- 0.3.0 es la última versión compatible con Python 2.x.

Versión | Rama / Etiqueta | Versión de TensorFlow | Versión de dm-reverb
--- | --- | --- | ---
Nocturna | [master](https://github.com/tensorflow/agents) | tf-nightly | dm-reverb-nightly
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
0.3.0 | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 y 2.0.0. |

<a id="Principles"></a>

## Principios

Este proyecto se rige por los [principios de IA de Google](https://github.com/tensorflow/agents/blob/master/PRINCIPLES.md). Al participar de este proyecto, usarlo o contribuir con él, se espera que adhiera a estos principios.

<a id="Citation"></a>

## Cita

Si usa este código, cítelo de la siguiente manera:

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
