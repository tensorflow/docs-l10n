# Compile o pacote Wheel do Python do TensorFlow Lite

Esta página descreve como compilar a biblioteca do Python `tflite_runtime` do TensorFlow Lite para dispositivos x86_64 e diversos dispositivos ARM.

As instruções abaixo foram testadas em um PC (AMD64) com Ubuntu 16.04.3 de 64 bits, macOS Catalina (x86_64) e imagem devel docker do TensorFlow [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Observação:** esse recurso está disponível a partir a versão 2.4.

#### Pré-requisitos

Você precisa instalar o CMake e de uma cópia do código fonte do TensorFlow. Confira mais detalhes na página [Compile o TensorFlow Lite com o CMake](https://www.tensorflow.org/lite/guide/build_cmake).

Para compilar o pacote PIP para sua estação de trabalho, você pode executar os seguintes comandos:

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**Observação:** se você tiver vários interpretadores do Python disponíveis, especifique a versão do Python exata pela variável  `PYTHON` (atualmente, há suporte ao Python 3.7 e superiores).

## Compilação cruzada para ARM

No caso de compilação cruzada para ARM, recomenda-se usar o Docker, pois facilita a configuração do ambiente de compilação cruzada. Além disso, você precisa de uma opção `target` para descobrir a arquitetura alvo.

Existe uma ferramenta helper no Makefile `tensorflow/lite/tools/pip_package/Makefile` disponível para invocar um comando de compilação usando um container Docker predefinido. Em uma máquina host Docker, você pode executar um comando de compilação da seguinte forma:

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**Observação:** há suporte ao Python versão 3.7 e superiores.

### Nomes de alvos disponíveis

O script `tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` requer um nome de alvo para descobrir a arquitetura alvo. Veja a lista de alvos com suporte:

Alvo | Arquitetura alvo | Comentários
--- | --- | ---
armhf | ARMv7 VFP com Neon | Compatível com Raspberry Pi 3 e 4
rpi0 | ARMv6 | Compatível com Raspberry Pi Zero
aarch64 | aarch64 (ARM de 64 bits) | [Coral Mendel Linux 4.0](https://coral.ai/) <br> Raspberry Pi com [Ubuntu Server 20.04.01 LTS de 64 bits](https://ubuntu.com/download/raspberry-pi)
native | Sua estação de trabalho | Compila com a otimização "-mnative"
<default></default> | Sua estação de trabalho | Alvo padrão

### Exemplos de builds

Veja alguns comandos que você pode usar.

#### Alvo armhf para Python 3.7

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### Alvo aarch64 para Python 3.8

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### Como usar uma toolchain personalizada?

Se os binários gerados não forem compatíveis com seu alvo, você precisa usar sua própria toolchain e fornecer sinalizadores de compilação personalizados (confira [aqui](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment) para entender seu ambiente alvo). Nesse caso, você precisa modificar `tensorflow/lite/tools/cmake/download_toolchains.sh` para usar sua própria toolchain. O script da toolchain define as duas variáveis abaixo para o script `build_pip_package_with_cmake.sh`.

Variável | Finalidade | Exemplo
--- | --- | ---
`ARMCC_PREFIX` | Define o prefixo da toolchain | arm-linux-gnueabihf-
`ARMCC_FLAGS` | Sinalizadores de compilação | -march=armv7-a -mfpu=neon-vfpv4

**Observação:** pode ser que `ARMCC_FLAGS` precise conter o caminho de inclusão da biblioteca do Python. Confira a referência em `download_toolchains.sh`.
