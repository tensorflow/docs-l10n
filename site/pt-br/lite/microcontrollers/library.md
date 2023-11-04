# Sobre a biblioteca do C++

A biblioteca do C++ do TensorFlow Lite para Microcontroladores faz parte do [repositório do TensorFlow](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro). Ela é fácil ler, de modificar, bem testada e fácil de integrar, além de ser compatível com o TensorFlow Lite comum.

Este documento descreve a estrutura básica da biblioteca do C++ e apresenta informações sobre como criar seu próprio projeto.

## Estrutura do arquivo

O diretório raiz [`micro`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro) tem uma estrutura relativamente simples. Porém, como fica localizado dentro do extenso repositório do TensorFlow, criamos scripts e arquivos de projeto pré-gerados que fornecem os arquivos fonte relevantes isoladamente dentro de diversos ambientes de desenvolvimento embarcados.

### Principais arquivos

Os arquivos mais importantes para usar o interpretador do TensorFlow Lite para Microcontroladores ficam localizados na raiz do projeto, acompanhados de testes:

```
[`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h)
can be used to provide the operations used by the interpreter to run the
model.
```

- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h) – gera como saída informações de depuração.
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h) – contém código para testar e executar modelos.

Confira detalhes do uso geral em [Introdução aos microcontroladores](get_started_low_level.md).

O sistema de compilação fornece implementações de determinados arquivos para plataformas específicas, que ficam localizadas no diretório com o nome da plataforma. Por exemplo: [`cortex-m`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_generic).

Diversos outros diretórios existem, incluindo:

- [`kernel`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels), que contém implementações de operações e o código subjacente.
- [`tools`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools), que contém ferramentas de compilação e suas saídas.
- [`examples`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples), que contém exemplos de código.

## Comece um novo projeto

Recomendamos usar o exemplo *Hello World* (Olá, mundo) como template para novos projetos. Para obter uma versão para a plataforma escolhida, basta seguir as instruções desta seção.

### Use a biblioteca do Arduino

Se você estiver usando o Arduino, o exemplo *Hello World* está incluído na biblioteca `Arduino_TensorFlowLite` do Arduino, que você pode instalar manualmente no Arduino IDE e no [Arduino Create](https://create.arduino.cc/).

Após adicionar a biblioteca, acesse `File -> Examples` (Arquivo -&gt; Exemplos). Você deverá ver um exemplo perto da parte inferior da lista com nome `TensorFlowLite:hello_world`. Selecione-o e clique em `hello_world` para carregar o exemplo. Em seguida, você pode salvar uma cópia do exemplo e usá-lo como base do seu próprio projeto.

### Gere projetos para outras plataformas

O TensorFlow Lite para Microcontroladores consegue gerar projetos standalone que contêm todos os arquivos fonte necessário usando um `Makefile`. No momento, os ambientes com suporte são Keil, Make e Mbed.

Para gerar esses projetos com o Make, clone o [repositório TensorFlow/tflite-micro](https://github.com/tensorflow/tflite-micro) e execute o seguinte comando:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

Vai levar alguns minutos, já que é preciso baixar algumas toolchains grandes para as dependências. Após o término, você verá algumas pastas criadas dentro de um caminho como `gen/linux_x86_64/prj/` (o caminho exato depende do sistema operacional do host). Essas pastas contêm o projeto gerado e os arquivos fonte.

Após executar o comando, você encontrará os projetos de *Hello World* em `gen/linux_x86_64/prj/hello_world`. Por exemplo: `hello_world/keil` conterá o projeto para Keil.

## Execute os testes

Para compilar a biblioteca e executar todos os testes de unidade, use o seguinte comando:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

Para executar um teste específico, use o comando abaixo, substituindo `<test_name>` pelo nome do teste:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

Os nomes dos testes estão disponíveis nos Makefiles do projeto. Por exemplo: `examples/hello_world/Makefile.inc` especifica os nomes de testes para o exemplo *Hello World*.

## Compile os binários

Para compilar um binário executável para um determinado projeto (como uma aplicação de exemplo), use o comando abaixo, substituindo `<project_name>` pelo projeto que deseja compilar:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

Por exemplo: o comando abaixo compilará um binário para a aplicação *Hello World*:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

Por padrão, o projeto será compilado para o sistema operacional do host. Para especificar um arquitetura alvo diferente, use `TARGET=` e `TARGET_ARCH=`. O exemplo abaixo mostra como compilar o exemplo *Hello World* para um cortex-m0 genérico:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m0 hello_world_bin
```

Quando um alvo é especificado, todos os arquivos fonte específicos para o alvo disponíveis serão usados no lugar do código original. Por exemplo: o subdiretório `examples/hello_world/cortex_m_generic` contém implementações dos arquivos `constants.cc` e `output_handler.cc` para SparkFun Edge, que serão usados quando o alvo `cortex_m_generic` for especificado.

Os nomes dos projetos estão disponíveis nos Makefiles do projeto. Por exemplo: `examples/hello_world/Makefile.inc` especifica os nomes dos binários para o exemplo *Hello World*.

## Kernels otimizados

Os kernels de referência na raiz de `tensorflow/lite/micro/kernels` são implementados em C/C++ puro e não incluem otimizações de hardware de plataformas específicas.

São fornecidas versões otimizadas dos kernels nos subdiretórios. Por exemplo: `kernels/cmsis-nn` contém diversos kernels otimizados que usam a biblioteca CMSIS-NN do Arm.

Para gerar projetos usando kernels otimizados, use o comando abaixo, substituindo `<subdirectory_name>` pelo nome do subdiretório que contém as otimizações:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

Para adicionar suas próprias otimizações, basta criar uma nova subpasta para elas. Sugerimos que sejam usados pull requests para novas implementações otimizadas.

## Gere a biblioteca do Arduino

Se você precisar gerar uma nova build da biblioteca, pode usar o seguinte script do repositório do TensorFlow:

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

A biblioteca resultante está disponível em `gen/arduino_x86_64/prj/tensorflow_lite.zip`.

## Portabilidade para novos dispositivos

Confira as orientações sobre como fazer a portabilidade do TensorFlow Lite para Microcontroladores para novas plataformas e dispositivos em [`micro/docs/new_platform_support.md`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md).
