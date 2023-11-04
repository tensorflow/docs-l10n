# Práticas recomendadas para teste do TensorFlow

Estas são as práticas recomendadas para testar código no [repositório TensorFlow](https://github.com/tensorflow/tensorflow).

## Antes de começar

Antes de contribuir ao codigo fonte de um projeto TensorFlow, por favor revise o arquivo `CONTRIBUTING.md` no repositório GitHub do projeto. Por exemplo, veja o arquivo [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) para o repositório TensorFlow corre. Todos os contribuidores de código precisam assinar um [Contributor License Agreement](https://cla.developers.google.com/clas) (CLA).

## Princípios gerais

### Dependa apenas do que você usar nas suas regras de BUILD

TensorFlow é uma biblioteca grande e depender do pacote completo ao escrever um teste de unidade para seus submódulos tem sido uma prática comum. No entanto, isso desativa a análise `bazel`, baseada em dependências. Isto significa que os sistemas de integração contínua não poderão eliminar de forma inteligente testes não relacionados para execuções de pré-envio/pós-envio. Se você depender apenas dos submódulos que está testando no seu arquivo `BUILD`, economizará tempo de todos os desenvolvedores do TensorFlow, além de recursos valiosos de computação.

No entanto, modificar sua dependência de build para omitir os alvos TF completos traz algumas limitações quanto ao que você pode importar no seu código Python. Você não poderá mais usar a instrução `import tensorflow as tf` nos seus testes de unidade. Mas esta é uma compensação que vale a pena, pois evita que todos os desenvolvedores executem milhares de testes desnecessários.

### Todo código deve ter testes unitários

Para qualquer código que você escrever, você também deve escrever seus testes unitários. Se você escrever um novo arquivo `foo.py`, deverá colocar seus testes unitários em `foo_test.py` e enviá-lo com a mesma alteração. Procure garantir mais de 90% de cobertura de teste incremental para todo o seu código.

### Evite usar regras de teste nativas do Bazel no TF

TF introduz muitas sutilezas ao executar testes. Trabalhamos para ocultar todas essas complexidades nas nossas macros Bazel. Para evitar ter que lidar com isso, use o seguinte em vez das regras de teste nativas. Observe que todas são definidas em `tensorflow/tensorflow.bzl`. Para testes CC, use `tf_cc_test`, `tf_gpu_cc_test`, `tf_gpu_only_cc_test`. Para testes python, use `tf_py_test` ou `gpu_py_test`. Se você precisar de algo realmente próximo da regra `py_test` nativa, use aquela definida em tensorflow.bzl. Você só precisa adicionar a seguinte linha no topo do seu arquivo BUILD: `load(“tensorflow/tensorflow.bzl”, “py_test”)`

### Esteja ciente de onde o teste é executado

Quando você escreve um teste, nossa infraestrutura de teste pode cuidar da execução de seus testes em CPU, GPU e aceleradores, se você os escrever de acordo. Temos testes automatizados que rodam em Linux, macOS, windows, que possuem sistemas com ou sem GPUs. Você simplesmente precisa escolher uma das macros listadas acima e usar tags para limitar onde elas serão executadas.

- A tag `manual` impedirá que seu teste seja executado em qualquer lugar. Isto inclui execuções de testes manuais que usam padrões tais como `bazel test tensorflow/…`

- `no_oss` excluirá a execução do seu teste na infraestrutura de teste oficial do TF OSS.

- As tags `no_mac` ou `no_windows` podem ser usadas para excluir seu teste dos conjuntos de testes relevantes do sistema operacional.

- A tag `no_gpu` pode ser usada para impedir que seu teste seja executado em suítes de testes de GPU.

### Verifique os testes executados nas suites de testes esperadas

TF tem muitos conjuntos de testes. Às vezes, a configuração deles pode ser confusa. Pode haver diferentes problemas que fazem com que seus testes sejam omitidos de builds contínuos. Portanto, você deve verificar se seus testes estão sendo executados conforme o esperado. Para fazer isso:

- Aguarde até que os pré-envios do seu pull request (PR) sejam executados até o fim.
- Role até a parte inferior do seu PR para ver as verificações de status.
- Clique no link “Details” do lado direito de qualquer verificação Kokoro.
- Verifique a lista “Targets” para encontrar os alvos recém-adicionados.

### Cada classe/unidade deve ter seu próprio arquivo de teste unitário

Classes de teste separadas nos ajudam a isolar melhor falhas e recursos. Elas levam a arquivos de teste muito mais curtos e fáceis de ler. Portanto, todos os seus arquivos Python devem ter pelo menos um arquivo de teste correspondente (para cada `foo.py`, deve haver ukm `foo_test.py`). Para testes mais elaborados, como testes de integração que exigem configurações diferentes, não há problema em adicionar mais arquivos de teste.

## Velocidade e tempos de execução

### A fragmentação deve ser usada o mínimo possível

Em vez de fragmentar, considere:

- Deixar seus testes menores
- Se o acima não for possível, divida os testes

A fragmentação ajuda a reduzir a latência geral de um teste, mas o mesmo pode ser alcançado dividindo os testes em alvos menores. A divisão de testes nos dá um nível mais preciso de controle em cada teste, minimizando execuções desnecessárias de pré-envio e reduzindo a perda de cobertura de um buildcop que desativa um alvo inteiro devido a um caso de teste com comportamento incorreto. Além disso, a fragmentação introduz custos ocultos que não são tão óbvios, como a execução de todo o código de inicialização de teste para todos os fragmentos. Esse problema nos foi encaminhado pelas equipes de infra-estrutura como uma fonte que cria carga extra.

### Testes menores são melhores

Quanto mais rápido seus testes forem executados, maior será a probabilidade de as pessoas os executarem. Um segundo adicional no seu teste pode se acumular em horas adicionais gastas na execução do teste pelos desenvolvedores e pela nossa infraestrutura. Tente fazer com que seus testes sejam executados em menos de 30 segundos (no modo não opcional!) E faça com que sejam pequenos. Marque seus testes como médios apenas como último recurso. A infra não executa testes grandes como pré-envios ou pós-envios! Portanto, só escreva um teste grande se você for cuidar de onde ele será executado. Algumas dicas para deixar os testes mais rápidos:

- Execute menos iterações de treinamento em seu teste
- Considere usar injeção de dependências para substituir dependências pesadas do sistema em teste por dependências falsas, mais simples.
- Considere usar dados de entrada menores em testes unitários
- Se nada mais funcionar, tente dividir seu arquivo de teste.

### Os tempos de teste devem ter como objetivo metade do tempo limite do tamanho do teste para evitar instabilidade

Com alvos de teste `bazel`, testes pequenos têm tempo limite de 1 minuto. O tempo limite de testes médios é de 5 minutos. Testes grandes simplesmente não são executados pela infra de teste do TensorFlow. No entanto, muitos testes não são determinísticos quanto ao tempo que demoram. Por vários motivos, seus testes podem levar mais tempo de vez em quando. E, se você marcar um teste que é executado em média por 50 segundos como pequeno, seu teste falhará se for agendado numa máquina com uma CPU antiga. Portanto, busque um tempo médio de execução de 30 segundos para testes pequenos. Tente buscar 2 minutos e 30 segundos de tempo médio de execução para testes médios.

### Reduza o número de amostras e aumente as tolerâncias para treinamentos

Testes de execução lenta desencorajam os contribuidores. A execução do treinamento em testes pode ser muito lenta. Prefira tolerâncias mais altas para poder usar menos amostras em seus testes e mantê-los suficientemente rápidos (2,5 minutos no máximo).

## Elimine o não-determinismo e instabilidade

### Escreva testes determinísticos

Os testes unitários devem sempre ser determinísticos. Todos os testes executados no TAP and guitar devem ser executados da mesma maneira todas as vezes, se não houver nenhuma alteração no código que os afete. Para garantir isso, a seguir estão alguns pontos a serem considerados.

### Sempre defina sementes em qualquer fonte de estocasticidade

Qualquer gerador de números aleatórios ou qualquer outra fonte de estocasticidade pode causar instabilidade. Portanto, cada um deles deve ser gerado com sementes. Além de tornar os testes menos fragmentados, isso permite que todos os testes sejam reproduzíveis. Diferentes maneiras de definir algumas sementes que você pode precisar definir nos testes TF são:

```python
# Python RNG
import random
random.seed(42)

# Numpy RNG
import numpy as np
np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)
```

### Evite usar `sleep` em testes multithread

Usar a função de `sleep` em testes pode ser uma das principais causas de instabilidade. Especialmente ao usar múltiplos threads, o uso de sleep para esperar por outro thread jamais será determístico. Isso se deve ao fato de o sistema não ser capaz de garantir qualquer ordem de execução de diferentes threads ou processos. Portanto, prefira construções de sincronização determinísticas, como os mutexes.

### Verifique se o teste está instável

A instabilidade faz com que os buildcops e os desenvolvedores percam muitas horas. Eles são difíceis de detectar e difíceis de depurar. Embora existam sistemas automatizados para detectar instabilidade, eles precisam acumular centenas de execuções de testes antes de poder colocar esses testes numa lista de rejeição (denylist) com precisão. Mesmo quando detectam tais testes, eles os colocam na lista de rejeição e a cobertura do teste é perdida. Portanto, os autores dos testes devem verificar se seus testes são instáveis ​​ao escrevê-los. Isto pode ser feito executando seu teste com o sinalizador: `--runs_per_test=1000`

### Use TensorFlowTestCase

O TensorFlowTestCase toma as precauções necessárias, como propagar todos os geradores de números aleatórios usados ​​para reduzir ao máximo a instabilidade. À medida que descobrimos e corrigimos mais fontes de instabilidade, todas elas serão adicionadas ao TensorFlowTestCase. Portanto, você deve usar TensorFlowTestCase ao escrever testes para tensorflow. O TensorFlowTestCase é definido aqui: `tensorflow/python/framework/test_util.py`

### Escreva testes herméticos

Os testes herméticos não necessitam de recursos externos. Eles vêm com tudo de que precisam e simplesmente iniciam quaisquer serviços falsos de que possam precisar. Quaisquer serviços que não sejam seus testes são fontes de não determinismo. Mesmo com 99% de disponibilidade de outros serviços, a rede pode falhar, a resposta do RPC pode ser atrasada e você pode acabar com uma mensagem de erro inexplicável. Os serviços externos podem ser, mas não limitados a, GCS, S3 ou qualquer site.
