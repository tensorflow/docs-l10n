# Compreensão do modelo com o painel da Ferramenta What-If

> **Aviso**: esta documentação só se aplica ao TensorBoard 2.11 e versões mais antigas, porque a Ferramenta What-If não é mais mantida ativamente. Em vez disso, confira a [Ferramenta de Interpretabilidade de Aprendizado (LIT)](https://pair-code.github.io/lit/).

![Ferramenta What-If](./images/what_if_tool.png)

A Ferramenta What-If (WIT) fornece uma interface fácil de usar para melhorar a compreensão sobre modelos de ML de regressão e classificação de caixa preta. Com o plugin, você pode realizar a inferência em um grande conjunto de exemplos e visualizar imediatamente os resultados de diversas formas. Além disso, os exemplos podem ser editados de forma manual ou programática e serem executados novamente através do modelo para ver os resultados das mudanças. Ela contém ferramentas para investigar o desempenho e a imparcialidade do modelo em subconjuntos de um dataset.

A finalidade da ferramenta é proporcionar uma maneira simples, intuitiva e avançada de explorar e investigar modelos de ML treinados em uma interface visual com absolutamente nenhum código necessário.

A ferramenta pode ser acessada pelo TensorBoard ou diretamente em um notebook do Jupyter ou do Colab. Para mais detalhes, demonstrações, tutoriais e informações específicas do uso da WIT no modo notebook, confira o [site da Ferramenta What-If](https://pair-code.github.io/what-if-tool).

## Requisitos

Para usar a WIT no TensorBoard, são necessárias duas coisas:

- Os modelos que você quer explorar precisam ser servidos usando o [TensorFlow Serving](https://github.com/tensorflow/serving) com a API de classificação, regressão ou previsão.
- O dataset que será inferido pelos modelos precisam estar em um arquivo TFRecord acessível ao servidor da Web do TensorBoard.

## Uso

Ao abrir o painel da Ferramenta What-If no TensorBoard, você verá uma tela de configuração onde fornecerá o host e a porta do servidor do modelo, o nome do modelo servido, o tipo de modelo e o caminho do arquivo TFRecords que será carregado. Depois de preencher essas informações e clicar em "Aceitar", a WIT carregará o dataset e realizará a inferência com o modelo, exibindo os resultados.

Para detalhes sobre os diferentes recursos da WIT e como eles podem ajudar na compreensão do modelo e nas investigações de imparcialidade, veja o tutorial no [site da Ferramenta What-If](https://pair-code.github.io/what-if-tool).

## Modelo e dataset de demonstração

Se você quiser testar a WIT no TensorBoard com um modelo pré-treinado, pode baixar e descompactar um dataset e modelo pré-treinado em https://storage.googleapis.com/what-if-tool-resources/uci-census-demo/uci-census-demo.zip. É um modelo de classificação binária que usa o dataset [UCI Census](https://archive.ics.uci.edu/ml/datasets/census+income) para prever se uma pessoa ganha mais de US$ 50.000 por ano. Esse dataset e tarefa de previsão são geralmente usados na pesquisa de modelos de aprendizado de máquina e imparcialidade.

Defina a variável de ambiente MODEL_PATH como o local do diretório do modelo resultante na sua máquina.

Instale o docker e o TensorFlow Serving seguindo a [documentação oficial](https://www.tensorflow.org/tfx/serving/docker).

Sirva o modelo usando o docker através de `docker run -p 8500:8500 --mount type=bind,source=${MODEL_PATH},target=/models/uci_income -e MODEL_NAME=uci_income -t tensorflow/serving`. Talvez seja necessário executar o comando com `sudo` dependendo da sua configuração do docker.

Agora, inicialize o TensorBoard e use o menu suspenso do painel para acessar a Ferramenta What-If.

Na tela de configuração, defina o endereço de inferência como "localhost:8500", o nome do modelo como "uci_income" e o caminho dos exemplos como o caminho completo do arquivo `adult.tfrecord` baixado. Depois, pressione "Accept" (Aceitar).

![Tela de configuração para a demonstração](./images/what_if_tool_demo_setup.png)

Exemplos do que você pode testar com a Ferramenta What-If nesta demonstração:

- Editar um único ponto de dados e ver a mudança de resultado na inferência.
- Explorar a relação entre características individuais no dataset e os resultados de inferência do modelo através de plotagens de dependências parciais.
- Dividir o dataset em subsets e comparar o desempenho das fatias.

Para uma análise detalhada dos recursos da ferramenta, confira o [tutorial da Ferramenta What-If](https://pair-code.github.io/what-if-tool/walkthrough.html).

Observe que a característica de verdade absoluta no dataset que esse modelo está tentando prever é chamada de "Target" (Alvo), então, ao usar a guia "Performance &amp; Fairness" (Desempenho e imparcialidade), "Target" é o que você quer especificar no menu suspenso da característica de verdade absoluta.
