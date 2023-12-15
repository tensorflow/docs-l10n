# Avaliação de modelos com o painel de controle de Fairness Indicators [Beta]

![Fairness Indicators](./images/fairness-indicators.png)

Fairness Indicators para o TensorBoard facilita a computação de métricas de imparcialidade frequentemente identificadas para classificadores *binários* e *multiclasse*. Com o plugin, você pode visualizar avaliações de imparcialidade para suas execuções e comparar facilmente o desempenho entre grupos.

Em especial, Fairness Indicators para o TensorBoard permite que você avalie e visualize o desempenho do modelo, dividido em grupos definidos de usuários. Sinta-se confiante em relação aos seus resultados com os intervalos de confiança e as avaliações de vários limites.

Várias ferramentas existentes para avaliar preocupações de imparcialidade não funcionam bem em modelos e datasets de grande escala. No Google, é importante termos ferramentas que funcionem em bilhões de sistemas de usuário. Com Fairness Indicators, você pode avaliar em qualquer tamanho de caso de uso, no ambiente do TensorBoard ou no [Colab](https://github.com/tensorflow/fairness-indicators).

## Requisitos

Para instalar Fairness Indicators para o TensorBoard, execute:

```
python3 -m virtualenv ~/tensorboard_demo
source ~/tensorboard_demo/bin/activate
pip install --upgrade pip
pip install fairness_indicators
pip install tensorboard-plugin-fairness-indicators
```

## Demonstração

Se você quiser testar Fairness Indicators no TensorBoard, pode baixar os resultados de avaliação de amostra da TensorFlow Model Analysis (eval_config.json, métricas e arquivos de plotagens), ou Análise de Modelo do TensorFlow, e um utilitário `demo.py` do Google Cloud Platform, [aqui](https://console.cloud.google.com/storage/browser/tensorboard_plugin_fairness_indicators/), usando o comando a seguir.

```
pip install gsutil
gsutil cp -r gs://tensorboard_plugin_fairness_indicators/ .
```

Navegue até o diretório que contém os arquivos baixados.

```
cd tensorboard_plugin_fairness_indicators
```

Esses dados de avaliação são baseados no [dataset Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), calculados usando a biblioteca [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) da TensorFlow Model Analysis. Ele também contém um arquivo de dados de resumo do TensorBoard para referência.

O utilitário `demo.py` escreve um arquivo de dados de resumo do TensorBoard, que será lido pelo TensorBoard para renderizar o painel de controle dos Fairness Indicators (Veja o [tutorial do TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/README.md) para mais informações sobre arquivos de dados de resumo).

Flags usadas com o utilitário `demo.py`:

- `--logdir`: diretório onde o TensorBoard escreverá o resumo
- `--eval_result_output_dir`: diretório com os resultados de avaliação da TFMA (baixados na última etapa)

Execute o utilitário `demo.py` para escrever os resultados de resumo no diretório do log:

`python demo.py --logdir=. --eval_result_output_dir=.`

Execute o TensorBoard:

Observação: para esta demonstração, execute o TensorBoard a partir do mesmo diretório que contém todos os arquivos baixados.

`tensorboard --logdir=.`

Isso iniciará uma instância local. Depois de iniciar essa instância, um link aparecerá no terminal. Abra o link no seu navegador para ver o painel de controle de Fairness Indicators.

### Colab de demonstração

[Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) contém uma demonstração completa para treinar e avaliar um modelo e visualizar os resultados da avaliação de imparcialidade no TensorBoard.

## Uso

Para usar Fairness Indicators com seus próprios dados e avaliações:

1. Treine um novo modelo e avalie usando a API `tensorflow_model_analysis.run_model_analysis` ou `tensorflow_model_analysis.ExtractEvaluateAndWriteResult` na [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py). Para conferir fragmentos de código de como fazer isso, veja o colab de Fairness Indicators [aqui](https://github.com/tensorflow/fairness-indicators).

2. Escreva o resumo de Fairness Indicators usando a API `tensorboard_plugin_fairness_indicators.summary_v2`.

    ```
    writer = tf.summary.create_file_writer(<logdir>)
    with writer.as_default():
        summary_v2.FairnessIndicators(<eval_result_dir>, step=1)
    writer.close()
    ```

3. Execute o TensorBoard

    - `tensorboard --logdir=<logdir>`
    - Selecione a nova execução da avaliação usando o menu suspenso no lado esquerdo do painel de controle para visualizar os resultados.
