# **Tutorial TFX Airflow**

## Visão geral

## Visão geral

Este tutorial foi desenvolvido para ajudar você a aprender a criar seus próprios pipelines de machine learning usando o TensorFlow Extended (TFX) e o Apache Airflow como orquestrador. Ele é executado no Vertex AI Workbench e mostra integração com TFX e TensorBoard, bem como interação com TFX em um ambiente Jupyter Lab.

### O que você vai fazer?

Você aprenderá como criar um pipeline de ML usando TFX

- Um pipeline TFX é um gráfico acíclico direcionado ou "DAG". Freqüentemente nos referiremos aos pipelines como DAGs.
- Os pipelines TFX são apropriados quando você for implantar um aplicativo de ML de produção
- Os pipelines TFX são apropriados quando os datasets são grandes ou podem crescer muito
- Os pipelines TFX são apropriados quando a consistência de treinamento/serviço é importante
- Os pipelines do TFX são apropriados quando o gerenciamento de versões para inferência é importante
- O Google usa pipelines TFX para ML em produção

Consulte o [Guia do usuário do TFX](https://www.tensorflow.org/tfx/guide) para saber mais.

Você seguirá um processo típico de desenvolvimento de ML:

- Ingerindo, compreendendo e limpando dados
- Engenharia de características
- Treinamento
- Analisando o desempenho de modelos
- Ensaboar, enxaguar, repetir
- Pronto para entrar em produção

## **Apache Airflow para orquestração de pipelines**

Os orquestradores do TFX são responsáveis ​​por agendar componentes do pipeline TFX com base nas dependências definidas pelo pipeline. O TFX foi projetado para ser portável para vários ambientes e frameworks de orquestração. Um dos orquestradores padrão suportados pelo TFX é o [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow). Este laboratório ilustra o uso do Apache Airflow para orquestração de um pipeline TFX. O Apache Airflow é uma plataforma para criar, agendar e monitorar workflows de forma programática. O TFX usa o Airflow para criar workflows como grafos acíclicos direcionados (DAGs) de tarefas. A sua rica interface de usuário facilita a visualização de pipelines rodando em produção, o monitoramento do progresso e a solução de problemas quando necessário. Os workflows do Apache Airflow são definidos em forma de código. Isso os torna mais fáceis de manter, versionáveis, testáveis ​​e colaborativos. O Apache Airflow é ideal para pipelines de processamento em lote. É leve e fácil de aprender.

Neste exemplo, executaremos um pipeline TFX numa instância configurando o Airflow manualmente.

Os outros orquestradores padrão suportados pelo TFX são Apache Beam e Kubeflow. O [Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) pode ser executado em múltiplos back-ends de processamento de dados (Beam Ruunners). O Cloud Dataflow é um desses beam runners que pode ser usado para executar pipelines TFX. O Apache Beam pode ser usado tanto para pipelines de streaming como para pipelines de processamento em lote.<br> O [Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow) é uma plataforma de ML de código aberto dedicada a tornar as implantações de workflows de aprendizado de máquina (ML) no Kubernetes simples, portáveis e escalonáveis. O Kubeflow pode ser usado como orquestrador para pipelines TFX quando eles precisam ser implantados em clusters Kubernetes. Além disso, você também pode usar seu próprio [orquestrador personalizado](https://www.tensorflow.org/tfx/guide/custom_orchestrator) para executar um pipeline TFX.

Leia mais sobre o Airflow [aqui](https://airflow.apache.org/).

## **Dataset Chicago Taxi**

![taxi.jpg](images/airflow_workshop/taxi.jpg)

Consulte o [Guia do usuário do TFX](https://www.tensorflow.org/tfx/guide) para saber mais.

Você usará o [dataset Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) disponibilizado pela cidade de Chicago.

Observação: Este tutorial cria um aplicativo que utiliza dados que foram modificados para uso a partir de sua fonte original obtida em www.cityofchicago.org, site oficial da cidade de Chicago. A cidade de Chicago não faz nenhuma reivindicação quanto ao conteúdo, precisão, atualidade ou integridade de qualquer um dos dados fornecidos neste site. Os dados fornecidos neste site estão sujeitos a alterações a qualquer momento. Entende-se que os dados fornecidos neste tutorial são utilizados por sua conta e risco.

### Objetivo do modelo – Classificação binária

O cliente dará uma gorjeta maior ou menor que 20%?

## Configure o projeto Google Cloud

**Antes de clicar no botão Start Lab** leia estas instruções. Os laboratórios são cronometrados e você não poderá pausá-los. O cronômetro, que começa quando você clica em **Start Lab**, mostra por quanto tempo os recursos do Google Cloud ficarão disponíveis para você.

Este laboratório prático permite que você mesmo realize as atividades de laboratório num ambiente de nuvem real, não num ambiente de simulação ou de demonstração. Isto é feito fornecendo credenciais novas e temporárias que você usa para fazer login e acessar o Google Cloud durante o laboratório.

**Do que você precisa**. Para concluir este laboratório, você precisará de:

- Acesso a um navegador de internet padrão (recomenda-se o navegador Chrome).
- Tempo para concluir o laboratório.

**Observação:** se você já tem sua conta ou projeto pessoal do Google Cloud, não use-o neste laboratório.

**Observação:** se você estiver usando um dispositivo Chrome OS, abra uma janela anônima para executar este laboratório.

**Como iniciar seu laboratório e fazer login no Console do Google Cloud** 1. Clique no botão **Start Lab**. Se você precisar pagar pelo laboratório, um pop-up será aberto para você selecionar sua forma de pagamento. À esquerda há um painel preenchido com as credenciais temporárias que você deve usar para este laboratório.

![qwiksetup1.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup1.png?raw=true)

1. Copie o nome de usuário e clique em **Open Google Console**. O laboratório inicializa o ambiente e abre outra aba que mostra a página **Sign in**.

![qwiksetup2.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup2.png?raw=true)

***Dica:*** abra as abas em janelas separadas, lado a lado.

![qwiksetup3.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup3.png?raw=true)

1. Na página **Sign in**, cole o nome de usuário que você copiou no painel esquerdo. Em seguida, copie e cole a senha.

***Importante:*** Você deve utilizar as credenciais do painel esquerdo. Não use suas credenciais do Google Cloud Training. Se você tiver sua própria conta do Google Cloud, não a use neste laboratório (evita cobranças).

1. Clique nas páginas seguintes:
2. Aceite os termos e condições.

- Não adicione opções de recuperação ou autenticação de dois fatores (porque esta é uma conta temporária).

- Não se inscreva para avaliações gratuitas.

Após alguns instantes, o Cloud Console será aberto nesta aba.

**Observação:** você poderá visualizar o menu com uma lista de produtos e serviços do Google Cloud clicando no **menu de navegação** no canto superior esquerdo.

![qwiksetup4.png](images/airflow_workshop/qwiksetup4.png)

### Ative o Cloud Shell

O Cloud Shell é uma máquina virtual carregada com ferramentas de desenvolvimento. Ela oferece um diretório inicial persistente de 5 GB e é executado no Google Cloud. O Cloud Shell fornece acesso de linha de comando aos recursos do Google Cloud.

No Cloud Console, na barra de ferramentas superior direita, clique no botão **Activate Cloud Shell**.

![qwiksetup5.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup5.png?raw=true)

Clique em **Continue**.

![qwiksetup6.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup6.png?raw=true)

Leva alguns minutos para provisionar e conectar-se ao ambiente. Quando estiver conectado, você já estará autenticado e o projeto estará definido como seu *PROJECT_ID*. Por exemplo:

![qwiksetup7.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/qwiksetup7.png?raw=true)

`gcloud` é a ferramenta de linha de comando do Google Cloud. Ela vem pré-instalada no Cloud Shell e suporta preenchimento usando tabulações.

Você pode listar o nome da conta ativa com este comando:

```
gcloud auth list
```

(Saída)

> ACTIVE: * ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net To set the active account, run: $ gcloud config set account `ACCOUNT`

Você pode listar o ID do projeto com este comando: `gcloud config list project` (Saída)

> [core] project = &lt;project_ID&gt;

(Example output)

> [core] project = qwiklabs-gcp-44776a13dea667a6

Para obter a documentação completa do gcloud, veja [visão geral da ferramenta de linha de comando gcloud](https://cloud.google.com/sdk/gcloud).

## Ative os serviços do Google Cloud

1. No Cloud Shell, use gcloud para ativar os serviços usados ​​no laboratório. `gcloud services enable notebooks.googleapis.com`

## Implante uma instância do Vertex Notebook

1. Clique no **menu de navegação** e navegue até **Vertex AI** e depois até **Workbench**.

![vertex-ai-workbench.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/vertex-ai-workbench.png?raw=true)

1. Na página Notebook instances, clique em **New notebook**.

2. No menu Customize da instância, selecione **TensorFlow Enterprise** e escolha a versão **TensorFlow Enterprise 2.x (with LTS)** &gt; **Without GPUs**.

![vertex-notebook-create-2.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/vertex-notebook-create-2.png?raw=true)

1. Na caixa de diálogo **New notebook instance**, clique no ícone de lápis para **Editar** as propriedades da instância.

2. Em **Instance name**, insira um nome para sua instância.

3. Em **Region**, selecione `us-east1` e para **Zone**, selecione uma zona dentro da região selecionada.

4. Role para baixo até Machine configuration e selecione **e2-standard-2** para Machine type.

5. Deixe os campos restantes com os valores default e clique em **Create**.

Após alguns minutos, o console da Vertex AI exibirá o nome da sua instância, seguido por **Open Jupyterlab**.

1. Clique em **Open JupyterLab**. Uma janela do JupyterLab será aberta em uma nova aba.

## Configure o ambiente

### Clone o repositório do laboratório

Em seguida, você clonará o repositório `tfx` na sua instância do JupyterLab. 1. No JupyterLab, clique no ícone **Terminal** para abrir um novo terminal.

{ql-infobox0}<strong>Observação:</strong> se solicitado, clique em <code>Cancel</code> para Build Recommended.{/ql-infobox0}

1. Para clonar o repositório Github do `tfx`, digite o seguinte comando e pressione **Enter**.

```
git clone https://github.com/tensorflow/tfx.git
```

1. Para confirmar que você clonou o repositório, dê um duplo-clique no diretório `tfx` e confirme que você pode ver seu conteúdo.

![repo-directory.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/repo-directory.png?raw=true)

### Instale as dependências do laboratório

1. Execute o seguinte para ir para a pasta `tfx/tfx/examples/airflow_workshop/taxi/setup/` e execute `./setup_demo.sh` para instalar as dependências do laboratório:

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

O código acima irá

- Instalar os pacotes necessários.
- Criar uma pasta `airflow` na pasta inicial.
- Copiar a pasta `dags` da pasta `tfx/tfx/examples/airflow_workshop/taxi/setup/` para a `~/airflow/`.
- Copiar o arquivo csv de `tfx/tfx/examples/airflow_workshop/taxi/setup/data` para `~/airflow/data`.

![airflow-home.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/airflow-home.png?raw=true)

## Configurando o servidor Airflow

### Crie uma regra de firewall para acessar o servidor Airflow no navegador

1. Acesse `https://console.cloud.google.com/networking/firewalls/list` e certifique-se de que o nome do projeto esteja selecionado corretamente
2. Clique na opção `CREATE FIREWALL RULE` na parte superior

![firewall-rule.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/firewall-rule.png?raw=true)

Na caixa de diálogo **Create a firewall**, siga as etapas listadas abaixo.

1. Para **Name**, coloque `airflow-tfx`.
2. Para **Priority**, selecione `1`.
3. Para **Targets**, selecione `All instances in the network`.
4. Para **Source IPv4 ranges**, selecione `0.0.0.0/0`
5. Para **Protocols and ports**, clique em `tcp` e digite `7000` na caixa ao lado de `tcp`
6. Clique `Create`.

![create-firewall-dialog.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/create-firewall-dialog.png?raw=true)

### Execute o servidor airflow a partir do seu shell

Na janela do Jupyter Lab Terminal, mude para o diretório inicial e execute o comando `airflow users create` para criar um usuário admin para o Airflow:

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

Em seguida, execute o comando `airflow webserver` e `airflow scheduler` para rodar o servidor. Escolha a porta `7000`, já que ela é permitida pelo firewall.

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### Obtenha seu ip externo

1. No Cloud Shell, use `gcloud` para obter o IP externo.

```
gcloud compute instances list
```

![gcloud-instance-ip.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/gcloud-instance-ip.png?raw=true)

## Executando um DAG/Pipeline

### Num navegador

Abra um navegador e acesse http://&lt;external_ip&gt;:7000

- Na página de login, insira o nome de usuário (`admin`) e a senha (`admin`) que você escolheu ao executar o comando `airflow users create`.

![airflow-login.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/airflow-login.png?raw=true)

O Airflow carrega DAGs de arquivos-fonte Python. Ele pega cada arquivo e o executa. Em seguida, ele carrega quaisquer objetos DAG desse arquivo. Todos os arquivos `.py` que definem objetos DAG serão listados como pipelines na página inicial do Airflow.

Neste tutorial, o Airflow verifica a pasta `~/airflow/dags/` em busca de objetos DAG.

Se você abrir `~/airflow/dags/taxi_pipeline.py` e rolar até o final, verá que ele cria e armazena um objeto DAG numa variável chamada `DAG`. Portanto, ele será listado como um pipeline na página inicial do Airflow, conforme mostrado abaixo:

![dag-home-full.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/dag-home-full.png?raw=true)

Se você clicar em taxi, será redirecionado para a visualização em grade do DAG. Você pode clicar na opção `Graph` na parte superior para obter a visualização do gráfico do DAG.

![airflow-dag-graph.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/airflow-dag-graph.png?raw=true)

### Acione o pipeline taxi

Na página inicial você verá os botões que podem ser usados ​​para interagir com o DAG.

![dag-buttons.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/dag-buttons.png?raw=true)

No cabeçalho **actions** , clique no botão **trigger** para acionar o pipeline.

Na página do **DAG** taxi, use o botão à direita para atualizar o estado da visualização do gráfico do DAG à medida que o pipeline é executado. Além disso, você pode ativar **Auto Refresh** para instruir o Airflow a atualizar automaticamente a visualização do gráfico conforme e quando o estado mudar.

![dag-button-refresh.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/dag-button-refresh.png?raw=true)

Você também pode usar a [CLI do Airflow](https://airflow.apache.org/cli.html) no terminal para ativar e acionar seus DAGs:

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### Aguardando a conclusão do pipeline

Depois de acionar o pipeline, na tela de visualização dos DAGs, você poderá observar o progresso do pipeline enquanto ele está em execução. À medida que cada componente é executado, a cor do contorno do componente no gráfico DAG muda para mostrar seu estado. Quando um componente terminar de ser processado, o contorno ficará verde escuro para mostrar que foi concluído.

![dag-step7.png](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/dag-step7.png?raw=true)

## Entendendo os componentes

Agora vamos dar uma olhada nos componentes desse pipeline em detalhes e examinaremos individualmente as saídas produzidas por cada etapa do pipeline.

1. No JupyterLab, vá para `~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/`

2. Abra **notebook.ipynb.**![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)

3. Continue o laboratório no notebook e execute cada célula clicando no ícone **Run** (<img src="images/airflow_workshop/f1abc657d9d2845c.png" width="28.00" alt="run-button.png">) na parte superior da tela. Você também pode executar o código numa célula com **SHIFT + ENTER**.

Leia a narrativa e tenha certeza que entendeu o que está acontecendo em cada célula.
