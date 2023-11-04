# tfds e o Google Cloud Storage

O Google Cloud Storage (GCS) pode ser usado com tfds por diversos motivos:

- Armazenamento de dados pré-processados
- Acesso de datases que possuem dados armazenados no GCS

## Acesso através do bucket TFDS GCS

Alguns conjuntos de dados estão disponíveis diretamente no nosso bucket do GCS [`gs://tfds-data/datasets/`](https://console.cloud.google.com/storage/browser/tfds-data) sem qualquer autenticação:

- Se `tfds.load(..., try_gcs=False)` (padrão), o dataset será copiado localmente em `~/tensorflow_datasets` durante `download_and_prepare`.
- Se `tfds.load(..., try_gcs=True)`, o dataset será transmitido diretamente do GCS ( `download_and_prepare` será ignorado).

Você pode verificar se um dataset está hospedado no bucket público com `tfds.is_dataset_on_gcs('mnist')`.

## Autenticação

Antes de iniciar, você deve decidir como deseja autenticar. Há três opções:

- sem autenticação (também conhecido como acesso anônimo)
- usando sua conta do Google
- usando uma conta de serviço (pode ser facilmente compartilhada com outras pessoas da sua equipe)

Você pode encontrar informações detalhadas na [documentação do Google Cloud](https://cloud.google.com/docs/authentication/getting-started)

### Instruções simplificadas

Se você executar do Colab, poderá autenticar com sua conta, mas executando:

```python
from google.colab import auth
auth.authenticate_user()
```

Se você executar na sua máquina local (ou numa VM), poderá autenticar com sua conta executando:

```shell
gcloud auth application-default login
```

Se você preferir fazer login com uma conta de serviço, baixe a chave em arquivo JSON e defina

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## Usando o Google Cloud Storage para armazenar dados pré-processados

Normalmente, quando você usa datasets do TensorFlow, os dados baixados e preparados serão armazenados em cache num diretório local (por padrão `~/tensorflow_datasets`).

Em alguns ambientes onde o disco local pode ser efêmero (um servidor em nuvem temporário ou um [notebook Colab](https://colab.research.google.com)) ou quando você precisa que os dados sejam acessíveis por múltiplas máquinas, é útil definir `data_dir` para um sistema de armazenamento em nuvem, como um bucket do Google Cloud Storage (GCS).

### Como?

[Crie um bucket do GCS](https://cloud.google.com/storage/docs/creating-buckets) e garanta que você (ou sua conta de serviço) tenha permissões de leitura/gravação nele (consulte as instruções de autorização acima)

Ao usar `tfds`, você poderá definir `data_dir` como `"gs://YOUR_BUCKET_NAME"`

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### Ressalvas:

- Essa abordagem funciona para datasets que usam apenas `tf.io.gfile` para acesso aos dados. Isto vsale para a maioria dos datasets, mas não para todos.
- Lembre-se de que acessar o GCS é acessar um servidor remoto e transmitir dados a partir dele, portanto, você poderá incorrer em custos de rede.

## Acessando datasets armazenados no GCS

Se os proprietários do dataset autorizaram acesso anônimo, você pode simplesmente executar o código tfds.load - e tudo funciona como um download comum da Internet.

Se o dataset exigir autenticação, use as instruções acima para decidir qual opção você deseja (conta própria versus conta de serviço) e comunique o nome da conta (também conhecido como e-mail) ao proprietário do dataset. Depois que for autorizado o acesso ao diretório GCS, você poderá executar o código de download do tfds.
