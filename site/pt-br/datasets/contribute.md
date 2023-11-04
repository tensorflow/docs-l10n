# Contribua para o repositório TFDS

Obrigado pelo seu interesse na nossa biblioteca! Estamos entusiasmados por ter uma comunidade tão motivada.

## Como começar

- Se você é novo no TFDS, a maneira mais fácil de começar é implementar um dos nossos [datasets solicitados](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22dataset+request%22+sort%3Areactions-%2B1-desc), focando nos mais populares. [Siga nosso guia](https://www.tensorflow.org/datasets/add_dataset) para mais instruções.
- Issues, solicitações de recursos, bugs,... têm um impacto muito maior do que a inclusão de novos datasets, pois eles beneficiam toda a comunidade TFDS. Veja a [lista de possíveis contribuições](https://github.com/tensorflow/datasets/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+-label%3A%22dataset+request%22+). Comece com aqueles rotulados com [contribution-welcome](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22), que são pequenos issues independentes e fáceis de começar a se envolver.
- Não hesite em assumir bugs que já foram atribuídos, mas que não são atualizados há algum tempo.
- Não há necessidade de atribuir o problema a você. Basta comentar sobre o problema quando vocÊ começar a trabalhar nele :)
- Não hesite em pedir ajuda se tiver interesse em um issue, mas não souber como começar. E envie um rascunho do pull request se desejar feedback antecipado.
- Para evitar duplicação desnecessária de trabalho, verifique a lista de [pull requests pendentes](https://github.com/tensorflow/datasets/pulls) e comente os issues nos quais você está trabalhando.

## Configuração

### Clonando o repositório

Para começar, clone ou baixe o repositório [Tensorflow Datasets](https://github.com/tensorflow/datasets) e instale o repositório localmente.

```sh
git clone https://github.com/tensorflow/datasets.git
cd datasets/
```

Instale as dependências de desenvolvimento:

```sh
pip install -e .  # Install minimal deps to use tensorflow_datasets
pip install -e ".[dev]"  # Install all deps required for testing and development
```

Observe que também há um `pip install -e ".[tests-all]"` para instalar todas as dependências específicas do dataset.

### Visual Studio Code

Ao desenvolver com [Visual Studio Code](https://code.visualstudio.com/) , nosso repositório vem com algumas [configurações pré-definidas](https://github.com/tensorflow/datasets/tree/master/.vscode/settings.json) para auxiliar no desenvolvimento (indentação correta, pylint,...).

Observação: ativar a descoberta de testes no VS Code pode falhar devido a alguns bugs do VS Code [#13301](https://github.com/microsoft/vscode-python/issues/13301) e [#6594](https://github.com/microsoft/vscode-python/issues/6594). Para resolver esses problemas, consulte os logs de descoberta de teste:

- Se você encontrar alguma mensagem de warning do TensorFlow, tente [esta correção](https://github.com/microsoft/vscode-python/issues/6594#issuecomment-555680813).
- Se a descoberta falhar devido à falta de um import que deveria ter sido instalado, envie um pull request para atualizar o pip install do `dev`.

## Checklist de pull requests

### Assine o CLA

As contribuições para este projeto devem ser acompanhadas de um Contrato de Licença de Contribuinte (CLA). Você (ou seu empregador) retém os direitos autorais de sua contribuição; isto simplesmente nos dá permissão para usar e redistribuir suas contribuições como parte do projeto. Acesse [https://cla.developers.google.com/](https://cla.developers.google.com/) para ver seus contratos atuais registrados ou para assinar um novo.

Geralmente, você só precisa enviar um CLA uma única vez; portanto, se já tiver enviado um (mesmo que seja para um projeto diferente), provavelmente não será necessário fazê-lo novamente.

### Siga práticas recomendadas

- A legibilidade é importante. O código deve seguir as melhores práticas de programação (evitar duplicação, fatorar em pequenas funções independentes, nomes de variáveis ​​explícitos,...)
- Quanto mais simples, melhor (por exemplo, a implementação é dividida em vários pull requests independentes e menores, que são mais fáceis de revisar).
- Inclua testes quando necessário; os testes existentes devem estar passando.
- Inclua [anotações de tipo](https://docs.python.org/3/library/typing.html)

### Verifique seu guia de estilo

Nosso estilo é baseado no [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md), que é baseado no [Guia de estilo PEP 8 Python](https://www.python.org/dev/peps/pep-0008). O novo código deve tentar seguir o[estilo de código Black](https://github.com/psf/black/blob/master/docs/the_black_code_style.md), mas com:

- Comprimento da linha: 80
- Recuo de 2 espaços em vez de 4.
- Aspas simples `'`

**Importante:** certifique-se de executar `pylint` em seu código para verificar se ele está formatado corretamente:

```sh
pip install pylint --upgrade
pylint tensorflow_datasets/core/some_file.py
```

Você pode tentar usar `yapf` para formatar automaticamente um arquivo, mas a ferramenta não é perfeita, então provavelmente você terá que aplicar as correções manualmente depois.

```sh
yapf tensorflow_datasets/core/some_file.py
```

Tanto o `pylint` quanto o `yapf` deveriam ter sido instalados com `pip install -e ".[dev]"` mas também podem ser instalados manualmente com `pip install`. Se você estiver usando o VS Code, essas ferramentas deverão ser integradas à sua interface do usuário.

### Docstrings e anotações de tipo

Classes e funções devem ser documentadas com docstrings e anotações de tipo. Os docstrings devem seguir o [estilo do Google](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods). Por exemplo:

```python
def function(x: List[T]) -> T:
  """One line doc should end by a dot.

  * Use `backticks` for code and tripple backticks for multi-line.
  * Use full API name (`tfds.core.DatasetBuilder` instead of `DatasetBuilder`)
  * Use `Args:`, `Returns:`, `Yields:`, `Attributes:`, `Raises:`

  Args:
    x: description

  Returns:
    y: description
  """
```

### Adicione e execute testes unitários

Certifique-se de que novos recursos sejam testados com testes de unidade. Você pode executar testes por meio da interface do VS Code ou da linha de comando. Por exemplo:

```sh
pytest -vv tensorflow_datasets/core/
```

`pytest` vs `unittest`: Historicamente, temos usado o módulo `unittest` para escrever testes. Novos testes devem utilizar preferencialmente `pytest` que é mais simples, flexível, moderno e utilizado pelas bibliotecas mais famosas (numpy, pandas, sklearn, matplotlib, scipy, six,...). Você pode ler o [guia do pytest](https://docs.pytest.org/en/stable/getting-started.html#getstarted) se não estiver familiarizado com o pytest.

Os testes para DatasetBuilders são especiais e estão documentados no [guia para adicionar um dataset](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md#test-your-dataset).

### Envie o pull request para avaliações!

Parabéns! Consulte a [Ajuda do GitHub](https://help.github.com/articles/about-pull-requests/) para mais informações sobre como usar pull requests.
