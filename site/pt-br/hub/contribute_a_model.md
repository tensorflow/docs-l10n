# Enviar um pull request

Esta página discute como enviar um pull request contendo arquivos de documentação Markdown para o repositório do GitHub [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev). Confira mais informações sobre como escrever arquivos Markdown no [guia de redação de documentação](writing_documentation.md).

**Observação:** se você quiser que seu modelo seja espelhado para outros hubs de modelos, use uma licença MIT, CC0 ou Apache. Caso não queira que seu modelo seja espelhado para outros hubs de modelos, use outra licença apropriada.

## Verificações do GitHub Actions

O repositório [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) usa o GitHub Actions para validar o formato dos arquivos em um pull request. O workflow usado para validar pull requests é definido em [.github/workflows/contributions-validator.yml](https://github.com/tensorflow/tfhub.dev/blob/master/.github/workflows/contributions-validator.yml). Você pode executar o script de validação em seu próprio branch fora do workflow, mas precisará ter todas as dependências do pacote pip corretas instaladas.

Quem estiver contribuindo pela primeira vez só pode executar verificações automatizadas com a aprovação de um mantenedor de repositório, de acordo com a [política do GitHub](https://github.blog/changelog/2021-04-22-github-actions-maintainers-must-approve-first-time-contributor-workflow-runs/). Aconselhamos que os publicadores enviem um pull request pequeno que corrija erros de ortografia ou que melhore a documentação do modelo, ou que enviem um pull request contendo somente sua página de publicador como o primeiro pull request para conseguirem executar verificações automatizadas em pull requests subsequentes.

Importante: seu pull request precisa ser aprovado nas verificações automatizadas antes de ser revisado!

## Como enviar o pull request

Os arquivos Markdown completos podem ser incluídos na branch principal de [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) por um dos seguintes métodos:

### Envio via linha de comando do git

Supondo que o caminho de arquivos Markdown identificado seja `assets/docs/publisher/model/1.md`, você pode seguir as etapas padrão do GitHub para criar um novo pull request com um arquivo recém-adicionado.

Comece bifurcando o repositório do TensorFlow no GitHub e depois criando um [pull request por essa bifurcação](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) no branch principal do TensorFlow Hub.

Veja abaixo os comandos típicos de linha de comando do git necessários para adicionar um novo arquivo ao branch principal do repositório bifurcado.

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### Envio via interface gráfica do GitHub

Uma forma mais direta de enviar é pela interface gráfica do GitHub. O GibHub permite a criação de pull requests para [novos arquivos](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files) ou [alterações de arquivos existentes](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository) diretamente pela interface gráfica.

1. Na [página do TensorFlow Hub no GitHub](https://github.com/tensorflow/tfhub.dev), pressione o botão`Create new file` (Criar novo arquivo).
2. Defina o caminho de arquivos correto: `assets/docs/publisher/model/1.md`
3. Copie e cole o Markdown existente.
4. Na parte inferior, selecione "Create a new branch for this commit and start a pull request" (Criar novo branch para este commit e iniciar um pull request).
