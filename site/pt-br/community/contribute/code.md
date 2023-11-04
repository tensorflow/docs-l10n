# Contribuir para o código do TensorFlow

Esteja você adicionando uma função de perda, melhorando a cobertura de teste, ou escrevendo um RFC para uma grande mudança de design, essa parte do guia do contribuidor o ajudará a começar. Obrigado pelo seu trabalho e interece em melhorar o TensorFlow.

## Antes de começar

Antes de contribuir ao codigo fonte de um projeto TensorFlow, por favor revise o arquivo `CONTRIBUTING.md` no repositório GitHub do projeto. Por exemplo, veja o arquivo [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) no núcleo do repositório TensorFlow. Todos os contribuidores de código precisam assinar um [Contributor License Agreement](https://cla.developers.google.com/clas) (CLA).

Para evitar trabalho duplicado, por vafor revise [current](https://github.com/tensorflow/community/tree/master/rfcs) ou [proposed](https://github.com/tensorflow/community/labels/RFC%3A%20Proposed) RFCs e contacte os desenvolvedores nos fóruns do TensorFlow ([developers@tensorflow.org](https://groups.google.com/u/1/a/tensorflow.org/g/developers)) antes de você começar a trabalhar em uma feature não trivial. Nós somos um pouco seletivos ao decidir adicionar novas funcionalidades, e a melhor maneira de contribuir e ajudar o projeto é trabalhar em problemas conhecidos.

## Issues para novos contribuidores

Novos contribuidores deveriam buscar as seguintes tags ao procurar por uma primeira contribuição a base de código do TensorFlow. Nós fortemente recomendamos que novos contribuidores abordem projetos “good first issue” e "contributions welcome" primeiro; isso ajuda o contribuidor a se familiarizar com o fluxo de contribuição, e para o núcleo de desenvolvedores  conhecer o contribuidor.

- [good first issue](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue)
- [contributions welcome](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome)

se você está interessado em recrutar um time para abordar um problema de larga escala ou uma nova feature, por favor email o [developers@ group](https://groups.google.com/a/tensorflow.org/g/developers) e revise nossa atual lista de RFCs.

## Revisão de código

Novas features, correção de bugs, e quaisquer outras alterações a base de código são sujeitas a revisão de código.

Revisão de código contribuido a projetos como pull request é um componente crucial para o desenvolvimento do TensorFlow. Nós encorajamos qualquer um a começar a revisar código submetido por outros desenvolvedores, especialmente se a feature é algo que você provavelmente vá usar.

Aqui estão algumas perguntas a serem lembradas durante o processo de revisão de código:

- *Queremos isso no TensorFlow?* É provável que seja usado? Você, como usuário do TensorFlow, gosta da mudança e pretende usá-la? Essa mudança está no escopo do TensorFlow? O custo de manutenção de um novo recurso valerá seus benefícios?

- *O código é concistente com a API do TensorFlow?* São funções públicas, classes e parâmetros bem nomeados e projetados intuitivamente?

- *Inclui documentação?* Todas as funções públicas, classes, parâmetros, tipos de retorno e atributos armazenados são nomeados de acordo com as convenções do TensorFlow e claramente documentados? A nova funcionalidade é descrita na documentação do TensorFlow e ilustrada com exemplos, sempre que possível? A documentação é renderizada corretamente?

- *O código é humanamente legível?* Possui pouca redundância? Os nomes das variáveis devem ser melhorados para maior clareza ou consistência? Comentários devem ser adicionados? Algum comentário deve ser removido como inútil ou irrelevante?

- *O código é eficiente?* Poderia ser reescrito facilmente para ser executado com mais eficiência?

- O código é *retrocompatível* com verções anteriores do TensorFlow?

- O novo código adicionará *novas dependências* em outras bibliotecas?

## Testar e melhorar cobertura de teste

Testes de unidade de alta qualidade são a base do processo de desnvolvimento do TensoFlow. Para esse propósito, utilizamos imagens Docker. As funções de teste são nomeadas apropiadamente e são responsaveis por checar a validade dos algoritomos, assim como as diferentes opções de código.

Todos os novos recursos e correções de bugs *devem* incluir cobertura de teste adequada. Também recebemos contribuições de novos casos de teste ou melhorias nos testes existentes. Se você descobrir que nossos testes existentes não estão completos - mesmo que isso não esteja causando um bug no momento - registre um problema e, se possível, uma solicitação pull.

Para obter os detalhes específicos dos procedimentos de teste em cada projeto do TensorFlow, consulte os arquivos `README.md` e `CONTRIBUTING.md` no repositório do projeto no GitHub.

De preocupações particulares em *testes adequados* :

- São, *todas as funções e classes públicas* testadas?
- Há um *conjunto razoável de parâmetros*, seus valores, tipos e combinações testados?
- Os testes validam que o *código está correto* e que está *fazendo o que a documentação diz que* o código deveria fazer?
- Se a alteração for uma correção de bug, um *teste de não regressão* está incluído?
- Os testes *passam na build de integração contínua*?
- Os testes *cobrem todas as linhas de código?* Se não, as exceções são razoáveis e explícitas?

Se você encontrar algum problema, considere ajudar o colaborador a entender esses problemas e resolvê-los.

## Melhorar mensagens de erro ou logs

Contribuições que melhoram mensagens de erro e registro são bem vindas.

## Fluxo de trabalho de contribuição

Contribuições de código—correções de bugs, novos desenvolvimentos, melhorias de teste—seguem um fluxo de trabalho centrado no GitHub. Para participar do desenvolvimento do TensorFlow, configure uma conta do GitHub. Então:

1. Faça um Fork do repositório no qual planeja trabalhar. Vá para a página do repositório do projeto e use o botão *Fork* . Isso criará uma cópia do repositório, sob seu nome de usuário. (Para obter mais detalhes sobre como fazer um branch de um repositório, consulte [este guia](https://help.github.com/articles/fork-a-repo/) .)

2. Clone o repositório para o seu sistema local.

    `$ git clone git@github.com:your-user-name/project-name.git`

3. Crie uma nova branch para armazenar seu trabalho.

    `$ git checkout -b new-branch-name`

4. Trabalhe em seu novo código. Escreva e execute testes.

5. Confirme suas alterações.

    `$ git add -A`

    `$ git commit -m "commit message here"`

6. Envie suas alterações para o repositório do GitHub.

    `$ git push origin branch-name`

7. Abra um *Pull Request*(PR). Acesse o repositório do projeto original no GitHub. Haverá uma mensagem sobre seu branch enviado recentemente, perguntando se você gostaria de abrir um Pull Request. Siga os prompts, *compare os repositórios* e envie o PR. Isso enviará um email para os committers. Você pode querer considerar o envio de um email para a lista de discussão para obter mais visibilidade. (Para mais detalhes, consulte o [guia do GitHub sobre PRs](https://help.github.com/articles/creating-a-pull-request-from-a-fork) .

8. Os mantenedores e outros contribuidores *revisarão seu PR*. Participe da conversa e tente *fazer as alterações solicitadas*. Assim que o PR for aprovado, o código será merged.

*Antes de trabalhar em sua próxima contribuição* , certifique-se de que seu repositório local esteja atualizado.

1. Defina o upstream remote. (Você só precisa fazer isso uma vez por projeto, não sempre.)

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. Alterne para a branch master local.

    `$ git checkout master`

3. Pull down as alterações do upstream.

    `$ git pull upstream master`

4. Envie as alterações para sua conta do GitHub. (Opcional, mas uma boa prática.)

    `$ git push origin master`

5. Crie uma nova branch se estiver iniciando um novo trabalho.

    `$ git checkout -b branch-name`

Recursos adicionais do `git` e do GitHub:

- [Documentação Git](https://git-scm.com/documentation)
- [Fluxo de desenvolvimento do Git](https://docs.scipy.org/doc/numpy/dev/development_workflow.html)
- [Resolução de conflitos de merge](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/) .

## Checklist do colaborador

- Leia as [diretrizes de contribuição](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) .
- Leia o [Código de Conduta](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md) .
- Certifique-se de ter assinado o [Contrato de licença do colaborador (CLA)](https://cla.developers.google.com/) .
- Verifique se suas alterações estão de acordo com as [diretrizes](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution) .
- Verifique se suas alterações são consistentes com o [estilo de código do TensorFlow](https://www.tensorflow.org/community/contribute/code_style) .
- [Execute os testes de unidade](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#running-unit-tests) .
