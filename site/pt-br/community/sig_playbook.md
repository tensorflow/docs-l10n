# Manual do SIG

## Escopo de um SIG

O TensorFlow hospeda *grupos de interesse especial* (Special Interest Groups, ou SIGs) para focar a colaboração em determinadas áreas específicas. Os SIGs fazem seu trabalho em público. Para participar e contribuir, analise o trabalho do grupo e entre em contato com o líder do SIG. As políticas de associação variam de acordo com cada SIG.

O escopo ideal para um SIG atende a um domínio bem definido, onde a maior parte da participação é da comunidade. Além disso, deve haver provas suficientes de que existem membros da comunidade dispostos a envolver-se e contribuir caso o grupo de interesse seja estabelecido.

Nem todos os SIGs terão o mesmo nível de energia, amplitude de escopo ou modelos de governança, portanto, espere alguma variabilidade.

Veja a lista completa dos [SIGs do TensorFlow](https://github.com/tensorflow/community/tree/master/sigs).

### Não objetivos: o que um SIG *não* é

O objetivo de um SIG é facilitar a colaboração no trabalho compartilhado. Um SIG, portanto:

- *Não é um fórum de suporte*: uma lista de discussão e um SIG não são a mesma coisa.
- *Não é necessário imediatamente*: no início da vida de um projeto, você pode não saber se você tem trabalho compartilhado ou colaboradores.
- *Não é trabalho não remunerado*: é preciso ter energia para fazer crescer e coordenar o trabalho de forma colaborativa.

Nossa abordagem para a criação de SIGs será conservadora – graças à facilidade de iniciar projetos no GitHub, há muitos caminhos onde a colaboração pode acontecer sem a necessidade de um SIG.

## Ciclo de vida de um SIG

### Pesquisa e consulta

Os proponentes de grupos deverão reunir evidências para aprovação, conforme especificado abaixo. Alguns caminhos possíveis a serem considerados são:

- Um problema ou conjunto de problemas bem definido que o grupo resolveria.
- Consulta com os membros da comunidade que seriam beneficiados, avaliando tanto o benefício como a vontade de se comprometerem.
- Para projetos existentes, evidências de que os colaboradores se preocupam com o tópico obtidas de issues e pull requests.
- Metas potenciais a serem alcançadas pelo grupo.
- Requisitos de recursos para administrar o grupo.

Mesmo que a necessidade de um SIG pareça evidente, a pesquisa e a consulta ainda são importantes para o sucesso do grupo.

### Criando o novo grupo

O novo grupo deverá seguir o processo abaixo para elaborar seu estatuto. Em particular, ele deverá demonstrar:

- Um propósito e benefício claros para o TensorFlow (seja em torno de um subprojeto ou área de aplicação)
- Dois ou mais colaboradores dispostos a atuar como líderes do grupo, existência de outros colaboradores e evidência de demanda pelo grupo
- Recursos necessários inicialmente (geralmente, lista de e-mails e videoconferências regulares).

A aprovação do grupo será dada por decisão da TF Community Team, definida como mantenedora do projeto tensorflow/comunidade. A equipe consultará outras partes interessadas conforme necessário.

Antes de entrar nas partes formais do processo, é aconselhável consultar a equipe da comunidade TensorFlow, community-team@tensorflow.org. É altamente provável que conversas e alguma iteração sejam necessárias antes que a solicitação do SIG esteja pronta.

A solicitação formal para o novo grupo deve ser feita enviando um estatuto como pull request para tensorflow/community, e incluindo a solicitação nos comentários do pull request (veja modelo abaixo). Após a aprovação, será feito merge do pull request do grupo e os recursos necessários serão criados.

### Solicitação de modelo para o novo SIG

Este modelo estará disponível no repositório da comunidade: [SIG-request-template.md](https://github.com/tensorflow/community/blob/master/governance/SIG-request-template.md).

### Estatuto

Cada grupo será estabelecido com um estatuto que será regido pelo código de conduta do TensorFlow. Os arquivos do grupo serão públicos. A adesão pode ser aberta a todos sem aprovação ou disponível mediante solicitação, dependendo da aprovação do administrador do grupo.

O estatuto deve nomear um administrador. Além de um administrador, o grupo deve incluir pelo menos uma pessoa como líder (pode ser a mesma pessoa), que servirá como ponto de contato para coordenação conforme necessário com a equipe da comunidade do TensorFlow.

Esse estatuto será publicado inicialmente na lista de discussão do grupo. O repositório da comunidade na organização do TensorFlow no GitHub arquivará esses documentos e políticas ([exemplo do Kubernetes](https://github.com/kubernetes/community)). À medida que qualquer grupo desenvolve as suas práticas e convenções, esperamos que as documente na parte relevante do repositório da comunidade.

### Colaboração e inclusão

Embora não seja obrigatório, o grupo deve optar por utilizar a colaboração através de teleconferências agendadas ou canais de chat para conduzir reuniões. Quaisquer reuniões desse tipo devem ser anunciadas na lista de discussão e as notas postadas posteriormente na lista de discussão. Reuniões regulares ajudam a impulsionar as responsabilidades e o progresso em um SIG.

Os membros da equipe da comunidade do TensorFlow monitorarão e incentivarão proativamente o grupo a discutir e agir conforme seja apropriado.

### Lançamento

Atividades necessárias:

- Notificar grupos de discussão geral do TensorFlow ([discut@](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) , [desenvolvedores@](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers)).
- Adicionar o SIG às páginas da comunidade no site do TensorFlow.

Atividades opcionais:

- Criação de uma postagem de blog para a comunidade de blogs do TensorFlow.

### Saúde e encerramento de um SIG

A equipe da comunidade do TensorFlow fará o possível para garantir a integridade dos SIGs. De tempos em tempos, ele solicitará ao líder do SIG que forneça um relatório do trabalho do SIG, que será usado para informar a comunidade mais ampla do TensorFlow sobre a atividade do grupo.

Se um SIG não tiver mais uma finalidade útil ou comunidade interessada, ele poderá ser arquivado e cessar sua operação. À equipe da comunidade TF reserva-se o direito de arquivar esses SIGs inativos, a fim de manter a saúde do projeto em geral, embora esse seja um resultado menos preferível. Um SIG também pode optar pela dissolução se reconhecer que atingiu o fim da sua vida útil.
