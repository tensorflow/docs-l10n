# O processo RFC do TensorFlow

Cada novo recurso do TensorFlow começa como um documento Request for Comments (RFC).

Um RFC é um documento que descreve um requisito e as alterações propostas que irão resolvê-lo. Especificamente, o RFC irá:

- Ser formatado de acordo com o [modelo RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
- Ser enviado como um pull request ao diretório [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).
- Estar sujeito a discussão e uma reunião de revisão antes da sua aceitação.

O objetivo de uma Request for Comments (RFC) do TensorFlow é envolver a comunidade do TensorFlow no desenvolvimento, obtendo feedback de partes interessadas e especialistas e comunicando amplamente as alterações de design.

## Como enviar um RFC

1. Antes de enviar uma RFC, discuta seus objetivos com os colaboradores e mantenedores do projeto e obtenha feedback antecipado. Use a lista de discussão de desenvolvedores do projeto em questão (developers@tensorflow.org ou a lista do SIG relevante).

2. Elabore seu RFC.

    - Leia os [critérios de revisão de projetos](https://github.com/tensorflow/community/blob/master/governance/design-reviews.md)
    - Siga o [modelo RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
    - Nomeie seu arquivo RFC `YYYYMMDD-descriptive-name.md`, onde `YYYYMMDD` é a data de envio e `descriptive-name` está relacionado ao título de seu RFC. (Por exemplo, se o seu RFC for intitulado *Parallel Widgets API* , você poderá usar o nome de arquivo `20180531-parallel-widgets.md`.
    - Se você tiver imagens ou outros arquivos auxiliares, crie um diretório no formato `YYYYMMDD-descriptive-name` onde possa armazenar esses arquivos.

    Depois de escrever o rascunho do RFC, obtenha feedback dos mantenedores e contribuidores antes de enviá-lo.

    Escrever código de implementação não é um requisito, mas pode ajudar a criar discussões.

3. Convoque um patrono.

    - O patrono deve ser um mantenedor do projeto.
    - Identifique o patrono na RFC, antes de postar o pull request.

    Você *pode* postar um RFC sem patrono, mas se dentro de um mês após a publicação do PR ainda não houver um patrono, ele será fechado.

4. Envie seu RFC como um pull request para [tensorflow/community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).

    Inclua a tabela de cabeçalho e o conteúdo da seção *Objective* no comentário do seu pull request, usando Markdown. Para usar como exemplo, veja [esta amostra de um RFC](https://github.com/tensorflow/community/pull/5). Inclua os handles GitHub dos coautores, revisores e patronos.

    No topo do pull request, identifique quanto tempo durará o período de comentários. Isto deve ocorrer *no mínimo duas semanas* após a publicação do pull request.

5. Envie um e-mail para a lista de discussão de desenvolvimento com uma breve descrição, um link para o pull request e uma solicitação de revisão. Siga o formato dos mailings anteriores, como você pode ver [neste exemplo](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/PIChGLLnpTE).

6. O patrono solicitará uma reunião do comitê de revisão, no máximo duas semanas após a publicação do pull request da RFC. Se a discussão for acalorada, espere até que esteja resolvida antes de seguir para a revisão. O objetivo da reunião de revisão é resolver questões menores; o consenso deve ser alcançado antecipadamente sobre as questões principais.

7. A reunião pode aprovar o RFC, rejeitá-la ou exigir alterações antes que ela possa ser considerada novamente. Os RFCs aprovados serão mesclados em [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs) e os RFCs rejeitados terão seus pull requests fechados.

## Participantes de um RFC

Muitas pessoas estão envolvidas no processo de um RFC:

- **Autor do RFC** – um ou mais membros da comunidade que escrevem um RFC e estão comprometidos em defendê-la durante todo o processo

- **Patrono do RFC** — um mantenedor que patrocina a RFC e irá orientá-la durante o processo de revisão.

- **Comitê de revisão** — um grupo de mantenedores que tem a responsabilidade de recomendar a adoção do RFC

- Qualquer **membro da comunidade** pode ajudar fornecendo feedback sobre se a RFC atenderá às suas necessidades.

### Patronos do RFC

Um patrono é o mantenedor do projeto responsável por garantir o melhor resultado possível do processo do RFC. Isto inclui:

- Advogar em defesa do design proposto.
- Orientar o RFC a aderir às convenções de design e estilo existentes.
- Orientar o comitê de revisão para chegar a um consenso produtivo.
- Se alterações forem solicitadas pelo comitê de revisão, certificar-se de que sejam feitas e buscar a aprovação subsequente dos membros do comitê.
- Se a RFC passar para a implementação:
    - Garantir que a implementação proposta esteja de acordo com o design.
    - Coordenar com as partes interessadas para uma implementação bem-sucedida.

### Comitês de revisão do RFC

O comitê de revisão decide por consenso se aprova, rejeita ou solicita alterações. Eles são responsáveis ​​por:

- Garantir que os itens substanciais do feedback público tenham sido levados em consideração.
- Adicionar suas notas de reunião como comentários ao pull request.
- Fornecer razões para suas decisões.

A constituição de um comitê de revisão pode mudar de acordo com o estilo de governança e liderança particular de cada projeto. Para o TensorFlow core, o comitê será composto por colaboradores do projeto TensorFlow que tenham experiência na área de domínio em questão.

### Membros da comunidade e o processo RFC

O propósito das RFCs é garantir que a comunidade seja bem representada e atendida por novas mudanças no TensorFlow. É responsabilidade dos membros da comunidade participar da revisão dos RFCs quando tiverem interesse no seu resultado.

Os membros da comunidade que estiverem interessados ​​em uma RFC devem:

- **Fornecer feedback** o mais rápido possível para permitir tempo adequado para consideração.
- **Ler os RFCs** atentamente antes de fornecer feedback.
- Serem **civilizados e construtivos**.

## Implementando novos recursos

Uma vez que o RFC é aprovado, a implementação poderá começar.

Se você estiver trabalhando num novo código para implementar um RFC:

- Certifique-se de compreender o recurso e o design aprovado na RFC. Faça perguntas e discuta a abordagem antes de começar o trabalho.
- Novos recursos devem incluir novos testes de unidade que verifiquem se o recurso funciona conforme o esperado. É uma boa ideia escrever esses testes antes de escrever o código.
- Siga o [Guia de estilo de código do TensorFlow](#tensorflow-code-style-guide)
- Adicione ou atualize a documentação relevante da API. Faça referência ao RFC na nova documentação.
- Siga quaisquer outras diretrizes descritas no arquivo `CONTRIBUTING.md` no repositório do projeto para o qual você está contribuindo.
- Execute testes unitários antes de enviar seu código.
- Trabalhe com o patrono do RFC para lançar o novo código com sucesso.

## Mantendo o nível elevado

Enquanto encorajamos e celebramos cada colaborador, o nível de aceitação de um RFC é mantido intencionalmente alto. Um novo recurso pode ser rejeitado ou precisar de revisão significativa em qualquer um dos estágios a seguir:

- Conversas iniciais sobre design na lista de discussão relevante.
- Falha em recrutar um patrono.
- Objeções críticas durante a fase de feedback.
- Falha em obter consenso durante a revisão do projeto.
- Preocupações levantadas durante a implementação (por exemplo: incapacidade de alcançar compatibilidade com versões anteriores, preocupações com manutenção).

Se o processo estiver funcionando bem, espera-se que os RFC falhem nas fases iniciais, e não nas posteriores. Uma RFC aprovada não é garantia de compromisso de implementação, e a aceitação de uma implementação proposta de RFC ainda está sujeita ao processo usual de revisão de código.

Se você tiver alguma dúvida sobre esse processo, sinta-se à vontade para perguntar na lista de discussão de desenvolvedores ou registrar um issue em [tensorflow/community](https://github.com/tensorflow/community/tree/master/rfcs).
