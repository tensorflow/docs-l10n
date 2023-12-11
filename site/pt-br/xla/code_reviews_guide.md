# Guia para revisões de código

O objetivo deste documento é explicar o raciocínio por trás da posição da equipe XLA em relação a revisões de código – uma posição que cresceu a partir de anos de experiência coletiva trabalhando em projetos de código aberto em geral e em XLA em particular.

Diferentes projetos de código aberto têm diferentes expectativas culturais sobre o quanto os revisores podem exigir dos autores de código. Em alguns projetos, os revisores pegarão uma pull request (PR) "quase correta", modificarão eles próprios e a enviarão. O XLA adota a abordagem oposta: esperamos que os autores repitam os PRs até que sejam bons o suficiente para serem enviados sem alterações adicionais.

A principal razão para esta abordagem é que queremos que os autores de PR aprendam a ser contribuidores de pleno direito do XLA. Se os próprios revisores resolverem os problemas descritos no PR, será muito mais difícil para os autores aprenderem. A abordagem XLA pode ser desafiadora tanto para revisores quanto para revisados, mas acreditamos que, em última análise, nos ajuda a fazer crescer a comunidade.

Aprender a ser um "colaborador XLA completo" não envolve apenas escrever código que não contenha bugs. Há muito mais para aprender sobre “como modificar o XLA”. Isto inclui:

- estilo de programação,
- quais casos extremos procurar,
- expectativas quanto à criação de testes,
- expectativas relacionadas a comentários e descrições de pull requests,
- e expectativas em torno da construção de infraestrutura para dar suporte às suas alterações.

À medida que você desenvolve o conhecimento do projeto e a confiança dos revisores, você pode esperar receber menos comentários, porque você estará naturalmente escrevendo um código mais alinhado com as expectativas do seu revisor.

Como muitos projetos de código aberto, o XLA conta com algumas pessoas altamente experientes e muitas pessoas relativamente novas. Aqueles de nós que são altamente experientes têm muitas demandas de nosso tempo. Para manter os PRs avançando em tempo hábil, você pode ajudar a reduzir o tempo necessário para os revisores e o número de iterações necessárias, seguindo as recomendações abaixo:

- *Revisar cuidadosamente e/ou fazer com que seu PR seja revisado por um colega antes de enviá-lo:* Tente remover o máximo de erros triviais (estilo de código, erros ortográficos e gramaticais, etc.) antes de enviar o PR para revisão. Certifique-se de que todos os testes estejam passando com sucesso.
- *Ler atentamente os comentários do revisor:* tente entender o que o revisor está pedindo e tente responder a todos os comentários antes de enviar uma nova versão.
- *Evitar discussões tangenciais (bikeshedding):* Discussões e desentendimentos técnicos são altamente valiosos e ninguém é perfeito. Porém, evite discussões que não façam diferença ou que sejam meramente estilísticas. Se você discordar do comentário de um revisor, tente detalhar seus motivos da forma mais precisa e abrangente possível para evitar longas discussões.
- *Evitar fazer as "perguntas de revisão mais frequentes" listadas abaixo:* Listamos algumas respostas a perguntas comuns e nossa justificativa abaixo.

Em geral, convidamos você a tentar fazer com que a revisão de seus PRs leve o mínimo de tempo possível. Assim vamos querer revisar suas alterações rapidamente!

Obrigado por contribuir com o XLA e boa programação!

## Perguntas de revisão mais comuns

### “Esta alteração de infraestrutura não está relacionada ao meu PR, por que devo fazer isso?”

A equipe XLA não possui uma equipe de infraestrutura dedicada, então cabe a todos nós construir bibliotecas auxiliares e evitar aumentar a dívida técnica. Consideramos que isto é um procedimento comum nas alterações do XLA e espera-se que todos participem. Geralmente construímos infraestrutura conforme necessário ao escrever código.

Os revisores do XLA podem solicitar que você construa alguma infraestrutura (ou faça uma grande alteração num PR) junto com um PR que você escreveu. Essa solicitação pode parecer desnecessária ou ortogonal à alteração que você está tentando fazer. Provavelmente, isto se deve a alguma incompatibilidade entre suas expectativas sobre a quantidade de infra que você precisa construir e as expectativas do revisor quanto à mesma.

Haver uma incompatibilidade de expectativas não é problema! Isso é esperado quando você é novo num projeto (e às vezes até acontece com os veteranos). É provável que os projetos nos quais você trabalhou no passado tenham expectativas diferentes. Isso também é normal e esperado e não significa que nenhum desses projetos tenha a abordagem errada; eles são apenas diferentes. Convidamos você a aceitar solicitações de infra junto com todos os outros comentários de revisão como uma oportunidade de saber o que esperamos deste projeto.

### "Posso tratar do seu comentário num PR futuro?"

Uma pergunta frequente relativamente aos pedidos de infraestrutura (ou outros pedidos de grande dimensão) nos PR é se a alteração deve ou não ser feita no PR original, ou se pode ser feita como seguimento num PR futuro.

Em geral, o XLA não permite que os autores de PR abordem os comentários da revisão como um PR de follow-up. Quando um revisor decide que algo precisa ser abordado neste PR, geralmente esperamos que os autores abordem isso no PR original, mesmo que o que seja solicitado seja uma grande alteração. Este padrão se aplica externamente e também internamente no Google.

Existem algumas razões pelas quais o XLA adota essa abordagem.

- *Confiança:* Ter conquistado a confiança do revisor é um componente chave. Num projeto de código aberto, os colaboradores podem aparecer ou desaparecer a qualquer momento. Depois que aprovamos um PR, os revisores não têm como garantir que os follow-ups prometidos sejam realmente realizados.

- *Impacto sobre outros desenvolvedores:* Se você enviou um PR abordando uma parte específica do XLA, há uma boa chance de que outras pessoas estejam olhando para essa mesma parte. Se aceitarmos dívida técnica no seu PR, todos que estiverem visualizando este arquivo serão impactados por essa dívida até que o follow-up seja enviado.

- *Largura de banda do revisor:* Adiar uma alteração para um follow-up impõe diversos custos aos nossos revisores já sobrecarregados. Os revisores provavelmente esquecerão do que se tratava este PR enquanto aguardam o follow-up, tornando a próxima revisão mais difícil. Além disso, os revisores terão que acompanhar os follow-ups que estão aguardando, certificando-se de que eles realmente aconteçam. Se a alteração puder ser feita de forma que seja verdadeiramente ortogonal ao PR original, para que algum outro revisor possa revisá-la, a largura de banda seria um problema menor. Na nossa experiência, isso raramente acontece.
