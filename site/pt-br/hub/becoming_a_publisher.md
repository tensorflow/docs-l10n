<!--* freshness: { owner: 'maringeo' } *-->

# Como se tornar publicador

## Termos de serviço

Ao enviar um modelo para publicação, você concorda com os Termos de Serviço do TensorFlow Hub, disponíveis em [https://tfhub.dev/terms](https://tfhub.dev/terms).

## Visão geral do processo de publicação

O processo completo de publicação é composto por três etapas:

1. Criar o modelo (veja como [exportar um modelo](exporting_tf2_saved_model.md))
2. Redigir a documentação (veja como [escrever a documentação do modelo](writing_model_documentation.md))
3. Criar uma solicitação de contribuição (veja como [contribuir](contribute_a_model.md))

## Formato específico Markdown da página de publicador

A documentação do publicador é declarada com o mesmo tipo de arquivos Markdown descritos no guia como [escrever a documentação do modelo](writing_model_documentation), com pequenas diferenças sintáticas:

O local correto do arquivo de publicador no repositório do TensorFlow Hub é: [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/&lt;nome_do_publicador&gt;/&lt;nome_do_publicador.md&gt;

Confira um exemplo mínimo de documentação de publicador:

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

O exemplo acima especifica o nome do publicador, uma breve descrição, o caminho do ícone a ser usado e uma documentação Markdown maior em formato livre.

### Diretrizes de nome de publicador

O nome de publicador pode ser seu nome de usuário do GitHub ou o nome da organização do GitHub que você gerencia.
