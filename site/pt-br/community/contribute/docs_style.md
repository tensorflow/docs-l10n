# Guia de estilo de documentação do TensorFlow

## Melhores práticas

- Concentre-se na intenção do usuário e no público-alvo.
- Use palavras do dia a dia e mantenha as frases curtas.
- A construção das frases, a redação e o uso de letras maiúsculas deve ser consistente.
- Use títulos e listas para facilitar o uso do documento como referência.
- O [Guia de estilo do Google Developer Docs](https://developers.google.com/style/highlights) é uma referência útil.

## Markdown

Com algumas exceções, a documentação do TensorFlow usa uma sintaxe Markdown semelhante ao [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/) (GFM). Esta seção explica as diferenças entre a sintaxe do GFM Markdown e o Markdown usado para a documentação do TensorFlow.

### Escreva sobre código

#### Símbolos e código dentro do texto

Coloque <code>`crases`</code> em volta dos seguintes símbolos quando usados ​​dentro do texto:

- Nomes de argumentos: <code>`input`</code>, <code>`x`</code>, <code>`tensor`</code>
- Nomes de tensores retornados: <code>`output`</code>, <code>`idx`</code>, <code>`out`</code>
- Tipos de dados: <code>`int32`</code>, <code>`float`</code>, <code>`uint8`</code>
- Referência a outros nomes de ops no texto: <code>`list_diff()`</code>, <code>`shuffle()`</code>
- Nomes de classe: <code>`tf.Tensor`</code>, <code>`Strategy`</code>
- Nomes de arquivo: <code>`image_ops.py`</code>, <code>`/path_to_dir/file_name`</code>
- Expressões ou condições matemáticas: <code>`-1-input.dims() &lt;= dim &lt;= input.dims()`</code>

#### Blocos de código

Use três crases para abrir e fechar um bloco de código. Opcionalmente, especifique a linguagem de programação após o primeiro grupo de crases, por exemplo:

<pre><code>
```python
# some python code here
```
</code></pre>

### Links em Markdown e notebooks

#### Links entre arquivos num repositório

Use links relativos entre arquivos num único repositório GitHub. Inclua a extensão do arquivo.

Por exemplo, **este arquivo que você está lendo** é do repositório [https://github.com/tensorflow/docs](https://github.com/tensorflow/docs). Portanto, ele pode usar caminhos relativos para vincular a outros arquivos no mesmo repositório como este:

- <code>\[Basics\]\(../../guide/basics.ipynb\)</code> produz [Basics](../../guide/basics.ipynb).

Esta é a abordagem preferida porque desta forma os links em [tensorflow.org](https://www.tensorflow.org), [GitHub](https://github.com/tensorflow/docs) {:.external} e [Colab](https://github.com/tensorflow/docs/tree/master/site/en/guide/bazics.ipynb) {:.external} funcionam. Além disso, o leitor permanece no mesmo site quando clica num link.

Observação: você deve incluir a extensão do arquivo — como `.ipynb` ou `.md` — para links relativos. Ele será renderizado em `tensorflow.org` sem extensão.

#### Links externos

Para links para arquivos que não estão no repositório atual, use os links Markdown padrão com a URI completa. Prefira vincular à URI [tensorflow.org](https://www.tensorflow.org) se estiver disponível.

Para vincular ao código-fonte, use um link começando com <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>, seguido pelo nome do arquivo começando na raiz do GitHub.

Ao criar um link para fora de [tensorflow.org](https://www.tensorflow.org), inclua um `{:.external}` no link Markdown para que o símbolo de "link externo" seja mostrado.

- `[GitHub](https://github.com/tensorflow/docs){:.external}` produz [GitHub](https://github.com/tensorflow/docs) {:.external}

Não inclua parâmetros de consulta de URI no link:

- Use: `https://www.tensorflow.org/guide/data`
- Não use: `https://www.tensorflow.org/guide/data?hl=en`

#### Imagens

As recomendações da seção anterior são para links que têm páginas como destino. Imagens são tratadas de maneira diferente.

Geralmente, você não deve fazer check-in de imagens no seu repositório e, em vez disso, adicionar a [equipe TensorFlow-Docs](https://github.com/tensorflow/docs) ao seu pull request e pedir que hospedem as imagens no [tensorflow.org](https://www.tensorflow.org). Isto ajuda a manter o tamanho do seu repositório pequeno.

Se você enviar imagens para o seu repositório, observe que alguns sistemas não lidam bem com caminhos relativos a imagens. Prefira usar uma URL completa apontando para a eventual localização da imagem em [tensorflow.org](https://www.tensorflow.org).

#### Links para a documentação da API

Os links para a API são convertidos quando o site é publicado. Para vincular à página de referência da API de um símbolo, coloque o caminho do símbolo entre crases:

- <code>`tf.data.Dataset`</code> produz [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

Caminhos completos são geralmente preferidos, exceto para caminhos longos. Os caminhos podem ser abreviados eliminando os componentes iniciais do caminho. Caminhos parciais serão convertidos em links se:

- Houver pelo menos um `.` no caminho, e
- O caminho parcial for único dentro do projeto.

Caminhos na API são vinculados **para cada projeto** com uma API Python publicada em [tensorflow.org](https://www.tensorflow.org). Você pode facilmente criar links para vários subprojetos a partir de um único arquivo agrupando os nomes da API com crases. Por exemplo:

- <code>`tf.metrics`</code>, <code>`tf_agents.metrics`</code>, <code>`text.metrics`</code> produz: `tf.metrics`, `tf_agents.metrics`, `text.metrics`.

Para símbolos com múltiplos aliases de caminho, há uma ligeira preferência pelo caminho que corresponde à página da API em [tensorflow.org](https://www.tensorflow.org). Todos os aliases serão redirecionados para a página correta.

### Matemática em Markdown

Você pode usar MathJax no TensorFlow ao editar arquivos Markdown, mas observe o seguinte:

- MathJax é renderizado corretamente em [tensorflow.org](https://www.tensorflow.org).
- MathJax não é renderizado corretamente no GitHub.
- A notação pode desanimar desenvolvedores que não têm familiaridade com ela.
- Por consistência, [tensorflow.org](https://www.tensorflow.org) segue as mesmas regras do Jupyter/Colab.

Use <code>$$</code> em torno de um bloco de MathJax:

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

Envolva expressões MathJax dentro do texto com <code>$ ... $</code>:

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $
</code></pre>

Este é um exemplo de uma expressão MathJax que aparece dentro do texto: $ 2 \times 2 = 4 $

Os delimitadores <code>\( ... \)</code> também funcionam para matemática dentro do texto, mas o formato com $ às vezes é mais legível.

Obseervação: Se você precisar usar um cifrão em texto ou expressões MathJax, escape-o com uma barra antecedendo o símbolo: `\$`. Os cifrões dentro de blocos de código (como nomes de variáveis ​​Bash) não precisam ser escapados.

## Estilo de prosa

Se você for escrever ou editar partes substanciais da documentação narrativa, leia o [Guia de estilo de documentação do desenvolvedor Google](https://developers.google.com/style/highlights).

### Princípios do bom estilo

- *Verifique a ortografia e a gramática de suas contribuições.* A maioria dos editores inclui um corretor ortográfico ou tem um plugin de verificação ortográfica disponível. Você também pode colar seu texto num Google Doc ou outro software de documentos para uma verificação ortográfica e gramatical mais robusta.
- *Use um tom casual e cordial.* Escreva a documentação do TensorFlow como se fosse uma conversa, como se você estivesse conversando com outra pessoa individualmente. Use um tom de apoio no artigo.

Observação: Ser menos formal não significa ser menos técnico. Simplifique sua prosa, não o conteúdo técnico.

- *Evite isenções de responsabilidade, opiniões e julgamentos de valor.* Palavras como “facilmente”, “apenas” e “simples” estão carregadas de suposições. Algo pode parecer fácil para você, mas será difícil para outra pessoa. Tente evitá-las sempre que possível.
- *Use frases simples e diretas, sem jargões complexos.* Frases compostas, cadeias de cláusulas e expressões idiomáticas específicas de uma região podem dificultar a compreensão e a tradução do texto. Se uma frase puder ser dividida em duas frases, provavelmente deve ser dividida. Evite usar ponto e vírgula. Use listas com marcadores quando apropriado.
- *Forneça contexto.* Não use abreviaturas sem explicá-las. Não mencione projetos que não sejam do TensorFlow sem vincular a eles. Explique por que o código foi escrito da maneira como foi.

## Guia de uso

### Ops

Em arquivos markdown, use `# ⇒` em vez de um único sinal de igual quando quiser mostrar o que uma operação retorna.

```python
# 'input' is a tensor of shape [2, 3, 5]
tf.expand_dims(input, 0)  # ⇒ [1, 2, 3, 5]
```

Nos notebooks, mostre o resultado em vez de adicionar um comentário (se a última expressão numa célula do notebook não for atribuída a uma variável, ela é exibida automaticamente).

Nos documentos de referência da API, prefira usar [doctest](docs_ref.md#doctest) para mostrar resultados.

### Tensores

Quando você estiver falando sobre um tensor em geral, não ponha letra maiúscula na palavra *tensor*. Ao falar sobre o objeto específico fornecido ou retornado de uma operação, você deve usar a palavra *Tensor* com a primeira letra maiúscula e incluir crases em volta dela, já que está falando sobre um objeto `Tensor`.

Não use a palavra *Tensors* (plural) para descrever vários objetos `Tensor`, a menos que você realmente esteja falando sobre um objeto `Tensors`. Em vez disso, diga "uma lista (ou coleção) de objetos `Tensor`".

Use a palavra *formato* para detalhar os eixos de um tensor e mostre o formato entre colchetes, cercado por crases. Por exemplo:

<pre><code>
If `input` is a three-axis `Tensor` with shape `[3, 4, 3]`, this operation
returns a three-axis `Tensor` with shape `[6, 8, 6]`.
</code></pre>

Como acima, prefira usar "axis" (eixo) ou "index" (índice) em vez de "dimension" (dimensão) ao falar sobre os elementos do formato de um `Tensor`. Caso contrário, fica fácil confundir “dimension” com a dimensão de um espaço vetorial. Um "vetor tridimensional" possui um único eixo com comprimento 3.
