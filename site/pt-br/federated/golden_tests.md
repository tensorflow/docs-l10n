# Testes golden

O TFF inclui uma pequena biblioteca chamada `golden` que facilita a escrita e a manutenção de testes golden.

## O que são testes golden? Quando devo usá-los?

Os testes golden são usados ​​quando você deseja que um desenvolvedor saiba que seu código alterou a saída de uma função. Eles violam muitas características de bons testes unitários, pois fazem promessas sobre as saídas exatas das funções, em vez de testar um conjunto específico de propriedades claras e documentadas. Às vezes não fica claro quando uma mudança em uma saída golden é "esperada" ou se está violando alguma propriedade que o teste golden procurou impor. Como tal, um teste unitário bem fatorado é geralmente preferível a um teste golden.

No entanto, os testes golden podem ser extremamente úteis para validar o conteúdo exato de mensagens de erro, diagnósticos ou código gerado. Nesses casos, os testes golden podem ser uma verificação útil de confiança de que quaisquer alterações na saída gerada "parecem corretas".

## Como devo escrever testes usando `golden` ?

`golden.check_string(filename, value)` é o ponto de entrada principal na biblioteca `golden`. Ele verificará a string `value` em relação ao conteúdo de um arquivo cujo último elemento do caminho é `filename`. O caminho completo para `filename` deve ser fornecido por meio de um argumento de linha de comando `--golden <path_to_file>`. Da mesma forma, esses arquivos devem ser disponibilizados para testes usando o argumento `data` da regra `py_test` BUILD. Use a função `location` para gerar um caminho relativo apropriado e correto:

```
py_string_test(
  ...
  args = [
    "--golden",
    "$(location path/to/first_test_output.expected)",
    ...
    "--golden",
    "$(location path/to/last_test_output.expected)",
  ],
  data = [
    "path/to/first_test_output.expected",
    ...
    "path/to/last_test_output.expected",
  ],
  ...
)
```

Por convenção, os arquivos golden devem ser colocados em um diretório irmão com o mesmo nome do alvo de teste, com o sufixo `_goldens`:

```
path/
  to/
    some_test.py
    some_test_goldens/
      test_case_one.expected
      ...
      test_case_last.expected
```

## Como atualizar arquivos `.expected`?

Os arquivos `.expected` podem ser atualizados executando o alvo de teste afetado com os argumentos `--test_arg=--update_goldens --test_strategy=local`. A diferença resultante deve ser verificada quanto a alterações imprevistas.
