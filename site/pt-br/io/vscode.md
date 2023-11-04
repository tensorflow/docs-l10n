# Configurando o Visual Studio Code

O Visual Studio Code (VSCode) é um editor de código gratuito executado nos sistemas operacionais macOS, Linux e Windows. Ele tem um conjunto de ferramentas elegante com suporte a desenvolvimento de Python e C++, depuração visual, integração com o git e muitos outros recursos interessantes. Devido à facilidade de uso e gerenciamento de extensões, é um excelente editor para desenvolvimento do TensorFlow IO. Porém, é necessário um certo esforço para configurá-lo corretamente. Como a configuração do VSCode é bem flexível, ele permite que os desenvolvedores compilem o projeto usando o Bezel e executem o código em depuradores Python e C++. A configuração base da ferramenta pode ser diferente dependendo do sistema operacional, mas a estratégia de configuração é parecida.

## Extensões

Para instalar uma extensão, clique no ícone de exibir extensões na barra lateral ou use o atalho Ctrl + Shift + X. Em seguida, pesquise a palavra-chave abaixo:

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) – Extensão oficial do C++ da Microsoft
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) – Extensão oficial do Python da Microsoft
- [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) – Outra extensão útil para desenvolvimento de Python

## Compilando projetos

O TensorFlow IO é compilado usando-se o comando build do Bazel:

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

Veja o arquivo [README](https://github.com/tensorflow/io#ubuntu-18042004) do projeto para conferir os detalhes de como configurar o ambiente de desenvolvimento no Ubuntu. O sinalizador --compilation_mode dbg indica que o binário produzido deve ter símbolos de depuração. Após você compilar o projeto pela linha de comando, pode configurar o VSCode para poder chamar o mesmo comando.

Abra View -&gt; Command Pallete (Exibir -&gt; Paleta de comandos) (**Ctrl + Shift + P**) e comece a digitar: "Tasks: Configure Build Task" (tarefas: configurar tarefa de compilação). Se for sua primeira vez, o editor vai sugerir a criação do arquivo tasks.json. Quando você tiver esse arquivo, cole o seguinte código json.

```jsonc
{
	"version": "2.0.0",
	"tasks": [
		{
			"label": "Build TF.IO (Debug)",
			"type": "shell",
			"command": "bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...",
			"group": {
				"kind": "build",
				"isDefault": true
			},
			"problemMatcher": []
		}
	]
}
```

Agora, você pode pressionar **Ctrl + Shift + B**, e o VSCode usará o comando acima para compilar o projeto. Ele usa sua própria janela de terminal, em que é possível clicar em todos os links. Portanto, quando houver um erro de compilação, você pode abrir o arquivo correspondente e navegar até a linha clicando no link na janela do terminal.

## Depurando projetos

Depurar código Python é trivial. Confira a documentação oficial para descobrir como configurar o VSCode para poder depurar: https://code.visualstudio.com/docs/python/debugging.

Porém, para depurar código C++, é preciso instalar o [GDB](https://www.gnu.org/software/gdb/) no seu sistema. Se você tiver um script `bq_sample_read.py` do Python que use a biblioteca `tensorflow-io` e que normalmente seja executado da seguinte maneira:

```sh
python3 bq_sample_read.py --gcp_project_id=...
```

Você pode executar no GDB usando o seguinte comando:

```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

Se a aplicação travar na fase de código C++, você pode executar `backtrace` no console do GDB para obter o stacktrace do erro.

O VSCode também tem suporte ao depurador do GDB. Ele permite adicionar pontos de interrupção (breakpoints), observar os valores de variáveis e executar o código passo a passo. Para adicionar a configuração de depuração, pressione o ícone Visualizar Depurador na barra lateral ou use o atalho **Ctrl + Shift + D**. Agora, pressione a seta para baixo pequena ao lado do botão de execução e selecione "Add Configuration..." (Adicionar configuração). Será criado um arquivo `launch.json`. Adicione a seguinte configuração dentro dele:

```jsonc
{
    "name": "(gdb) Launch",
    "type": "cppdbg",
    "request": "launch",
    "program": "/usr/bin/python3",
    "args": ["bq_sample_read.py", "--gcp_project_id=..."],
    "stopAtEntry": false,
    "cwd": "${workspaceFolder}",
    "environment": [
        {
            /* path to your bazel-bin folder */
            "name": "TFIO_DATAPATH",
            "value": "/usr/local/google/home/io/bazel-bin"
        },
        {
            /* other env variables to use */
            "name": "GOOGLE_APPLICATION_CREDENTIALS",
            "value": "..."
        }
    ],
    "externalConsole": false,
    "MIMode": "gdb",
    "setupCommands": [
        {
            "description": "Enable pretty-printing for gdb",
            "text": "-enable-pretty-printing",
            "ignoreFailures": true
        }
    ]
}
```

Se tudo estiver configurado corretamente, você deverá poder usar *Run -&gt; Start Debugging* (Executar -&gt; Começar depuração) (**F5**) ou *Run -&gt; Run Without Debugging* (Executar -&gt; Executar sem depuração) (**Ctrl + F5**). O seu código será executado com o depurador:

![VSCode debugger](./images/vscode_debugger.png)

Para simplificar ainda mais a experiência de depuração, você pode configurar o GDB para ignorar as bibliotecas padrão do C++, o que permite ignorar o código que não importa para você. Para fazer isso, crie um arquivo `~/.gdbinit` com o seguinte conteúdo:

```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## Formatando arquivos

Você sempre pode reformatar o arquivo C++ ou Python da seguinte forma: *Clique com o botão direito do mouse -&gt; Format Document* (Formatar documento) (**Ctrl + Shift + I**), mas o VSCode usa uma convenção de estilo diferente. Felizmente, é fácil de alterar.

Para formatar Python, confira https://donjayamanne.github.io/pythonVSCodeDocs/docs/formatting/

Para formatar C++, faça o seguinte:

- Acesse *Preferences -&gt; Settings* (Preferências -&gt; Ajustes)
- Procure "C_Cpp.clang_format_fallbackStyle"
- Modifique o arquivo `file:setting.json` diretamente e adicione o seguinte conteúdo:

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
