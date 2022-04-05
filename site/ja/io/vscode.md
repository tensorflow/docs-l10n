# Visual Studio コードを構成する

Visual Studio コード（VSCode）は、macOS、Linux、および Windows オペレーティングシステムで実行する無料のコードエディタです。Python と C++ 開発、ビジュアルデバッグ、Git との統合、その他多数の興味深い機能をサポートする魅力的なツールサポートを備えています。使いやすさと拡張機能の管理しやすさにより、TensorFlow IO 開発用の優れたエディタと言えますが、適切に構成するには多少の努力が必要となります。VSCode 構成は非常に柔軟であるため、開発者は Bazel を使用してプロジェクトをコンパイルし、Python と C++ デバッガでコードを実行することができます。基本のツールセットアップはオペレーティングシステムによって異なりますが、構成のアプローチは類似しています。

## 拡張機能

拡張機能をインストールするには、サイドバーの拡張機能ビューアイコン（Extensions）をクリックするか、Ctrl+Shift+X のショートカットを使用し、次のキーワードを検索します。

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) - Microsoft が提供する公式の C++ 拡張機能
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Microsoft が提供する公式の Python 拡張機能
- [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) - Python 開発に有用なもう 1 つの拡張機能

## プロジェクトをコンパイルする

TensorFlow IO は Bazel ビルドコマンドを使用してコンパイルされています。

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

Ubuntu で開発環境をセットアップする方法については、プロジェクトの [README](https://github.com/tensorflow/io#ubuntu-18042004) ファイルをご覧ください。上記の --compilation_mode dbg フラグは、生成されるバイナリにデバッグシンボルを含むことを示します。コマンドラインからプロジェクトをコンパイルできるようになったら、同じコマンドを VSCode から呼び出せるように構成することができます。

表示 -&gt;コマンド パレット（**Ctrl+Shift+P**）を開き、"Tasks: Configure Build Task" と入力し始めます。初めてこれを行う場合、エディタから tasks.json ファイルを作成するように提案されます。これを作成したら、次の json を貼り付けます。

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

これで、**Ctrl+Shift+B** を押すと、VSCode は上記のコマンドを使用して、プロジェクトを構築できるようになりました。VSCode には独自のターミナルウィンドウがあり、すべてのリンクがクリック可能です。そのため、コンパイルエラーが発生した場合は、ターミナルウィンドウ内のリンクをクリックするだけで、対応するファイルを開いて、エラーのある行に移動することができます。

## プロジェクトをデバッグする

Python コードのデバッグの作業は簡単です。VSCode でデバックを行えるよう構成する方法は、公式ドキュメントに従ってください（https://code.visualstudio.com/docs/python/debugging）。

ただし、C++ コードのデバッグには、システムに [GDB](https://www.gnu.org/software/gdb/) がインストールされている必要があります。`tensorflow-io` ライブラリを使用し、通常は次のように実行される `bq_sample_read.py` Python スクリプトがあったとします。

```sh
python3 bq_sample_read.py --gcp_project_id=...
```

この場合、次のコードを使用して、GDB で実行することができます。

```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

C++ コードの段階でアプリケーションがクラッシュする場合、GDB コンソールで `backtrace` を実行して、エラーのスタックトレースを取得することができます。

VSCode には、GDB デバッガサポートもあります。ブレイクポイントを追加できるため、変数の値を観察し、ステップごとにコードを実行することができます。デバッグ構成を追加するには、サイドバーのデバッグビューアイコン（デバッグ）をクリックするか、**Ctrl+Shift+D** ショートカットを使用します。ここで、再生ボタンの隣にある小さな下向きの矢印を押し、"構成の追加..." を選択します。すると、`launch.json` ファイルが作成されるので、それに次の config を追加してください。

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

すべてが正しく構成されると、*実行 -&gt; デバッグの開始*（**F5**）または*実行 -&gt; デバッグなしで実行する*（**Ctrl + F5**）を使用できるようになり、コードがデバッガで実行されるようになります。

![VSCode debugger](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/io/images/vscode_debugger.png)

デバッグのエクスペリエンスをさらに単純化するには、標準 C++ ライブラリをスキップするように GDB を構成することができます。こうすることで、必要でないコードを無視できるようになります。これを行うには、次の内容で `~/.gdbinit` ファイルを作成します。

```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## ファイルの書式を設定する

C++ または Python ファイルは、*右クリック -&gt; ドキュメントのフォーマット*（**Ctrl + Shift + I**）を使ってフォーマットし直すことはできますが、VSCode が使用するスタイル規則は異なります。幸いにも、この変更は簡単に行えます。

Python の書式については、 https://donjayamanne.github.io/pythonVSCodeDocs/docs/formatting/ をご覧ください。

C++ の書式については、次の操作を行います。

- *環境設定 -&gt; 設定* に移動します。
- 「C_Cpp.clang_format_fallbackStyle」を検索します。
- 次の内容を追加して、`file:setting.json` ファイルを直接変更します。

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
