# 配置 Visual Studio Code

Visual Studio Code (VSCode) 是一款免费的代码编辑器，可在 macOS、Linux 和 Windows 操作系统上运行。它具有良好的工具支持能力，支持 Python 和 C++ 开发，可视化调试，与 Git 集成，以及很多其他有趣的功能。得益于易用性和扩展程序管理，它是一款不错的 TensorFlow IO 开发编辑器。但是，您需要花一点精力来正确配置编辑器。由于 VSCode 配置非常灵活，它允许开发者使用 Bazel 编译项目，并在 Python 和 C++ 调试器下运行代码。在不同的操作系统上，基本工具设置可能有所不同，但是配置方法应该大同小异。

## 扩展程序

要安装扩展程序，请点击边栏上的扩展程序视图图标 (Extensions)，或者使用快捷键 Ctrl+Shift+X。然后，请搜索以下关键词。

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) - Microsoft 发布的官方 C++ 扩展程序
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Microsoft 发布的官方 Python 扩展程序
- [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) - 用于 Python 开发的另一个实用的扩展程序

## 编译项目

TensorFlow IO 使用 Bazel 构建命令进行编译：

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

有关在 Ubuntu 上如何设置开发环境的详细信息，请查看项目[自述](https://github.com/tensorflow/io#ubuntu-18042004)文件。其中的 --compilation_mode dbg 标记表示产生的二进制文件应包含调试符号。一旦您可以从命令行编译项目，还可以对 VSCode 进行配置，以便调用相同的命令。

打开 View->Command Pallete (**Ctrl+Shift+P**)，然后开始输入：“Tasks: Configure Build Task”。如果您是首次执行此操作，编辑器会建议创建 tasks.json 文件。创建该文件后，请粘贴以下 json 内容：

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

现在，您可以按 **Ctrl+Shift+B**，VSCode 随即会使用上面的命令构建项目。它会使用自己的终端窗口，其中所有链接都可以点击。如果发生编译错误，您可以点击终端窗口中的链接，打开相应的文件，然后找到发生错误的代码行。

## 调试项目

调试 Python 代码易如反掌，请参考官方文档，了解如何配置 VSCode，以便启用调试：https://code.visualstudio.com/docs/python/debugging。

但是，调试 C++ 代码要求在系统上安装 [GDB](https://www.gnu.org/software/gdb/)。如果您有使用 `tensorflow-io` 库的 `bq_sample_read.py` Python 脚本，则通常按以下方式执行：

```sh
python3 bq_sample_read.py --gcp_project_id=...
```

使用以下命令，您可以在 GDB 下执行：

```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

如果应用在 C++ 代码阶段崩溃，您可以在 GDB 控制台中运行 `backtrace`，获取错误的堆栈跟踪。

VSCode 还提供了 GDB 调试器支持。它允许添加断点，观察变量的值，以及逐步执行代码。要添加调试配置，请按边栏上的调试视图图标（Debug），或者使用快捷键 **Ctrl+Shift+D**。然后，请按 Play 按钮旁的向下小箭头，并选择“Add Configuration…”。现在，它会创建一个 `launch.json` 文件，请将以下配置添加到该文件中：

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

如果全部配置正确，您应该可以执行 *Run -> Start Debugging* (**F5**) 或 *Run -> Run Without Debugging* (**Ctrl + F5**)。这样就会在调试器下运行您的代码：

![VSCode debugger](./images/vscode_debugger.png)

为了进一步简化调试体验，您可以将 GDB 配置为跳过标准 C++ 库。这样，您就可以忽略不需要关注的代码。为此，请创建一个包含以下内容的 `~/.gdbinit` 文件：

```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## 格式化文件

通过*点击右键 -> Format Document* (**Ctrl + Shift + I**)，您随时可以重新格式化 C++ 或 Python 文件，但是，VSCode 使用不同的样式惯例。幸运地是，这很容易更改。

对于 Python 格式化，请参阅 https://donjayamanne.github.io/pythonVSCodeDocs/docs/formatting/

对于 C++ 格式化，请按以下步骤操作：

- 转到 *Preferences -> Settings*
- 搜索“C_Cpp.clang_format_fallbackStyle”
- 通过添加以下内容直接修改 `file:setting.json` 文件

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
