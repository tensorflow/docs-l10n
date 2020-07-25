# Configuring Visual Studio Code

Visual Studio is a free code editor, which runs on the macOS, Linux, and Windows operating systems.

It has nice tooling for Python and C++ development, visual debugger, git integration, and many more
useful features. It is a great editor to use for TensorFlow IO development, but it takes some effort
to configure it properly. VSCode configuration is very flexible, it allows compiling project using
bazel, and running code under Python and C++ debuggers. This manual is for Linux, other OSes
might have specifics, but approach should be similar.


## Extensions
To install an extension click the extensions view icon (Extensions) on the Sidebar, or use the shortcut Ctrl+Shift+X.
Then searh for keyword below.

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) - Official C++ extension from Microsoft
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Official Python extension from Microsoft
- [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) - another useful extension for Python development

## Compiling projects
TensorFlow IO is compiled using bazel build command:

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

See project [README](https://github.com/tensorflow/io#ubuntu-18042004) file for details on how to setup development environment in Ubuntu.
--compilation_mode dbg flag here indicates that produced binary should have debug symbols.
Once you can compile project from command line, you can also configure VSCode to be able to invoke same command.

Open View->Command Pallete (Ctrl+Shift+P) and start typing: "Tasks: Configure Build Task".
If you are doing this for the first time, editor is going to suggest creating tasks.json file.
Once you have it, paste following json:

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

Now, you can press "Ctrl+Shift+B", and VSCode is going to use the command above to build the project.
It uses its own terminal window, where all links are clickable. So when compilation error occurs, you can
just click the link, and editor will open corresponding file and navigate to the line.

## Debugging projects
Debugging Python code is trivial, follow official documentation to figure out how to configure VSCode to enable that: https://code.visualstudio.com/docs/python/debugging
Debugging C++ code requires GDB to be installed on your system.
If you have a bq_sample_read.py python script that is using tensorflow-io library that is normally
executed like:
```sh
python3 bq_sample_read.py --gcp_project_id=...
```

In order to execute it under GDB, run following:
```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

If application crashes in C++ code, you can run ```backtrace``` in GDB console to get stacktrace.

VSCode also has GDB debugger support, it allows adding breakpoints, see values of variables and step through the code.
To add debug configuration press the Debug View icon (Debug) on the Sidebar, or use the shortcut Ctrl+Shift+D. Here press the little down arrow next to the play button and select "Add Configuration...".
It will create launch.json file, add following config here:

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

If everything is configured correctly, you should be able to do Run -> Start Debugging (F5) or Run -> Run Without Debugging (Ctrl + F5). This will run your code under debugger:

![VSCode debugger](./images/vscode_debugger.png)

One other thing worth doing to simplify debugging experience is configuting GDB to skip standard C++ libraries, so you don't step into code you don't care about. In order to do this, create ```~/.gdbinit``` file with following content:
```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## Formatting files
You can always reformat C++ or Python file by Right Click -> Format Document (Ctrl + Shift + I), but VSCode uses different style conention. Luckily it is easy to change.

For Python formatting, see https://donjayamanne.github.io/pythonVSCodeDocs/docs/formatting/

To configure C++ formatter, do following:

- Go Preferences -> Settings
- Search C_Cpp.clang_format_fallbackStyle
- Modify the file:setting.json directly
- Add following

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
