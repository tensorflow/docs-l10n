# Visual Studio Code 구성하기

Visual Studio Code(VSCode)는 macOS, Linux 및 Windows 운영 체제에서 실행되는 무료 코드 편집기입니다. Python 및 C++ 개발, 시각적 디버깅, git과의 통합 및 기타 다양하고 흥미로운 기능을 지원하는 우아한 도구 지원이 제공됩니다. 사용 및 확장 관리가 용이하다는 점에서 TensorFlow IO 개발을 위한 편집기로 매우 훌륭합니다. 그러나 적합하게 구성하려면 약간의 노력이 필요합니다. VSCode 구성은 매우 유연하기 때문에 개발자는 bazel을 사용하여 프로젝트를 컴파일하고 Python 및 C++ 디버거에서 코드를 실행할 수 있습니다. 기본 도구 설정은 운영 체제에 따라 다를 수 있지만 구성 방법은 비슷합니다.

## 확장

확장을 설치하려면 사이드바에서 확장 보기 아이콘(확장)을 클릭하거나 단축키 Ctrl+Shift+X를 사용합니다. 그런 다음 아래 키워드를 검색합니다.

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) - Microsoft의 공식 C++ 확장
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Microsoft의 공식 Python 확장
- [Python 확장 팩](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) - Python 개발을 위한 또 다른 유용한 확장

## 프로젝트 컴파일하기

TensorFlow IO는 bazel build 명령을 사용하여 컴파일됩니다.

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

Ubuntu에서 개발 환경을 설정하는 방법에 대한 자세한 내용은 프로젝트 [README](https://github.com/tensorflow/io#ubuntu-18042004) 파일을 참조하세요. --여기서 compilation_mode dbg 플래그는 생성된 바이너리에 디버그 기호가 있어야 함을 나타냅니다. 명령줄에서 프로젝트를 컴파일할 수 있으면 동일한 명령을 호출할 수 있도록 VSCode를 구성할 수도 있습니다.

View-&gt;Command Pallete(**Ctrl+Shift+P**)를 열고 "Tasks: Configure Build Task" 입력을 시작합니다. 처음 이 작업을 수행하는 경우 편집기에서 tasks.json 파일을 생성하도록 제안합니다. 파일이 준비되었으면 다음 json을 붙여넣습니다.

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

이제 **Ctrl+Shift+B**를 누르면 VSCode가 위의 명령을 사용하여 프로젝트를 빌드합니다. 여기서 모든 링크를 클릭할 수 있는 자체 터미널 창이 사용됩니다. 따라서 컴파일 오류가 발생하면 해당 파일을 열고 터미널 창에서 링크를 클릭하여 해당 행으로 이동합니다.

## 프로젝트 디버깅하기

Python 코드 디버깅은 간단합니다. 공식 설명서에 따라 VSCode를 구성하는 방법을 알아보세요(https://code.visualstudio.com/docs/python/debugging).

그러나 C++ 코드를 디버깅하려면 시스템에 [GDB](https://www.gnu.org/software/gdb/)를 설치해야 합니다. `tensorflow-io` 라이브러리를 사용하고 일반적으로 다음과 같은 방식으로 실행되는 `bq_sample_read.py` Python 스크립트가 있는 경우:

```sh
python3 bq_sample_read.py --gcp_project_id=...
```

다음을 사용하여 GDB에서 실행할 수 있습니다.

```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

C++ 코드 단계에서 애플리케이션이 충돌하는 경우 GDB 콘솔에서 `backtrace`를 실행하여 오류의 스택 추적을 가져올 수 있습니다.

VSCode에는 GDB 디버거 지원도 있습니다. 중단점을 추가하고 변수 값을 관찰하며 단계별로 코드를 실행할 수 있습니다. 디버그 구성을 추가하려면 사이드바에서 Debug View 아이콘(Debug)을 누르거나 단축키 **&nbsp;Ctrl+Shift+D**를 사용합니다. 여기에서 재생 버튼 옆에 있는 작은 아래쪽 화살표를 누르고 "Add Configuration..."를 선택합니다. 이제 `launch.json` 파일이 생성되며 여기에 다음 구성을 추가합니다.

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

모두 올바르게 구성되었으면 *Run -&gt; Start Debugging*(**F5**) 또는 *Run -&gt; Run Without Debugging*(**Ctrl + F5**)를 수행할 수 있습니다. 그러면 디버거에서 코드가 실행됩니다.

![VSCode 디버거](./images/vscode_debugger.png)

디버깅 환경을 더욱 단순화하기 위해 표준 C++ 라이브러리를 건너뛰도록 GDB를 구성할 수 있습니다. 이렇게 하면 신경 쓰지 않아도 되는 코드를 무시할 수 있습니다. 이를 위해 다음 내용으로 `~/.gdbinit` 파일을 만듭니다.

```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## 파일 형식 지정하기

*마우스 오른쪽 버튼 클릭 -&gt; Format Document*(**Ctrl + Shift + I**)을 사용하여 언제든지 C++ 또는 Python 파일의 형식을 변경할 수 있지만 VSCode는 다른 스타일 규칙을 사용합니다. 다행인 것은 변경이 쉽다는 것입니다.

Python 형식 지정에 대해서는 https://donjayamanne.github.io/pythonVSCodeDocs/docs/formatting/을 참조하세요.

C++ 형식 지정의 경우 다음을 수행합니다.

- *Preferences -&gt; Settings*로 이동합니다.
- "C_Cpp.clang_format_fallbackStyle"을 검색합니다.
- 다음 내용을 추가하여 직접 `file:setting.json` 파일을 수정합니다.

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
