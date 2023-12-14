# Configurar el código de Visual Studio

Visual Studio Code (VSCode) es un editor de código gratuito que se ejecuta en los sistemas operativos macOS, Linux y Windows. Tiene soporte de herramientas elegante que admite el desarrollo de Python y C++, la depuración visual, la integración con git y muchas funciones interesantes más. Debido a su facilidad de uso y gestión de extensiones, es un excelente editor para el desarrollo de TensorFlow IO. Sin embargo, hay que esforzarse un poco para configurarlo correctamente. Dado que la configuración de VSCode es muy flexible, permite a los desarrolladores compilar proyectos con bazel y ejecutar el código en los depuradores de Python y C++. La configuración de la herramienta básica puede diferir según los sistemas operativos, pero el enfoque de configuración debe ser similar.

## Extensiones

Para instalar una extensión, haga clic en el icono de vista de extensiones (Extensiones) en la barra lateral o use el acceso directo Ctrl+Shift+X. Luego busque una de las siguientes palabras clave.

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools): extensión oficial de C++ de Microsoft
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): extensión oficial de Python de Microsoft
- [Paquete de extensión de Python](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack): otra extensión útil para el desarrollo de Python

## Compilar proyectos

TensorFlow IO se compila mediante el comando de compilación de bazel:

```sh
bazel build -s --verbose_failures --compilation_mode dbg //tensorflow_io/...
```

Consulte el archivo [README](https://github.com/tensorflow/io#ubuntu-18042004) del proyecto para obtener detalles sobre cómo configurar el entorno de desarrollo en Ubuntu. Aquí, el indicador --compilation_mode dbg significa que el binario producido debe tener símbolos de depuración. Una vez que se pueda compilar el proyecto desde la línea de comandos, también puede configurar VSCode para poder invocar el mismo comando.

Abra View (Ver)-&gt; Command Pallete (Paleta de comandos) (**Ctrl+Shift+P**) y comience a escribir: "Tasks: Configure Build Task" (Tareas: Configurar tarea de compilación). Si está haciendo esto por primera vez, el editor le sugerirá crear el archivo task.json. Una vez que lo tenga pegue el siguiente json:

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

Ahora, puede presionar **Ctrl+Shift+B** y VSCode usará el comando anterior para generar el proyecto. Usará su propia ventana de terminal, donde se puede hacer clic en todos los enlaces. Entonces, cuando ocurre un error de compilación, puede abrir el archivo correspondiente y navegar hasta la línea con tan solo hacer clic en el enlace en la ventana del terminal.

## Proyectos de depuración

Depurar el código Python es trivial; siga la documentación oficial para descubrir cómo configurar VSCode para habilitarlo: https://code.visualstudio.com/docs/python/debugging.

Sin embargo, la depuración del código C++ requiere que [GDB](https://www.gnu.org/software/gdb/) esté instalado en su sistema. Si tiene un script de Python `bq_sample_read.py` que usa la biblioteca `tensorflow-io` y normalmente se ejecuta de la siguiente manera:

```sh
python3 bq_sample_read.py --gcp_project_id=...
```

Puede ejecutarlo bajo GDB con lo siguiente:

```sh
gdb -ex r --args python3 bq_sample_read.py --gcp_project_id=...
```

Si la aplicación falla en la fase de código C++, puede ejecutar `backtrace` en la consola GDB para obtener el seguimiento de la pila del error.

VSCode también admite el depurador GDB. Permite agregar puntos de interrupción, observar valores de variables y recorrer el código paso a paso. Para agregar una configuración de depuración, presione el ícono Vista de depuración (Depurar) en la barra lateral o use el acceso directo **Ctrl+Shift+D**. Aquí, presione la pequeña flecha hacia abajo al lado del botón de reproducción y seleccione "Agregar configuración...". Ahora creará un archivo `launch.json` y debe agregarle la siguiente configuración:

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

Si todo está configurado correctamente, debería poder ejecutar *Run (Ejecutar) -&gt; Start Debugging (Iniciar depuración)* (**F5**) o *Run (Ejecutar) -&gt; Run Without Debugging (Ejecutar sin depurar)* (**Ctrl + F5**). Esto ejecutará su código en el depurador:

![VSCode debugger](./images/vscode_debugger.png)

Para simplificar aún más la experiencia de depuración, puede configurar GDB para omitir las bibliotecas estándar de C++. Esto le permite ignorar el código que no le interesa. Para hacer esto, cree un archivo `~/.gdbinit` con el siguiente contenido:

```
skip -gfi /usr/include/c++/*/*/*
skip -gfi /usr/include/c++/*/*
skip -gfi /usr/include/c++/*
```

## Formatear archivos

Siempre puede volver a formatear un archivo C++ o Python al hacer *clic derecho -&gt; Format Document (Formatear documento)* (**Ctrl + Shift + I**), pero VSCode usa una convención de estilo diferente. Por suerte, es fácil de cambiar.

Para formatear Python, consulte https://donjayamane.github.io/pythonVSCodeDocs/docs/formatting/

Para formatear C++, haga lo siguiente:

- Vaya a *Preferences (Preferencias) -&gt; SEttings (Configuración)*
- Busque "C_Cpp.clang_format_fallbackStyle"
- Modifique el archivo `file:setting.json` directamente agregando el siguiente contenido

```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google}"
```
