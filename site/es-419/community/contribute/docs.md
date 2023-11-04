# Cómo contribuir a la documentación de TensorFlow

TensorFlow agradece las contribuciones a la documentación: si mejora la documentación, mejorará la biblioteca de TensorFlow. La documentación disponible en tensorflow.org se divide en las siguientes categorías:

- *Referencia de la API*: [los documentos de referencia de la API](https://www.tensorflow.org/api_docs/) se generan a partir de cadenas de documentos en el [código fuente de TensorFlow](https://github.com/tensorflow/tensorflow).
- *Documentación narrativa*: se trata de [tutoriales](https://www.tensorflow.org/tutorials), [guías](https://www.tensorflow.org/guide) y otros escritos que no forman parte del código de TensorFlow. Esta documentación está disponible en el repositorio de GitHub [tensorflow/docs](https://github.com/tensorflow/docs).
- *Traducciones de la comunidad*: son guías y tutoriales traducidos por la comunidad. Todas las traducciones de la comunidad se encuentran en el repositorio [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site).

Algunos [proyectos de TensorFlow](https://github.com/tensorflow) mantienen los archivos fuente de documentación cerca del código en un repositorio separado, por lo general en un directorio `docs/`. Consulte el archivo `CONTRIBUTING.md` del proyecto o comuníquese con el responsable del mantenimiento para contribuir.

Si desea participar en la comunidad de documentos de TensorFlow, haga lo siguiente:

- Consulte el repositorio de GitHub [tensorflow/docs](https://github.com/tensorflow/docs).
- Siga la etiqueta [docs](https://discuss.tensorflow.org/tag/docs) en el [Foro de TensorFlow](https://discuss.tensorflow.org/).

## Referencia de la API

Para obtener más información, utilice la [guía para colaboradores de documentos de la API de TensorFlow](docs_ref.md). Allí se indica cómo encontrar el [archivo fuente](https://www.tensorflow.org/code/tensorflow/python/) y editar la <a href="https://www.python.org/dev/peps/pep-0257/" class="external">cadena de documentación</a> del símbolo. Muchas páginas de referencia de la API en tensorflow.org incluyen un enlace al archivo fuente donde se define el símbolo. Las cadenas de documentos son compatibles con <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a> y es posible acceder a una vista previa (aproximada) con cualquier <a href="http://tmpvar.com/markdown.html" class="external">visor de Markdown</a>.

### Versiones y ramas

La versión de [referencia de la API](https://www.tensorflow.org/api_docs/python/tf) del sitio es, por defecto, la versión binaria estable más reciente, que coincide con el paquete instalado con `pip install tensorflow`.

El paquete TensorFlow predeterminado se genera a partir de la rama estable `rX.x` en el repositorio principal <a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>. La documentación de referencia se genera a partir de comentarios de código y cadenas de documentación en el código fuente de <a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>, <a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a> y <a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>.

Las versiones anteriores de la documentación de TensorFlow están disponibles como [ramas rX.x](https://github.com/tensorflow/docs/branches) en el repositorio TensorFlow Docs. Estas ramas se agregan cuando se publica una nueva versión.

### Cómo crear documentos para la API

Nota: Este paso no es necesario para editar ni para acceder a una vista previa de las cadenas de documentos de la API, solo para generar el HTML que se usa en tensorflow.org.

#### Referencia de Python

El paquete `tensorflow_docs` incluye el generador de [documentos de referencia de la API de Python](https://www.tensorflow.org/api_docs/python/tf). Instalación:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

Para generar los documentos de referencia de TensorFlow 2, use el script `tensorflow/tools/docs/generate2.py`:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

Nota: Este script utiliza el paquete de TensorFlow *instalado* para generar documentos y solo funciona con TensorFlow 2.x.

## Documentación narrativa

Las [guías](https://www.tensorflow.org/guide) y los [tutoriales](https://www.tensorflow.org/tutorials) de TensorFlow se escriben como archivos <a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> y blocs de notas interactivos de <a href="https://jupyter.org/" class="external">Jupyter</a>. Los blocs de notas se pueden ejecutar en un navegador si se usa <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>. Los documentos narrativos de [tensorflow.org](https://www.tensorflow.org) se crean a partir de la rama <code>master</code> <a>tensorflow/docs</a>. Las versiones anteriores están disponibles en GitHub en las ramas de versión `rX.x`

### Cambios simples

La forma más sencilla de actualizar la documentación de los archivos Markdown es usar el [editor de archivos basado en la web](https://github.com/tensorflow/docs/tree/master/site/en) de GitHub. Navegue por el repositorio [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) para encontrar el archivo Markdown que se corresponde aproximadamente con la estructura de la URL de <a href="https://www.tensorflow.org">tensorflow.org</a>. En la esquina superior derecha de la vista del archivo, haga clic en el icono del lápiz <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> para abrir el editor de archivos. Edite el archivo y, a continuación, envíe una nueva solicitud de extracción.

### Configure un repositorio de Git local

Para editar varios archivos o ejecutar actualizaciones más complejas, es mejor utilizar un flujo de trabajo de Git local para crear una solicitud de extracción.

Nota: <a href="https://git-scm.com/" class="external">Git</a> es el sistema de control de versiones (VCS) de código abierto que se utiliza para hacer un seguimiento de los cambios en el código fuente. <a href="https://github.com" class="external">GitHub</a> es un servicio en línea que proporciona herramientas de colaboración que funcionan con Git. Consulte la <a href="https://help.github.com" class="external">Ayuda de GitHub</a> para configurar su cuenta de GitHub y empezar a trabajar.

Los siguientes pasos de Git solo son necesarios cuando se configura un proyecto local por primera vez.

#### Bifurque el repositorio tensorflow/docs

En la página de GitHub de <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>, haga clic en el botón *Fork* (Bifurcar) <svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> para crear su propia copia del repositorio en su cuenta de GitHub. Una vez que se haya efectuado la bifurcación, usted será responsable de mantener su copia del repositorio actualizada con el repositorio ascendente de TensorFlow.

#### Clone su repositorio

Descargue una copia de *su* repositorio remoto <var>username</var>/docs en su máquina local. Este es el directorio de trabajo donde hará los cambios:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### Agregue un repositorio ascendente para mantenerlo actualizado (opcional)

Para mantener su repositorio local sincronizado con `tensorflow/docs`, agregue un repositorio remoto *ascendente* para descargar los últimos cambios.

Nota: Asegúrese de actualizar su repositorio local *antes* de iniciar una contribución. Las sincronizaciones periódicas con el canal ascendente reducen la posibilidad de que se produzca un <a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">conflicto de fusión</a> cuando envíe su solicitud de extracción.

Agregue un repositorio remoto:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

Para actualizar:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### Flujo de trabajo de GitHub

#### 1. Cree una nueva rama

Después de actualizar su repositorio desde `tensorflow/docs`, cree una nuea rama desde la rama *master* local:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. Haga cambios

Edite archivos en su editor preferido y siga la [guía de estilo de la documentación de TensorFlow](./docs_style.md).

Confirme el cambio de su archivo:

<pre class="prettyprint lang-bsh">
# View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

Agregue más confirmaciones, según sea necesario.

#### 3. Cree una solicitud de extracción

Cargue su rama local en su repositorio remoto de GitHub (github.com/<var>username</var>/docs):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

Una vez que se complete la inserción, se mostrará un mensaje con una URL para enviar automáticamente una solicitud de extracción al repositorio ascendente. De lo contrario, vaya al repositorio <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> (o a su propio repositorio) y GitHub le pedirá que cree una solicitud de extracción.

#### 4. Revise

Los responsables del mantenimiento y otros contribuyentes revisarán su solicitud de extracción. Participe en el debate y realice los cambios solicitados. Cuando se apruebe su solicitud de extracción, se fusionará en el repositorio de documentos ascendente de TensorFlow.

Éxito: sus cambios han sido aceptados en la documentación de TensorFlow.

Hay un paso de publicación independiente para actualizar [tensorflow.org](https://www.tensorflow.org) desde el repositorio de GitHub. Normalmente, los cambios se agrupan en lotes y el sitio se actualiza con un ritmo regular.

## Blocs de notas interactivos

Aunque es posible editar el archivo JSON del bloc de notas con el <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">editor de archivos basado en la web</a> de GitHub, no se recomienda ya que un JSON con errores de formato puede dañar el archivo. Asegúrese de probar el bloc de notas antes de enviar una solicitud de extracción.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> es un entorno de bloc de notas alojado que facilita la edición (y la ejecución) de la documentación de los blocs de notas. Para cargar los blocs de notas de GitHub en Google Colab, basta con pasar la ruta a la URL de Colab. Por ejemplo, el bloc de notas ubicado en GitHub aquí: <a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br> se puede cargar en Google Colab en esta URL: <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

Hay una extensión <a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a> de Chrome que sustituye la URL cuando se navega por un bloc de notas en GitHub. Esto es útil cuando se abre un bloc de notas en la bifurcación de repositorio, porque los botones superiores siempre enlazan a la rama `master` de TensorFlow Docs.

### Formateo de los blocs de notas

Una herramienta de formateo de blocs de notas hace que los archivos fuente de Jupyter sean consistentes y más fáciles de revisar. Dado que los entornos de creación de blocs de notas difieren en lo que respecta a la salida de archivos, sangría, metadatos y otros campos no especificados, `nbfmt` usa valores predeterminados opinados con preferencia por el flujo de trabajo de TensorFlow docs Colab. Para dar formato a un bloc de notas, instale las <a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools/" external="class">herramientas de bloc de notas de documentos de TensorFlow</a> y ejecute la herramienta `nbfmt`:

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

Para los proyectos de documentación de TensorFlow, los blocs de notas *sin* celdas de salida se ejecutan y prueban; los blocs de notas *con* celdas de salida guardadas se publican tal cual. `nbfmt` respeta el estado del bloc de notas y usa la opción `--remove_outputs` para eliminar explícitamente las celdas de salida.

Para crear un nuevo bloc de notas, copie y edite la <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">plantilla del bloc de notas de documentos de TensorFlow</a>.

### Edición en Colab

Dentro del entorno de Google Colab, haga doble clic en las celdas para editar texto y bloques de código. Las celdas de texto usan Markdown y se deben apegar a la [guía de estilo de los documentos de TensorFlow](./docs_style.md) .

Descargue archivos de bloc de notas de Colab con *Archivo &gt; Descargar .pynb*. Confirme este archivo en su [repositorio Git local](##set_up_a_local_git_repo) y envíe una solicitud de extracción.

Para crear un nuevo bloc de notas, copie y edite la <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">plantilla de bloc de notas de TensorFlow</a>.

### Flujo de trabajo Colab-GitHub

En lugar de descargar un archivo de bloc de notas y usar un flujo de trabajo de Git local, puede editar y actualizar su bifurcación de repositorio de GitHub directamente desde Google Colab:

1. En su repositorio <var>username</var>/docs bifurcado, use la IU web de GitHub para <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">crear una nueva rama</a>.
2. Navegue hasta el archivo del bloc de notas para editarlo.
3. Abra el bloc de notas en Google Colab: use el intercambio de URL o la extensión *Open in Colab* de Chrome.
4. Edite el bloc de notas en Colab.
5. Confirme los cambios en su repositorio desde Colab con *Archivo &gt; Guardar una copia en GitHub....* El cuadro de diálogo para guardar debería vincular el repositorio y la rama correspondientes. Agregue un mensaje de confirmación significativo.
6. Después de guardar, busque su repositorio o el repositorio <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>, GitHub debería solicitarle que cree una solicitud de extracción.
7. Los responsables del mantenimiento revisan la solicitud de extracción.

Éxito: sus cambios han sido aceptados en la documentación de TensorFlow.

## Traducciones

El equipo de TensorFlow trabaja con la comunidad y los proveedores para facilitar traducciones para tensorflow.org. Las traducciones de blocs de notas y otros contenidos técnicos se encuentran en el repositorio de GitHub <a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a>. Envíe solicitudes de extracción a través del <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">proyecto TensorFlow GitLocalize</a>.

Los documentos en inglés son la *fuente de la verdad* y las traducciones deben seguir estas guías con la mayor exactitud posible. Dicho esto, las traducciones están escritas para las comunidades a las que sirven. Si la terminología, la redacción, el estilo o el tono en inglés no se traducen a otro idioma, utilice una traducción apropiada para el lector.

El apoyo lingüístico se determina en función de una serie de factores que incluyen, entre otros, las métricas y la demanda del sitio, el apoyo de la comunidad, el <a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">dominio del inglés</a>, las preferencias del público y otros indicadores. Dado que cada idioma admitido conlleva un costo, los idiomas que no reciben mantenimiento se eliminan. La inclusión de nuevos idiomas se anunciará en el <a class="external" href="https://blog.tensorflow.org/">blog de TensorFlow</a> o en <a class="external" href="https://twitter.com/TensorFlow">Twitter</a>.

Si su idioma de preferencia no tiene soporte, le recomendamos que se encargue de mantener una bifurcación comunitaria para contribuyentes de código abierto. Estos no se publican en tensorflow.org.
