# Enviar una solicitud de incorporación

En esta página explica cómo enviar una solicitud de incorporación que tiene archivos de documentación de Markdown a [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev). Repositorio de GitHub. Para obtener más información sobre cómo escribir archivos Markdown en primer lugar, consulte la [guía para escribir documentación](writing_documentation.md).

**Nota:** Si quiere que su modelo se refleje en otros hubs de modelos, use una licencia MIT, CC0 o Apache. Si no quiere que su modelo se refleje en otros hubs de modelos, use otra licencia adecuada.

## Verificaciones de GitHub Actions

El repositorio [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) usa GitHub Actions para validar el formato de los archivos en una solicitud de incorporación (PR). El flujo de trabajo que se usa para validar las PR se define en [.github/workflows/contributions-validator.yml](https://github.com/tensorflow/tfhub.dev/blob/master/.github/workflows/contributions-validator.yml). Puede ejecutar la secuencia de comandos del validador en su propia rama fuera del flujo de trabajo, pero deberá asegurarse de tener instaladas todas las dependencias correctas del paquete pip.

Quienes contribuyen por primera vez solo pueden ejecutar verificaciones automáticas con la aprobación de un responsable del repositorio, según la [política de GitHub](https://github.blog/changelog/2021-04-22-github-actions-maintainers-must-approve-first-time-contributor-workflow-runs/). Se alienta a que los editores envíen una pequeña PR para corregir errores tipográficos, mejorar la documentación del modelo o enviar una PR que contenga solo su página del editor como su primera PR para poder ejecutar verificaciones automáticas en las PR posteriores.

Importante: ¡Su solicitud de incorporación debe pasar las verificaciones automatizadas antes de que se revise!

## Enviar la PR

Los archivos completos de Markdown se pueden incorporar en la rama principal de [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) mediante uno de los siguientes métodos.

### Envío de Git CLI

Suponiendo que la ruta identificada del archivo de markdown es `assets/docs/publisher/model/1.md`, puede seguir los pasos estándar de Git[Hub] para crear una nueva solicitud de incorporación con un archivo recién agregado.

Para empezar, se debe bifurcar el repositorio de GitHub de TensorFlow Hub y luego crear una [solicitud de incorporación desde esa bifurcación](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) en la rama principal de TensorFlow Hub.

A continuación, se muestran los comandos típicos de CLI git que se necesitan para agregar un nuevo archivo a una rama principal del repositorio bifurcado.

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### Envío de la GUI de GitHub

Una forma un poco más fácil de enviar es a través de la interfaz gráfica de usuario de GitHub. GitHub permite crear solicitudes de incorporación para [archivos nuevos](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files) o [ediciones de archivos](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository) directamente a través de la GUI.

1. En la [página de GitHub de TensorFlow Hub](https://github.com/tensorflow/tfhub.dev), presione el botón `Create new file` (Crear archivo nuevo).
2. Establezca la ruta de archivo correcta: `assets/docs/publisher/model/1.md`
3. Copie y pegue el markdown existente.
4. En la parte inferior, seleccione "Create a new branch for this commit and start a pull request" (Crear una rama nueva para esta confirmación e iniciar una solicitud de incorporación).
