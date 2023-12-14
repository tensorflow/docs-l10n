<!--* freshness: { owner: 'maringeo' } *-->

# Cómo ser un editor

## Términos de servicio

Al enviar un modelo para publicación, acepta los Términos de servicio de TensorFlow Hub en [https://tfhub.dev/terms](https://tfhub.dev/terms).

## Descripción general del proceso de publicación

El proceso completo de publicación consiste en:

1. Crear el modelo (vea cómo [exportar un modelo](exporting_tf2_saved_model.md))
2. Escribir documentación (consulte cómo [escribir la documentación del modelo)](writing_model_documentation.md)
3. Crear una solicitud de publicación (consulte cómo [contribuir](contribute_a_model.md))

## Formato de Markdown de la página del editor

La documentación del editor se declara en el mismo tipo de archivos de markdown que se describe en la guía de [escribir la documentación del modelo](writing_model_documentation), con pequeñas diferencias sintácticas.

La ubicación correcta para el archivo del editor en el repositorio de TensorFlow Hub es: [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/&lt;publisher_name&gt;/&lt;publisher_name.md&gt;

Vea el ejemplo mínimo de documentación del editor:

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

En el ejemplo anterior se especifica el nombre del editor, una breve descripción, la ruta al icono que se usará y una documentación de markdown de formato libre más extensa.

### Pautas sobre el nombre del editor

Su nombre de editor puede ser su nombre de usuario de GitHub o el nombre de la organización de GitHub que administra.
