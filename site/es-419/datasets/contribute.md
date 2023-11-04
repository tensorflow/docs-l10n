# Contribuir al repositorio TFDS

¡Gracias por su interés en nuestra biblioteca! Nos encanta tener una comunidad tan motivada.

## Empecemos

- Si TFDS es nuevo para usted, la forma más simple de comenzar es implementar uno de nuestros [conjuntos de datos solicitados](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22dataset+request%22+sort%3Areactions-%2B1-desc), centrándose en los más solicitados. [Siga nuestra guía](https://www.tensorflow.org/datasets/add_dataset) para obtener instrucciones.
- Los problemas, las solicitudes de funciones, los errores... afectan mucho más que agregar nuevos conjuntos de datos, ya que benefician a toda la comunidad TFDS. Vea la [lista de posibles contribuciones](https://github.com/tensorflow/datasets/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+-label%3A%22dataset+request%22+). Comience con los que están etiquetados con [contribución de bienvenida](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22), que son pequeños temas independientes y fáciles para comenzar.
- No dude en tratar los errores que ya están asignados, y que no han sido actualizados desde hace tiempo.
- No es necesario que le asignen el problema. Simplemente comente el problema cuando empieces a trabajar en él :)
- No dude en pedir ayuda si le interesa un problema pero no sabe cómo empezar. Y envíe un borrador de solicitud de cambios si desea recibir comentarios anticipados.
- Para evitar la duplicación innecesaria de trabajo, consulte la lista de [solicitudes de cambios pendientes](https://github.com/tensorflow/datasets/pulls) y comente los problemas en los que esté trabajando.

## Preparación

### Clonar el repositorio

Para comenzar, clone o descargue el repositorio de [Tensorflow Datasets](https://github.com/tensorflow/datasets) e instálelo localmente.

```sh
git clone https://github.com/tensorflow/datasets.git
cd datasets/
```

Instale las dependencias de desarrollo:

```sh
pip install -e .  # Install minimal deps to use tensorflow_datasets
pip install -e ".[dev]"  # Install all deps required for testing and development
```

Tenga en cuenta que también hay un `pip install -e ".[tests-all]"` para instalar todos los departamentos específicos del conjunto de datos.

### Visual Studio Code

Al desarrollar con código con [Visual Studio Code](https://code.visualstudio.com/), nuestro repositorio viene con algunas [configuraciones predefinidas](https://github.com/tensorflow/datasets/tree/master/.vscode/settings.json) para ayudar con el desarrollo (sangría correcta, pylint,...).

Nota: es posible que no se pueda habilitar el descubrimiento de pruebas en VS Code debido a algunos errores de VS Code [#13301](https://github.com/microsoft/vscode-python/issues/13301) y [#6594](https://github.com/microsoft/vscode-python/issues/6594). Para resolver los problemas, puede consultar los registros de descubrimiento de pruebas:

- Si le sale un mensaje de advertencia de TensorFlow, pruebe [esta solución](https://github.com/microsoft/vscode-python/issues/6594#issuecomment-555680813).
- Si el descubrimiento no funciona porque falta una importación que debería haberse instalado, envíe una solicitud de cambio para actualizar la instalación `dev` pip.

## Lista de verificación de la solicitud de cambios

### Firme el Acuerdo de licencia de colaboradores

El Acuerdo de licencia de colaboradores (CLA, por sus siglás en inglés) deben acompañar las contribuciones de este proyecto. Usted (o su empleador) conserva los derechos de autor de su contribución; esto simplemente nos da permiso para usar y redistribuir sus contribuciones como parte del proyecto. Dirígase a [https://cla.developers.google.com/](https://cla.developers.google.com/) para ver sus acuerdos actuales archivados o para firmar uno nuevo.

Por lo general, solo necesita enviar un CLA una vez, por lo que si ya envió uno (incluso si fue para un proyecto diferente), probablemente no necesite hacerlo de nuevo.

### Siga las mejores prácticas

- La legibilidad es importante. El código debe seguir las mejores prácticas de programación (evite duplicaciones, factorizar en pequeñas funciones independientes, nombres explícitos de variables,...)
- Cuanto más simple, mejor (por ejemplo, la implementación se divide en varias solicitudes de cambio más pequeñas e independientes, que es más fácil de revisar).
- Agregue pruebas cuando sea necesario; se deben aprobar las pruebas existentes.
- Agregar [anotaciones escritas](https://docs.python.org/3/library/typing.html)

### Verifíque su guía de estilo

Nuestro estilo se basa en la [Guía de estilo de Google Python](https://github.com/google/styleguide/blob/gh-pages/pyguide.md), que se basa en la [guía de estilo de Python PEP 8](https://www.python.org/dev/peps/pep-0008). En el nuevo código, se debería intentar seguir el [estilo del código de Black](https://github.com/psf/black/blob/master/docs/the_black_code_style.md) pero con:

- Longitud de línea: 80
- Sangría de 2 espacios en lugar de 4.
- Comilla simple `'`

**Importante:** asegúrese de ejecutar `pylint` en su código para verificar que esté formateado correctamente:

```sh
pip install pylint --upgrade
pylint tensorflow_datasets/core/some_file.py
```

Puede probar `yapf` para formatear automáticamente un archivo, pero la herramienta no es perfecta, por lo que probablemente tendrá que aplicar correcciones manuales después.

```sh
yapf tensorflow_datasets/core/some_file.py
```

Tanto `pylint` como `yapf` deberían haberse instalado con `pip install -e ".[dev]"`, pero también se pueden instalar manualmente con `pip install`. Si está usando VS Code, se deben integrar esas herramientas en la interfaz de usuario.

### Cadenas de documentos y anotaciones escritas

Las clases y funciones deben documentarse con cadenas de documentación y anotaciones escritas. Las cadenas de documentos deben seguir el [estilo de Google](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods). Por ejemplo:

```python
def function(x: List[T]) -> T:
  """One line doc should end by a dot.

  * Use `backticks` for code and tripple backticks for multi-line.
  * Use full API name (`tfds.core.DatasetBuilder` instead of `DatasetBuilder`)
  * Use `Args:`, `Returns:`, `Yields:`, `Attributes:`, `Raises:`

  Args:
    x: description

  Returns:
    y: description
  """
```

### Agregar y ejecutar pruebas unitarias

Asegúrese de que las nuevas funciones se prueben con pruebas unitarias. Puede ejecutar pruebas a través de la interfaz de VS Code o la línea de comandos. Por ejemplo:

```sh
pytest -vv tensorflow_datasets/core/
```

`pytest` vs `unittest`: históricamente, hemos usado el módulo `unittest` para escribir pruebas. Las nuevas pruebas deberían usar preferentemente `pytest`, que es más simple, flexible, moderna y que la usan las bibliotecas más famosas (numpy, pandas, sklearn, matplotlib, scipy, six,...). Puede leer la [guía de pytest](https://docs.pytest.org/en/stable/getting-started.html#getstarted) si no está familiarizado con pytest.

Las pruebas para DatasetBuilders son especiales y están documentadas en la [guía para agregar un conjunto de datos](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md#test-your-dataset).

### ¡Envíe la solicitud de cambios para revisiones!

¡Felicitaciones! Consulte la [Ayuda de GitHub](https://help.github.com/articles/about-pull-requests/) para obtener más información sobre el uso de solicitudes de cambios.
