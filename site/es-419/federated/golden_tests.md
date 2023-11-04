# Prueba dorada

TFF incluye una pequeña biblioteca llamada `golden` que facilita la escritura y el mantenimiento de pruebas doradas.

## ¿Qué son las pruebas doradas? ¿Cuándo debo usarlas?

Las pruebas doradas se utilizan cuando el objetivo es que un desarrollador sepa que su código alteró la salida de una función. Estas pruebas vulneran muchas características de las buenas pruebas unitarias porque hacen promesas sobre las salidas exactas de las funciones, en lugar de probar un conjunto específico de propiedades claras y documentadas. A veces no está claro cuándo se "espera" un cambio en una salida dorada o si se está violando alguna propiedad que la prueba dorada pretendía hacer cumplir. En este sentido, suele preferirse una prueba unitaria bien documentada a una prueba dorada.

Sin embargo, las pruebas doradas pueden ser extremadamente útiles para validar el contenido exacto de mensajes de error, diagnósticos o código generado. En estos casos, las pruebas doradas pueden ser útiles para comprobar si algún cambio en la salida generada "parece correcto".

## ¿Cómo debo usar `golden` para escribir pruebas?

`golden.check_string(filename, value)` es el principal punto de entrada a la biblioteca `golden`. Verifica la cadena de `value` con el contenido de un archivo cuyo último elemento de la ruta es `filename`. La ruta completa a `filename` debe proporcionarse mediante el argumento `--golden <path_to_file>` de la línea de comandos. Del mismo modo, estos archivos deben estar disponibles para las pruebas mediante el argumento `data` de la regla BUILD de `py_test`. Use la función `location` para generar una ruta relativa correspondiente:

```
py_string_test(
  ...
  args = [
    "--golden",
    "$(location path/to/first_test_output.expected)",
    ...
    "--golden",
    "$(location path/to/last_test_output.expected)",
  ],
  data = [
    "path/to/first_test_output.expected",
    ...
    "path/to/last_test_output.expected",
  ],
  ...
)
```

Por convención, los archivos dorados deben colocarse en un directorio hermano con el mismo nombre que su objetivo de prueba, con el sufijo `_goldens`:

```
path/
  to/
    some_test.py
    some_test_goldens/
      test_case_one.expected
      ...
      test_case_last.expected
```

## ¿Cómo actualizo los archivos `.expected`?

Los archivos `.expected` se pueden actualizar mediante la ejecución del objetivo de prueba afectado con los argumentos `--test_arg=--update_goldens --test_strategy=local`. La diferencia resultante se debe comprobar para detectar cambios imprevistos.
