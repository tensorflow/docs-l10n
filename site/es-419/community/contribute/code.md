# Cómo contribuir con el código de TensorFlow

Ya sea para agregar una función de pérdida, mejorar la cobertura de las pruebas o escribir una RFC para un cambio de diseño importante, esta parte de la guía del colaborador le ayudará a ponerse en marcha. Le agradecemos su trabajo y su interés en mejorar TensorFlow.

## Antes de empezar

Antes de contribuir con código fuente a un proyecto de TensorFlow, revise el archivo `CONTRIBUTING.md` en el repositorio de GitHub del proyecto. Por ejemplo, consulte el archivo [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) en el repositorio principal de TensorFlow. Todos los contribuyentes de código deben firmar un [Acuerdo de licencia de colaborador](https://cla.developers.google.com/clas) (CLA, por sus siglas en inglés).

Para evitar duplicar el trabajo, revise los RFC [actuales](https://github.com/tensorflow/community/tree/master/rfcs) o [propuestos](https://github.com/tensorflow/community/labels/RFC%3A%20Proposed) y comuníquese con los desarrolladores en los foros de TensorFlow ([developers@tensorflow.org](https://groups.google.com/u/1/a/tensorflow.org/g/developers)) antes de comenzar a trabajar en una característica importante. Somos un poco selectivos a la hora de agregar nuevas funciones y la mejor manera de contribuir y ayudar al proyecto es trabajar en problemas conocidos.

## Problemas para nuevos contribuyentes

Los nuevos contribuyentes deben buscar las siguientes etiquetas cuando busquen una primera contribución al código base de TensorFlow. Recomendamos enfáticamente que los nuevos contribuyentes aborden primero los proyectos de “good first issue” (buen primer problema) y “contributions welcome” (contribuciones bienvenidas); esto ayuda al colaborador a familiarizarse con el flujo de trabajo de contribución y a que los principales desarrolladores se familiaricen con el colaborador.

- [buen primer problema](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue)
- [contribuciones bienvenidas](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome)

Si desea reclutar un equipo para que lo ayude a abordar un problema a gran escala o una nueva característica, envíe un correo electrónico a [developers@ group](https://groups.google.com/a/tensorflow.org/g/developers) y revise nuestra lista actual de RFC.

## Revisión de código

Las nuevas características, correcciones de errores y cualquier otro cambio a la base del código están sujetos a la revisión del código.

Revisar el código que se contribuye al proyecto como solicitudes de incorporación es un componente crucial del desarrollo de TensorFlow. Invitamos a todos a comenzar a revisar el código enviado por otros desarrolladores, especialmente si la característica es una que probablemente utilizará.

A continuación, les dejamos algunas preguntas a tener en cuenta durante el proceso de revisión del código:

- *¿Queremos esto en TensorFlow?* ¿Es probable que se utilice? A usted, como usuario de TensorFlow, ¿le gusta el cambio y tiene intención de utilizarlo? ¿Este cambio se encuentra dentro del alcance de TensorFlow? ¿El costo de mantener una nueva característica valdrá la pena por sus beneficios?

- *¿El código es coherente con la API de TensorFlow?* ¿Las funciones, las clases y los parámetros públicos están bien nombrados y diseñados de forma intuitiva?

- *¿Incluye documentación?* ¿Todas las funciones públicas, las clases, los parámetros, los tipos de rendimiento y los atributos almacenados se nombraron de acuerdo con las convenciones de TensorFlow y están claramente documentados? ¿Se describen las nuevas funciones en la documentación de TensorFlow y se ilustran con ejemplos, siempre que sea posible? ¿La documentación se muestra correctamente?

- *¿El código es legible por humanos?* ¿Tiene bajos niveles de redundancia? ¿Deberían mejorarse los nombres de las variables para lograr una mayor claridad o coherencia? ¿Deberían agregarse comentarios? ¿Debería eliminarse algún comentario por considerarlo inútil o superfluo?

- *¿El código es eficiente?* ¿Podría reescribirse fácilmente para que se ejecute de manera más eficiente?

- ¿El código es *compatible* con versiones anteriores de TensorFlow?

- ¿El nuevo código agregará *nuevas dependencias* en otras bibliotecas?

## Cómo probar y mejorar la cobertura de las pruebas

Las pruebas unitarias de alta calidad son la piedra angular del proceso de desarrollo de TensorFlow. Para ello, utilizamos imágenes de Docker. Las funciones de prueba tienen nombres apropiados y son responsables de comprobar la validez de los algoritmos, así como diferentes opciones del código.

Todas las nuevas funciones y correcciones de errores *deben* incluir una cobertura de prueba adecuada. También agradecemos contribuciones de nuevos casos de prueba o mejoras a las pruebas existentes. Si descubre que nuestras pruebas existentes no están completas, incluso si eso no está generando errores actualmente, presente un problema y, si es posible, una solicitud de incorporación.

Para obtener detalles específicos de los procedimientos de prueba en cada proyecto de TensorFlow, consulte los archivos `README.md` y `CONTRIBUTING.md` en el repositorio del proyecto en GitHub.

Es especialmente importante que *las pruebas sean adecuadas*:

- ¿Se prueban *todas las funciones y clases públicas*?
- ¿Se prueba un *conjunto razonable de parámetros*, sus valores, tipos de valores y combinaciones?
- ¿Las pruebas validan que el *código es correcto* y que *hace lo que la documentación indica* que el código debe hacer?
- Si el cambio es una corrección de errores, ¿se incluye una *prueba de no regresión*?
- ¿Las pruebas *pasan la compilación de integración continua*?
- ¿Las pruebas *cubren cada línea de código*? De no ser así, ¿las excepciones son razonables y explícitas?

Si encuentra algún problema, considere en la posibilidad de ayudar al colaborador a comprender esos problemas y resolverlos.

## Cómo mejorar los registros o los mensajes de error

Agradecemos contribuciones que mejoren el registro y los mensajes de error.

## Flujo de trabajo de contribución

Las contribuciones al código (correcciones de errores, nuevos desarrollos, mejoras a las pruebas) siguen un flujo de trabajo centrado en GitHub. Para participar en el desarrollo de TensorFlow, configure una cuenta de GitHub. Luego, siga estos pasos:

1. Bifurque el repositorio en el que planea trabajar. Vaya a la página del repositorio del proyecto y use el botón *Fork* (Bifurcar). Esto creará una copia del repositorio, con su nombre de usuario. (Si desea obtener más información sobre cómo bifurcar un repositorio, consulte [esta guía](https://help.github.com/articles/fork-a-repo/)).

2. Clone el repositorio en su sistema local.

    `$ git clone git@github.com:your-user-name/project-name.git`

3. Cree una nueva rama para guardar su trabajo.

    `$ git checkout -b new-branch-name`

4. Trabaje en su nuevo código. Escriba y ejecute pruebas.

5. Confirme sus cambios.

    `$ git add -A`

    `$ git commit -m "commit message here"`

6. Envíe los cambios a su repositorio de GitHub.

    `$ git push origin branch-name`

7. Abra una *solicitud de incorporación* (PR). Vaya al repositorio del proyecto original en GitHub. Habrá un mensaje sobre la rama que envió recientemente, donde se le pregunta si desea abrir una solicitud de incorporación. Siga las indicaciones, *compare entre repositorios* y envíe la PR. Esto enviará un correo electrónico a los encargados de la confirmación. Es posible que quiera enviar un correo electrónico a la lista de correo para conseguir más visibilidad. (Si desea obtener más información al respecto, consulte la [guía de GitHub sobre solicitudes de incorporación](https://help.github.com/articles/creating-a-pull-request-from-a-fork).

8. Los encargados del mantenimiento y otros contribuyentes *revisarán sus solicitudes de incorporación*. Participe en la conversación e intente *realizar los cambios solicitados*. Una vez que se apruebe la PR, el código se fusionará.

*Antes de trabajar en su próxima contribución*, asegúrese de que su repositorio local esté actualizado.

1. Configure el acceso remoto al sector de exploración y producción. (Solo tiene que hacer esto una vez por proyecto, no todas las veces).

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. Cambie a la rama maestra local.

    `$ git checkout master`

3. Descargue los cambios desde el sector de exploración y producción.

    `$ git pull upstream master`

4. Envíe los cambios a su cuenta de GitHub. (Esto es opcional, pero es una buena práctica).

    `$ git push origin master`

5. Cree una nueva rama si está comenzando un nuevo trabajo.

    `$ git checkout -b branch-name`

Recursos `git` y GitHub adicionales:

- [Documentación de git](https://git-scm.com/documentation)
- [Flujo de trabajo de desarrollo de git](https://docs.scipy.org/doc/numpy/dev/development_workflow.html)
- [Cómo resolver conflictos de fusión](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)

## Lista de verificación del contribuyente

- Lea las [pautas de contribución](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).
- Lea el [Código de conducta](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).
- Asegúrese de haber firmado el [Acuerdo de licencia de colaborador (CLA)](https://cla.developers.google.com/).
- Compruebe si sus cambios cumplen con las [pautas](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Compruebe si sus cambios cumplen con el [estilo de codificación de TensorFlow](https://www.tensorflow.org/community/contribute/code_style).
- [Ejecute las pruebas unitarias](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#running-unit-tests).
