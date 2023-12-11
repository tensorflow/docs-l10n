# El proceso RFC de TensorFlow

Todas las nuevas características de TensorFlow comienzan como una Solicitud de comentario (RFC, por sus siglas en inglés).

Una RFC es un documento que describe un requisito y los cambios propuestos para resolverlo. En concreto, la RFC deberá cumplir con las siguientes condiciones:

- Seguirá el formato de la [plantilla de RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
- Se enviará como solicitud de incorporación al directorio [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).
- Se someterá a debate y a una reunión de revisión antes de su aceptación.

El propósito de una Solicitud de comentario (RFC) de TensorFlow es involucrar a la comunidad de TensorFlow en el desarrollo, a través de la obtención de comentarios de las partes interesadas y de los expertos, y de una amplia comunicación de los cambios de diseño.

## Cómo enviar una RFC

1. Antes de enviar una RFC, analice sus objetivos con los contribuyentes y los responsables del mantenimiento del proyecto y obtenga sus primeros comentarios. Utilice la lista de correo de desarrolladores del proyecto en cuestión (developers@tensorflow.org, o la lista del SIG correspondiente).

2. Redacte un borrador de su RFC.

    - Lea los [criterios de revisión del diseño.](https://github.com/tensorflow/community/blob/master/governance/design-reviews.md)
    - Siga la [plantilla de RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
    - nombre su archivo RFC `YYYYMMDD-descriptive-name.md`, donde `YYYYMMDD` corresponde a la fecha de envío y `descriptive-name` se relaciona con el título de su RFC. (Por ejemplo, si su RFC se titula *API de widgets paralelos*, puede usar el nombre de archivo `20180531-widgets-paralelos.md`.
    - Si tiene imágenes u otros archivos auxiliares, cree un directorio con el formato `YYYYMMDD-descriptive-name` para almacenar esos archivos.

    Tras redactar el borrador de la RFC, solicite los comentarios de los responsables del mantenimiento y de los colaboradores antes de enviarlo.

    No es obligatorio escribir el código de implementación, pero puede ser útil para los debates sobre el diseño.

3. Reclute un patrocinador.

    - Un patrocinador debe ser un responsable del mantenimiento del proyecto.
    - Identifique al patrocinador en la RFC, antes de publicar la PR.

    *Puede* publicar una RFC sin un patrocinador, pero si en el plazo de un mes de publicación de la PR sigue sin patrocinador, se cerrará.

4. Envíe su RFC como una solicitud de incorporación a [tensorflow/community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).

    Incluya la tabla de cabecera y el contenido de la sección *Objetivo* en el comentario de su solicitud de incorporación, con ayuda de Markdown. Consulte [esta RFC de ejemplo](https://github.com/tensorflow/community/pull/5). Incluya los identificadores de GitHub de coautores, revisores y patrocinadores.

    En la parte superior de la PR, indique la duración del periodo de comentarios. Deberá ser como *mínimo dos semanas* a partir de la publicación de la PR.

5. Envíe un correo electrónico a la lista de correo de desarrolladores con una breve descripción, un vínculo a la PR y una solicitud de revisión. Siga el formato de los correos anteriores, como se puede ver en [este ejemplo](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/PIChGLLnpTE).

6. El patrocinador solicitará una reunión del comité de revisión, no antes de dos semanas después de la publicación de la RFC PR. Si el debate es intenso, espere a que se haya calmado antes de pasar a la revisión. El objetivo de la reunión de revisión es resolver cuestiones menores; las cuestiones importantes deben consensuarse de antemano.

7. En la reunión se podrá aprobar la RFC, rechazarla o exigir cambios antes de que pueda ser considerada nuevamente. Las RFC aprobadas se fusionarán en [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs), y se cerrarán las PR de las RFC rechazadas.

## Participantes de la RFC

Muchas personas participan en el proceso de RFC:

- **Autor de la RFC**: uno o más miembros de la comunidad que escriben una RFC y se comprometen a defenderla durante el proceso.

- **Patrocinador de la RFC**: un responsable del mantenimiento que patrocina la RFC y que le servirá de guía a través del proceso de revisión de la RFC.

- **Comité de revisión**: grupo de responsables del mantenimiento que tienen la responsabilidad de recomendar la adopción de la RFC.

- Cualquier **miembro de la comunidad** puede ayudar con comentarios sobre si la RFC satisfará sus necesidades.

### Patrocinadores de las RFC

Un patrocinador es un responsable del mantenimiento del proyecto que se encarga de garantizar el mejor resultado posible del proceso de RFC. Esto incluye lo siguiente:

- Abogar por el diseño propuesto.
- Orientar en la redacción de la RFC para que se adhiera a las convenciones de diseño y estilo existentes.
- Guiar al comité de revisión para llegar a un consenso productivo.
- Si el comité de revisión solicita cambios, el patrocinador debe asegurarse de que se realicen y solicitar la aprobación posterior de los miembros del comité.
- En caso de que la RFC pase a la implementación:
    - Garantizar que la implementación propuesta se adhiera al diseño.
    - Coordinar con las partes correspondientes para lograr una correcta implementación.

### Comités de revisión de RFC

El comité de revisión decide por consenso si aprueba, rechaza o solicita cambios. Son responsables de lo siguiente:

- Asegurarse de que se hayan tenido en cuenta los puntos importantes de los comentarios del público.
- Agregar sus notas de la reunión como comentarios a la PR.
- Justificar sus decisiones.

La constitución de un comité de revisión puede cambiar en función del estilo particular de gobernanza y liderazgo de cada proyecto. Para TensorFlow principal, el comité estará formado por contribuyentes al proyecto de TensorFlow que tengan experiencia en el área de dominio en cuestión.

### Los miembros de la comunidad y el proceso de RFC

El propósito de las RFC es garantizar que la comunidad esté bien representada y se beneficie de los nuevos cambios en TensorFlow. Es responsabilidad de los miembros de la comunidad participar en la revisión de las RFC cuando tengan interés en el resultado.

Los miembros de la comunidad que estén interesados ​​en una RFC deben hacer lo siguiente:

- **Ofrecer comentarios** lo antes posible para que haya tiempo suficiente para estudiarlos.
- **Leer las RFC** detenidamente antes de ofrecer comentarios.
- Ser **civilizados y ofrecer comentarios constructivos**.

## Cómo implementar nuevas características

Una vez que se aprueba una RFC, puede comenzar su implementación.

Si está trabajando en un código nuevo para implementar una RFC:

- Asegúrese de que entiende la función y el diseño que se aprueban en la RFC. Haga preguntas y analice el enfoque antes de comenzar a trabajar.
- Las nuevas características deben incluir nuevas pruebas unitarias que sirvan para verificar que la característica funciona según lo esperado. Es conveniente escribir estas pruebas antes de escribir el código.
- Siga la [Guía de estilo del código de TensorFlow](#tensorflow-code-style-guide)
- Agregue o actualice la documentación relevante de la API. Haga referencia a la RFC en la nueva documentación.
- Siga cualquier otra directriz que figure en el archivo `CONTRIBUTING.md` del repositorio del proyecto al que está contribuyendo.
- Ejecute pruebas unitarias antes de enviar su código.
- Trabaje con el patrocinador de la RFC para lograr un correcto desarrollo del nuevo código.

## Cómo mantener la vara alta

Si bien animamos y agradecemos a todos los colaboradores, el nivel de exigencia para aceptar una RFC es muy alto. Una nueva función puede ser rechazada o sufrir una revisión importante en cualquiera de estas etapas:

- Conversaciones iniciales sobre el diseño en la lista de correo correspondiente.
- Imposibilidad de conseguir un patrocinador.
- Objeciones críticas durante la fase de comentarios.
- Falta de consenso durante la revisión del diseño.
- Preocupaciones planteadas durante la implementación (por ejemplo: incapacidad para lograr compatibilidad con versiones anteriores, preocupaciones sobre el mantenimiento).

Si este proceso funciona bien, se espera que las RFC se rechacen en las primeras etapas y no en las últimas. Una RFC aprobada no garantiza un compromiso de implementación, y la aceptación de una propuesta de implementación de una RFC sigue estando sujeta al proceso habitual de revisión del código.

Si tiene alguna pregunta sobre este proceso, no dude en enviarla a la lista de correo de desarrolladores o bien presente una incidencia en [tensorflow/community](https://github.com/tensorflow/community/tree/master/rfcs).
