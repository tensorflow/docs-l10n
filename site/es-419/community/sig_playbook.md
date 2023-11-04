# Cuaderno de estrategias de SIG

## Alcance de un SIG

TensorFlow ofrece *Grupos de Interés Especial* (SIG) para centrar la colaboración en áreas específicas. Los SIG desarrollan su trabajo en público. Para unirse y contribuir, revise el trabajo del grupo y póngase en contacto con el líder del SIG. Las políticas de afiliación varían en función del SIG.

El marco ideal para un SIG responde a un campo bien definido, en el que la mayoría de la participación sea de la comunidad. Además, debe haber pruebas suficientes de que hay miembros de la comunidad dispuestos a participar y contribuir en caso de que se establezca el grupo de interés.

No todos los SIG tendrán el mismo nivel de energía, alcance o modelos de gobernanza, de manera que cabe esperar cierta variabilidad.

Consulte la lista completa de [SIG de TensorFlow](https://github.com/tensorflow/community/tree/master/sigs).

### Los no objetivos: qué *no* es un SIG

El objetivo de los SIG es facilitar la colaboración en trabajos compartidos. Es decir que un SIG no es nada de lo siguiente:

- *un foro de apoyo*: una lista de correo y un SIG no son lo mismo.
- *necesario inmediatamente*: al principio de un proyecto, es posible que no sepa si tiene trabajo compartido o colaboradores.
- *trabajo no remunerado*: se requiere energía para crecer y coordinar el trabajo en colaboración.

Nuestro enfoque de la creación de SIG es conservador: gracias a la facilidad para iniciar proyectos en GitHub, existen muchas vías de colaboración sin necesidad de un SIG.

## Ciclo de vida de un SIG

### Investigación y consulta

Quienes propongan los grupos deberán reunir la evidencia suficiente para la aprobación, como se especifica más adelante. Algunas de las vías posibles a tener en cuenta son las siguientes:

- Un problema o conjunto de problemas bien definidos que el grupo resolvería.
- Consulta a los miembros de la comunidad que se beneficiarían, en la que se evalúen tanto los beneficios como su disposición a comprometerse.
- En el caso de los proyectos existentes, pruebas de que los colaboradores están interesados en el tema.
- Posibles objetivos del grupo.
- Recursos necesarios para el funcionamiento del grupo.

Aunque la necesidad de un SIG parezca clara, la investigación y la consulta siguen siendo importantes para el éxito del grupo.

### Cómo crear el nuevo grupo

El nuevo grupo debe seguir el siguiente proceso de constitución. En particular, debe demostrar lo siguiente:

- Un propósito y un beneficio claros para TensorFlow (ya sea en torno a un subproyecto o a un área de aplicación).
- Dos o más colaboradores dispuestos a actuar como líderes del grupo, existencia de otros colaboradores y pruebas de demanda del grupo.
- Recursos que se necesitarán inicialmente (normalmente, lista de correo y videoconferencia periódica).

La aprobación del grupo será concedida por decisión del TF Community Team, definido como los responsables del mantenimiento del proyecto tensorflow/comunidad. El equipo consultará a otras partes interesadas según sea necesario.

Antes de dar paso a las formalidades del proceso, es aconsejable consultar con el equipo de la comunidad TensorFlow, community-team@tensorflow.org. Es muy probable que sea necesario entablar conversaciones e iteraciones antes de que la solicitud de SIG esté lista.

Para solicitar formalmente un nuevo grupo, se debe enviar una carta como PR a tensorflow/community e incluir la solicitud en los comentarios del PR (consulte la plantilla a continuación). Una vez aprobado, el PR para el grupo se fusionará y se crearán los recursos necesarios.

### Plantilla de solicitud de un nuevo SIG

La plantilla estará disponible en el repositorio de la comunidad: [SIG-request-template.md](https://github.com/tensorflow/community/blob/master/governance/SIG-request-template.md).

### Constitución

Cada grupo se establecerá con un estatuto y se regirá por el código de conducta de TensorFlow. Los archivos del grupo serán públicos. La afiliación puede estar abierta a todos sin aprobación, o disponible previa solicitud, pendiente de la aprobación del administrador del grupo.

El estatuto debe nombrar a un administrador. Además de un administrador, el grupo debe incluir al menos una persona como líder (puede ser la misma persona), que servirá como punto de contacto para coordinar con el equipo de la comunidad TensorFlow cuando sea necesario.

Este documento se publicará inicialmente en la lista de correo del grupo. El repositorio de la comunidad en la organización TensorFlow GitHub archivará dichos documentos y políticas ([ejemplo de Kubernetes](https://github.com/kubernetes/community)). A medida que un grupo desarrolle sus prácticas y convenciones, se espera que las documente en la parte correspondiente del repositorio de la comunidad.

### Colaboración e inclusión

Si bien no es obligatorio, el grupo puede optar por utilizar la colaboración a través de conferencias telefónicas programadas o canales de chat para llevar a cabo las reuniones. Dichas reuniones deberán anunciarse en la lista de correo y, posteriormente, se publicarán en ella las notas correspondientes. Las reuniones periódicas ayudan a impulsar la responsabilidad y el progreso en un SIG.

Los miembros del equipo de la comunidad de TensorFlow controlarán y alentarán proactivamente al grupo para que debatan y actúen como corresponda.

### Lanzamiento

Actividades obligatorias:

- Notificar a los grupos de debate generales de TensorFlow ([discuss@](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss), [developers@](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers)).
- Agregar SIG a las páginas de la comunidad en el sitio web de TensorFlow.

Actividades opcionales:

- Crear una publicación de blog en la comunidad de blogs de TensorFlow.

### Mantenimiento y cierre de los SIG

El equipo de la comunidad de TensorFlow hará todo lo posible para garantizar el mantenimiento de los SIG. De vez en cuando se pedirá al líder del SIG que entregue un informe del trabajo del SIG, que se usará para informar acerca de la actividad del grupo a toda de la comunidad de TensorFlow.

Si un SIG deja de tener un propósito útil o una comunidad interesada, es posible que se archive y deje de funcionar. El equipo de la comunidad de TF se reserva el derecho de archivar tales SIG inactivos, con el fin de mantener en buen estado el proyecto en general, aunque no sea un resultado deseable. Un SIG también puede optar por disolverse si reconoce que ha llegado al final de su vida útil.
