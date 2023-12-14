# Guía de revisión de códigos

El propósito de este documento es explicar el razonamiento detrás de la postura del equipo de XLA sobre las revisiones de código, una postura que ha crecido a partir de años de experiencia colectiva de trabajo en proyectos de código abierto en general y de XLA en particular.

Los distintos proyectos de código abierto tienen diferentes expectativas culturales respecto a cuánto pueden exigir los revisores a los autores del código. En algunos proyectos, los revisores toman una solicitud de incorporación (PR) "mayormente correcta", la modifican ellos mismos y la envían. En XLA se adopta el enfoque opuesto: esperamos que los autores iteren las solicitudes de incorporación hasta que sean lo suficientemente buenas para enviarlas sin cambios adicionales.

La razón principal de este enfoque es que queremos que los autores de las solicitudes de incorporación aprendan a ser contribuyentes de XLA de pleno derecho. Si los revisores solucionan por sí mismos los problemas en las solicitudes de incorporación, entonces será mucho más difícil que los autores aprendan. El enfoque de XLA puede ser un desafío tanto para los revisores como para quienes se someten a la revisión, pero creemos que, en última instancia, nos ayuda a hacer crecer la comunidad.

Aprender a ser un "contribuyente de XLA de pleno derecho" no se trata solo de escribir código que no tenga errores. Hay muchas más cosas que aprender sobre "cómo modificar XLA". Esto incluye lo siguiente:

- estilo de codificación,
- qué casos extremos se deben tener en cuenta,
- expectativas en torno a las pruebas escritas,
- expectativas en torno a comentarios y descripciones de las solicitudes de incorporación,
- y expectativas en torno a la compilación de infraestructura para respaldar los cambios.

A medida que desarrolle conocimiento sobre el proyecto y la confianza de los revisores, es de esperar que reciba menos comentarios, porque naturalmente escribirá un código más acorde con las expectativas del revisor.

Como muchos proyectos de código abierto, XLA tiene algunas personas con mucha experiencia y mucha gente relativamente nueva. Quienes tenemos mucha experiencia dedicamos mucho tiempo. Para que las PR sigan avanzando de manera oportuna, puede ayudar a reducir el tiempo que necesitan los revisores y la cantidad de iteraciones requeridas si sigue estos pasos:

- *Revisar cuidadosamente y/o pedirle a un colega revise la PR antes de enviarla:* intente eliminar tantos errores comunes como sea posible (estilo de código, errores ortográficos y gramaticales, etc.) antes de enviar la PR para que la revisen. Asegúrese de que se pasen todas las pruebas.
- *Leer atentamente los comentarios de su revisor:* intente comprender lo que pide el revisor y abordar todos los comentarios antes de publicar una nueva versión.
- *Evitar discusiones tangenciales (bikeshedding):* las discusiones y los desacuerdos técnicos son muy valiosos y nadie es perfecto. Sin embargo, evite discusiones que no marquen la diferencia o que sean meramente estilísticas. Si no está de acuerdo con el comentario de un revisor, trate detallar sus motivos de la manera más precisa y completa posible para evitar largas discusiones de ida y vuelta.
- *Evitar hacer las "preguntas de revisión más frecuentes" que se enumeran a continuación:* enumeramos algunas respuestas a las preguntas más comunes y damos nuestra justificación.

En general, le recomendamos que intente que la revisión de sus PR nos lleve el menor tiempo posible. Entonces querremos revisar sus cambios rápidamente.

¡Gracias por contribuir a XLA! Que tenga una excelente sesión de programación.

## Preguntas de revisión más frecuentes

### "Este cambio de infraestructura no está relacionado con mi solicitud de incorporación, ¿por qué debería hacerlo?"

El equipo de XLA no tiene un equipo de infraestructura dedicado, por lo que todos debemos compilar bibliotecas de ayuda y evitar deudas técnicas. Consideramos que es una tarea habitual a la hora de introducir cambios en XLA y se espera que todos participen. Por lo general, compilamos la infraestructura a medida que la necesitamos cuando escribimos código.

Los revisores de XLA pueden pedirle que compile alguna infraestructura (o que haga un cambio importante en una PR) junto con una PR que haya escrito. Esta solicitud puede parecer innecesaria u ortogonal al cambio que intenta realizar. Es probable que esto se deba a una discrepancia entre sus expectativas sobre la cantidad de infraestructura que debe compilar y las expectativas del revisor al respecto.

¡Es normal que las expectativas no coincidan! Eso es lo que se espera cuando se es nuevo en un proyecto (y a veces incluso nos pasa después de mucho tiempo). Es probable que los proyectos en los que haya trabajado en el pasado tengan expectativas diferentes. ¡Eso también está bien y es de esperarse! Esto no significa que algunos de estos proyectos tengan el enfoque equivocado; simplemente son diferentes. Lo invitamos a aceptar las solicitudes de infraestructura junto con todos los demás comentarios de revisión como una oportunidad para conocer qué esperamos de este proyecto.

### "¿Puedo abordar su comentario en una próxima PR?"

Una pregunta frecuente con respecto a las solicitudes de infraestructura (u otras solicitudes grandes) en las PR es si el cambio debe hacerse en la PR original o si se puede hacer como seguimiento en una futura PR.

En general, XLA no permite que los autores de PR aborden los comentarios de la revisión como PR de seguimiento. Cuando un revisor decide que es necesario abordar algo en esta PR, generalmente esperamos que los autores lo aborden en la PR original, incluso si lo que se solicita es un cambio grande. Este estándar se aplica externamente y también internamente dentro de Google.

Hay varias razones por las que XLA adopta este enfoque.

- *Confianza:* ganarse la confianza del revisor es un componente clave. En un proyecto de código abierto, los contribuyentes pueden aparecer o desaparecer libremente. Después de aprobar una PR, los revisores no tienen forma de garantizar que los seguimientos prometidos realmente se lleven a cabo.

- *Incidencia en otros desarrolladores:* si envió una PR que aborda una parte específica de XLA, es muy probable que otras personas estén viendo la misma parte. Si aceptamos la deuda técnica en su PR, todos los que vean este archivo se verán afectados por esta deuda hasta que se envíe el seguimiento.

- *Ancho de banda de los revisores:* aplazar un cambio a un seguimiento implica múltiples inconvenientes para nuestros revisores, que de por sí ya están sobrecargados. Los revisores probablemente se olvidarán de qué se trataba esta PR mientras esperan el seguimiento, lo que dificultará la siguiente revisión. Además, los revisores deberán llevar un registro de los seguimientos que están esperando, para asegurarse de que realmente se lleven a cabo. Si el cambio se puede hacer de manera que sea verdaderamente ortogonal a la PR original para que otro revisor pueda revisarla, el ancho de banda sería un problema menor. En nuestra experiencia, este rara vez es el caso.
