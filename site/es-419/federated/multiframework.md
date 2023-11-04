# Compatibilidad con múltiples marcos en TensorFlow Federated

TensorFlow Federated (TFF) fue diseñado para ofrecer compatibilidad con una amplia gama de cálculos federados, expresados a través de una combinación de operadores federados de TFF que modelan la comunicación distribuida y la lógica de procesamiento local.

Actualmente, la lógica de procesamiento local se puede expresar por medio de las API de TensorFlow (a través de `@tff.tf_computation`) en el frontend y se ejecuta a través del tiempo de ejecución de TensorFlow en el backend. Sin embargo, nuestro objetivo es admitir muchos otros marcos de frontend y backend (que no sean TensorFlow) para cálculos locales, incluidos marcos que no sean de ML (por ejemplo, para lógica expresada en SQL o lenguajes de programación de propósito general).

En esta sección, se incluye información sobre lo que sigue:

- Mecanismos que proporciona TFF para admitir marcos alternativos y cómo puede agregar compatibilidad con su tipo preferido de frontend o backend a TFF.

- Implementaciones experimentales de compatibilidad con marcos que no son de TensorFlow, con ejemplos.

- Hoja de ruta futura aproximada para que estas capacidades superen la fase experimental.
