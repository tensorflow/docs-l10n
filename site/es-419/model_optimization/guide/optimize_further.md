# Optimizar aún más

Cuando los modelos preoptimizados y las herramientas posentrenamiento no satisfacen su caso de uso, el siguiente paso es probar las diferentes herramientas durante el entrenamiento.

Las herramientas durante el entrenamiento aprovechan la función de pérdida del modelo sobre los datos de entrenamiento de modo que el modelo pueda "adaptarse" a los cambios causados por la técnica de optimización.

El punto de partida para usar nuestras API de entrenamiento es un script de entrenamiento de Keras, que opcionalmente se puede inicializar desde un modelo de Keras preentrenado para realizar ajustes adicionales.

Herramientas durante el entrenamiento disponibles para que las pruebe:

- [Eliminación de pesos](./pruning/)
- [Cuantización](./quantization/training)
- [Agrupación de pesos](./clustering/)
- [Optimización colaborativa](./combine/collaborative_optimization)
