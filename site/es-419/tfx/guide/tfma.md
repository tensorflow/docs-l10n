# Cómo mejorar la calidad del modelo con TensorFlow Model Analysis

## Introducción

A medida que modifica su modelo durante el desarrollo, debe verificar si sus cambios están mejorando el modelo. Quizás limitarse a comprobar la precisión no sea suficiente. Por ejemplo, si tiene un clasificador para un problema en el que el 95 % de sus instancias son positivas, es posible que pueda mejorar la precisión si se limita a predecir siempre resultados positivos, pero no tendrá un clasificador muy sólido.

## Descripción general

El objetivo de TensorFlow Model Analysis es proporcionar un mecanismo para la evaluación de modelos en TFX. El análisis de modelos de TensorFlow le permite realizar evaluaciones de modelos en la canalización de TFX y ver las métricas y los gráficos resultantes en un bloc de notas Jupyter. En concreto, puede proporcionar lo siguiente:

- [Métricas](../model_analysis/metrics) calculadas en todo el conjunto de datos de entrenamiento y retención, así como evaluaciones del día siguiente
- Seguimiento de métricas a lo largo del tiempo
- Rendimiento de calidad del modelo en diferentes segmentos de características
- [Validación del modelo](../model_analysis/model_validations) para garantizar que el modelo mantenga un rendimiento constante

## Siguientes pasos

Pruebe nuestro [tutorial de TFMA](../tutorials/model_analysis/tfma_basic).

Consulte nuestra página de [GitHub](https://github.com/tensorflow/model-analysis) para obtener detalles sobre [métricas y gráficos](../model_analysis/metrics) admitidos y las [visualizaciones](../model_analysis/visualizations) de blocs de notas asociados.

Consulte las guías de [instalación](../model_analysis/install) y [introducción](../model_analysis/get_started) para obtener información y ejemplos sobre cómo [configurar](../model_analysis/setup) una canalización independiente. Recuerde que TFMA también se usa dentro del componente [Evaluator](evaluator.md) en TFX, por lo que estos recursos también serán útiles para iniciarse en TFX.
