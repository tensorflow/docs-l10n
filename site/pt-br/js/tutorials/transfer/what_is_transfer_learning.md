# O que é aprendizado por transferência?

Modelos sofisticados de aprendizado profundo têm milhões de parâmetros (pesos), e treiná-los do zero costuma exigir uma grande quantidade de dados e recursos computacionais. O aprendizado por transferência é uma técnica que pega um atalho ao aproveitar um pedaço de um modelo já treinado em uma tarefa relacionada e reutilizá-lo em um novo modelo.

Por exemplo, o próximo tutorial nesta seção mostrará como criar seu próprio reconhecedor de imagens que aproveita um modelo já treinado para reconhecer milhares de diferentes tipos de objetos dentro de imagens. Você pode adaptar os conhecimentos existentes do modelo pré-treinado para reconhecer suas próprias classes de imagens usando muito menos dados de treinamento do que os necessários para o modelo original.

Essa técnica é útil para desenvolver novos modelos bem como para personalizar modelos em ambientes com restrição de recursos, como navegadores e dispositivos móveis.

Na maioria das vezes, ao fazermos aprendizado por transferência, não ajustamos os pesos do modelo original. Em vez disso, removemos a camada final e treinamos um novo modelo (geralmente, bem superficial) baseado na saída do modelo truncado. É essa técnica que será demonstrada nos tutoriais desta seção.

- [Crie um classificador de imagem baseado em aprendizado por transferência](image_classification)
- [Crie um reconhecedor de áudio baseado em aprendizado por transferência](audio_recognizer)
