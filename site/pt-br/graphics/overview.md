# Visão geral

Nos últimos anos, temos visto uma ascensão de camadas gráficas diferenciáveis originais que podem ser acopladas às arquiteturas de redes neurais. De transformadores espaciais a renderizadores gráficos diferenciáveis, essas novas camadas usam os conhecimentos de visão computacional e pesquisa gráfica adquiridos ao longo dos anos para criar arquiteturas de rede novas e mais eficientes. Modelar explicitamente restrições e priors geométricos em redes neurais abre as portas para arquiteturas que podem ser treinadas de maneira robusta, eficiente e, acima de tudo, autosupervisionadas.

De forma geral, um pipeline de computação gráfica requer uma representação de objetos tridimensionais e sua posição absoluta na cena, uma descrição do material de composição, luzes e uma câmera. Em seguida, a descrição dessa cena é interpretada por um renderizador para gerar uma renderização sintética.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/graphics.jpg" width="600">
</div>

Comparativamente, um sistema de visão computacional começaria por uma imagem e tentaria inferir os parâmetros da cena, o que permite prever quais objetos estão na cena, de quais materiais são feitos e a posição e orientação tridimensionais.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv.jpg" width="600">
</div>

Geralmente, treinar sistemas de aprendizado de máquina capazes de resolver essas tarefas complexas de visão tridimensional requer grandes quantidades da dados. Como fazer a rotulação de dados é um processo caro e complexo, é importante ter mecanismos para desenvolver modelos de aprendizado de máquina que consigam compreender o mundo tridimensional e que possam ser treinados sem muita supervisão. Combinar técnicas de visão computacional e computação gráfica proporciona uma oportunidade única de aproveitar as grandes quantidades de dados não rotulados prontamente disponíveis. Conforme ilustrado na imagem acima, pode-se conseguir isso usando análise por síntese, em que o sistema de visão extrai os parâmetros da cena, e o sistema gráfico renderiza uma imagem com base neles. Se a renderização coincidir com a imagem original, o sistema de visão terá extraído com precisão os parâmetros da cena. Nessa configuração, a visão computacional e a computação gráfica andam de mãos dadas, formando um único sistema de aprendizado de máquina similar a um autoencoder, que pode ser treinado de uma forma autosupervisionada.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv_graphics.jpg" width="600">
</div>

O Tensorflow Graphics está sendo desenvolvido para ajudar a enfrentar esses tipos de desafios e, para isso, oferece um conjunto de camadas de gráficos diferenciáveis e geometria (por exemplo, câmeras, modelos de reflexividade, transformações espaciais, convoluções de malhas) e funcionalidades de exibição tridimensional (por exemplo, TensorBoard 3D), que podem ser usadas para treinar e depurar os modelos de aprendizado de máquina da sua escolha.
