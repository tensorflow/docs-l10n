# Conceitos do aprendizado de máquina quântico

O <a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">experimento de quântica além do clássico</a> do Google usou 53 qubits *ruidosos* para demonstrar que poderia realizar um cálculo em um computador quântico em 200 segundos que levaria 10 mil anos no maior computador clássico usando algoritmos existentes. Isso marca o início da era computacional <a href="https://quantum-journal.org/papers/q-2018-08-06-79/" class="external">Quântica de Escala Intermediária Ruidosa</a> (NISQ). Nos próximos anos, os dispositivos quânticos com dezenas a centenas de qubits ruidosos devem se tornar realidade.

## Computação quântica

A computação quântica recorre a propriedades da mecânica quântica para calcular problemas que estão fora do alcance dos computadores clássicos. Um computador quântico usa *qubits*. Os qubits são como bits regulares em um computador, mas também podem ser colocados em uma *superposição* e compartilharem *emaranhamento* uns com os outros.

Os computadores clássicos realizam operações clássicas determinísticas ou emulam processos probabilísticos usando métodos de amostragem. Ao usar a superposição e o emaranhamento, os computadores quânticos podem realizar operações quânticas que são difíceis de emular em grande escala com computadores clássicos. Algumas ideias de uso da computação quântica NISQ incluem a otimização, a simulação quântica, a criptografia e o aprendizado de máquina.

## Aprendizado de máquina quântico

O *aprendizado de máquina quântico* (QML) é baseado em dois conceitos: *dados quânticos* e *modelos clássicos-quânticos híbridos*.

### Dados quânticos

Os *dados quânticos* são qualquer fonte de dados que ocorre em um sistema quântico natural ou artificial. Isso pode ser dados gerados por um computador quântico, como as amostras coletadas do <a href="https://www.nature.com/articles/s41586-019-1666-5" class="external">processador Sycamore</a> para a demonstração do Google da supremacia quântica. Os dados quânticos exibem superposição e emaranhamento, levando a distribuições de probabilidade conjunta que exigiriam uma quantidade exponencial de recursos computacionais clássicos para a representação ou o armazenamento. O experimento da supremacia quântica mostrou que é possível obter amostras de uma distribuição de probabilidade conjunta extremamente complexa de 2^53 de espaço de Hilbert.

Os dados quânticos gerados por processadores NISQ são ruidosos e geralmente emaranhados logo antes da medição. Técnicas heurísticas de aprendizado de máquina podem criar modelos que maximizam a extração de informações clássicas úteis de dados emaranhados ruidosos. A biblioteca do TensorFlow Quantum (TFQ) fornece primitivos para desenvolver modelos que desemaranham e generalizam correlações em dados quânticos, gerando oportunidades para descobrir novos algoritmos quânticos ou melhorar os existentes.

Os seguintes são exemplos de dados quânticos que podem ser gerados ou simulados em um dispositivo quântico:

- *Simulação química* — extraia informações sobre estruturas e dinâmicas químicas com possíveis aplicações na ciência dos materiais, química computacional, biologia computacional e descoberta de medicamentos.
- *Simulação quântica de matéria* — modele e projete a supercondutividade de alta temperatura ou outros estados exóticos de matéria que exibem efeitos quânticos de muitos corpos.
- *Controle quântico* — modelos clássicos-quânticos híbridos podem ser treinados variacionalmente para realizar melhor controle de loop aberto ou fechado, calibração e mitigação de erros. Isso inclui estratégias de detecção e correção de erros para dispositivos e processadores quânticos.
- *Redes de comunicação quânticas* — use o aprendizado de máquina para discriminar entre estados quânticos não ortogonais, com aplicação ao design e à construção de repetidores quânticos estruturados, receptores quânticos e unidades de purificação.
- *Metrologia quântica* — as medições de alta precisão aprimoradas com quântica, como sensores e imagens, são feitas inerentemente em sondas que são dispositivos quânticos de pequena escala e podem ser projetados ou melhorados por modelos quânticos variacionais.

### Modelos clássicos-quânticos híbridos

Um modelo quântico pode representar e generalizar dados com uma origem mecânica quântica. Como os processadores quânticos de curto prazo ainda são bastante pequenos e ruidosos, os modelos quânticos não podem generalizar os dados quânticos usando apenas esses processadores. Os processadores NISQ precisam trabalhar em conjunto com coprocessadores clássicos para serem eficazes. O TensorFlow já é compatível com a computação heterogênea em CPUs, GPUs e TPUs, então é usado como a plataforma de base para experimentar com algoritmos clássicos-quânticos híbridos.

Uma *rede neural quântica* (QNN) é usada para descrever um modelo computacional quântico parametrizado que é melhor executado em um computador quântico. Esse termo é geralmente intercambiável com *circuito quântico parametrizado* (PQC).

## Pesquisa

Durante a era NISQ, algoritmos quânticos com speedups reconhecidos em relação a algoritmos clássicos — como o <a href="https://arxiv.org/abs/quant-ph/9508027" class="external">algoritmo de fatoração de Shor</a> ou o <a href="https://arxiv.org/abs/quant-ph/9605043" class="external">algoritmo de busca de Grover</a> — ainda não são possíveis em uma escala significativa.

Uma meta do TensorFlow Quantum é ajudar a descobrir algoritmos para a era NISQ, com interesse especial no seguinte:

1. *Uso do aprendizado de máquina clássico para aprimorar algoritmos NISQ.* A esperança é que técnicas do aprendizado de máquina clássico possam melhorar nossa compreensão da computação quântica. No <a href="https://arxiv.org/abs/1907.05415" class="external">meta-aprendizado para redes neurais quânticas por redes neurais recorrentes clássicas</a>, uma rede neural recorrente (RNN) é usada para descobrir que a otimização dos parâmetros de controle para algoritmos como QAOA e VQE são mais eficientes do que otimizadores simples prontos para uso. E o <a href="https://www.nature.com/articles/s41534-019-0141-3" class="external">aprendizado de máquina para controle quântico</a> usa o aprendizado por reforço para ajudar a mitigar erros e produzir portas quânticas de maior qualidade.
2. *Modelagem de dados quânticos com circuitos quânticos.* A modelagem clássica de dados quânticos é possível se você tem uma descrição exata da origem dos dados. No entanto, às vezes isso não é possível. Para resolver esse problema, você pode tentar modelar no próprio computador quântico e medir/observar estatísticas importantes. As <a href="https://www.nature.com/articles/s41567-019-0648-8" class="external">redes neurais convolucionais quânticas</a> mostram um circuito quântico projetado com uma estrutura análoga a uma rede neural convolucional (CNN) para detectar diferentes fases topológicas da matéria. O computador quântico armazena os dados e o modelo. O processador clássico vê apenas as amostras de medição da saída do modelo, e nunca os próprios dados. Na <a href="https://arxiv.org/abs/1711.07500" class="external">renormalização de emaranhamento robusto em um computador quântico ruidoso</a>, os autores aprendem a comprimir informações sobre sistemas quânticos de muitos corpos usando um modelo DMERA.

Outras áreas de interesse no aprendizado de máquina quântico incluem:

- Modelagem de dados puramente clássicos em computadores quânticos.
- Algoritmos clássicos inspirados em quântica.
- <a href="https://arxiv.org/abs/1810.03787" class="external">Aprendizado supervisionado com classificadores quânticos</a>.
- Aprendizado adaptativo em camadas para a rede neural quântica.
- <a href="https://arxiv.org/abs/1909.12264" class="external">Aprendizado de dinâmica quântica</a>.
- <a href="https://arxiv.org/abs/1910.02071" class="external">Modelagem generativa de estados quânticos misturados</a>.
- <a href="https://arxiv.org/abs/1802.06002" class="external">Classificação com redes neurais quânticas em processadores de curto prazo</a>.
