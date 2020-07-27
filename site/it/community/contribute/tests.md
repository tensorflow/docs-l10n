# Buone pratiche di testing per TensorFlow

Queste sono le pratiche raccomandate per il testing del codice nel
[repository TensorFlow](https://github.com/tensorflow/tensorflow).

## Prima di iniziare

Prima di iniziare il tuo contributo al codice sorgente di un progetto TensorFlow, leggi il
file `CONTRIBUTING.md` nel repository GitHub del progetto. (Per esempio, guarda
[il file CONTRIBUTING.md per il repository del core di TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).)
A tutti i contributori è richiesto di firmare un
[Contributor License Agreement (Accordo di Licenza di Contribuzione n.d.t.)](https://cla.developers.google.com/clas) (CLA).

## Principi Generali

### Dipendi solo da ciò che metti nelle tue regole di BUILD

TensorFlow è una grossa libreria, e mettere la dipendenza dal package completo quando
stai scrivendo uno unit test per uno dei suoi sotto-moduli è una pratica comune. Ciò, però,
inibisce l'analisi `bazel` basata sulle dipendenze. Ciò significa che i sistemi di integrazione 
continua (continuos integration n.d.t.) non possono eliminare intelligentemente i test irrilevanti per 
per le prove delle pre/post condizioni. Se inserisci nel tuo file `BUILD`
solo le dipendenze dai sotto-moduli che stai testando, farai risparmiare tempo a tutti gli
sviluppatori di TensorFlow, ed un sacco di costosa potenza di calcolo.

Comunque, modificare le tue dipendenze di compilazione omettendo l'intero TF, comporta alcune
limitazioni a ciò che puoi importare nel tuo codice Python. Non potrai più usare
l'enunciato `import tensorflow as tf` nei tuoi test di unità. Ma questo è comunque
compromesso utile, perché evita agli sviluppatori di eseguire migliaia di test non necessari.

### Scrivi degli unit test per tutto il codice

Qualsiasi codice tu scriva, dovresti scrivere anche i relativi unit test. Se scrivi un nuovo
file `foo.py`, dovresti mettere i suoi unit test in `foo_test.py` e sottoporli 
all'interno della stessa richiesta di cambiamento. Puntando ad una copertura di test incrementali di 
più del 90% del tuo codice.

### In TF, evita di usare delle regole native di bazel test

TF ha mole sottigliezze nell'esecuzione dei test. Abbiamo lavorato per nascondere tutte
queste complessità nelle nostre bazel macro. Per evitare di averci a che fare, invece
delle regole native di test, usa quelle che seguono. Nota che esse sono tutte 
definite in `tensorflow/tensorflow.bzl`
Per i test CC, usa `tf_cc_test`, `tf_gpu_cc_test`, `tf_gpu_only_cc_test`.
Per i test Python, usa `tf_py_test` o `gpu_py_test`.
Se hai necessità di qualcosa di veramente vicino ad una regola nativa `py_test`, usa
una di quelle definite in tensorflow.bzl. Hai solo bisogno di aggiungere la linea seguente
all'inizio del file BUILD: `load(“tensorflow/tensorflow.bzl”, “py_test”)`

### Fai attenzione a dove viene eseguito il test

Quando scrivi un test, se li scrivi appropriatamente, la nostra infrastruttura di test
può eseguirli su CPU, GPU o acceleratori. Abbiamo test automatizzati
che girano su Linux, MacOs e Windows, che non usano GPUs. Devi semplicemente
usare una delle macro elencate sopra, e quindi usare dei tag per limitare dove i test
vengano eseguiti.

* il tag `manual` esclude che il vostro test sia eseguito. Ciò implica l'esecuzione
di test manuale che usa pattern come `bazel test tensorflow/…`

* il tag `no_oss` escluderà che il tuo test sia eseguito all'interno dell'infrastruttura 
di test ufficiale TF OSS.

* i tag `no_mac` e `no_windows` possono essere usati per escludere il tuo test da ambienti
relativi a specifici sistemi operativi.
* il tag `no_gpu` può essere usato per escludere che il tuo test sia eseguito 
all'interno degli ambienti di test con GPU.

### Verifica che i test vengano eseguiti negli ambienti previsti

TF ha abbastanza pochi ambienti di test. Talvolta, ci si può confondere durante l'impostazione.
Possono verificarsi diversi problemi che possono causare che i tuoi test siano omessi dalla 
compilazione continua (continuous build n.d.t.). Di conseguenza, dovresti verificare che i 
test siano eseguiti come atteso
Per farlo:

* Attendi che i presubmit sulle tue richieste di Pull (PR) arrivino a completamento.
* Scorri fino alla fine della tua PR per vedere lo stato dei controlli.
* Clicca sul link “Dettagli” a destra di ogni controllo Kokoro.
* Verifica la lista “Risultati” per trovare il tuo risultato appena aggiunto.

### Ogni classe/unità deve avere il proprio file di unit test

Separare le classi di test ci aiuta a isolare meglio i malfunzionamenti e le risorse.
Ciò rende i file di test più corti e facili da leggere. Di conseguenza, tutti i tuoi file
Python devono avere almeno un corrispondente file di test (Per ogni `foo.py`, ci dovrebbe
essere `foo_test.py`). Per test più elaborati, come test di integrazione che
richiedono setup diversi, va bene aggiungere più file di test.

## velocità e tempi di esecuzione

### Lo sharding dovrebbe essere usato il meno possibile

Invece dello sharding considera di:
* Rendere il tuo test più piccolo
* Se il punto precedente non è realizzabile, dividi il test

Lo sharding aiuta a ridurre la latenza complessiva del test, ma lo stesso risultato può
essere raggiunto spezzando i test in segmenti più piccoli. Scindere i test ci da un 
migliore livello di controllo su ciascun test, minimizzando l'esecuzione di presubmit
e riducendo la perdita di copertura dovuta all'eventuale distrazione di un revisore del codice,
dovuta ad un caso di test con comportamento errato. Quindi, lo sharding impica costi nascosti 
non così ovvi quando arriva a girare tutto il codice di inizializzazione dei test
per tutti gli shard. Questo problema ci è stato trasferito dai team infrastrutturali come
una sorgente di carico extra.

### Test più piccoli sono migliori

Più veloce è l'esecuzione del tuo test, più facilmente la gente sarà in grado di eseguirlo.
Un secondo in più del tuo test può accumulare ore di tempo extra speso
dagli sviluppatori e dall'infrastruttura, eseguendo il tuo test. Prova a fare in modo che 
i tuoi test vengano eseguiti in meno di 30 secondi (non è un'opzione!), e rendili più piccoli.
Marca i tuoi test come medi solo come ultima possibilità.
L'infrastruttura non esegue ogni test grande come presubmit o postsubmit! 
Quindi, scrivi un test ampio solo se stai decidendo dove andrà eseguito. 
Di seguito, alcuni suggerimenti per rendere i test più veloci:

* Nei tuoi test, esegui meno iterazioni di addestramento
* Considera di aggiungere dipendenze da semplici simulacri per rimpiazzare dipendenze forti del sistema
sotto test.
* Considera di usare dati di ingresso più piccoli per gli unit test
* Se nient'altro funziona, prova a scindere i tuoi file di test.

### I tempi di test dovrebbero puntare a durare la metà della soglia di di timeout 
per evitare interruzioni.

Con test target `bazel`, i test piccoli hanno timeout di 1 minuto. Il timeout dei test medi
è di 5 minuti. I test grandi non sono proprio eseguiti dall'infrastruttura di test di
TensorFlow. Comunque, molti test non hanno una durata deterministica. 
Per varie ragioni i tuoi test possono richiedere più tempo da una volta all'altra.
Inoltre, se fai un test che dura come minimo 50 secondi in media, il tuo test
sarà interrotto se schedulato su una macchina con una vecchia CPU Quindi punta 
ad un tempo medio di esecuzione di 30 secondi per test piccoli e punta ad un tempo medio di 
2 minuti e 30 secondi per i test medi.

### Nell'addestramento, riduci il numero di campioni ed aumenta la tolleranza

Lunghi tempi di test deprimono i contributori. L'esecuzione dell'addestramento in test
può essere molto lento. Opta per tolleranze più ampie in modo da usare meno campioni nei tuoi
test, così potrai mantenere i tuoi test sufficientemente rapidi (2.5 minuti massimo).

## Elimina il non-determinismo e l'indeterminatezza

### Scrivi test deterministici

Gli unit test devono essere sempre deterministici. Tutti i test che girano su TAP o Guitar,
in assenza di modifiche al codice, devono girare tutte le volte nello stesso modo. 
Per assicurarlo, qui sotto ci sono alcuni punti da considerare.

### Seminate sempre ogni sorgente di stocasticità

Ogni generatore di numeri casuali, o ogni altra sorgente di stocasticità può causare 
indeterminatezza. Di conseguenza, ogni aspetto di questo tipo dovrebbe essere evitato.
Oltre a permettere di eseguire test con minore ideterminatezza, ciò rende tutti i test
riproducibili. I vari modi in cui potete impostare i vari semi di cui potete aver
bisogno nei test di TF sono:

```python
# Python RNG
import random
random.seed(42)

# Numpy RNG
import numpy as np
np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)
```

### Evita di usare `sleep` nei test di codice multithread

Usare la funzione `sleep` nei test può essere una grossa fonte di indeterminatezza. 
Specialmente quando si usano thread multipli, usare sleep per attendere un altro thread
non può essere mai deterministico. Ciò è dovuto al fatto che il sistema non è in grado di garantire
alcun ordinamento dell'esecuzione di thread o processi diversi. Di conseguenza,
è sono preferibili costrutti di sincronizzazione deterministici come i mutex.

### Controllare se il test è indeterminato

LLe indeterminatezze fanno perdere molte ore agli sviluppatori ed ai revisori del codice
Sono difficili da diagnosticare e difficili da correggere. Anche pensando di avere sistemi
automatici di diagnosi delle incertezze, esse necessitano di accumulare centinaia di
esecuzioni dei test prima di poter escludere accuratamente i test. 
Anche quando essi li identifichino, essi escludono i tuoi test e la copertura del test si perde.
Di conseguenza, gli autori dei test devono controllare se i loro test siano indeterminati. 
Ciò può essere fatto facilmente eseguendo il test con il flag: `--runs_per_test=1000`

### Usa TensorFlowTestCase

TensorFlowTestCase prende le precauzioni necessarie, come impostare i semi di tutti i 
generatori di numeri casuali, per ridurre il più possibile l'indeterminatezza. Man mano che
scopriamo possibili sorgenti di indeterminatezza, le aggiungiamo a TensorFlowTestCase.
Di conseguenza, dovresti usare TensorFlowTestCase nella scrittura di test per tensorflow.
TensorFlowTestCase è definito qui: `tensorflow/python/framework/test_util.py`

### Scrivere test ermetici

I test ermetici non necessitano di risorse esterne. Essi sono inclusi in un pacchetto che
contiene ogni cosa di cui hanno bisogno, e semplicemente avviano ogni simulacro di servizio di cui
possano aver bisogno. Ogni servizio diverso da quello del tuo test può essere fonte di non determinismo.
Anche con il 90% della disponibilità di altri servizi, la rete può non essere disponibile, 
la risposta ad una rpc può essere ritardata, e tu puoi finire col trovarti un inesplicabile
messaggio di errore.
Servizi esterni possono essere, ma non limitarsi a: GCS, S3 o ogni altro sito web.
