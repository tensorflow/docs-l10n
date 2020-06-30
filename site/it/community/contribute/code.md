# Contribuire al codice TensorFlow

Sia che tu voglia aggiungere una funzione obiettivo, migliorare la copertura di un test, oppure scrivere una
RFC per un cambiamento rilevante di progettazione, questa parte della guida del contributore ti sarà d'aiuto
per cominciare. Grazie per il lavoro e l'interesse nel miglioramento di TensorFlow.

## Prima di cominciare

Prima di contribuire al codice sorgente di un progetto TensorFlow, cortesemente consulta il file `CONTRIBUTING.md` nel repository GitHub del progetto. (Per esempio, vedi
[file CONTRIBUTING.md nel repository centrale di TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).) A tutti i contributori è richiesto di firmare un [Accordo di Licenza di Contribuzione](https://cla.developers.google.com/clas) (CLA).

Per evitare attività duplicate, prima di cominciare a lavorare su una funzionalità non banale, cortesemente consulta [RFC correnti](https://github.com/tensorflow/community/tree/master/rfcs) e contatta gli sviluppatori nei forum TensorFlow. Talvolta siamo selettivi quando dobbiamo decidere se aggiungere una funzionalità, ed il modo migliore di contribuire ed aiutare il progetto è lavorare su problemi conosciuti. 

## Issue per il nuovo contributore

I nuovi contributori alla ricerca di un primo issue per un contributo al codice TensorFlow, dovrebbero cercare i seguenti tag. Raccomandiamo fortemente che i nuovi contributori affrontino inizialmente progetti “easy” e problematiche "good first issue"; questo aiuta il contributore a familiarizzare con il flusso di lavoro di contribuzione, ed agli sviluppatori del core di conoscere il contributore.

- `good first issue`
- `easy`
- `contributions welcome`

Se siete interessati a reclutare un team per aiutarvi ad affrontare un problema ampio o una nuova funzionalità, mandate una email a [gruppo sviluppatori@](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers) e osservate la nostra lista attuale di RFC. 


## Revisione del codice

Nuove funzioni, eliminazione di difetti, ed ogni altro cambiamento nel codice base sono soggetti a revisione del codice.

La revisione del codice ricevuto come contributo in un progetto attraverso le "pull request" è una componente fondamentale dello sviluppo di TensorFlow. Incoraggiamo chiunque a cominciare rivedendo codice sottoposto da altri sviluppatori, specialmente in quei casi in cui il codice implementa qualcosa che ti piacerebbe usare.

Qui alcune domande da tenere a mente durante il processo di revisione del codice:

*   *Vogliamo questa cosa in TensorFlow?* Potrebbe piacere usarla? Come utente TensorFlow, ti piacerebbe questa modifica e la useresti? Si tratta di un cambiamento nell'ambito di TensorFlow? Il costo di manutenzione di questa nuova funzione sarebbe maggiore dei suoi benefici?
*   *Il codice è consistente con le API di TensorFlow?* Le funzioni pubbliche, le classi, ed i parametri sono nominati in modo appropriato e progettati intuitivamente?
*   *E' inclusa la documentazione?* Le funzioni pubbliche, le classi, i tipi di ritorno e gli attributi memorizzati sono nominati in accordo con le convenzioni di TensorFlow e sono documentati chiaramente? La nuova funzionalità è descritta nella documentazione di TensorFlow ed illustrata, ovunque possibile, con esempi? La documentazione si legge in modo appropriato?

*   *Il codice è facilmente leggibile?* Si evitano ridondanze? I nomi delle variabili dovrebbero essere migliorati per aumentare la chiarezza o la consistenza? Dovrebbero essere aggiunti commenti? Ci sono commenti inutili o estranei che potrebbero essere rimossi?
*   *Il codice è efficiente?* Potrebbe essere riscritto facilmente per essere più efficiente?
*   IL codice è *retro-compatibile* con le versioni precedenti di TensorFlow?
*   Il nuovo codice aggiungerà *nuove dipendenze* on other libraries?

## Test e miglioramento della copertura dei test

Lo Unit testing di alta qualità è una pietra miliare del processo di sviluppo di TensorFlow. Per questo scopo usiamo immagini Docker. Le funzioni di test hanno nomi appropriati, e sono responsabili del controllo della validità degli algoritmi, così come delle varie opzioni del codice.

Tutte le nuove funzioni e le correzioni di difetti *devono* includere test con adeguata copertura. Perciò accogliamo anche contributi di nuovi casi di test o migliorie ai test esistenti. Se scoprite che i nostri attuali test non sono completi — anche se ciò, al momento, non causa un difetto — per cotesia, sottoponete un issue e, se possibile una pull request.

Per tutti i dettagli particolari sulle procedure di test in ciascun progetto TensorFlow, vedere i file `README.md` e `CONTRIBUTING.md` nel rispettivo repository su GitHub.

Di particolare importanza è fare *testing adeguato*:

*   E stata testata *ogni funzione pubblica di ogni classe*? 
*   Sono stati testati *insiemi ragionevoli di parametri*, di valori, tipi e in ragionevoli combinazioni? 
*   I test sono in grado di validare che *il codice è corretto*, e che esso faccia *ciò che la documentazione dice* che esso faccia?
*   Se la modifica è la correzione di un difetto, è incluso un *test di non regressione*?
*   Il test *supera il build in continuous integration*?
*   Il test *copre tutte le linee del codice?* Se no, le eccezioni sono ragionevoli ed esplicite?

Se individuate dei problemi, considerate la possibilità di supportare un contributore a capire questi problemi ed a risolverli. 


## Miglioramento dei messaggi di errore o dei log

Accogliamo contributi che migliorano la messaggistica di errore e il logging. 


## Flusso dei contributi

Tutti i contributi al codice—correzioni, nuovi sviluppi, miglioramenti ai test—seguono un flusso incentrato su GitHub. Per partecipare allo sviluppo in TensorFlow, aprite un account GitHub. Quindi:

1.  Fate un Fork del repository su cui avete intenzione di lavorare.
    Andate sulla pagina del repository di progetto e usate il pulsante *Fork*. Ciò creerà una copia del
    repository, con il vostro nome utente. (Per maggiori dettagli su come fare il fork di un repository vedere
    [questa guida](https://help.github.com/articles/fork-a-repo/).)

2.  Clonate il repository sul vostro sistema locale.

    `$ git clone git@github.com:your-user-name/project-name.git`

3.  Definite un nuovo branch per isolare il vostro lavoro.

    `$ git checkout -b new-branch-name`

4.  Lavorate sul vostro nuovo codice. Scrivere ed eseguite i test.

5.  Fate il commit delle vostre modifiche.

    `$ git add -A`

    `$ git commit -m "commit message here"`

6.  Fate un push delle vostre modifiche sul vostro repository GitHub.

    `$ git push origin branch-name`

7.  Aprite una *Pull Request* (PR). Andando su repository originale del progetto su GitHub. Ci sarà un messaggio che evidenzia il branch di cui avete fatto il push di recente, che vi chiederà se avete l'intenzione di aprire una pull request. Scegliete, *confronto tra repository* (*compare across repositories* n.d.t), e sottoponete la PR. Ciò produrrà l'invio di una email ai committer. Per dare maggiore visibilità potete considerare la possibilità di inviare anche un'email alla mailing list. (Per maggiori dettagli, vedere la [guida di GitHub sulle PR](https://help.github.com/articles/creating-a-pull-request-from-a-fork). 

8.  I maintainer ed altri contributori *revisioneranno la vostra PR*. Partecipate alla conversazione, e provate ad *eseguire ogni richiesta di cambiamento*. Una volta che la PR è approvata, il codice sarà incorporato.

*Prima di mettervi a lavorare sul vostro prossimo contributo*, assicuratevi che il vostro repository locale sia aggiornato.

1. Impostate l'upsteram remoto. (Basta farlo una sola volta per ogni progetto, non ogni volta.)

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. Spostatevi sul vostro branch locale principale.

    `$ git checkout master`

3. Scaricate i cambiamenti dal riferimento remoto.

    `$ git pull upstream master`

4. Promuovete i cambiamenti nel vostro account GitHub. (E' opzionale, ma si tratta di una buona pratica.)

    `$ git push origin master`

5. Genrate un nuovo branch se dovete iniziare un nuovo lavoro.

    `$ git checkout -b branch-name`

Risorse aggiuntive `git` e GitHub:

*   [Documentazione git](https://git-scm.com/documentation)
*   [Documentazione sul flusso di sviluppo git](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
*   [Soluzione dei conflitti di merge](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/).


## Checklist del contributore

*   Leggere le linee guida per la contribuzione.
*   Leggere il codice di condotta.
*   Assicurarsi di aver firmato l'Accordo di Licenza di Contribuzione (CLA).
*   Verificare se vostri cambiamenti sono consistenti con le linee guida.
*   Verificare se i vostri cambiamenti sono consistenti con lo stile di codifica TensorFlow.
*   Eseguire gli unit test.
