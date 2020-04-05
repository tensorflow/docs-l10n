# Contributo al codice TensorFlow

Sia che tu voglia aggiungere una funzione perdita, migliorare la copertura di un test, oppure scrivere una
RFC per un cambiamento rilevante di progettazione, questa parte della guida del contributore ti sarà d'aiuto
per cominciare. Grazie per il lavoro e l'interesse nel miglioramento di TensorFlow.

## Prima di cominciare

Prima di contribuire al codice sorgente di un progetto TensorFlow, cortesemente consulta il file `CONTRIBUTING.md` nel repository GitHub del progetto. (Per esempio, vedi
[file CONTRIBUTING.md nel repository centrale di TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).) A tutti i contributori è richiesto di firmare un [Accordo di Licenza di Contribuzione](https://cla.developers.google.com/clas) (CLA).

Per evitare attività duplicate, prima di cominciare a lavorare su una funzionalità non banale, cortesemente consulta [RFCs correnti](https://github.com/tensorflow/community/tree/master/rfcs) e contatta gli sviluppatori nei forum TensorFlow. Talvolta siamo selettivi quando dobbiamo decidere se aggiungere una funzionalità, ed il modo migliore di contribuire ed aiutare il progetto è lavorare su problemi conosciuti. 

## Temi per il nuovo contributore

I nuovi contributori alla ricerca di un primo tema per un contributo al codice TensorFlow, dovrebbero cercare i seguenti tag. Raccomandiamo fortemente che i nuovi contributori affrontino inizialmente progetti “easy” e problematiche "good first issue"; questo aiuta il contributore a familiarizzarsi con il flusso di lavoro della contribuzione, ed agli sviluppatori del core di conoscere il contributore.

- `good first issue`
- `easy`
- `contributions welcome`

Se siete interessati a reclutare un team per aiutarvi ad affrontare un problema ampio o una nuova funzionalità, cortemente mandate una email a [gruppo sviluppatori@](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers) e osservate la nostra lista attuale di RFC. 


## Revisione del codice

Nuove funzioni, eliminazione di difetti, ed ogni altro cambiamento nel codice base sono soggetti a revisione del codice.

La revisione del codice ricevuto come contributo in un progetto attraverso le "pull request" è una componente fondamentale dello sviluppo di TensorFlow. Incoraggiamo chiunque a cominciare rivedendo codice sottoposto da altri sviluppatori, specialmente in quei casi in cui il codice implementa qualcosa che ti piacerebbe usare.

Qui alcune domande da tenere a mente durante il processo di revisione del codice:

*   *Vogliamo questa cosa in TensorFlow?* Potrebbe piacere usarla? Come utente TensorFlow, ti piacerebbe questa modifica e la useresti? Si tratta di un cambiamento nell'ambito di TensorFlow? Il costo di manutenzione di questa nuova funzione sarebbe maggiore dei suoi benefici?
*   *Il codice è consistente con le API di TensorFlow?* Le funzioni pubbliche, le classi, ed i parametri sono nominati in modo appropriato e progettati intuitivamente?
*   *E' inclusa la documentazione?* Le funzioni pubbliche, le classi, i tipi di ritorno e gli attributi memorizzati sono nominati in accordo con le convenzioni di TensorFlow e sono documentati chiaramente? La nuova funzionalità è descritta nella documentazione di TensorFlow ed illustrata, ovunque possibile, con esempi? La documentazione si legge in modo appropriato?

*   *Il codice è leggibile da una persona?* Si evitano ridondanze? I nomi delle variabili dovrebbero essere migliorati per aumentare la chiarezza o la consistenza? Dovrebbero essere aggiunti commenti? Ci sono commenti inutili o estranei che potrebbero essere rimossi?
*   *Il codice è efficiente?* Potrebbe essere riscritto facilmente per essere più efficiente?
*   IL codice è *retro-compatibile* con le versioni precedenti di TensorFlow?
*   Il nuovo codice aggiungerà *nuove dipendenze* on other libraries?

## Test e miglioramento della copertura dei test

Unit testing di alta qualità è una pietra miliare del processo di sviluppo di TensorFlow. Per questo scopo usiamo immagini Docker. Le funzioni di test hanno nomi appropriati, e sono responsabili del controllo della validità degli algoritmi, così come delle varie opzioni del codice.

Tutt le nuove funzioni e le correzioni di difetti *devono* includere test con adeguata copertura. Perciò accogliamo anche contributi di nuovi casi di test o migliorie ai test esistenti. Se scoprite che i nostri attuali test non sono completi — anche se ciò, al momento, non causa un difetto — per cotesia, sottoponete un issue e, se possibile una pull request.

Per tutti i dettagli particolari sulle procedure di test in ciascun progetto TensorFlow, vedere i file `README.md` e `CONTRIBUTING.md` nel rispettivo repository su GitHub.

Di particolare importanza è fare *testing adeguato*:

*   E stata testata *ogni funzione pubblica di ogni classe*? 
*   Sono stati testati *insiemi ragionevoli di parametri*, di valori, tipi e in ragionevoli combinazioni? 
*   I test sono in grado di validare che *il codice è corretto*, e che esso faccia *ciò che la documentazione dice* che esso faccia?
*   Se la modifica è la correzione di un difetto, è incluso un *test di non regressione*?
*   Il test *supera il build in continuous integration*?
*   Il test *copre tutte le linee del codice?* se no, le eccezioni sono ragionevoli ed esplicite?

Se trovate dei problemi, cortesemente, considerate la possibilità di supportare un contributore a capire questi problemi ed a risolverli. 


## Improve error messages or logs

We welcome contributions that improve error messages and logging. 


## Contribution workflow

Code contributions—bug fixes, new development, test improvement—all follow a GitHub-centered workflow. To participate in TensorFlow development, set up a GitHub account. Then:

1.  Fork the repo you plan to work on.
    Go to the project repo page and use the *Fork* button. This will create a copy of the
    repo, under your username. (For more details on how to fork a repository see
    [this guide](https://help.github.com/articles/fork-a-repo/).)

2.  Clone down the repo to your local system.

    `$ git clone git@github.com:your-user-name/project-name.git`

3.  Create a new branch to hold your work.

    `$ git checkout -b new-branch-name`

4.  Work on your new code. Write and run tests.

5.  Commit your changes.

    `$ git add -A`

    `$ git commit -m "commit message here"`

6.  Push your changes to your GitHub repo.

    `$ git push origin branch-name`

7.  Open a *Pull Request* (PR). Go to the original project repo on GitHub. There will be a message about your recently pushed branch, asking if you would like to open a pull request. Follow the prompts, *compare across repositories*, and submit the PR. This will send an email to the committers. You may want to consider sending an email to the mailing list for more visibility. (For more details, see the [GitHub guide on PRs](https://help.github.com/articles/creating-a-pull-request-from-a-fork). 

8.  Maintainers and other contributors will *review your PR*. Please participate in the conversation, and try to *make any requested changes*. Once the PR is approved, the code will be merged.

*Before working on your next contribution*, make sure your local repository is up to date.

1. Set the upstream remote. (You only have to do this once per project, not every time.)

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. Switch to the local master branch.

    `$ git checkout master`

3. Pull down the changes from upstream.

    `$ git pull upstream master`

4. Push the changes to your GitHub account. (Optional, but a good practice.)

    `$ git push origin master`

5. Create a new branch if you are starting new work.

    `$ git checkout -b branch-name`

Additional `git` and GitHub resources:

*   [Git documentation](https://git-scm.com/documentation)
*   [Git development workflow](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
*   [Resolving merge conflicts](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/).


## Contributor checklist

*   Read contributing guidelines.
*   Read the Code of Conduct.
*   Ensure you have signed the Contributor License Agreement (CLA).
*   Check if your changes are consistent with the guidelines.
*   Check if your changes are consistent with the TensorFlow coding style.
*   Run unit tests.
