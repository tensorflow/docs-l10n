# Contribuire alla documentazione TensorFlow

TensorFlow accoglie contributi alla documentazione—se migliorate la 
documentazione, migliorate la libreria TensorFlow in se. La documentazione
tensorflow.org ricade nelle seguenti categorie:

* *API reference* —La [Documentazione di riferimento sulle API](https://www.tensorflow.org/api_docs/)
  è generata da docstring nel
  [Codice sorgente TensorFlow](https://github.com/tensorflow/tensorflow).
* *Documentazione discorsiva* —Questi sono [tutorial](https://www.tensorflow.org/tutorials),
  [guide](https://www.tensorflow.org/guide), ed altri scritti che non sono parti
  del codice TensorFlow. Questa documentazione si trova nel
  repository GitHub [tensorflow/docs](https://github.com/tensorflow/docs).
* *Traduzioni della comunità* —Queste sono guide e tutorial tradotte dalla
  comunità. Tutte le traduzioni della comunità si trovano nel
  repository [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site).

Alcuni [progetti TensorFlow](https://github.com/tensorflow) tengono i file sorgenti 
della documentazione vicino al codice in un repository separato, di solito in una 
directory `docs/`. Vedere il file del progetto `CONTRIBUTING.md` o contattare il 
responsabile della manutenzione per contribuire.

Per partecipare alla comunità docs di TensorFlow:

* Guarda il repository GitHub [tensorflow/docs](https://github.com/tensorflow/docs) GitHub.
* Aderisci a [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs).

## API reference

Per modificare la documentazione di riferimento, trovare il 
[file sorgente](https://www.tensorflow.org/code/tensorflow/python/)
e cambiare le 
<a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a> dei simboli.
Molte pagine di riferimento per le API in tensorflow.org sono collegate al codice sorgente
ove è definito un simbolo. I docstring supportano il
<a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a>
e possono essere visti (approssimativamente) in anteprima usando un qualsiasi
<a href="http://tmpvar.com/markdown.html" class="external">Markdown previewer</a>.

Per la qualità della documentazione di riferimento e come essere coinvolti nella comunità
degli sprint di Docs, vedere gli
[avvisi di Docs per le API TensorFlow 2](https://docs.google.com/document/d/1e20k9CuaZ_-hp25-sSd8E8qldxKPKQR-SkwojYr_r-U/preview).

### Versioni e rami

La versione del sito [API reference](https://www.tensorflow.org/api_docs/python/tf)
punta, per default, all'ultima versione stabile del codice binario—che corrisponde al pacchetto
installato con `pip install tensorflow`.

Il pacchetto di default di TensorFlow è compilato dal ramo stabile `rX.x` del repository principale
<a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>. 
La documentazione di riferimento è generata dai commenti del codice
e dalle docstring nel codice sorgente per
<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>,
<a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>, e
<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>.

Versioni precedenti della documentazione TensorFlow sono disponibili come
nel repository Docs di tensorFlow come [rami rX.x](https://github.com/tensorflow/docs/branches).
Questi rami vengono aggiunti ogni volta che viene rilasciata una nuova versione.

### Compilare la documentazione sulle API

Nota: Questo passaggio non è richiesto per modificare o vedere in anteprima le docstring delle API,
 ma solo per generare l'HTML usato su tensorflow.org.

#### Python reference

Il pacchetto `tensorflow_docs` include il generatore per la
[documentazione di riferimento per le PI Python](https://www.tensorflow.org/api_docs/python/tf).
Per installarla usare:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

Per generare la documentazione di riferimento TensorFlow 2, usare lo script:
`tensorflow/tools/docs/generate2.py`

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

Nota: Questo script usa il pacchetto TensorFlow *installato* per generare i documenti e
funziona solo per TensorFlow 2.x.


## Documentazione discorsiva

Le [guide](https://www.tensorflow.org/guide) ed i
[tutorial](https://www.tensorflow.org/tutorials) TensorFlow sono scritti come file
<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a>
e notebook
<a href="https://jupyter.org/" class="external">Jupyter</a> interattivi. I notebook
possono essere eseguiti nel vostro browser usando
<a href="https://colab.research.google.com/notebooks/welcome.ipynb"
   class="external">Google Colaboratory</a>.
I documenti discorsivi su [tensorflow.org](https://www.tensorflow.org) sono compilati
dal ramo `master` 
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>. 
Le versioni più vecchie sono disponibili in GitHub sui rami dei rilasci `rX.x`.

### Modifiche semplici

Il modo più facile per fare aggiornamenti diretti e correzioni è usare l'
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">editor di file di GitHub via web</a>.
Navigate il repository [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)
per localizzare il file Markdown o il notebook che corrisponde sommariamente alla 
struttura dell' URL <a href="https://www.tensorflow.org">tensorflow.org</a>. Nell'angolo
in alto a destra della pagina, cliccate sull'icona della matita
<svg version="1.1" width="14" height="16" viewBox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path></svg>
per aprire l'editor di file. Modificate il file, e fate una nuova richiesta di pull.

### Inizializzare un repository Git locale

Per modifiche a più file o aggiornamenti più complessi, è meglio lavorare su un Git locale
per creare una richiesta di pull.

Nota: <a href="https://git-scm.com/" class="external">Git</a> è un sistema di controllo delle versioni
(VCS) a codice aperto, usato per tracciare le modifiche al codice sorgente.
<a href="https://github.com" class="external">GitHub</a> è un servizio online
che fornisce strumenti collaborativi per lavorare con Git. Vedere l'
<a href="https://help.github.com" class="external">Aiuto GitHub</a> per inizializzare
il vostro account GitHub e iniziare.

I passi seguenti su Git sono necessari solo la prima volta che inizializzate il vostro progetto locale.

#### Fare un fork del repository tensorflow/docs

Per creare la vostra copia del repository, all'interno del vostro account GitHub,
cliccate il pulsante *Fork* <svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path></svg>
sulla pagina GitHub
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>. 
Una volta eseguito il fork, voi siete responsabili di tenere aggiornata la vostra copia 
rispetto al repository TensorFlow originale.

#### Fare un clone del vostro repository

Scaricate una copia del *vostro* repository di documentazione <var>username</var>/docs remoto, sulla vostra macchina
locale. Questa è la directory di lavoro dove farete i cambiamenti:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:<var>username</var>/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### Aggiungere un repository origine (upstream n.d.t.) per rimanere aggiornati (opzionale)

Per mantenere sincronizzato il vostro repository locale con `tensorflow/docs`, aggiungete un' *origine*
remota per scaricare gli aggiornamenti più recenti.

Nota: Accertatevi di aggiornare il vostro repository locale *prima* di iniziare un contributo.
Sincronizzazioni regolari con l'origine riducono la probabilità di un 
<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">conflitto di merge</a>
quando inoltrate la vostra richiesta di pull.

Aggiungere un repository remoto:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git remote add <var>upstream</var> git@github.com:tensorflow/docs.git</code>

# View remote repos
<code class="devsite-terminal">git remote -v</code>
origin    git@github.com:<var>username</var>/docs.git (fetch)
origin    git@github.com:<var>username</var>/docs.git (push)
<var>upstream</var>  git@github.com:tensorflow/docs.git (fetch)
<var>upstream</var>  git@github.com:tensorflow/docs.git (push)
</pre>

Aggiornare:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout master</code>
<code class="devsite-terminal">git pull <var>upstream</var> master</code>

<code class="devsite-terminal">git push</code>  # Push changes to your GitHub account (defaults to origin)
</pre>

### Modalità di lavoro su GitHub

#### 1. Creare un nuovo ramo

Dopo che avete aggiornato il vostro repository da `tensorflow/docs`, create un nuovo ramo dal 
rano *master* locale:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout -b <var>feature-name</var></code>

<code class="devsite-terminal">git branch</code>  # List local branches
  master
* <var>feature-name</var>
</pre>

#### 2. Apportare i cambiamenti

Modificate i file con il vostro editor preferito e seguite le
[guide di stile TensorFlow](./docs_style.md).

Fate commit delle vostre modifiche:

<pre class="prettyprint lang-bsh">
# View changes
<code class="devsite-terminal">git status</code>  # See which files have changed
<code class="devsite-terminal">git diff</code>    # See changes within files

<code class="devsite-terminal">git add <var>path/to/file.md</var></code>
<code class="devsite-terminal">git commit -m "Your meaningful commit message for the change."</code>
</pre>

Se necessario, aggiungete altri commit.

#### 3. Create una richiesta di pull

Caricate il vostro ramo locale sul vostro repository GitHub remoto
(github.com/<var>username</var>/docs):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

Dopo aver completato il push, un messaggio potrebbe presentare un URL per 
sottomettere automaticamente una richiesta di pull al repository origine. Se ciò non accade,
andate sul repository <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
—o sul vostro repository—e GitHub vi proporrà di creare una nuova richiesta di pull.

#### 4. Revisioni

I manutentori ed altri contributori revisioneranno la vostra richiesta di pull.
Partecipate alla discussione e fate i cambiamenti che vi vengono richiesti. quando la vostra
richiesta di pull sarà approvata, sarà inserita nel repository della documentazione TensorFlow originale.

Successo: I vostri cambiamenti sono stati accettati nella documentazione TensorFlow.

Per aggiornare il repository 
[tensorflow.org](https://www.tensorflow.org) da quello GitHub c'è un passo di pubblicazione diverso.
Di solito, i cambiamenti sono accorpati, ed il sito è aggiornato con cadenza regolare.

## Notebook interattivi

Non è raccomandato modificare il file JSON dei notebook  
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">l'editor web</a> 
di GitHub, perché, benché sia possibile, il JSON malformato può danneggiare il file stesso. 
Quindi, provate un notebook, prima di sottomettere una richiesta di pull.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>
è un ambiente di esecuzione di notebook che rende facile modificare ed eseguire notebook di documentazione.
I notebook, in GitHub, sono caricati in Google Colab passando il loro percorso
all'URL di Colab, per esempio, il notebook che in GitHub si trova qui:
<a href="https&#58;//github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https&#58;//github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br/>
può essere caricato in Google Colab con questo URL:
<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

C'è un'estensione di Chrome 
<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Apri in Colab</a>
che realizza questa sostituzione di URL quando sta navigando un notebook in
GitHub. Ciò è utile quando state aprendo un notebook nel vostro fork di repository, perché 
il bottone in alto punte sempre al ramo `master` di TensorFlow Docs.

### Modificare in Colab

All'interno dell'ambiente Google Colab, fate doppio click sulle celle per modificare il testo
ed i blocchi di codice. Le celle di testo usano Markdown e dovrebbero seguire le
[linee guida di stile di TensorFlow](./docs_style.md).

Scaricate i file notebook da Colab con *File > Download .pynb*. Fate commit
nel vostro [repository Git locale](##set_up_a_local_git_repo) e mandate una richiesta di pull.

Per creare un nuovo notebook, copiate e modificate il
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class"> template di notebook TensorFlow</a>.

### Modalità di lavoro Colab-GitHub

Invece di scaricare un file notebook ed usare Git locale, potete
modificare ed aggiornare la vostra copia di repository GitHub direttamente con Google Colab:

1. Nel vostro repository, ottenuto tramite fork, in <var>username</var>/docs, usate l'interfaccia web di GitHub per
   <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">creare un nuovo ramo</a>.
2. Spostatevi sul file notebook per modificarlo.
3. Aprite il notebook in Google Colab: usanfo la conversione di URL o l'estensione Chrome: *Apri in Colab*.
4. Modificate il notebook in Colab.
5. Fate commit dei cambiamenti al vostro repository da Colab con
   *File > Save a copy in GitHub...*, ove la finestra di dialogo di salvataggio dovrebbe collegarsi
   al ramo del repository appropriato, ed aggiungete un messaggio di commit significativo.
6. Dopo il salvataggio, andate al vostro repository o sul repository
   <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>,
   e GitHub dovrebbe chiedervi di creare la richiesta di pull.
7. La richiesta di pull è revisionata dai manutentori.

Successo: I vostri cambiamenti sono stati accettati nella documentazione TensorFlow.


## Community translations

Community translations are a great way to make TensorFlow accessible all over
the world. To update a translation, find or add a file in the
[language directory](https://github.com/tensorflow/docs/tree/master/site) that
matches the same directory structure of the `en/` directory. The English docs
are the *source-of-truth* and translations should follow these guides as close
as possible. That said, translations are written for the communities they serve.
If the English terminology, phrasing, style, or tone does not translate to
another language, please use a translation appropriate for the reader.

Note: The API reference is *not* translated for tensorflow.org.

There are language-specific docs groups that make it easier for translation
contributors to organize. Please join if you're an author, reviewer, or just
interested in building out TensorFlow.org content for the community:

* Chinese (Simplified): [docs-zh-cn@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)
* Italian: [docs-it@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-it)
* Japanese: [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)
* Korean: [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
* Russian: [docs-ru@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru)
* Turkish: [docs-tr@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-tr)

### Review notifications

All documentation updates require a review. To collaborate more efficiently with
the TensorFlow translation communities, here are some ways to keep on top of
language-specific activity:

* Join a language group listed above to receive an email for any *created* pull
  request that touches the <code><a
  href="https://github.com/tensorflow/docs/tree/master/site">site/<var>lang</var></a></code>
  directory for that language.
* Add your GitHub username to the `site/<lang>/REVIEWERS` file to get
  automatically comment-tagged in a pull request. When comment-tagged, GitHub
  will send you notifications for all changes and discussion in that pull
  request.

### Keep code up-to-date in translations

For an open source project like TensorFlow, keeping documentation up-to-date is
challenging. After talking with the community, readers of translated content
will tolerate text that is a little out-of-date, but out-of-date code is
frustrating. To make it easier to keep the code in sync, use the
[nb-code-sync](https://github.com/tensorflow/docs/blob/master/tools/nb_code_sync.py)
tool for the translated notebooks:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">./tools/nb_code_sync.py [--lang=en] site/<var>lang</var>/notebook.ipynb</code>
</pre>

This script reads the code cells of a language notebook and check it against the
English version. After stripping the comments, it compares the code blocks and
updates the language notebook if they are different. This tool is particularly
useful with an interactive git workflow to selectively add hunks of the file to
the commit using: `git add --patch site/lang/notebook.ipynb`
