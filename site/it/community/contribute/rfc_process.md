# Il processo di RFC di TensorFlow

Ogni nuova funzionalità di TensorFlow inizia la sua vita con una Richiesta di Commento (RFC).

Una RFC è un documento che descrive un requisito ed i cambiamenti proposti che lo indirizzeranno.
In particolare, la RFC:

*   Sarà formattata secondo il 
    [template RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
*   Sarà sottoposta come una richiesta di pull (PR n.d.t.) alla directory
    [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).
*   Sarà soggetta a discussione ed ad un incontro di revisione prima dell'accettazione.

La proposta di una Richiesta di Commenti (RFC) per TensorFlow significa ingaggiare la
comunità TensorFlow per lo sviluppo, richiedendo un parere di ritorno a stakeholder ed
esperti, comunicando a largo spettro le modifiche di progettazione.

## Come sottoporre una RFC

1.  Prima di sottoporre una RFC, per ottenere un parere iniziale, discuti i tuoi obiettivi
    con i contributori al progetto e con i manutentori. Usa la lista di distribuzione degli sviluppatori
    per notizie sul progetto (developers@tensorflow.org), o per la lista delle SIG rilevanti.

2.  Bozza della tua RFC.

    *   Segui il
        [template RFC](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md).
    *   Dai un nome al tuo file RFC `YYYYMMDD-nome-descrittivo.md`, dove `YYYYMMDD` è
        la data di sottomissione, e `nome-descrittivo` fa riferimento al titolo della
        tua RFC. (Per esempio, se la tua RFC si intitola _Parallel Widgets API_,
        puoi usare il nome file `20180531-parallel-widgets.md`.
    *   Se hai immagini o altri file ausiliari, crea una directory del tipo
        `YYYYMMDD-nome-descrittivo` in cui mettere tali file.

    Dopo aver scritto la bozza della RFC, ottieni un feedback da manutentori e contributori
    prima di sottometterla.

    Scrivere del codice di implementazione non è un requisito, ma può aiutare gli scambi 
    di progettazione.

3.  Reclutare uno Sponsor.

    *   Uno sponsor deve essere un manutentore del progetto.
    *   Identifica lo sponsor nella RFC, prima di pubblicare la PR.

    _Puoi_ pubblicare una RFC senza uno sponsor, ma se entro un mese dalla pubblicazione
    la PR non avrà ancora uno sponsor, essa sarà chiusa.

4.  Sottoponi la RFC come una richiesta di pull a
    [tensorflow/community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs).

    Includi la tabella di intestazione e il contenuto della sezione _Obiettivo_ nel
    commento della tua richiesta di pull, usando Markdown. Come esempio, vedi
    [questo esempio di RFC](https://github.com/tensorflow/community/pull/5). Includi
    i riferimenti GitHub di co-autori, revisori, e sponsor.

    All'inizio della PR indica quanto sarà lungo il periodo per i commenti.
    _Almeno due settimane_ dalla pubblicazione della PR.

5.  Manda un'email alla lista di distribuzione degli sviluppatori con una breve descrizione,
    un link alla PR ed una richiesta di revisione. Segui il formato delle mail precedenti, come 
    mostrato in 
    [questo esempio](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/PIChGLLnpTE).

6.  Lo sponsor chiederà un incontro del comitato di revisione, non prima di due 
    settimane dalla data di pubblicazione della PR relativa alla RFC. Se c'è una discussione,
    aspetta la conclusione prima di iniziare la revisione. Lo scopo degli incontri di revisione
    è risolvere problemi minori; il consenso su problemi più ampi deve essere raggiunto prima.

7.  l'incontro può approvare la RFC, rifiutarla, o richiedere modifiche prima di 
    prenderla nuovamente in considerazione. Le RFC approvate saranno fatte confluire in
    [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs),
    e le RCF rifiutate vedranno chiudere le relative PR.

## Partecipanti alla RFC

Nel trattamento di una RFC sono coinvolte molte persone:

*   **l'autore della RFC** — uno o più membri della comunità che hanno scritto la RFC e sono
    responsabili di promuoverla durante il processo

*   **lo sponsor della RFC** — un maintainer che sostiene la RFC e la accompagnerà
    durante il processo di revisione

*   **comitato ri revisione** — un gruppo di manutentori che hanno la responsabilità di
    raccomandare l'adozione della RFC

*   Ogni **membro della comunità** può essere d'aiuto fornendo un feedback su qualsiasi aspetto
    della RFC incontri le sue esigenze.

### Sponsor della RFC

Uno sponsor è un manutentore del progetto responsabile dell'assicurazione del miglior risultato
possibile del trattamento della RFC. Ciò include:

*   Sostegno per il progetto proposto.
*   Guida affinché la RFC aderisca a progetti e convenzioni di stile esistenti.
*   Guida del comitato di revisione per raggiungere un consenso proficuo.
*   Se il comitato di revisione chiede delle revisioni, assicurare che queste vengano fatte e
    cercare la successiva approvazione dai membri del comitato.
*   Se la RFC passa all'implementazione:
    *   Assicurare che l'implementazione aderisca alla progettazione.
    *   Coordinarsi con i soggetti appropriati per arrivare con successo all'implementazione.

### Comitati di revisione delle RFC

Il comitato di revisione decide di comune accordo se approvare, respingere o richiedere
modifiche. Essi sono responsabili di:

*   Assicurare che pervengano sostanziosi feedback pubblici.
*   Aggiungere le proprie note durante gli incontri come commenti alla PR.
*   Fornire ragioni per le decisioni.

La costituzione di un comitato di revisione può cambiare a seconda del particolare
stile di governo e leadership di ciascun progetto. Per il core di TensorFlow, il
comitato sarà composto di contributori al progetto TensorFlow che abbiano
esperienza nell'are di dominio di interesse.

### Membri della comunità e trattamento delle RFC

Lo scopo delle RFC è assicurare che la comunità sia be rappresentata e servita dai
cambiamenti a TensorFlow. E' responsabilità dei membri della comunità partecipare 
alla revisione delle RFC abbiano interesse all'argomento.

I membri della comunità interessati ad una RFC dovrebbero:

*   **Fornire feedback** prima possibile per dare tempo sufficiente a che siano
    presi in considerazione.
*   **Leggere le RFC** a fondo prima di fornire feedback.
*   Essere **civili e costruttivi**.

## Implementazione di nuove funzionalità

Una volta che una RFC sia stata approvata, può iniziare l'implementazione.

Se state lavorando a nuovo codice per implementare una RFC:

*   Assicuratevi di aver compreso la funzione e la progettazione approvati nella RFC.
    Fate domande e discutete l'approccio prima di iniziare a lavorare.
*   Le nuove funzioni devono includere nuove unità di test che verifichino che le funzioni
    lavorino come atteso. E' una buona idea scrivere queste unità di test prima di scrivere il codice.
*   Seguite la [Guida di Stile del Codice TensorFlow](#tensorflow-code-style-guide)
*   Aggiungete o modificate la documentazione sulle API di interesse. Nella nuova documentazione
    fate riferimento alla RFC.
*   Seguite ogni altra linea guida presente nel file `CONTRIBUTING.md` che trovate nel repository 
    del progetto a cui state contribuendo.
*   Eseguite il test di unità prima di sottoporre il vostro codice.
*   Lavorate con lo sponsor della RFC per completare il nuovo codice in modo soddisfacente.

## Tenere alta la soglia

Se, da un lato, incoraggiamo e sosteniamo ogni contributo, la soglia per l'accettazione di una  RFC
è intenzionalmente tenuta alta. Una nuova funzionalità può essere rifiutata o richiedere una revisione 
significativa ad ognuno dei seguenti passi:

*   Conversazioni iniziali sulla progettazione sulle mailing list di rilievo.
*   Insuccesso nel trovare uno sponsor.
*   Obiezioni critiche durante la fase di feedback.
*   Fallimento nel raggiungere il consenso durante la revisione della progettazione.
*   Comparsa di problemi durante l'implementazione (per esempio: impossibilità di garantire
    la compatibilità verso il passato, problemi di manutenzione, ecc.).

Se questo processo funziona bene, ci si può aspettare che le RFC possano fallire,
nelle prime fasi anziché nelle ultime. Una RFC approvata non ha comunque la garanzia
di un viatico per l'implementazione, e l'accettazione di una proposta di implementazione
è ancora soggetta al consueto processo di revisione del codice.

Per qualsiasi domanda relativa a questo processo, sentitevi liberi di chiedere alla
mailing list degli sviluppatori o aggiungere un punto in
[tensorflow/community](https://github.com/tensorflow/community/tree/master/rfcs).
