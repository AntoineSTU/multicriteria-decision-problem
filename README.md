# Projet Systèmes de décision

## Participants

Le groupe est composé de:

- Romain DUCRET
- Nicolas LAUBE
- Antoine STUTZ

## Sujet

L'objectif de ce cours est de construire des solveurs pour deux problèmes.

### Inv-MR-Sort

Le premier problème est intitulé MR-Sort:

> ### **MR-Sort**
>
> **Étant donné:**
>
> - un ensemble de classes : $C^0 ... C^p$
> - un ensemble de de matières : $m_1 ... m_n$
> - une note maximale $N$ (les notes allant de 0 à $N$)
> - un ensemble de de frontières (notes limites) : $\forall h\in \llbracket 1,h \rrbracket, b^h\in [0..N]^n$
> - un ensemble de poids : $w_0 ... w_n \in \mathbb{R}^+$
> - une constante : $\lambda \in [0, 1]$
>
> **Sous les contraintes:**
>
> - $\forall h < h'\in \llbracket 1,p\rrbracket , \forall i \in \llbracket 1,n\rrbracket , b^h_i \leq b^{h'}_i$,
> - $\sum_{i=1}^{n} w_i = 1$
>
> **On définit:**
>
> $u\in [0,N]^n$ est dans la classe $C^h$ ssi $\sum_{\{i\in \llbracket 0,N\rrbracket /u_i \geq b^h_i\}} w_i \geq > \lambda$ et $\sum_{\{i\in \llbracket 0,N\rrbracket /u_i \geq b^{h+1}_i\}} w_i < \lambda$

Ce problème est résolu linéairement à l'aide du solveur Gurobi.

### Inv-NCS

Le deuxième problème est intitulé NCS:

> ### **NCS**
>
> **Étant donné:**
>
> - un ensemble de classes $C^0 ... C^p$
> - un ensemble de de matières $m_1 ... m_n$
> - une note maximale $N$ (les notes allant de 0 à $N$)
> - un ensemble de de frontières (notes limites) $\forall h\in \llbracket 1,p\rrbracket , b^h\in \llbracket 0,N\rrbracket ^n$,
> - un ensemble de coalitions suffisantes $T$
>
> **Sous les contraintes:**
>
> - $\forall h < h'\in \llbracket 1,p\rrbracket , \forall i \in \llbracket 1,n\rrbracket , b^h_i \leq b^{h'}_i$,
>
> **On définit:**
>
> $u\in \llbracket 0,N\rrbracket ^n$ est dans la classe $C^h$ ssi $\{i\in \llbracket 1,n\rrbracket /u_i \geq b^h_i\}\in T$ et $\{i\in \llbracket 1,n\rrbracket /u_i \geq b^{h+1}_i\}\notin T$

Une variante avec des intervalles pour traiter des données non monotones:

> ### **NCS**
>
> **Étant donné:**
>
> - un ensemble de classes $C^0 ... C^p$
> - un ensemble de de matières $m_1 ... m_n$
> - une note maximale $N$ (les notes allant de 0 à $N$)
> - un ensemble de de frontières max et min (notes limites) $\forall h\in \llbracket 1,p\rrbracket , b^h_{min} \in \llbracket 0,N\rrbracket ^n et \: b^h_{max} \in \llbracket 0,N\rrbracket ^n$,
> - un ensemble de coalitions suffisantes $T$
>
> **Sous les contraintes:**
>
> - $\forall h < h'\in \llbracket 1,p\rrbracket , \forall i \in \llbracket 1,n\rrbracket , [b^{h'}_{i, min}, b^{h'}_{i, max}] \subset [b^h_{i, min}, b^h_{i, max}]$,
>
> **On définit:**
>
> $u\in \llbracket 0,N\rrbracket ^n$ est dans la classe $C^h$ ssi $\{i\in \llbracket 1,n\rrbracket / b^h_{i, min} \leq u_i \leq b^h_{i, max} \}\in T$ et $\{i\in \llbracket 1,n\rrbracket / b^{h+1}_{i, min} \leq u_i \leq b^{h+1}_{i, max} \}\notin T$

Ces problèmes sont résolus à l'aide du solveur SAT gophersat.

**Remarque:** ici on définit les notes comme des entiers, mais tant que les frontières sont définies par des entiers on peut prendre les notes comme des réels quitte à les arrondir à l'entier inférieur.

## Structure de fichiers

```
├── README.md
│
├── src
│   ├── mr_sort        <- Toutes les classes nécessaires pour traiter les problèmes type MR Sort.
│   ├── ncs            <- Toutes les classes nécessaires pour traiter les problèmes type NCS.
│   ├── test           <- Les fichiers de test.
│   └── config.py      <- Les paramètres et constantes utilisés dans le projet.
│
├── .gitignore
│
└── requirements.txt
```

Les fichiers de codes sont construits similairement pour les deux problèmes (dans le dossier `mr_sort` pour le premier problème et le dossier `ncs` pour le second):

- Les fichier `{}_solver.py` implémentent les solveurs, une classe avec une méthode _solve_
- Les fichier `[{}_]generator.py` implémentent les générateurs de données sous forme de classe
- Les fichier `[{}_]classifier.py` implémentent les classifieurs de données, selon les paramètres donnés du problème

Si aucun préfixe n'est spécifié, il s'agit du problème multiclasse standard (ni binaire, ni avec intervalles).

Les fichiers de test `{}_test.py` se trouvent dans le dossier tests. Ils lancent des tests unitaires, sous pytest.

D'autres fichiers nécessaires au solveur peuvent être présents.

## Lancer le code

### Tests

Pour lancer les tests unitaires, il suffit de taper la commande `pytest` dans le cmd.

### Solveur

Pour lancer le solveur, il faut

- Se mettre dans le dossier racine (**projet-systemes-de-decision**)
- Importer la classe **Solver** du fichier correspondant: `from src.**.{}_solver import Solver`
- Initialiser le solver avec les paramètres désirés:

  ```python
  ##################
  ## Pour MR-Sort ##
  ##################

  # Cas binaire rigide
  s = BinarySolver(
    nb_grades:int = ...,
    nb_students:int = ...,
  )

  # Cas binaire relaxé
  s = RelaxedBinarySolver(
    nb_grades:int = ...,
    nb_students:int = ...,
  )

  # Cas multiclasse
  s = MulticlassSolver(
    nb_categories:int = ...,
    nb_grades:int = ...,
    nb_students:int = ...,
  )

  ##############
  ## Pour NCS ##
  ##############

  # Cas rigide
  s = RigidNcsSolver(
    nb_categories:int = ...,
    nb_grades:int = ...,
    max_grade:int = ...,
  )

  # Cas relaxé
  s = RelaxedNcsSolver(
    nb_categories:int = ...,
    nb_grades:int = ...,
    max_grade:int = ...,
  )

  # Cas intervalles relaxé
  s = RelaxedIntervalNcsSolver(
    nb_categories:int = ...,
    nb_grades:int = ...,
    max_grade:int = ...,
  )
  ```

- Lancer le solver sur les données désirées:

  ```python
  # Pour rigid et relaxed binary MR-Sort
  s.solve(accepted, refused)
  # où accepted (resp. refused) est une liste de listes l_u de notes, où chaque l_u correspond aux notes d'un étudiant accepté (resp. refusé)

  # Pour les autres solveurs
  s.solve(experiences)
  # où experiences est sous la forme
  # {{i: liste d'ensembles de notes qui correpondent à la classe i}}, la classe 0 symbolisant l'absence de classe
  ```

- Les résultats sont sous la forme:

  ```python
  # Pour MR-Sort
  {
      "borders": # liste des frontières correspondant aux différentes classes,
      "poids": # liste des poids affectés aux notes validées,
      "lam": # facteur d'acceptation de l'élève dans la catégorie
  }

  # Pour NCS rigide
  {
      "borders": # liste des frontières correspondant aux différentes classes,
      "valid_set": # liste des ensembles de matières acceptés pour entrer dans une catégorie
  }

  # Pour NCS relaxé
  {
      "borders": # liste des frontières correspondant aux différentes classes (sous forme de liste pour NCS basique et de couple de liste pour NCS avec intervalles),
      "valid_set": # liste des ensembles de matières acceptés pour entrer dans une catégorie,
      "discarded_data": # liste des étudiants non pris en compte
  }
  ```

### Generator

Même marche à suivre que pour le solver (la classe à importer s'appelle **Generator**).

Les méthodes sont les suivantes:

- **reset_parameters**: pour réinitialiser la classe avec de nouveaux paramètres
- **get_parameters**: pour récupérer les paramètres
- **random_parameters**: pour réinitialiser la classe avec des paramètres aléatoires
- **generate**: pour générer des données selon les paramètres donnés précédemment

Pour avoir plus d'info sur ces méthodes, il suffit de lancer la commande `help(Generator)` dans l'interpréteur Python.

### Classifier

Même marche à suivre que pour le solver (la classe à importer s'appelle **Generator**).

Les méthodes sont les suivantes:

- **reset_parameters**: pour réinitialiser la classe avec de nouveaux paramètres
- **classify**: pour classifier des données selon les paramètres donnés précédemment
- **classify_one**: pour classifier un ensemblde de notes (renvoie sa classe)

Pour avoir plus d'info sur ces méthodes, il suffit de lancer la commande `help(Classifier)` dans l'interpréteur Python.
