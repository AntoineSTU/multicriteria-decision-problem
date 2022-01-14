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

> **Étant donné:**
>
> - un ensemble de classes : $C^0 ... C^p$
> - un ensemble de de matières : $m_1 ... m_n$
> - une note maximale $N$ (les notes allant de 0 à $N$)
> - un ensemble de de frontières (notes limites) : $\forall k\in \llbracket 1,k \rrbracket, b^k\in [0..N]^n$,
> - un ensemble de poids : $w_0 ... w_n \in \mathbb{R}^+$
> - une constante : $\lambda \in [0, 1]$
>
> **Sous les contraintes:**
>
> - $\forall k < k'\in \llbracket 1,p\rrbracket , \forall i \in \llbracket 1,n\rrbracket , b^k_i \leq b^{k'}_i$,
> - $\sum_{i=1}^{n} w_i = 1$
>
> **On définit:**
>
> $x\in [0,N]^n$ est dans la classe $C^k$ ssi $\sum_{\{i\in \llbracket 0,N\rrbracket /x_i \geq b^k_i\}} w_i \geq > \lambda$ et $\sum_{\{i\in \llbracket 0,N\rrbracket /x_i \geq b^{k+1}_i\}} w_i < \lambda$

Ce problème est résolu linéairement à l'aide du solveur Gurobi.

### Inv-NCS

Le deuxième problème est intitulé NCS:

> **Étant donné:**
>
> - un ensemble de classes $C^0 ... C^p$
> - un ensemble de de matières $m_1 ... m_n$
> - une note maximale $N$ (les notes allant de 0 à $N$)
> - un ensemble de de frontières (notes limites) $\forall k\in \llbracket 1,p\rrbracket , b^k\in \llbracket 0,N\rrbracket ^n$,
> - un ensemble de coalitions suffisantes $T$
>
> **Sous les contraintes:**
>
> - $\forall k < k'\in \llbracket 1,p\rrbracket , \forall i \in \llbracket 1,n\rrbracket , b^k_i \leq b^{k'}_i$,
>
> **On définit:**
>
> $x\in \llbracket 0,N\rrbracket ^n$ est dans la classe $C^k$ ssi $\{i\in \llbracket 1,n\rrbracket /x_i \geq b^k_i\}\in T$ et $\{i\in \llbracket 1,n\rrbracket /x_i \geq b^{k+1}_i\}\notin T$

Ce problème est résolu à l'aide du solveur SAT gophersat.

**Remarque:** ici on définit les notes comme des entiers, mais tant que les frontières sont définies par des entiers on peut prendre les notes comme des réels quitte à les arrondir à l'entier inférieur.

## Structure de fichiers

Les fichiers de codes sont construits similairement pour les deux problèmes (dans le dossier `mr_sort` pour le premier problème et le dossier `ncs` pour le second):

- Le fichier `solver.py` implémente le solveur, une classe avec une méthode _solve_
- Le fichier `generator.py` implémente un générateur de données sous forme de classe
- Le fichier `classifier.py` implémente un classifieur de données, selon les paramètres donnés du problème
- Le fichier `{}_test.py` implémente des tests unitaires, sous pytest

D'autres fichiers nécessaires au solveur peuvent être présents.

## Lancer le code

### Tests

Pour lancer les tests unitaires, il suffit de taper la commande `pytest` dans le cmd.

### Solveur

Pour lancer le solveur, il faut

- Se mettre dans le dossier racine (**projet-systemes-de-decision**)
- Importer la classe **Solver** du fichier correspondant: `from src.**.solver import Solver`
- Initialiser le solver avec les paramètres désirés:

  ```python
  # Pour MR-Sort

  # Pour NCS
  s = Solver(
      nb_categories:int = ...,
      nb_grades:int = ...,
      max_grade:int = ...,
  )
  ```

- Lancer le solver sur les données désirées:

  ```python
  # Pour MR-Sort

  # Pour NCS
  s.solve(experiences)
  # où experiences est sous la forme {{i: liste d'ensembles de notes qui correpondent à la classe i}}, la classe 0 symbolisant l'absence de classe
  ```

- Les résultats sont sous la forme:

  ```json
  // Pour MR-Sort

  // Pour NCS
  {
      "borders": list des frontières correspondant aux différentes classes,
      "valid_set": liste des ensembles de matières acceptés pour entrer dans une catégorie
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

Pour avoir plus d'info sur ces méthodes, il suffit de lancer la commande `help(Generator)` dans l'interpréteur Python.
