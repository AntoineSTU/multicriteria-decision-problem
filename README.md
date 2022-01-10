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

```
Définition du problème
```

Ce problème est résolu linéairement à l'aide du solveur Gurobi.

### Inv-NCS

Le deuxième problème est intitulé NCS:

```
Définition du problème
```

Ce problème est résolu à l'aide du solveur SAT gophersat.

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
