from typing import List, Optional
import numpy as np
import random as rd
from math import floor
from itertools import combinations, chain
from src.ncs.classifier import Classifier


class Generator:
    def __init__(self, **kwargs):
        """
        Pour initialiser le générateur
        Appelle reset_parameter
        """
        return self.reset_parameters(**kwargs)

    def reset_parameters(
        self,
        max_grade: float = 20,
        borders: List[List[int]] = [[12, 12, 12, 12, 12]],
        valid_set: List[List[int]] = [[1, 3], [2, 4], [1, 2, 5]],
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param nb_grades: le nombre de paramètres (notes)
        :param max_grade: la note maximale à générer
        :param nb_categories: le nombre de catégories que l'on consifère (sans compter la catégorie nulle)
        :param borders: les notes limites pour être évaluées positivement
        :params valid_set: les ensembles de matières possibles pour valider les catégories
        :return: None
        """
        self.nb_grades = len(borders[0])
        self.max_grade = max_grade
        self.nb_categories = len(borders)
        self.borders = borders
        self.valid_set = valid_set
        self.classifier = Classifier(borders=self.borders, valid_set=self.valid_set,)
        self.__complete_valid_set()

    def get_parameters(self):
        """
        Renvoie les paramètres de génération des données
        :return: {"nb_grades": le nombre de paramètres (notes), "max_grade": la note maximale à générer, "borders": les notes limites pour être évaluées positivement, "valid_set": les ensembles de matières pour valider les catégories}
        """
        return {
            "nb_grades": self.nb_grades,
            "max_grade": self.max_grade,
            "nb_categories": self.nb_categories,
            "borders": np.array(self.borders),
            "valid_set": np.array(self.valid_set, dtype=object),
        }

    def random_parameters(self):
        """
        Génère des paramètres de génération des données aléatoires
        :return: None
        """
        nb_grades = rd.randint(3, 5)
        max_grade = rd.randint(5, 100)
        nb_categories = rd.randint(1, 5)
        borders = []
        before = [self.max_grade for _ in range(nb_grades)]
        for _ in range(nb_categories):
            before = [floor(rd.randint(0, before[i])) for i in range(nb_grades)]
            borders.append(before)
        borders.reverse()
        all_combinations = list(
            chain.from_iterable(
                combinations(range(1, nb_grades + 1), r)
                for r in range(1, nb_grades + 1)
            )
        )
        valid_set = rd.sample(all_combinations, rd.randint(1, 4))
        return self.reset_parameters(max_grade, borders, valid_set)

    def generate(self, nb_data: int, noise_var: Optional[float] = None):
        """
        Pour générer nb_data nouvelles données, avec du bruit
        :param nb_data: nombre de données à générer
        :return: les données sous la forme {i: np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])} avec i le numéro de la catégorie (0 étant la catégorie "rejetée")
                 Si raw est true, renvoie {"raw": les données brutes, "classified": les données classifiées}
        """
        to_int_vect = np.vectorize(int)
        data = to_int_vect(
            np.random.rand(nb_data, self.nb_grades) * (self.max_grade + 1)
        )
        results = self.classifier.classify(data)
        if noise_var is not None:
            results = {
                k: [
                    np.clip(
                        np.floor(x + np.random.normal(0, noise_var)).astype(int),
                        0,
                        self.max_grade,
                    )
                    for x in v
                ]
                for k, v in results.items()
            }
        return results

    def __complete_valid_set(self):
        """
        Pour compléter l'ensemble de validation 
        :return: None
        """
        all_sets = []
        for valid_elt in self.valid_set:
            other_elts = [i for i in range(1, self.nb_grades + 1) if i not in valid_elt]
            combi = list(
                chain.from_iterable(
                    combinations(other_elts, r) for r in range(len(other_elts) + 1)
                )
            )
            ext_combi = [list(x) + list(valid_elt) for x in combi]
            all_sets.extend([tuple(sorted(elt)) for elt in ext_combi])
        self.valid_set = list(dict.fromkeys(all_sets))
