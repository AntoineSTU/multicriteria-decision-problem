from typing import List
import numpy as np
import random as rd
from math import floor
from itertools import combinations, chain


class Generator:
    def __init__(self, **kwargs):
        """
        Pour initialiser le générateur
        Appelle reset_parameter
        """
        return self.reset_parameters(**kwargs)

    def reset_parameters(
        self,
        nb_grades: int = 4,
        max_grade: float = 20,
        border: List[int] = [12, 12, 12, 12],
        valid_set: List[List[int]] = [[1, 3], [2, 4]],
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param nb_grades: le nombre de paramètres (notes)
        :param max_grade: la note maximale à générer
        :param border: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère l'acceptation de l'entrée
        :return: None
        """
        self.nb_grades = nb_grades
        self.max_grade = max_grade
        self.border = border
        self.valid_set = valid_set
        self.__complete_valid_set()

    def get_parameters(self):
        """
        Renvoie les paramètres de génération des données
        :return: {"nb_grades": le nombre de paramètres (notes), "max_grade": la note maximale à générer, "border": les notes limites pour être évaluées positivement, "poids": les poids associés aux différentes notes, "lam": le critère l'acceptation de l'entrée}
        """
        return {
            "nb_grades": self.nb_grades,
            "max_grade": self.max_grade,
            "border": np.array(self.border),
            "valid_set": np.array(self.valid_set),
        }

    def random_parameters(self):
        """
        Génère des paramètres de génération des données aléatoires
        :return: None
        """
        nb_grades = rd.randint(3, 10)
        max_grade = rd.randint(5, 100)
        border = [floor(rd.random() * (max_grade + 1)) for _ in range(nb_grades)]
        all_combinations = list(
            chain.from_iterable(
                combinations(range(1, nb_grades + 1), r)
                for r in range(1, nb_grades + 1)
            )
        )
        valid_set = rd.sample(all_combinations, rd.randint(1, 4))
        return self.reset_parameters(nb_grades, max_grade, border, valid_set)

    def generate(self, nb_data: int):
        """
        Pour générer nb_data nouvelles données, avec du bruit
        :param nb_data: nombre de données à générer
        :param noise_var: la variance du bruit blanc à ajouter aux données
        :return: les données sous la forme {"accepted": np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]), "rejected" : np.array([[1, 5, 6, 7, 8]])}
        """
        to_int_vect = np.vectorize(int)
        data = to_int_vect(
            np.random.rand(nb_data, self.nb_grades) * (self.max_grade + 1)
        )
        results = []
        for i, row in enumerate(data >= self.border):
            valid_grades = tuple([x + 1 for x in list(np.where(row)[0])])
            if valid_grades in self.valid_set:
                results.append(i)
        mask = np.zeros(data.shape[0], dtype=bool)
        mask[results] = True
        accepted = data[mask, :]
        rejected = data[~mask, :]
        return {0: rejected, 1: accepted}

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
