from typing import List, Dict
from itertools import combinations, chain
import numpy as np


class Classifier:
    def __init__(self, **kwargs):
        """
        Pour initialiser le générateur
        Appelle reset_parameter
        """
        return self.reset_parameters(**kwargs)

    def reset_parameters(
        self,
        borders: List[List[int]] = [[12, 12, 12, 12, 12]],
        valid_set: List[List[int]] = [[1, 3], [2, 4], [1, 2, 5]],
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param borders: les notes limites pour être évaluées positivement
        :params valid_set: les ensembles de matières possibles pour valider les catégories
        :return: None
        """
        self.nb_grades = len(borders[0])
        self.nb_categories = len(borders)
        self.borders = borders
        self.valid_set = valid_set
        self.__complete_valid_set()

    def classify(self, data: List[List[int]]) -> Dict[int, List[List[int]]]:
        """
        To classify elements according to the parameters
        :param data: les ensembles de notes à classer
        :return: les ensembles de notes classés
        """
        data = np.array(data)
        results = {}
        for k in range(self.nb_categories, 0, -1):
            res_k = []
            for i, row in enumerate(data >= self.borders[k - 1]):
                valid_grades = tuple([x + 1 for x in list(np.where(row)[0])])
                if valid_grades in self.valid_set:
                    res_k.append(i)
            mask = np.zeros(data.shape[0], dtype=bool)
            mask[res_k] = True
            results[k] = data[mask, :]
            data = data[~mask, :]
        results[0] = data
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
