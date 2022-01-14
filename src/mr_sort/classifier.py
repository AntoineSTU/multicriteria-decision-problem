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
        borders: List[List[int]],
        poids: List[int],
        lam: float,
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param borders: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère l'acceptation de l'entrée
        :return: None
        """
        self.nb_categories = len(borders)
        self.nb_grades = len(borders[0])
        self.borders = borders
        self.poids = poids
        self.lam = lam

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
                tot_sum = sum([self.poids[j] for j in list(np.where(row)[0])])
                if tot_sum >= self.lam:
                    res_k.append(i)
            mask = np.zeros(data.shape[0], dtype=bool)
            mask[res_k] = True
            results[k] = data[mask, :]
            data = data[~mask, :]
        results[0] = data
        return results
