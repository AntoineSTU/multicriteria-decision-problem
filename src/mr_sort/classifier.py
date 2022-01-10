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
        self, border: List[int], poids: List[int], lam: float,
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param borders: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère l'acceptation de l'entrée
        :return: None
        """
        self.nb_grades = len(border)
        self.border = border
        self.poids = poids
        self.lam = lam

    def classify(self, data: List[List[int]]) -> Dict[int, List[List[int]]]:
        """
        To classify elements according to the parameters
        :param data: les ensembles de notes à classer
        :return: les ensembles de notes classés
        """
        results = ((data >= self.border) * self.poids).sum(axis=1) > self.lam
        accepted = data[results, :]
        rejected = data[~results, :]
        return {"accepted": accepted, "rejected": rejected}
