from typing import List, Dict
from itertools import combinations, chain
import numpy as np


class BinaryClassifier:
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
        :param border: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère l'acceptation de l'entrée
        :return: None
        """
        self.nb_grades = len(border)
        self.border = border
        self.poids = poids
        self.lam = lam

    def classify(self, data: List[List[int]]) -> Dict[str, List[List[int]]]:
        """
        To classify elements according to the parameters
        :param data: les ensembles de notes à classer
        :return: les ensembles de notes classés
        """
        results = ((data >= self.border) * self.poids).sum(axis=1) > self.lam
        accepted = data[results, :]
        rejected = data[~results, :]
        return {"accepted": accepted, "rejected": rejected}

    def classify_one(self, grades: List[int]) -> str:
        """
        To classify elements according to the parameters
        :param data: l'ensemble de notes à classer
        :return: la classe correspondante
        """
        result = ((grades >= self.border) * self.poids).sum(axis=0) > self.lam
        return "accepted" if result else "rejected"
