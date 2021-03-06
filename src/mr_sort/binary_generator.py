from typing import List, Optional
import numpy as np
import random as rd
from math import floor
from src.mr_sort.binary_classifier import BinaryClassifier


class BinaryGenerator:
    def __init__(self, **kwargs):
        """
        Pour initialiser le générateur
        Appelle reset_parameter
        """
        self.nb_grades = None
        self.max_grade = None
        self.border = None
        self.poids = None
        self.lam = None
        self.classifier = None
        self.set_parameters(**kwargs)

    def set_parameters(
        self,
        max_grade: float = 20,
        border: List[int] = [12, 12, 12, 12, 12],
        poids: List[int] = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        lam: float = 0.6,
    ):
        """
        Pour redéfinir les paramètres de génération du modèle
        :param nb_grades: le nombre de paramètres (notes)
        :param max_grade: la note maximale à générer
        :param border: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère d'acceptation de l'entrée
        :return: None
        """
        self.nb_grades = len(border)
        self.max_grade = max_grade
        self.border = border
        self.poids = poids
        self.lam = lam
        self.classifier = BinaryClassifier(
            border=self.border, poids=self.poids, lam=self.lam
        )

    def random_parameters(self):
        """
        Génère des paramètres de génération des données aléatoires
        :return: None
        """
        self.nb_grades = rd.randint(3, 10)
        self.max_grade = rd.randint(5, 100)
        self.border = [
            floor(rd.random() * (self.max_grade + 1)) for _ in range(self.nb_grades)
        ]
        poids = [rd.random() for _ in range(self.nb_grades)]
        total = sum(poids)
        self.poids = [p / total for p in poids]
        self.lam = rd.random()
        self.classifier = BinaryClassifier(
            border=self.border, poids=self.poids, lam=self.lam
        )

    def get_parameters(self):
        """
        Renvoie les paramètres de génération des données
        :return: {"nb_grades": le nombre de paramètres (notes), "max_grade": la note maximale à générer, "border": les notes limites pour être évaluées positivement, "poids": les poids associés aux différentes notes, "lam": le critère l'acceptation de l'entrée}
        """
        return {
            "nb_grades": self.nb_grades,
            "max_grade": self.max_grade,
            "border": np.array(self.border),
            "poids": np.array(self.poids),
            "lam": self.lam,
        }

    def generate(self, nb_data: int, noise: Optional[float] = None):
        """
        Pour générer nb_data nouvelles données, avec du bruit
        :param nb_data: nombre de données à générer
        :param noise_var: la variance du bruit blanc à ajouter aux données
        :param raw: si les données brutes doivent être renvoyées ou non
        :return: les données sous la forme {"accepted": np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]), "rejected" : np.array([[1, 5, 6, 7, 8]])}
        """
        data = np.random.rand(nb_data, self.nb_grades) * self.max_grade
        results = self.classifier.classify(data)
        accepted = results["accepted"]
        rejected = results["rejected"]
        if noise is not None:
            accepted = accepted + np.random.normal(0, noise, accepted.shape)
            rejected = rejected + np.random.normal(0, noise, rejected.shape)
        classified = {"accepted": accepted, "rejected": rejected}

        return classified
