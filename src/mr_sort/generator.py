from typing import Any, List, Dict, Optional
import numpy as np
import random as rd
from src.mr_sort.classifier import Classifier


class Generator:
    def __init__(self, **kwargs) -> None:
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
        borders: List[List[int]] = [[12, 12, 12, 12, 12]],
        poids: List[int] = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        lam: float = 0.6,
    ) -> None:
        """
        Pour redéfinir les paramètres de génération du modèle
        :param nb_grades: le nombre de paramètres (notes)
        :param max_grade: la note maximale à générer
        :param borders: les notes limites pour être évaluées positivement
        :param poids: les poids associés aux différentes notes
        :param lam: le critère d'acceptation de l'entrée
        :return: None
        """
        self.nb_grades = len(borders[0])
        self.max_grade = max_grade
        self.nb_categories = len(borders)
        self.borders = borders
        self.poids = poids
        self.lam = lam
        self.classifier = Classifier(borders=borders, poids=poids, lam=lam)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Renvoie les paramètres de génération des données
        :return: {"nb_grades": le nombre de paramètres (notes), "max_grade": la note maximale à générer, "nb_categories": le nombre de catégories, "borders": les notes limites pour être évaluées positivement, "poids": les poids associés aux différentes notes, "lam": le critère l'acceptation de l'entrée}
        """
        return {
            "nb_grades": self.nb_grades,
            "max_grade": self.max_grade,
            "nb_categories": self.nb_categories,
            "borders": np.array(self.borders),
            "poids": np.array(self.poids),
            "lam": self.lam,
        }

    def random_parameters(self):
        """
        Génère des paramètres de génération des données aléatoires
        :return: None
        """
        nb_grades = rd.randint(3, 10)
        max_grade = rd.randint(5, 100)
        nb_categories = rd.randint(1, 5)
        borders = []
        before = [max_grade for _ in range(nb_grades)]
        for _ in range(nb_categories):
            before = [rd.random() * before[i] for i in range(nb_grades)]
            borders.append(before)
        poids = [rd.random() for _ in range(nb_grades)]
        total = sum(poids)
        poids = [p / total for p in poids]
        lam = rd.random()
        return self.set_parameters(max_grade, borders, poids, lam)

    def generate(self, nb_data: int, noise_var: float = 0):
        """
        Pour générer nb_data nouvelles données, avec du bruit
        :param nb_data: nombre de données à générer
        :param noise_var: la variance du bruit blanc à ajouter aux données
        :param raw: si les données brutes doivent être renvoyées ou non
        :return: les données sous la forme {"accepted": np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]), "rejected" : np.array([[1, 5, 6, 7, 8]])}
        """
        data = np.random.rand(nb_data, self.nb_grades) * self.max_grade
        results = self.classifier.classify(data)
        classified = {
            k: [x + np.random.normal(0, noise_var) for x in v]
            for k, v in results.items()
        }
        return classified
