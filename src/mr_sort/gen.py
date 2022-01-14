from typing import Any, List, Dict, Optional
import numpy as np
import random as rd
from src.mr_sort.classifier import Classifier
from nptyping import NDArray


class Generator:
    def __init__(
        self,
        nb_categories: int = 2,
        nb_grades: int = 5,
        max_grade: float = 20,
        borders: List[List[int]] = [[12, 12, 12, 12, 12]],
        weights: List[int] = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
        lam: float = 0.6,
    ) -> None:
        """
        Initialize the generator
        """
        assert nb_categories >= 2, "Number of categories should be greater than 2."
        assert nb_grades >= 0, "There should at least one grade."
        assert max_grade > 0, "Grades should be positive."
        assert (
            len(borders) == nb_categories - 1 and len(borders[0]) == nb_grades - 1
        ), "Shape of border should be (nb_categories - 1, nb_grades - 1)"
        assert sum(weights) - 1 < 1e-7, "Sum of weights should be equal to 1"
        assert (
            1 >= lam >= 0.5
        ), "Lambda should be equal greater than 0.5 and lower than 1."

        self.nb_categories = nb_categories
        self.nb_grades = nb_grades
        self.max_grade = max_grade
        self.borders = borders
        self.weights = weights
        self.lam = lam

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

    def set_rd_params(self) -> None:
        """
        Génère des paramètres de génération des données aléatoires
        :return: None
        """
        self.nb_grades = rd.randint(3, 10)
        self.max_grade = rd.randint(5, 100)
        nb_categories = rd.randint(1, 5)
        borders = []
        before = [self.max_grade for _ in range(self.nb_grades)]
        for _ in range(nb_categories):
            before = [rd.random() * before[i] for i in range(self.nb_grades)]
            borders.append(before)

        self.borders = borders
        weights = [rd.random() for _ in range(self.nb_grades)]
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.lam = rd.random()

    def generate(
        self, nb_data: int, noise_var: Optional[float] = None, raw: bool = False
    ) -> Dict[int, List[List[int]]]:
        """
        Pour générer nb_data nouvelles données, avec du bruit
        :param nb_data: nombre de données à générer
        :param noise_var: la variance du bruit blanc à ajouter aux données
        :param raw: si les données brutes doivent être renvoyées ou non
        :return: les données sous la forme {"accepted": np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]), "rejected" : np.array([[1, 5, 6, 7, 8]])}
        """

        students_grades = np.random.rand(nb_data, self.nb_grades) * self.max_grade

        classified_students = self.classify(students_grades)

        if noise_var is not None:
            classified = {
                k: [x + np.random.normal(0, noise_var) for x in v]
                for k, v in classified_students.items()
            }

        if raw:
            return {"raw": students_grades, "classified": classified_students}
        return classified

    def __classify(
        self, students_grades: NDArray[(Any, Any)]
    ) -> Dict[int, NDArray[(Any, Any)]]:
        classified_students = {}

        for category in range(self.nb_categories):

            students_in_category = []
            for i, row in enumerate(students_grades >= self.borders[category - 1]):
                tot_sum = sum([self.weights[j] for j in list(np.where(row)[0])])
                if tot_sum >= self.lam:
                    students_in_category.append(i)
            mask = np.zeros(students_grades.shape[0], dtype=bool)
            mask[students_in_category] = True
            classified_students[category] = students_grades[mask, :]
            students_grades = students_grades[~mask, :]
        classified_students[0] = students_grades

        return classified_students
