import pytest
from typing import Any, Dict
from src.ncs.generator import Generator
from src.ncs.classifier import Classifier
from src.ncs.solver import NcsSolver


def eval_solver(
    gen_params: Dict[str, Any], solver_params: Dict[str, Any], ecart: float = 0.2
) -> None:
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :return: None
    """
    # Generate data
    g = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        valid_set=gen_params["valid_set"],
    )
    data_true_classified = g.generate(20)
    classifier_solver = Classifier(
        borders=solver_params["borders"], valid_set=solver_params["valid_set"]
    )
    # Verify data
    nb_class_false = 0
    nb_class_tot = 0
    for category_real, grades_set_real in data_true_classified.items():
        nb_class_tot += len(grades_set_real)
        for grades_real in grades_set_real:
            category_classified = classifier_solver.classify_one(grades_real)
            if category_classified != category_real:
                nb_class_false += 1
    assert nb_class_false / nb_class_tot <= ecart


def test_basic():
    """
    Données simples
    """
    # Création des objets
    g = Generator()
    gen_params = g.get_parameters()
    s = NcsSolver(
        nb_categories=1,
        nb_grades=gen_params["nb_grades"],
        max_grade=gen_params["max_grade"],
    )

    # Génération des données d'entraînement et résolution
    data = g.generate(200)
    solver_params = s.solve(data)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params)


def test_all():
    """
    Paramètres random
    """
    g = Generator()
    for _ in range(10):
        # Création des objets
        g.random_parameters()
        gen_params = g.get_parameters()
        s = NcsSolver(
            nb_categories=gen_params["nb_categories"],
            nb_grades=gen_params["nb_grades"],
            max_grade=gen_params["max_grade"],
        )

        # Génération des données d'entraînement et résolution
        data = g.generate(200)
        solver_params = s.solve(data)

        # Génération des données de test et test
        eval_solver(gen_params=gen_params, solver_params=solver_params)
