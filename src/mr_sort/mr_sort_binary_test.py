from typing import Dict, Any
import pytest
from src.mr_sort.binary_generator import BinaryGenerator
from src.mr_sort.binary_classifier import BinaryClassifier
from src.mr_sort.binary_solver import BinarySolver
from src.mr_sort.relaxed_binary_solver import RelaxedBinarySolver


def eval_solver(
    gen_params: Dict[str, Any], solver_params: Dict[str, Any], ecart: float = 0.5
):
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :param noise: le bruit à ajouter sur les données générées
    :return: None
    """
    # Generate data
    g = BinaryGenerator(
        max_grade=gen_params["max_grade"],
        border=gen_params["border"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )
    data_true_classified = g.generate(20)
    classifier_solver = BinaryClassifier(
        border=solver_params["border"],
        poids=solver_params["poids"],
        lam=solver_params["lam"],
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
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    generator = BinaryGenerator()
    generator.set_parameters()
    gen_params = generator.get_parameters()

    gen_data = generator.generate(100)
    refused = gen_data["rejected"]
    accepted = gen_data["accepted"]

    solver = BinarySolver(
        nb_courses=gen_params["nb_grades"], nb_students=len(accepted) + len(refused),
    )

    # Génération des données d'entraînement et résolution
    solver_params = solver.solve(accepted, refused)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params, ecart=0.2)


def test_noisy():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    generator = BinaryGenerator()
    generator.set_parameters()
    gen_params = generator.get_parameters()

    gen_data = generator.generate(100, noise=0.1)
    refused = gen_data["rejected"]
    accepted = gen_data["accepted"]

    solver = RelaxedBinarySolver(
        nb_courses=gen_params["nb_grades"], nb_students=len(accepted) + len(refused),
    )

    # Génération des données d'entraînement et résolution
    solver_params = solver.solve(accepted, refused)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params)


def test_rd_params():
    """
    Paramètres random
    """
    generator = BinaryGenerator()
    for _ in range(2):
        # Création des objets
        generator = BinaryGenerator()
        generator.random_parameters()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100)
        refused = gen_data["rejected"]
        accepted = gen_data["accepted"]

        solver = RelaxedBinarySolver(
            nb_courses=gen_params["nb_grades"],
            nb_students=len(accepted) + len(refused),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(accepted, refused)

        # Génération des données de test et test
        eval_solver(gen_params=gen_params, solver_params=solver_params)


def test_noisy_rd_params():
    """
    Paramètres random
    """
    for _ in range(2):
        # Création des objets
        generator = BinaryGenerator()
        generator.random_parameters()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100, noise=0.1)
        refused = gen_data["rejected"]
        accepted = gen_data["accepted"]

        solver = RelaxedBinarySolver(
            nb_courses=gen_params["nb_grades"],
            nb_students=len(accepted) + len(refused),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(accepted, refused)

        # Génération des données de test et test
        eval_solver(gen_params=gen_params, solver_params=solver_params)
