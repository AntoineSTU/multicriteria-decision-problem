from typing import Dict
import pytest
from src.mr_sort.generator import Generator
from src.mr_sort.classifier import Classifier
from src.mr_sort.multiclass_solver import MulticlassSolver


def eval_solver(gen_params, solver_params, ecart: float = 0.2):
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :param noise: le bruit à ajouter sur les données générées
    :return: None
    """
    # Generate data
    g = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )
    data_true_classified = g.generate(20)
    classifier_solver = Classifier(
        borders=solver_params["borders"],
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
    generator = Generator()
    generator.set_parameters()
    gen_params = generator.get_parameters()

    gen_data = generator.generate(100)
    solver = MulticlassSolver(
        nb_categories=gen_params["nb_categories"],
        nb_grades=gen_params["nb_grades"],
        nb_students=sum([len(l) for l in gen_data.values()]),
    )

    # Génération des données d'entraînement et résolution
    data = generator.generate(100)
    solver_params = solver.solve(data)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params)


def test_noisy():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    generator = Generator()
    generator.set_parameters()
    gen_params = generator.get_parameters()

    gen_data = generator.generate(100, noise_var=0.1)

    solver = MulticlassSolver(
        nb_categories=gen_params["nb_categories"],
        nb_grades=gen_params["nb_grades"],
        nb_students=sum([len(l) for l in gen_data.values()]),
    )

    # Génération des données d'entraînement et résolution
    data = generator.generate(100, noise_var=0.1)
    solver_params = solver.solve(data)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params)


def test_rd_params():
    """
    Paramètres random
    """
    generator = Generator()
    for _ in range(2):
        # Création des objets
        generator = Generator()
        generator.random_parameters()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100)

        solver = MulticlassSolver(
            nb_categories=gen_params["nb_categories"],
            nb_grades=gen_params["nb_grades"],
            nb_students=sum([len(l) for l in gen_data.values()]),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(gen_data)

        # Génération des données de test et test
        eval_solver(
            gen_params=gen_params, solver_params=solver_params,
        )


def test_noisy_rd_params():
    """
    Paramètres random
    """
    for _ in range(2):
        # Création des objets
        generator = Generator()
        generator.random_parameters()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100, noise_var=0.1)

        solver = MulticlassSolver(
            nb_categories=gen_params["nb_categories"],
            nb_grades=gen_params["nb_grades"],
            nb_students=sum([len(l) for l in gen_data.values()]),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(gen_data)

        # Génération des données de test et test
        eval_solver(gen_params=gen_params, solver_params=solver_params)
