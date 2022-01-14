from typing import Dict, List, Optional
import pytest
from src.mr_sort.generator import Generator
from src.mr_sort.classifier import Classifier
from src.mr_sort.binary_solver import BinarySolver
from src.mr_sort.relaxed_binary_solver import RelaxedBinarySolver
from pprint import pprint

# def compare_params(real_params, comp_params, ecart: float = 0.1):
#     """
#     Compare les paramètres réels et estimés
#     :param real_params: les paramètres à partir desquels ont été générées les données
#     :param comp_params: les paramètres calculées par le solver
#     :param ecart: écart relatif pour la validation des résultats
#     :return: None
#     """
#     assert (
#         real_params["border"] - comp_params["border"] < 0.1 * real_params["border"]
#     ).all()
#     assert (
#         real_params["poids"] - comp_params["poids"] < 0.1 * real_params["poids"]
#     ).all()
#     assert real_params["lam"] - comp_params["lam"] < 0.1 * real_params["lam"]


def eval_solver(
    gen_params, solver_params, ecart: float = 0.5, noise: Optional[float] = None
):
    """
    Compare les résultats réels et estimés
    :param real_results: les données générées
    :param comp_results: les données calculées par le solver
    :return: None
    """
    # create new data
    generator = Generator()
    generator.set_parameters(**gen_params)
    eval_data = generator.generate(100, noise=noise)
    true = eval_data["classified"]

    classifier_computed = Classifier(**solver_params)
    pred = classifier_computed.classify(eval_data["raw"])

    true_accepted = [
        sorted([mark.round(2) for mark in student_marks])
        for student_marks in true["accepted"]
    ]
    pred_accepted = [
        sorted([mark.round(2) for mark in student_marks])
        for student_marks in pred["accepted"]
    ]

    true_refused = [
        sorted([mark.round(2) for mark in student_marks])
        for student_marks in true["rejected"]
    ]
    pred_refused = [
        sorted([mark.round(2) for mark in student_marks])
        for student_marks in pred["rejected"]
    ]

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    for accepted_student in pred_accepted:
        if accepted_student in true_accepted:
            tp += 1
        else:
            fp += 1
    for refused_student in pred_refused:
        if refused_student in true_refused:
            tn += 1
        else:
            fn += 1

    error = (fp + fn) / (tp + fp + tn + fn)

    pprint(solver_params)
    pprint(gen_params)

    assert error <= ecart


def test_basic():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    generator = Generator()
    generator.set_parameters()
    gen_params = generator.get_parameters()

    gen_data = generator.generate(100)
    refused = gen_data["classified"]["rejected"]
    accepted = gen_data["classified"]["accepted"]

    solver = BinarySolver(
        nb_courses=gen_params["nb_grades"], nb_students=len(accepted) + len(refused),
    )

    # Génération des données d'entraînement et résolution
    data = g.generate(100)
    params_returned = s.solve(data)

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

    gen_data = generator.generate(100, noise=0.1)
    refused = gen_data["classified"]["rejected"]
    accepted = gen_data["classified"]["accepted"]

    solver = RelaxedBinarySolver(
        nb_courses=gen_params["nb_grades"], nb_students=len(accepted) + len(refused),
    )

    # Génération des données d'entraînement et résolution
    data = g.generate(100, noise_var=0.1)
    params_returned = s.solve(data)

    # Génération des données de test et test
    eval_solver(gen_params=gen_params, solver_params=solver_params, noise=0.1)


def test_rd_params():
    """
    Paramètres random
    """
    generator = Generator()
    for _ in range(2):
        # Création des objets
        generator = Generator()
        generator.set_rd_params()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100)
        refused = gen_data["classified"]["rejected"]
        accepted = gen_data["classified"]["accepted"]

        solver = RelaxedBinarySolver(
            nb_courses=gen_params["nb_grades"],
            nb_students=len(accepted) + len(refused),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(accepted, refused)

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
        generator.set_rd_params()
        gen_params = generator.get_parameters()

        gen_data = generator.generate(100, noise=0.1)
        refused = gen_data["classified"]["rejected"]
        accepted = gen_data["classified"]["accepted"]

        solver = RelaxedBinarySolver(
            nb_courses=gen_params["nb_grades"],
            nb_students=len(accepted) + len(refused),
        )

        # Génération des données d'entraînement et résolution
        solver_params = solver.solve(accepted, refused)

        # Génération des données de test et test
        eval_solver(gen_params=gen_params, solver_params=solver_params, noise=0.1)
