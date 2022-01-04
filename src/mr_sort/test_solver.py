import pytest
from src.mr_sort.generator import Generator
from src.mr_sort.solver import Solver
import random as rd


def compare_params(real_params, comp_params, ecart: float = 0.1):
    """
    Compare les paramètres réels et estimés
    :param real_params: les paramètres à partir desquels ont été générées les données
    :param comp_params: les paramètres calculées par le solver
    :param ecart: écart relatif pour la validation des résultats
    :return: None
    """
    assert (
        real_params["border"] - comp_params["border"] < 0.1 * real_params["border"]
    ).all()
    assert (
        real_params["poids"] - comp_params["poids"] < 0.1 * real_params["poids"]
    ).all()
    assert real_params["lam"] - comp_params["lam"] < 0.1 * real_params["lam"]


def test_basic():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    generator = Generator()
    parameters = generator.get_parameters()
    data = generator.generate(100)

    solver = Solver(
        nb_courses=parameters["nb_grades"],
        nb_accepted=len(data["accepted"]),
        nb_refused=len(data["rejected"]),
    )

    params_returned = solver.solve(data["accepted"], data["rejected"])
    compare_params(parameters, params_returned)


def test_basic_var_1():
    """
    Données simples avec variance de 0.1 pour le bruit blanc
    """
    generator = Generator()
    parameters = generator.get_parameters()
    data = generator.generate(100, noise_var=0.1)

    solver = Solver(
        nb_courses=parameters["nb_grades"],
        nb_accepted=len(data["accepted"]),
        nb_refused=len(data["rejected"]),
    )

    params_returned = solver.solve(data["accepted"], data["rejected"])
    compare_params(parameters, params_returned)


def test_all():
    """
    Paramètresolver random
    """
    g = Generator()
    for _ in range(10):
        g.random_parameters()
        parameters = g.get_parameters()
        data = g.generate(rd.randint(10, 100))

        s = Solver(
            nb_courses=parameters["nb_grades"],
            nb_accepted=len(data["accepted"]),
            nb_refused=len(data["rejected"]),
        )

        params_returned = s.solve(data["accepted"], data["rejected"])
        compare_params(parameters, params_returned)


def test_all_var1():
    """
    Paramètres random + bruit
    """
    g = Generator()
    for _ in range(10):
        g.random_parameters()
        parameters = g.get_parameters()
        data = g.generate(rd.randint(20, 100), noise_var=0.1)

        s = Solver(
            nb_courses=parameters["nb_grades"],
            nb_accepted=len(data["accepted"]),
            nb_refused=len(data["rejected"]),
        )

        params_returned = s.solve(data["accepted"], data["rejected"])
        compare_params(parameters, params_returned)
