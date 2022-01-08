import pytest
from src.ncs.generator import Generator
from src.ncs.solver import Solver
import random as rd


def compare_params(real_params, comp_params):
    """
    Compare les paramètres réels et estimés
    :param real_params: les paramètres à partir desquels ont été générées les données
    :param comp_params: les paramètres calculées par le solver
    :param ecart: écart relatif pour la validation des résultats
    :return: None
    """
    assert (real_params["border"] == comp_params["border"]).all()
    assert sorted(real_params["valid_set"]) == sorted(comp_params["valid_set"])


def test_basic():
    """
    Données simples
    """
    g = Generator()
    parameters = g.get_parameters()
    s = Solver(
        nb_categories=1,
        nb_grades=parameters["nb_grades"],
        max_grade=parameters["max_grade"],
    )
    data = g.generate(200)

    params_returned = s.solve(data)
    compare_params(parameters, params_returned)


# Remarque : on ne peut pas faire de test randomisé simplement, car plusieurs définitions de frontière et d'ensembles validants peuvent ccorrespondre à la même réalité.
# Exemple : sur 4 notes, mettre une frontière à toutes mais ne devoir valider que la première est équivalent à devoir tout valider et la seule frontière non nulle est la matière à valider

# def test_2():
#     """
#     Paramètres random
#     """
#     g = Generator()
#     for _ in range(10):
#         g.random_parameters()
#         parameters = g.get_parameters()
#         s = Solver(
#             nb_categories=1,
#             nb_grades=parameters["nb_grades"],
#             max_grade=parameters["max_grade"],
#         )
#         data = g.generate(rd.randint(100, 500))

#         params_returned = s.solve(data)
#         compare_params(parameters, params_returned)
