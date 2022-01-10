from typing import Dict, List
import pytest
from src.mr_sort.generator import Generator
from src.mr_sort.classifier import Classifier
from solver import Solver
import random as rd


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


def compare_results(
    real_results: Dict[int, List[List[int]]],
    comp_results: Dict[int, List[List[int]]],
    ecart: float = 0.1,
):
    """
    Compare les résultats réels et estimés
    :param real_results: les données générées
    :param comp_results: les données calculées par le solver
    :return: None
    """
    nb_invalid = 0
    nb_tot = 0
    real_results = {k: sorted([tuple(x) for x in v]) for k, v in real_results.items()}
    comp_results = {k: sorted([tuple(x) for x in v]) for k, v in comp_results.items()}
    for k, v_real in real_results.items():
        if len(v_real) > 0:
            v_comp = comp_results[k]
            for i, x_real in enumerate(v_real):
                x_comp = v_comp[i]
                if x_real != x_comp:
                    nb_invalid += 1
                nb_tot += 1
    assert nb_invalid / nb_tot <= ecart


def test_basic():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    g = Generator()
    parameters = g.get_parameters()
    s = Solver()

    # Génération des données d'entraînement et résolution
    data = g.generate(100)
    params_returned = s.solve(data["accepted"], data["rejected"])

    # Génération des données de test et test
    res = g.generate(20, raw=True)
    test_true = res["classified"]
    classifier_computed = Classifier(**params_returned)
    test_comp = classifier_computed.classify(res["raw"])
    compare_results(test_true, test_comp)


def test_basic_var_1():
    """
    Données simples avec variance de 0 pour le bruit blanc
    """
    # Création des objets
    g = Generator()
    parameters = g.get_parameters()
    s = Solver()

    # Génération des données d'entraînement et résolution
    data = g.generate(100, noise_var=0.1)
    params_returned = s.solve(data["accepted"], data["rejected"])

    # Génération des données de test et test
    res = g.generate(20, noise_var=0.1, raw=True)
    test_true = res["classified"]
    classifier_computed = Classifier(**params_returned)
    test_comp = classifier_computed.classify(res["raw"])
    compare_results(test_true, test_comp)


def test_all():
    """
    Paramètres random
    """
    g = Generator()
    for _ in range(10):
        # Création des objets
        g.random_parameters()
        parameters = g.get_parameters()
        s = Solver()

        # Génération des données d'entraînement et résolution
        data = g.generate(100)
        params_returned = s.solve(data)

        # Génération des données de test et test
        res = g.generate(20, raw=True)
        test_true = res["classified"]
        classifier_computed = Classifier(**params_returned)
        test_comp = classifier_computed.classify(res["raw"])
        compare_results(test_true, test_comp)


def test_all():
    """
    Paramètres random
    """
    g = Generator()
    for _ in range(10):
        # Création des objets
        g.random_parameters()
        parameters = g.get_parameters()
        s = Solver()

        # Génération des données d'entraînement et résolution
        data = g.generate(100, noise_var=0.1)
        params_returned = s.solve(data)

        # Génération des données de test et test
        res = g.generate(20, noise_var=0.1, raw=True)
        test_true = res["classified"]
        classifier_computed = Classifier(**params_returned)
        test_comp = classifier_computed.classify(res["raw"])
        compare_results(test_true, test_comp)
