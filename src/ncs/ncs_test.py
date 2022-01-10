import pytest
from src.ncs.generator import Generator
from src.ncs.classifier import Classifier
from src.ncs.solver import Solver


def compare_results(real_results, comp_results):
    """
    Compare les résultats réels et estimés
    :param real_results: les données générées
    :param comp_results: les données calculées par le solver
    :return: None
    """
    real_results = {k: sorted([tuple(x) for x in v]) for k, v in real_results.items()}
    comp_results = {k: sorted([tuple(x) for x in v]) for k, v in comp_results.items()}
    for k, v_real in real_results.items():
        if len(v_real) > 0:
            v_comp = comp_results[k]
            for i, x_real in enumerate(v_real):
                x_comp = v_comp[i]
                assert x_real == x_comp


def test_basic():
    """
    Données simples
    """
    # Création des objets
    g = Generator()
    parameters = g.get_parameters()
    s = Solver(
        nb_categories=1,
        nb_grades=parameters["nb_grades"],
        max_grade=parameters["max_grade"],
    )

    # Génération des données d'entraînement et résolution
    data = g.generate(200)
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
        s = Solver(
            nb_categories=parameters["nb_categories"],
            nb_grades=parameters["nb_grades"],
            max_grade=parameters["max_grade"],
        )

        # Génération des données d'entraînement et résolution
        data = g.generate(1000)
        params_returned = s.solve(data)

        # Génération des données de test et test
        res = g.generate(20, raw=True)
        test_true = res["classified"]
        classifier_computed = Classifier(**params_returned)
        test_comp = classifier_computed.classify(res["raw"])
        compare_results(test_true, test_comp)
