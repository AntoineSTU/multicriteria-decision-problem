from numpy import append
from sklearn.metrics import f1_score, confusion_matrix
from typing import Dict, Any
import time
import seaborn as sns
import matplotlib.pyplot as plt

from src.mr_sort.binary_solver import BinarySolver
from src.mr_sort.binary_generator import BinaryGenerator
from src.mr_sort.binary_classifier import BinaryClassifier


def eval_binary(gen_params: Dict[str, Any], solver_params: Dict[str, Any]):
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :param noise: le bruit à ajouter sur les données générées
    :return: None
    """
    # Generate data
    generator = BinaryGenerator(
        max_grade=gen_params["max_grade"],
        border=gen_params["border"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = BinaryClassifier(
        border=solver_params["border"],
        poids=solver_params["poids"],
        lam=solver_params["lam"],
    )
    # Verify data
    true_classes = []
    pred_classes = []

    for category_real, grades_set_real in data_true_classified.items():
        for grades_real in grades_set_real:
            print(classifier_solver.classify_one(grades_real))
            if classifier_solver.classify_one(grades_real) == "accepted":

                pred_classes.append(1)
            else:
                pred_classes.append(0)
            print(category_real)
            if category_real == "accepted":

                true_classes.append(1)
            else:
                true_classes.append(0)
    return f1_score(true_classes, pred_classes)


def show_multiclass_matrix_confusion(
    gen_params: Dict[str, float], solver_params: Dict[str, float]
):
    """
    Returns the matrix confusion matrix in multiclass mode.
    """
    generator = BinaryGenerator(
        max_grade=gen_params["max_grade"],
        border=gen_params["border"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = BinaryClassifier(
        border=solver_params["border"],
        poids=solver_params["poids"],
        lam=solver_params["lam"],
    )
    # Verify data

    pred_classes = []
    true_classes = []

    for category_real, grades_set_real in data_true_classified.items():
        for grades_real in grades_set_real:
            category_classified = classifier_solver.classify_one(grades_real)
            pred_classes.append(category_classified)
            true_classes.append(category_real)

    confusion_mat = confusion_matrix(pred_classes, true_classes)
    ax = sns.heatmap(confusion_mat, annot=True, cmap="Blues")
    plt.show()


def confusion_matrix_mr_sort():
    """Compares dataset size"""
    num_grades = 5
    num_students = 200

    start = time.time()

    generator = BinaryGenerator(
        border=[12, 12, 12, 12, 12],
        poids=[1.0 / num_grades for _ in range(num_grades)],
        lam=0.6,
    )

    true_params = generator.get_parameters()

    data = generator.generate(num_students)

    solver = BinarySolver(nb_grades=num_grades, nb_students=num_students)

    pred_params = solver.solve(
        accepted_j_i=data["accepted"], refused_j_i=data["rejected"]
    )
    print("f1 ", eval_binary(true_params, pred_params))
    print("time ", time.time() - start)

    show_multiclass_matrix_confusion(true_params, pred_params)


if __name__ == "__main__":
    confusion_matrix_mr_sort()
