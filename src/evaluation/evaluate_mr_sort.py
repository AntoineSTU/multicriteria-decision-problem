from tqdm import tqdm
from typing import Dict, Any

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import time
import pandas as pd
from src.evaluation.evaluate_ncs import confusion_matrix_ncs

from src.mr_sort.binary_generator import BinaryGenerator
from src.mr_sort.generator import Generator
from src.mr_sort.binary_classifier import BinaryClassifier
from src.mr_sort.classifier import Classifier
from src.mr_sort.generator import Generator
from src.mr_sort.multiclass_solver import MulticlassSolver

plt.rcParams.update({"font.size": 28})


COLORS = ["red", "green", "blue", "orange", "black", "purple", "yellow"]


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
            pred_classes.append(classifier_solver.classify_one(grades_real))
            true_classes.append(category_real)
    return f1_score(true_classes, pred_classes)


def evaluate_multiclass(gen_params: Dict[str, float], solver_params: Dict[str, float]):
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :param noise: le bruit à ajouter sur les données générées
    :return: None
    """
    # Generate data
    generator = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )

    test_data = generator.generate(200)

    solver_classifier = Classifier(
        borders=solver_params["borders"],
        poids=solver_params["poids"],
        lam=solver_params["lam"],
    )

    true_classes = []
    pred_classes = []

    for category_real, grades_set_real in test_data.items():
        for grades_real in grades_set_real:
            true_classes.append(category_real)
            pred_classes.append(solver_classifier.classify_one(grades_real))

    return f1_score(true_classes, pred_classes, average="macro")


def show_multiclass_matrix_confusion(
    gen_params: Dict[str, float], solver_params: Dict[str, float]
):
    """
    Returns the matrix confusion matrix in multiclass mode.
    """
    generator = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        poids=gen_params["poids"],
        lam=gen_params["lam"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = Classifier(
        borders=solver_params["borders"],
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


def compare_on_num_courses():
    """Compares dataset size"""
    num_grades_to_eval = [2, 3, 4, 6]

    _, ax = plt.subplots(1, figsize=(8, 6))

    for i, num_grades in tqdm(
        enumerate(num_grades_to_eval), total=len(num_grades_to_eval)
    ):
        x = []
        y = []

        for num_students in [10, 50, 75, 100, 125, 150, 175, 200]:

            generator = Generator()

            generator.set_parameters(
                max_grade=20,
                borders=[
                    [10 for _ in range(num_grades)],
                    [16 for _ in range(num_grades)],
                ],
                poids=[1 / num_grades for _ in range(num_grades)],
            )

            data = generator.generate(num_students)

            true_params = generator.get_parameters()

            solver = MulticlassSolver(
                nb_grades=num_grades, nb_students=num_students, nb_categories=2
            )

            pred_params = solver.solve(data)

            x.append(num_students)
            y.append(evaluate_multiclass(true_params, pred_params))

        ax.plot(x, y, color=COLORS[i], label=f"{num_grades} grades")

    plt.legend(
        loc="lower right",
        # title="Evoluation of F1-score depending on number of students in train dataset",
        frameon=False,
    )
    plt.show()


def compare_time_num_students():
    """Compares dataset size"""
    num_category = 2
    num_students = 50
    num_courses = 5
    students = [10, 20, 50, 70, 90, 150, 200, 250, 300, 350, 400, 450, 500]
    x = []
    y = []
    for num_students in tqdm(students, total=len(students)):

        start_time = time.time()

        generator = Generator()

        generator.set_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
            poids=[1 / num_courses for _ in range(num_courses)],
        )

        data = generator.generate(num_students)

        solver = MulticlassSolver(
            nb_grades=num_courses, nb_students=num_students, nb_categories=2
        )

        solver.solve(data)

        x.append(num_students)
        y.append(time.time() - start_time)

    plt.xlabel("Number of students")
    plt.ylabel("Time taken (seconds)")

    plt.legend(
        loc="lower right",
        title="Execution time depending on number of students",
        frameon=False,
    )
    plt.plot(x, y)

    plt.show()


def compare_time_num_categories():
    """Compares dataset size"""
    num_categories = [2, 3, 4, 5, 6]
    num_students = 50
    num_grades = 5

    x = []
    y = []

    for num_category in tqdm(num_categories, total=len(num_categories)):

        start_time = time.time()

        generator = Generator()

        generator.set_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_grades)]
                for j in range(num_category)
            ],
            poids=[1 / num_grades for _ in range(num_grades)],
        )

        data = generator.generate(num_students)

        solver = MulticlassSolver(
            nb_grades=num_grades, nb_students=num_students, nb_categories=num_category
        )

        solver.solve(data)

        x.append(num_category)
        y.append(time.time() - start_time)

    plt.xlabel("Number of categories")
    plt.ylabel("Time taken (seconds)")

    plt.plot(x, y, color="red", label=f"{num_category} grades")

    plt.legend(
        loc="upper right",
        title="Execution time depending on number of categories",
        frameon=False,
    )
    plt.show()


def compare_time_num_grades():
    """Compares dataset size"""
    num_category = 3
    num_students = 50
    num_grades = 5
    ggra = [2, 3, 4, 5, 6, 7]

    x = []
    y = []

    for num_grades in tqdm(ggra, total=len(ggra)):

        start_time = time.time()

        generator = Generator()

        generator.set_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_grades)]
                for j in range(num_category)
            ],
            poids=[1 / num_grades for _ in range(num_grades)],
        )

        data = generator.generate(num_students)

        solver = MulticlassSolver(
            nb_grades=num_grades, nb_students=num_students, nb_categories=num_category
        )

        solver.solve(data)

        x.append(num_grades)
        y.append(time.time() - start_time)

    plt.xlabel("Number of grades")
    plt.ylabel("Time taken (seconds)")

    plt.plot(x, y, color="red", label=f"{num_category} grades")

    plt.legend(
        loc="upper right",
        title="Execution time depending on number of grades",
        frameon=False,
    )
    plt.show()


def compare_on_num_categories():
    """Compares dataset size"""
    num_categories = [2]

    num_grades = 10

    _, ax = plt.subplots(1, figsize=(8, 6))

    for i, num_category in tqdm(enumerate(num_categories), total=len(num_categories)):
        x = []
        y = []

        for num_students in tqdm([100]):

            generator = Generator()

            generator.set_parameters(
                max_grade=20,
                borders=[
                    [j * 20 / num_category for _ in range(num_grades)]
                    for j in range(num_category)
                ],
                poids=[1 / num_grades for _ in range(num_grades)],
            )

            data = generator.generate(num_students, noise_var=2)

            true_params = generator.get_parameters()

            solver = MulticlassSolver(
                nb_grades=num_grades,
                nb_students=num_students,
                nb_categories=num_category,
            )

            pred_params = solver.solve(data)

            x.append(num_students)
            y.append(evaluate_multiclass(true_params, pred_params))

        ax.plot(x, y, color=COLORS[i], label=f"{num_category} number of categories")

    plt.legend(
        loc="lower right",
        # title="Evaluation of F1-score depending on number of classes in train dataset",
        frameon=False,
    )
    plt.show()


def confusion_matrix_mr_sort():
    """Compares dataset size"""
    num_grades = 5
    num_students = 200
    num_categories = 1

    start = time.time()

    generator = Generator()

    generator.set_parameters(
        max_grade=20,
        borders=[
            [12 / num_categories for _ in range(num_grades)]
            # for j in range(num_categories)
        ],
        poids=[1.0 / num_grades for _ in range(num_grades)],
    )
    true_params = generator.get_parameters()

    data = generator.generate(
        num_students,
    )

    solver = MulticlassSolver(
        nb_grades=num_grades, nb_categories=num_categories, nb_students=num_students
    )

    pred_params = solver.solve(data)
    print("f1 ", evaluate_multiclass(true_params, pred_params))
    print("time ", time.time() - start)

    show_multiclass_matrix_confusion(true_params, pred_params)


if __name__ == "__main__":

    # compare_time_num_categories()
    # compare_time_num_grades()

    # compare_on_num_courses()
    # confusion_matrix_mr_sort()
    compare_time_num_categories()

    # confusion_matrix_mr_sort() # ok
