from typing import Dict, Any
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

from src.ncs.classifier import Classifier
from src.ncs.generator import Generator
from src.ncs.solver import NcsSolver

plt.rcParams.update({"font.size": 28})


COLORS = ["red", "green", "blue", "orange", "black", "purple", "yellow"]


def evaluate_ncs(gen_params: Dict[str, Any], solver_params: Dict[str, Any]) -> None:
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :return: None
    """
    # Generate data
    generator = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        valid_set=gen_params["valid_set"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = Classifier(
        borders=solver_params["borders"], valid_set=solver_params["valid_set"]
    )
    pred_classes = []
    true_classes = []
    for category_real, grades_set_real in data_true_classified.items():
        for grades_real in grades_set_real:
            pred_classes.append(classifier_solver.classify_one(grades_real))
            true_classes.append(category_real)

    return f1_score(pred_classes, true_classes, average="macro")


def confusion_matrix_ncs(
    gen_params: Dict[str, Any], solver_params: Dict[str, Any]
) -> None:
    # Generate data
    generator = Generator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        valid_set=gen_params["valid_set"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = Classifier(
        borders=solver_params["borders"], valid_set=solver_params["valid_set"]
    )
    pred_classes = []
    true_classes = []
    for category_real, grades_set_real in data_true_classified.items():
        for grades_real in grades_set_real:
            pred_classes.append(classifier_solver.classify_one(grades_real))
            true_classes.append(category_real)

    result = confusion_matrix(pred_classes, true_classes)
    sns.heatmap(result, annot=True, cmap="Blues")
    plt.show()


def compare_on_num_categories():
    """Compares dataset size"""
    num_categories = [2, 3, 4, 5, 6, 7, 8]

    num_grades = 10

    _, ax = plt.subplots(1, figsize=(8, 6))

    for i, num_category in tqdm(enumerate(num_categories), total=len(num_categories)):
        x = []
        y = []

        for num_students in tqdm(
            [10, 25, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250]
        ):

            generator = Generator()

            generator.reset_parameters(
                max_grade=20,
                borders=[
                    [j * 20 / num_category for _ in range(num_grades)]
                    for j in range(num_category)
                ],
            )

            data = generator.generate(num_students)

            true_params = generator.get_parameters()

            solver = NcsSolver(
                nb_grades=num_grades, nb_categories=num_category, max_grade=20
            )

            pred_params = solver.solve(data)

            x.append(num_students)
            y.append(evaluate_ncs(true_params, pred_params))

        ax.plot(x, y, color=COLORS[i], label=f"{num_category} number of categories")

    plt.legend(
        loc="lower right",
        # title="Evaluation of F1-score depending on number of classes in train dataset",
        frameon=False,
    )
    plt.show()


def compare_on_num_courses():
    """Compares dataset size"""
    num_grades_to_eval = [3, 4, 6, 8, 10]

    _, ax = plt.subplots(1, figsize=(8, 6))

    for i, num_grades in tqdm(
        enumerate(num_grades_to_eval), total=len(num_grades_to_eval)
    ):
        x = []
        y = []

        for num_students in [10, 25, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250]:

            generator = Generator()

            generator.reset_parameters(
                max_grade=20,
                borders=[[j * 20 / 2 for _ in range(num_grades)] for j in range(2)],
            )

            data = generator.generate(num_students)

            true_params = generator.get_parameters()

            solver = NcsSolver(nb_grades=num_grades, nb_categories=2, max_grade=20)

            pred_params = solver.solve(data)

            x.append(num_students)
            y.append(evaluate_ncs(true_params, pred_params))

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
    students = [i for i in range(500) if i > 10]
    x = []
    y = []
    for num_students in tqdm(students, total=len(students)):

        start_time = time.time()

        generator = Generator()

        generator.reset_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
        )

        data = generator.generate(num_students)

        solver = NcsSolver(nb_grades=num_courses, nb_categories=2, max_grade=20)

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


def compare_time_lambda():
    """Compares dataset size"""
    num_category = 2
    num_students = 200
    num_courses = 5
    students = [i for i in range(500) if i > 10]
    x = []
    y = []
    for i in range(1, 10):
        lam = max(0.5, i / 10)

        start_time = time.time()

        generator = Generator()

        generator.reset_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
            lam=lam,
        )

        data = generator.generate(num_students)

        solver = NcsSolver(nb_grades=num_courses, nb_categories=2, max_grade=20)

        solver.solve(data)

        x.append(lam)
        y.append(time.time() - start_time)

    plt.xlabel("Value of lambda")
    plt.ylabel("Time taken (seconds)")

    plt.legend(
        loc="lower right",
        title="Execution time depending on lambda parameter",
        frameon=False,
    )
    plt.plot(x, y)

    plt.show()


def compare_time_num_categories():
    """Compares dataset size"""
    num_category = 2
    num_courses = 5
    num_students = 200
    num_categories = [i for i in range(100) if i >= 2]
    x = []
    y = []
    for num_category in tqdm(num_categories, total=len(num_categories)):

        start_time = time.time()

        generator = Generator()

        generator.reset_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
        )

        data = generator.generate(num_students)

        solver = NcsSolver(
            nb_grades=num_courses, nb_categories=num_category, max_grade=20
        )

        solver.solve(data)

        x.append(num_category)
        y.append(time.time() - start_time)

    plt.xlabel("Number of categories")
    plt.ylabel("Time taken (seconds)")

    plt.legend(
        loc="lower right",
        title="Execution time depending on number of categories",
        frameon=False,
    )
    plt.plot(x, y)

    plt.show()


def compare_time_num_grades():
    """Compares dataset size"""
    num_category = 2
    num_courses = 5
    num_grades = [i for i in range(13) if i > 0]
    num_students = 200
    num_categories = [i for i in range(50) if i >= 2]
    x = []
    y = []
    for num_courses in tqdm(num_grades, total=len(num_grades)):

        start_time = time.time()

        generator = Generator()

        generator.reset_parameters(
            max_grade=20,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
        )

        data = generator.generate(num_students)

        solver = NcsSolver(
            nb_grades=num_courses, nb_categories=num_category, max_grade=20
        )

        solver.solve(data)

        x.append(num_courses)
        y.append(time.time() - start_time)

    plt.xlabel("Number of grades")
    plt.ylabel("Time taken (seconds)")

    plt.legend(
        loc="upper left",
        title="Execution time depending on number of grades",
        frameon=False,
    )
    plt.plot(x, y)

    plt.show()


def compare_time_max_grade():
    """Compares dataset size"""
    num_category = 2
    num_courses = 5
    max_grades = [20 + i * 10 for i in range(30)]
    num_students = 200
    x = []
    y = []
    for max_grade in tqdm(max_grades, total=len(max_grades)):

        start_time = time.time()

        generator = Generator()

        generator.reset_parameters(
            max_grade=max_grade,
            borders=[
                [j * 20 / num_category for _ in range(num_courses)]
                for j in range(num_category)
            ],
        )

        data = generator.generate(num_students)

        solver = NcsSolver(
            nb_grades=num_courses, nb_categories=num_category, max_grade=max_grade
        )

        solver.solve(data)

        x.append(max_grade)
        y.append(time.time() - start_time)

    plt.xlabel("Maximum grade")
    plt.ylabel("Time taken (seconds)")

    plt.legend(
        loc="upper left",
        title="Execution time depending on maximum grade.",
        frameon=False,
    )
    plt.plot(x, y)

    plt.show()


def get_matrix_conf():
    """Compares dataset size"""
    num_grades = 5
    num_students = 200
    num_categories = 4

    start = time.time()

    generator = Generator()

    generator.reset_parameters(
        max_grade=20,
        borders=[
            [j * 20 / num_categories for _ in range(num_grades)]
            for j in range(num_categories)
        ],
    )
    true_params = generator.get_parameters()

    data = generator.generate(num_students)

    solver = NcsSolver(nb_grades=num_grades, nb_categories=num_categories, max_grade=20)

    pred_params = solver.solve(data)
    print("f1 ", evaluate_ncs(true_params, pred_params))
    print("time ", time.time() - start)

    confusion_matrix_ncs(true_params, pred_params)


if __name__ == "__main__":

    # compare_on_num_categories()
    # compare_on_num_courses()
    # compare_time_num_students()
    # compare_time_num_categories()
    # compare_time_num_grades()
    # get_matrix_conf()
    # compare_time_num_categories()
    compare_time_lambda()
