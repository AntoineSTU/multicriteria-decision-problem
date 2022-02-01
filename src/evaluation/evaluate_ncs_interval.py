from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

from src.ncs.solver_relaxed_interval import RelaxedIntervalNcsSolver
from src.ncs.generator_interval import IntervalGenerator
from src.ncs.classifier_interval import IntervalClassifier

plt.rcParams.update({"font.size": 28})


def evaluate_ncs(gen_params: Dict[str, Any], solver_params: Dict[str, Any]) -> None:
    """
    Compare les résultats réels et estimés
    :param gen_params: les paramètres de génération
    :param comp_results: les paramètres du solver
    :param ecart: l'accuracy à vérifier sur les résultats
    :return: None
    """
    # Generate data
    generator = IntervalGenerator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        valid_set=gen_params["valid_set"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = IntervalClassifier(
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
    generator = IntervalGenerator(
        max_grade=gen_params["max_grade"],
        borders=gen_params["borders"],
        valid_set=gen_params["valid_set"],
    )
    data_true_classified = generator.generate(200)
    classifier_solver = IntervalClassifier(
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


def compare_time_max_grade():
    """Compares dataset size"""
    num_category = 1
    num_courses = 5
    max_grades = [20 + i * 10 for i in range(12)]
    num_students = 200
    x = []
    y = []
    for max_grade in tqdm(max_grades, total=len(max_grades)):

        start_time = time.time()

        generator = IntervalGenerator()

        generator.reset_parameters(
            max_grade=max_grade,
        )

        data = generator.generate(num_students)

        solver = RelaxedIntervalNcsSolver(
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

    generator = IntervalGenerator()

    generator.random_parameters()

    while (
        generator.get_parameters()["nb_categories"] != num_categories
        or generator.get_parameters()["nb_grades"] != num_grades
    ):
        generator.random_parameters()

    true_params = generator.get_parameters()

    data = generator.generate(num_students)

    solver = RelaxedIntervalNcsSolver(
        nb_grades=num_grades, nb_categories=num_categories, max_grade=20
    )

    pred_params = solver.solve(data)

    confusion_matrix_ncs(true_params, pred_params)

    print("f1 ", evaluate_ncs(true_params, pred_params))
    print(time.time() - start)


if __name__ == "__main__":
    # compare_time_max_grade()
    get_matrix_conf()
