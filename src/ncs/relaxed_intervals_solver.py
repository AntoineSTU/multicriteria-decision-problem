from src.ncs.interval_generator import IntervalGenerator
from typing import Any, Dict, List, Tuple
from itertools import combinations, chain
import subprocess
from src import config


class RelaxedIntervalNcsSolver:
    def __init__(self, nb_categories: int, nb_grades: int, max_grade: int):
        """
        Pour initialiser le solver
        :param nb_categories: le nombre de catégories (mentions). Ne pas compter "not pass" (cas par défaut)
        :param nb_grades: le nombre de notes différentes (matières)
        :param max_grade: la note maximale
        """
        self.max_grade = max_grade
        self.Categories = list(range(1, nb_categories + 1))  # Les mentions
        self.Criteria = list(range(1, nb_grades + 1))  # Les matières
        self.Possible_grades = list(range(max_grade + 1))  # Les notes
        self.Possible_valid = list(
            chain.from_iterable(
                combinations(self.Criteria, r) for r in range(len(self.Criteria) + 1)
            )
        )  # Les validation possibles

    def solve(self, experiences: Dict[int, Any]) -> Dict[str, Any]:
        """
        Pour trouver les frontières et ensembles de validation
        :param experiences: toutes les expériences dans les différentes classes. Sous la forme de dictionnaire {i: list d'ensembles de notes qui correpondent à la classe i}
        :return: les frontières et les ensembles de validation trouvés par le programme
        """

        ################################
        ### Définition des variables ###
        ################################

        vars_x = [
            ("x", (i, h, k))
            for i in self.Criteria
            for h in self.Categories
            for k in self.Possible_grades
        ]
        vars_y = [("y", tuple(sorted(B))) for B in self.Possible_valid]
        vars_z = [
            ("z", (h, n_u))
            for h in self.Categories + [0]
            for n_u, _ in enumerate(experiences[h])
        ]

        v2i = {v: i + 1 for i, v in enumerate(vars_x)}  # numérotation qui commence à 1
        v2i.update({v: i + len(vars_x) + 1 for i, v in enumerate(vars_y)})
        v2i.update({v: i + len(vars_x) + len(vars_y) + 1 for i, v in enumerate(vars_z)})
        i2v = {i: v for v, i in v2i.items()}

        ##############################
        ### Définition des clauses ###
        ##############################

        # Clause 1
        clause_1 = [
            [v2i["x", (i, h, kp)], -v2i["x", (i, h, k)], -v2i["x", (i, h, kpp)]]
            for i in self.Criteria
            for h in self.Categories
            for k in self.Possible_grades
            for kp in self.Possible_grades
            for kpp in self.Possible_grades
            if kpp > kp > k
        ]

        # Clause 2
        clause_2 = [
            [v2i["x", (i, h, k)], -v2i["x", (i, hp, k)]]
            for i in self.Criteria
            for h in self.Categories
            for hp in self.Categories
            for k in self.Possible_grades
            if hp > h
        ]

        # Clause 3
        clause_3 = [
            [v2i["y", tuple(sorted(Bp))], -v2i["y", tuple(sorted(B))]]
            for Bp in self.Possible_valid
            for B in chain.from_iterable(
                combinations(Bp, r) for r in range(len(Bp) + 1)
            )
        ]

        # Clause 4
        clause_4 = [
            [-v2i["x", (i, h, u[i - 1])] for i in B]
            + [-v2i["y", tuple(sorted(B))]]
            + [-v2i["z", (h - 1, n_u)]]
            for B in self.Possible_valid
            for h in self.Categories
            for n_u, u in enumerate(experiences[h - 1])
        ]

        # Clause 5
        clause_5 = [
            [v2i["x", (i, h, a[i - 1])] for i in B]
            + [v2i["y", tuple(sorted([i for i in self.Criteria if i not in B]))]]
            + [-v2i["z", (h, n_a)]]
            for B in self.Possible_valid
            for h in self.Categories
            for n_a, a in enumerate(experiences[h])
        ]

        # Goals
        goals = [
            [v2i["z", (h, n_u)]]
            for h in self.Categories + [0]
            for n_u, _ in enumerate(experiences[h])
        ]

        ######################################
        ### Solve the problem using Dimacs ###
        ######################################

        all_clauses = clause_1 + clause_2 + clause_3 + clause_4 + clause_5
        nb_var = len(vars_x) + len(vars_y) + len(vars_z)
        dimacs = self.__clauses_to_dimacs(all_clauses, goals, nb_var)

        self.__write_dimacs_file(dimacs, config.DIMACS_WORKINGFILE_PATH_RELAXED)
        result = self.__exec_gophersat(config.DIMACS_WORKINGFILE_PATH_RELAXED)
        return self.__format_res(result, i2v)

    def __format_res(
        self, res: Tuple[bool, List[int]], i2v: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Pour formater les résultats
        :param result: les résultats de gophersat
        :param i2v: pour matcher un numéro avec une variable
        :result: les variables associées avec leur valeur booléenne
        """
        res_vars = {i2v[abs(v)]: v > 0 for v in res[1] if v != 0}
        border = [
            ([self.max_grade for _ in self.Criteria], [0 for _ in self.Criteria])
            for _ in self.Categories
        ]
        for var, var_val in res_vars.items():
            if var[0] == "x" and var_val:
                # Rappel: x sous la forme ("x", (critère, catégorie, note))
                border[var[1][1] - 1][0][var[1][0] - 1] = min(
                    border[var[1][1] - 1][0][var[1][0] - 1], var[1][2]
                )
                border[var[1][1] - 1][1][var[1][0] - 1] = max(
                    border[var[1][1] - 1][1][var[1][0] - 1], var[1][2]
                )
        valid_set = []
        for var, var_val in res_vars.items():
            if var[0] == "y" and var_val:
                valid_set.append(var[1])
        discarded_data = []
        for var, var_val in res_vars.items():
            if var[0] == "z" and not var_val:
                h, n_u = var[1]
                discarded_data.append((h, n_u))
        return {
            "borders": border,
            "valid_set": valid_set,
            "discarded_data": discarded_data,
        }

    @staticmethod
    def __clauses_to_dimacs(
        clauses: List[List[int]], goals: List[List[int]], numvar: int
    ) -> str:
        """
        Pour créer le fichier Dimacs à sauvegarder
        :param clauses: les clauses sous forme normale conjonctive
        :param goals: les buts sous forme normale conjonctive
        :param numvar: le nombre de variables
        :param top: le poids des hard clauses
        :return: la transcription en format dimacs
        """
        top = str(len(goals) + 1)
        dimacs = (
            "c This is it\np wcnf "
            + str(numvar)
            + " "
            + str(len(clauses))
            + " "
            + top
            + "\n"
        )
        for clause in clauses:
            dimacs += top + " "
            for atom in clause:
                dimacs += str(atom) + " "
            dimacs += "0\n"
        for goal in goals:
            dimacs += "1 "
            for atom in goal:
                dimacs += str(atom) + " "
            dimacs += "0\n"
        return dimacs

    @staticmethod
    def __write_dimacs_file(dimacs: str, filename: str) -> None:
        """
        Pour sauvegarder le fichier Dimacs
        :param dimacs: le problème en format dimacs
        :param filename: où enregistrer le fichier dimacs
        :return: None
        """
        with open(filename, "w", newline="") as cnf:
            cnf.write(dimacs)

    @staticmethod
    def __exec_gophersat(
        filename: str, encoding: str = "utf-8"
    ) -> Tuple[bool, List[int]]:
        """
        Pour exécuter Gophersat sur le fichier Dimacs
        :param filename: où le fichier dimacs se trouve
        :param encoding: l'encoding du fichier
        :return: si le problème est resoluble et les valeurs booléennes correspondant aux variables
        """
        cmd = config.GOPHERSAT_PATH

        result = subprocess.run(
            [cmd, "--verbose", filename],
            stdout=subprocess.PIPE,
            check=True,
            encoding=encoding,
        )
        string = str(result.stdout)
        lines = string.splitlines()

        nb_clauses_unsatisfied = int(lines[-3].split(" ")[1])
        model = lines[-1][2:].replace("x", "").split(" ")

        return (
            nb_clauses_unsatisfied,
            [int(x) for x in model if x != "" and int(x) != 0],
        )


if __name__ == "__main__":
    g = IntervalGenerator()
    gen_params = g.get_parameters()
    s = RelaxedIntervalNcsSolver(
        nb_categories=1,
        nb_grades=gen_params["nb_grades"],
        max_grade=gen_params["max_grade"],
    )

    # Génération des données d'entraînement et résolution
    data = g.generate(200)
    solver_params = s.solve(data)
    print(solver_params)
