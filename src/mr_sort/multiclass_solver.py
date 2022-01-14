from typing import Dict, List
from unicodedata import category
import gurobipy as gp
import logging

from src.mr_sort.generator import Generator


class MulticlassSolver:
    """
    Solver in the case of two categories.
    Here, the categories are 'Accepted', 'Refused'

    """

    def __init__(
        self, nb_courses: int, nb_students: int, nb_categories: int = 2
    ) -> None:
        """Initialize solver"""

        assert nb_courses >= 1, nb_students >= 1
        assert nb_categories >= 2

        self.nb_categories = nb_categories

        self.nb_courses = nb_courses
        self.nb_students = nb_students

        self.model = gp.Model("MR sort")

        self.w_i = self.model.addMVar(shape=(nb_courses,), name="weights (nb_courses)")

        self.lambda_ = self.model.addVar(name="lambda_", lb=0)

        # x(student, class)

        self.x_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories),
            vtype=gp.GRB.BINARY,
            name="maximizer",
        )

        # w(student, class, lesson)
        self.c_i_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories, self.nb_courses),
            name="continuous weights (nb_courses, nb_students, nb_categories)",
            lb=0,
        )

        # b(class, lesson)
        self.b_i_h = self.model.addMVar(
            shape=(self.nb_categories - 1, self.nb_courses),
            name="boundaries (nb_courses, nb_categories)",
        )

        self.d_i_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories, self.nb_courses),
            vtype=gp.GRB.BINARY,
            name="deltas (nb_courses, nb_students, nb_categories)",
        )

        self.model.update()

        ####################################
        # Problem representation variables #
        ####################################

        # weights of courses

        # boudaries between validated/non-validated courses

        # acceptance criteria

        ###############################
        # Objectif function variables #
        ###############################

        # Alpha is continuous in [0, 1]

        ####################
        # Useful variables #
        ####################
        self.small = 1e-4
        self.large = 100

    def solver(self, classified_students: Dict[int, List[List[int]]]):
        # Create the variables

        # Add constraints
        self.model.addConstr(gp.quicksum(self.w_i) == 1)  # somme des poids = 1
        self.model.addConstr(self.lambda_ <= 1)  # lambda inférieur à 1

        offset = 0

        # Contraintes pour les étudiants acceptés
        for category in range(self.nb_categories):
            for j, student_grades in enumerate(classified_students[category]):
                j = j + offset
                for h in range(self.nb_categories):
                    for i in range(len(student_grades)):
                        # Condition w_i >= w_s_h_i
                        self.model.addConstr(self.w_i[i] >= self.c_i_j_h[j, h, i])

                        # Condition delta_s_h_i >= w_s_h_i >= delta_s_h_i + w_i - 1
                        self.model.addConstr(
                            self.d_i_j_h[j, h, i] >= self.c_i_j_h[j, h, i]
                        )
                        self.model.addConstr(
                            self.c_i_j_h[j, h, i]
                            >= self.d_i_j_h[j, h, i] + self.w_i[i] - 1
                        )

                        if h != 0:
                            # Condition M(delta_s_h_i -1) <= s_i - b_h_i < M delta_s_h_i
                            self.model.addConstr(
                                self.large * (self.d_i_j_h[j, h, i] - 1)
                                <= student_grades[i] - self.b_i_h[h - 1, i]
                            )
                            self.model.addConstr(
                                student_grades[i] - self.b_i_h[h - 1, i]
                                <= self.large * self.d_i_j_h[j, h, i] - self.small
                            )

                        # Student is in all classes below his
                        if h <= category:
                            score = gp.quicksum(self.c_i_j_h[j, h])
                            self.model.addConstr(
                                (score - self.lambda_)
                                - self.large * (1 - self.x_j_h[j, h])
                                >= 0
                            )

                        # Student is in all classes above his
                        if h > category:
                            score = gp.quicksum(self.c_i_j_h[j, h])
                            self.model.addConstr(
                                (score - self.lambda_) + self.large * self.x_j_h[j, h]
                                <= -self.small
                            )
            offset += len(classified_students[category])

        # Objective
        final_opti = gp.quicksum(
            [gp.quicksum(self.x_j_h[s_idx]) for s_idx in range(self.nb_students)]
        )
        self.model.setObjective(final_opti, gp.GRB.MAXIMIZE)

        # Solve
        self.model.optimize()

        if self.model.status == gp.GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise ValueError("Model was proven to be infeasible.")
        if self.model.status == gp.GRB.INF_OR_UNBD:
            raise ValueError("Model was proven to be either infeasible or unbounded.")
        if self.model.status == gp.GRB.UNBOUNDED:
            raise ValueError("Model was proven to be unbounded.")
        if self.model.status != gp.GRB.OPTIMAL:
            logging.warning("Model status code: %s", self.model.status)
            raise ValueError("Model didn't find optimal solution.")

        return {
            "lam": self.lambda_.X,
            "border": self.b_i_h.X,
            "poids": self.w_i.X,
        }

    def solve(self, classified_students: Dict[int, List[List[int]]]):
        """Find the right parameters"""

        # sum of weights should be equal to one
        self.model.addConstr(gp.quicksum(self.w_i) == 1)
        self.model.addConstr(self.lambda_ <= 1)

        offset = 0

        for category, key in enumerate(classified_students):
            students = classified_students[key]

            for j in range(len(students)):

                for h in range(self.nb_categories):
                    for i in range(self.nb_courses):

                        self.model.addConstr(
                            self.c_i_j_h[i, j + offset, h] <= self.w_i[i]
                        )

                        self.model.addConstr(
                            self.c_i_j_h[i, j + offset, h]
                            <= self.d_i_j_h[i, j + offset, h]
                        )

                        self.model.addConstr(
                            self.c_i_j_h[i, j + offset, h]
                            >= self.d_i_j_h[i, j + offset, h] - 1 + self.w_i[i]
                        )

                        if category >= h:

                            # a_j is at least as good as profile h
                            self.model.addConstr(
                                (
                                    gp.quicksum(self.c_i_j_h[:, j + offset, h])
                                    >= self.lambda_
                                    + (1 - self.x_j_h[j + offset, h]) * self.large
                                ),
                                name="better than",
                            )

                        if category < h:

                            # a_j is less good than profile b_h
                            self.model.addConstr(
                                (
                                    gp.quicksum(self.c_i_j_h[:, j + offset, h])
                                    + self.x_j_h[j + offset, h] * self.large
                                    + self.small
                                    <= self.lambda_
                                ),
                                name="less good than",
                            )

                        if h != 0:

                            # d_i_j_l == 1 <=> g_j(a_j) >= b_i_l for l = h-1
                            self.model.addConstr(
                                (
                                    self.large * (self.d_i_j_h[i, j + offset, h] - 1)
                                    <= students[j, i] - self.b_i_h[i, h - 1]
                                )
                            )

                            # d_i_j_l == 0 <=> g_j(a_j) < b_i_l for l = h
                            self.model.addConstr(
                                (
                                    students[j, i] - self.b_i_h[i, h - 1]
                                    <= self.large * self.d_i_j_h[i, j + offset, h]
                                    - self.small
                                )
                            )
            offset += len(students)

        # Goal function
        self.model.setObjective(
            gp.quicksum(
                [
                    self.x_j_h[j, h]
                    for j in range(self.nb_students)
                    for h in range(self.nb_categories)
                ]
            ),
            gp.GRB.MAXIMIZE,
        )

        self.model.optimize()


if __name__ == "__main__":
    import numpy as np

    NB_DATA = 300

    solver = MulticlassSolver(nb_courses=5, nb_students=NB_DATA, nb_categories=3)

    generator = Generator()
    generator.set_parameters(
        max_grade=20,
        borders=[[10 for i in range(5)], [16 for i in range(5)]],
    )

    data = generator.generate(NB_DATA)

    params = solver.solver(data)

    print(params)
