from typing import Dict, List
from unicodedata import category
import gurobipy as gp
import logging

from src.mr_sort.generator import Generator


class MulticlassSolver:
    """
    Solver in the case of multiple categories.
    """

    def __init__(
        self, nb_courses: int, nb_students: int, nb_categories: int = 2
    ) -> None:
        """Initialize solver"""
        nb_categories += 1

        assert nb_courses >= 1, nb_students >= 1
        assert nb_categories >= 2

        ####################
        # Useful variables #
        ####################
        self.small = 1e-4
        self.large = 100

        self.nb_categories = nb_categories
        self.nb_courses = nb_courses
        self.nb_students = nb_students

        ####################
        # Initialize model #
        ####################

        self.model = gp.Model("MR sort")

        ####################################
        # Problem representation variables #
        ####################################

        # weights of courses
        self.w_i = self.model.addMVar(shape=(nb_courses,), name="weights (nb_courses)")

        self.lambda_ = self.model.addVar(name="lambda_", lb=0)

        # acceptance criteria
        self.x_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories),
            vtype=gp.GRB.BINARY,
            name="maximizer",
        )

        # continuous delta
        self.c_i_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories, self.nb_courses),
            name="continuous weights (nb_courses, nb_students, nb_categories)",
            lb=0,
        )

        # boudaries between validated/non-validated courses
        self.b_i_h = self.model.addMVar(
            shape=(self.nb_categories - 1, self.nb_courses),
            name="boundaries (nb_courses, nb_categories)",
        )

        # delta
        self.d_i_j_h = self.model.addMVar(
            shape=(self.nb_students, self.nb_categories, self.nb_courses),
            vtype=gp.GRB.BINARY,
            name="deltas (nb_courses, nb_students, nb_categories)",
        )

        self.model.update()

    def solve(self, classified_students: Dict[int, List[List[int]]]):

        self.model.addConstr(gp.quicksum(self.w_i) == 1)
        self.model.addConstr(self.lambda_ <= 1)

        offset = 0

        self.model.addConstrs(
            self.w_i[i] >= self.c_i_j_h[j, h, i]
            for i in range(self.nb_courses)
            for j in range(self.nb_students)
            for h in range(self.nb_categories)
        )

        self.model.addConstrs(
            self.d_i_j_h[j, h, i] >= self.c_i_j_h[j, h, i]
            for i in range(self.nb_courses)
            for j in range(self.nb_students)
            for h in range(self.nb_categories)
        )
        self.model.addConstrs(
            self.c_i_j_h[j, h, i] >= self.d_i_j_h[j, h, i] + self.w_i[i] - 1
            for i in range(self.nb_courses)
            for j in range(self.nb_students)
            for h in range(self.nb_categories)
        )

        for category in range(self.nb_categories):
            for j, student_grades in enumerate(classified_students[category]):
                j = j + offset
                for h in range(self.nb_categories):
                    for i in range(len(student_grades)):

                        if h != 0:
                            self.model.addConstr(
                                self.large * (self.d_i_j_h[j, h, i] - 1)
                                <= student_grades[i] - self.b_i_h[h - 1, i]
                            )
                            self.model.addConstr(
                                student_grades[i] + self.small - self.b_i_h[h - 1, i]
                                <= self.large * self.d_i_j_h[j, h, i]
                            )

                        if h <= category:
                            # student should better
                            self.model.addConstr(
                                gp.quicksum(self.c_i_j_h[j, h])
                                >= self.lambda_ + self.large * (1 - self.x_j_h[j, h])
                            )

                        if h > category:
                            # student should worse
                            self.model.addConstr(
                                gp.quicksum(self.c_i_j_h[j, h])
                                + self.large * self.x_j_h[j, h]
                                + self.small
                                <= self.lambda_
                            )
            offset += len(classified_students[category])

        ###############################
        # Objectif function variables #
        ###############################

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
            "borders": self.b_i_h.X,
            "poids": self.w_i.X,
        }


if __name__ == "__main__":
    import numpy as np

    NB_DATA = 300
    NB_COURSES = 5

    solver = MulticlassSolver(
        nb_courses=NB_COURSES, nb_students=NB_DATA, nb_categories=3
    )

    generator = Generator()
    generator.set_parameters(
        max_grade=20,
        borders=[[10 for i in range(NB_COURSES)], [16 for i in range(NB_COURSES)]],
    )

    data = generator.generate(NB_DATA)

    params = solver.solve(data)

    print(params)
