from nptyping import NDArray
import gurobipy as gp
import numpy as np


class Solver:
    def __init__(self, nb_courses: int, nb_accepted: int, nb_refused: int) -> None:
        assert nb_courses >= 1
        self.nb_courses = nb_courses
        self.nb_accepted = nb_accepted
        self.nb_refused = nb_refused

        self.model = gp.Model("MR sort")

        self.weights = self.model.addMVar(shape=nb_courses)
        self.weights_s_acc = self.model.addMVar(
            shape=(nb_courses, nb_accepted), vtype=gp.GRB.CONTINUOUS
        )
        self.weights_s_ref = self.model.addMVar(
            shape=(nb_courses, nb_refused), vtype=gp.GRB.CONTINUOUS
        )

        self.boundary = self.model.addMVar(shape=nb_courses)
        self.lamb = self.model.addVar()
        self.delta_acc = self.model.addMVar(
            shape=(nb_courses, nb_accepted), vtype=gp.GRB.BINARY
        )
        self.delta_ref = self.model.addMVar(
            shape=(nb_courses, nb_refused), vtype=gp.GRB.BINARY
        )
        self.model.update()

    def solve(
        self, accepted_students: NDArray[float], refused_students: NDArray[float]
    ):

        lamb_ref = np.stack(np.array([self.lamb] * self.nb_refused), axis=0)

        self.model.addConstr(gp.quicksum(weight for weight in self.weights) == 1)

        for student_i in range(self.nb_accepted):
            self.model.addConstr(
                gp.quicksum(
                    self.weights_s_acc[i, student_i] for i in range(self.nb_courses)
                )
                >= self.lamb,
                name="constraint on accepted students",
            )
        for student_i in range(self.nb_refused):
            self.model.addConstr(
                gp.quicksum(
                    self.weights_s_ref[i, student_i] for i in range(self.nb_courses)
                )
                <= self.lamb,  # < strict is not supported with gurobi
                name="constraint on refused students",
            )

        M = 100

        self.model.addConstrs(
            (
                M * (self.delta_acc[course, student] - 1)
                <= accepted_students[student, course] - self.boundary[course]
            )
            for course in range(self.nb_courses)
            for student in range(self.nb_accepted)
        )

        self.model.addConstrs(
            (
                accepted_students[student, course] - self.boundary[course]
                <= M * self.delta_acc[course, student]
            )
            for course in range(self.nb_courses)
            for student in range(self.nb_accepted)
        )

        self.model.addConstrs(
            (
                M * (self.delta_acc[course, student] - 1)
                <= refused_students[student, course] - self.boundary[course]
            )
            for course in range(self.nb_courses)
            for student in range(self.nb_refused)
        )

        self.model.addConstrs(
            (
                refused_students[student, course] - self.boundary[course]
                <= M * self.delta_acc[course, student]
            )
            for course in range(self.nb_courses)
            for student in range(self.nb_refused)
        )

        self.model.addConstrs(
            (self.weights[course] >= self.weights_s_acc[course, student])
            for course in range(self.nb_courses)
            for student in range(self.nb_accepted)
        )
        self.model.addConstrs(
            (self.weights_s_acc[course, student] >= 0)
            for course in range(self.nb_courses)
            for student in range(self.nb_accepted)
        )
        self.model.addConstrs(
            (self.weights[course] >= self.weights_s_ref[course, student])
            for course in range(self.nb_courses)
            for student in range(self.nb_refused)
        )
        self.model.addConstrs(
            (self.weights_s_ref[course, student] >= 0)
            for course in range(self.nb_courses)
            for student in range(self.nb_refused)
        )
        # self.model.addConstr(weights_ref >= self.weights_s_ref)
        # Goal function
