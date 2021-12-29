from nptyping import NDArray
import gurobipy
import numpy as np

class MRSort_Solver:
    def __init__(self, nb_courses: int, nb_accepted: int, nb_refused) -> None:
        assert nb_courses >= 1
        self.model = gurobipy.Model("MR sort")

        self.weights = self.model.addMVar(shape=nb_courses)
        self.weights_s_acc = self.model.addMVar(shape=(nb_courses, nb_accepted))
        self.weights_s_ref = self.model.addMVar(shape=(nb_courses, nb_refused))

        self.frontier = self.model.addMVar(shape=nb_courses - 1)
        self.lamb = self.model.addVar()
        self.delta_acc = self.model.addMVar(shape=(nb_courses, nb_accepted), vtype=GRB.BOOL)
        self.delta_ref = self.model.addMVar(shape=(nb_courses, nb_refused), vtype=GRB.BOOL)

        self.model.update()

    def solve(self, accepted_students: NDArray[float], refused_students: NDArray[float]):
        nb_accepted = len(accepted_students)
        nb_refused = len(refused_students)

        lamb_acc = np.stack([self.lamb] * nb_accepted, axis=0)
        lamb_ref = np.stack([self.lamb] * nb_refused, axis=0)

        self.model.addConstr(np.sum(self.weights_s, axis=0) >= lamb_acc)
        self.model.addConstr(np.sum(self.weights_s, axis= 0) < lamb_ref)

        self.model.addConstr(np.sum(self.weights) == 1)

        M = 100
        M_acc = np.stack([M] * nb_accepted, axis=0)
        M_ref = np.stack([M] * nb_refused, axis=0)

        self.model.addConstr(M_acc * (self.delta_acc - np.ones(shape=()) <= accepted_students - self.frontier < M_acc * self.delta_acc))
        self.model.addConstr(M_ref * (self.delta_ref - np.ones(shape=()) <= refused_students - self.frontier < M_ref * self.delta_ref))

        weights_acc = np.stack([self.weights] * nb_accepted, axis=0)
        weights_ref = np.stack([self.weights] * nb_refused, axis=0)
        self.model.addConstr(weights_acc >= self.weights_s_acc >= 0)
        self.model.addConstr(weights_ref >= self.weights_s_ref)
        # Goal function
