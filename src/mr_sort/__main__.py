from src.mr_sort.solver import Solver
from src.mr_sort.generator import Generator

generator = Generator()
parameters = generator.get_parameters()
data = generator.generate(100, noise_var=0.1)

solver = Solver(
    nb_courses=parameters["nb_grades"],
    nb_accepted=len(data["accepted"]),
    nb_refused=len(data["rejected"]),
)

params_returned = solver.solve(data["accepted"], data["rejected"])
