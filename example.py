import numpy as np

from evolution import Evolution
from fitness_functions import test_fitness

# ! These are example answers and results
# ! This would be the place where you would load you own data
# answers should be a np.array of dimensions N_MEASUREMENTS x N_QUESTIONS
# results should be a np.array of length N_QUESTIONS
answers = np.round(np.random.rand(1000, 25) * 2).astype(int)
results = np.sum(answers, axis=1) * 2

# n_epochs determines how many times you evolve the question sets
n_epochs = 400

# n_switches determines how many questions are switched during each mutation
n_switches = 5

# n_questions_1 determines the amount of questions in the 1st stage
# n_questions_2 determines the amount of questions in the 2nd stage
n_questions_1 = 12
n_questions_2 = 8

# pool_size determines how big the population of question sets is
pool_size = 100

# Create the Evolution object
evo = Evolution(
    answers=answers,
    fitness_func=test_fitness,
    n_questions_1=n_questions_1,
    n_questions_2=n_questions_2,
    n_switches=n_switches,
    pool_size=pool_size,
    results=results
)

# Evolve n_epochs
for i in range(n_epochs):
    evo.step()
    print(evo.best().question_selection)

print(f"Best solution is: {evo.best().question_selection}")
