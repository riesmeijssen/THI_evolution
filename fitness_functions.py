import numpy as np

from questionset import Questionset


def fitness_1_stage(qs: Questionset, answers: np.ndarray, results: np.ndarray) -> float:
    """
    The fitness function for 1-stage methods like THI-10, THI-12 or THI-20.
    It scores the model based on the norm of the difference between the predicted
    and actual values.
    """
    predicted_THI_values = (
        answers.dot(qs.question_selection) * 25 / sum(qs.question_selection)
    )
    return float(np.linalg.norm(predicted_THI_values - results))


def fitness_3_stage(qs: Questionset, answers: np.ndarray, results: np.ndarray) -> float:
    """
    The fitness function for 3-stage models like THI-12/8/5.
    It scores the model based on the norm of the difference between the predicted
    and actual values.
    """
    questions_stage_1 = qs.question_selection == 1
    questions_stage_2 = (qs.question_selection == 1) | (qs.question_selection == 2)
    predicted_THI_values_stage_1 = (
        answers.dot(questions_stage_1) * 25 / sum(questions_stage_1)
    )
    predicted_THI_values_stage_2 = (
        answers.dot(questions_stage_2) * 25 / sum(questions_stage_2)
    )

    stencil_1 = predicted_THI_values_stage_1 <= 16
    stencil_2 = (predicted_THI_values_stage_2 <= 36) & ~stencil_1
    stencil_3 = ~(stencil_1 | stencil_2)

    predicted_THI = (
        predicted_THI_values_stage_1 * stencil_1
        + predicted_THI_values_stage_2 * stencil_2
        + results * stencil_3
    )
    return float(np.linalg.norm(predicted_THI - results))


def test_fitness(qs: Questionset, answers: np.ndarray, results: np.ndarray) -> int:
    """
    Use this function to test if the genetic alogorithm works.
    It should result in question selection with higher numbers at the beginning.
    """
    t = 0
    for i in range(25):
        t += qs.question_selection[i] * i
    return t
