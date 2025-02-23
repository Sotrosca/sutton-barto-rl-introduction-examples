import time

import numpy as np

from tabular_solutions import armed_bandit


def n_armed_bandit_experiment(
    agents_epsilons,
    steps,
    tasks_qty,
    execute_task_func=None,
    execute_task_params=None,
):
    if execute_task_func is None or execute_task_params is None:
        raise ValueError("execute_task_func and execute_task_params must be provided")
    agents_avg_reward_by_step = []
    agent_optimal_action_chosen_by_step = []
    start_time = time.time()
    for agent_eps in agents_epsilons:
        sum_rewards_by_step = np.zeros(steps)
        qty_action_chosen_by_step = np.zeros(steps)
        for _ in range(tasks_qty):
            rewards_by_step, optimal_action_chosen_by_step = execute_task_func(
                **execute_task_params
            )
            sum_rewards_by_step += rewards_by_step
            qty_action_chosen_by_step += optimal_action_chosen_by_step
        agents_avg_reward_by_step.append(sum_rewards_by_step / tasks_qty)
        agent_optimal_action_chosen_by_step.append(
            qty_action_chosen_by_step / tasks_qty
        )
    end_time = time.time()
    print(f"Experiment took {(end_time - start_time) / 60:.2f} minutes")
    return agents_avg_reward_by_step, agent_optimal_action_chosen_by_step
