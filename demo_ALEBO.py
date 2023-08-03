import numpy as np
from ax.modelbridge.strategies.alebo import ALEBOStrategy
# Run the optimization loop with that strategy
# This will take about 30 mins to run

from ax.service.managed_loop import optimize

from CUATRO.test_functions.high_dim.RB import RB

dim=100
RB_inst = RB(dim)

def RB_evaluation_function(parameterization):
    # Evaluates RB on the first two parameters of the parameterization.
    # Other parameters are unused.
    dim = RB_inst.dim
    x = np.array([parameterization[f"x{i}"] for i in range(dim)])
    return {"objective": (RB_inst.rosenbrock_higher(x), 0.0)} # 0.0 denotes 0 standard deviation, aka deterministic data

parameters = [
    {"name": f"x{i}", "type": "range", "bounds": [-5.0, 5.0], "value_type": "float"} for i in range(dim)
]

alebo_strategy = ALEBOStrategy(D=dim, d=4, init_size=5)

best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name="test",
    objective_name="objective",
    evaluation_function=RB_evaluation_function,
    minimize=True,
    total_trials=100,
    generation_strategy=alebo_strategy,
)

print(values)
# print(experiment.trials.values()[-1])

# fig = plt.figure(figsize=(7, 5))
# matplotlib.rcParams.update({'font.size': 16})
# plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
# plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
# plt.xlim([0, len(fX)])
# plt.ylim([0, 30])
# plt.title("10D Levy function")
# 
# plt.tight_layout()
# plt.show()

