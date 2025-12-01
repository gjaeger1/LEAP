"""An example of solving a reinforcement learning problem by using evolution to
tune the weights of a neural network with synchronous parallel fitness evaluation.

This example demonstrates LEAP's synchronous distributed evaluation using dask.
The synchronous approach evaluates all offspring in parallel across available
CPU cores (or cluster nodes if configured), waiting for all evaluations to
complete before proceeding to the next generation.

Key differences from the sequential version:
1. Import dask.distributed.Client for parallel processing
2. Import DistributedIndividual and synchronous module from leap_ec.distrib
3. Use DistributedIndividual class instead of default Individual
4. Replace ops.evaluate with synchronous.eval_pool() in the pipeline
5. Wrap the generational_ea call in a dask Client context manager

For cluster/supercomputer usage:
- Start dask-scheduler and dask-worker processes externally
- Point the Client to the scheduler file: Client(scheduler_file='scheduler.json')
"""

import os
import sys

import gymnasium as gym
import numpy as np
from distributed import Client
from gymnasium import spaces
from matplotlib import pyplot as plt

from leap_ec import Individual, Representation, ops, probe, test_env_var
from leap_ec.algorithm import generational_ea
from leap_ec.distrib import DistributedIndividual, synchronous
from leap_ec.executable_rep import executable, neural_network, problems
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


##############################
# Function build_probes()
##############################
def build_probes(genomes_file):
    """Set up probes for writings results to file and terminal and
    displaying live metric plots."""
    assert genomes_file is not None

    probes = []

    # Print fitness stats to stdout
    probes.append(probe.FitnessStatsCSVProbe(stream=sys.stdout))

    # Save genome of the best individual to a file
    # Note: DistributedIndividual includes additional attributes like
    # uuid, birth_id, start_eval_time, stop_eval_time, hostname, pid
    probes.append(
        probe.AttributesCSVProbe(
            stream=genomes_file, best_only=True, do_fitness=True, do_genome=True
        )
    )

    # Open a figure to plot a fitness curve to
    plt.figure()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.title("Best-of-Generation Fitness")
    probes.append(
        probe.FitnessPlotProbe(ylim=(0, 1), xlim=(0, 1), modulo=1, ax=plt.gca())
    )

    # Open a figure to plot the best-of-gen network graph to
    plt.figure()
    probes.append(
        neural_network.GraphPhenotypeProbe(
            modulo=1, ax=plt.gca(), weights=True, weight_multiplier=3.0
        )
    )

    return probes


##############################
# Entry point
##############################
if __name__ == "__main__":
    # Parameters
    runs_per_fitness_eval = 5
    simulation_steps = 2500
    pop_size = 25
    num_hidden_nodes = 15
    mutate_std = 0.05
    gui = False  # Set to False to avoid rendering issues with parallel workers

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == "True":
        generations = 2
    else:
        generations = 1000  # Reduced for demonstration purposes

    # Load the OpenAI Gym simulation
    # Note: render_mode is set to None for parallel execution
    # GUI rendering doesn't work well with parallel workers
    environment = gym.make("CartPole-v1", render_mode=None)

    # Representation
    num_inputs = 4
    num_actions = environment.action_space.n
    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = executable.WrapperDecoder(
        wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(
            shape=(num_inputs, num_hidden_nodes, num_actions)
        ),
        decorator=executable.ArgmaxExecutable,
    )

    # Create a dask client for parallel fitness evaluations
    # By default, this uses all available CPU cores on the local machine
    # For cluster usage, use: Client(scheduler_file='scheduler.json')
    print("Starting dask client for parallel fitness evaluations...")
    with Client() as client:
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print(f"Number of workers: {len(client.scheduler_info()['workers'])}")

        with open("./genomes_parallel.csv", "w") as genomes_file:
            generational_ea(
                max_generations=generations,
                pop_size=pop_size,
                # Solve a problem that executes agents in the
                # environment and obtains fitness from it
                problem=problems.EnvironmentProblem(
                    runs_per_fitness_eval,
                    simulation_steps,
                    environment,
                    "reward",
                    gui=gui,
                ),
                representation=Representation(
                    initialize=create_real_vector(
                        bounds=([[-1, 1]] * decoder.wrapped_decoder.length)
                    ),
                    decoder=decoder,
                    # Use DistributedIndividual for tracking parallel evaluation metadata
                    individual_cls=DistributedIndividual,
                ),
                # The operator pipeline.
                pipeline=[
                    ops.sus_selection,
                    ops.clone,
                    mutate_gaussian(
                        std=mutate_std, bounds=(-1, 1), expected_num_mutations=1
                    ),
                    # Replace ops.evaluate with synchronous.eval_pool
                    # This distributes fitness evaluations across dask workers
                    # and waits for all to complete before continuing
                    synchronous.eval_pool(client=client, size=pop_size),
                    *build_probes(genomes_file),  # Inserting all the probes at the end
                ],
            )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != "True":
        plt.show()

    plt.close("all")
