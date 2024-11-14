import neat
import numpy as np
from minecraft_agent_env import MinecraftAgentEnv


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = MinecraftAgentEnv()
    state, _ = env.reset()

    done = False
    total_reward = 0

    while not done:
        state_tensor = np.array(state, dtype=np.float32)
        action_values = net.activate(state_tensor)
        action = np.argmax(action_values)

        next_state, reward, done, result = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'neat_config.ini'
)

pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_population, n=100)
print('\nBest genome:\n{!s}'.format(winner))