import pygame
import pygame.display
import pygame.image
import pygame.math
import pygame.draw
import pygame.transform
import pygame.event
import pygame.sprite

import os
import sys
import neat
import neat.config
import neat.nn

from game_play.car import Car
from game_play.const import SCREEN, TRACK
from game_play.sprite import CarGroupSingle


def remove(
    index: int,
    cars: list[CarGroupSingle],
        ge: list[neat.DefaultGenome],
        nets: list[neat.nn.FeedForwardNetwork]):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.config.Config):
    cars: list[CarGroupSingle[Car]] = []
    ge: list[neat.DefaultGenome] = []
    nets: list[neat.nn.FeedForwardNetwork] = []

    for _, genome in genomes:
        # print(genome)
        cars.append(CarGroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i, cars, ge, nets)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update
        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


# Setup NEAT Neural Network
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
