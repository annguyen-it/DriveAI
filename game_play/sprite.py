import pygame.sprite

from game_play.car import Car


class CarGroupSingle(pygame.sprite.GroupSingle):
    sprite: Car

    def __init__(self, sprite: Car):
        pygame.sprite.GroupSingle.__init__(self, sprite)
