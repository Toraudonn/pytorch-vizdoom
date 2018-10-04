from doom.game import play
from utils.args import parse_arguments
import argparse


if __name__ == '__main__':
    game_args = parse_arguments()
    play(game_args)