"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from tetris import Tetris
#
#
# def get_args():
#     parser = argparse.ArgumentParser(
#         """Implementation of Deep Q Network to play Tetris""")
#
#     parser.add_argument("--width", type=int, default=10, help="The common width for all images")
#     parser.add_argument("--height", type=int, default=20, help="The common height for all images")
#     parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
#     parser.add_argument("--fps", type=int, default=300, help="frames per second")
#     parser.add_argument("--saved_path", type=str, default="trained_models")
#     parser.add_argument("--output", type=str, default="output.mp4")
#
#     args = parser.parse_args()
#     return args


def test():
    saved_path  = "saved_models"
    output = "output.mp4"
    fps  = 20
    block_size = 30
    height = 20
    width = 10
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris_3131".format(saved_path))
    else:
        model = torch.load("{}/tetris_3131".format(saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=width, height=height, block_size=block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                          (int(1.5 * width * block_size), height * block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    # opt = get_args()
    test()