from dynamic import DynamicModel
from beam import GetCompletions, InitBeam,GetSavedKeystrokes
from model import MetaModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
#from AMSGrad import AMSGrad

print('Loading Model.....')
m = MetaModel('../referit_experiment_img_4_3')
m.MakeSessionAndRestore(2)

print('Loading Image.....')
example_image = np.load('../data/referit/img_val/51.npy')
vgg_feat = m.ComputeVGG(example_image)
plt.imshow(example_image)
plt.show()

print('Done')
prefix = ''

import curses, time

def main(stdscr):
    """checking for keypress"""
    stdscr.nodelay(True)  # do not wait for input when calling getch
    return stdscr.getch()

while True:

    key = curses.wrapper(main)
    if key == -1:
        pass
    else:
        if key == 10:
            print('Resetting query....')
            prefix = ''
        else:
            prefix = prefix + chr(key)
            print(list(GetCompletions(['<S>'] + list(prefix), vgg_feat, m, branching_factor=4, beam_size=100))[-1])

        #print("key:", chr(key)) # prints: 'key: 97' for 'a' pressed
                                        # '-1' on no presses


    time.sleep(.1)