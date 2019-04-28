import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0],'..'))
from query_completion.beam import GetCompletions
from query_completion.model import MetaQACModel
from util.vgg.vgg_net import channel_mean


print('Loading Model.....')
m = MetaQACModel('query_experiment_refer')
m.MakeSessionAndRestore(2)

print('Loading Image.....')
example_image = np.load('data/visual/processed_images_224/830.npy') #290/830/258 a, ca
plt.imshow(example_image)
plt.show()
#Building 2567  b, ar
#Arm/area ar
# Bed 2106, 2132
m.Lock(example_image-channel_mean)

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
            comp_list = list(GetCompletions(['<S>'] + list(prefix), m, branching_factor=4, beam_size=100))
            top_queries = [''.join(q.words[1:-1]) for q in comp_list[:-4:-1]]
            #print(top_queries)
            print('Query: {0:20}  Completion: {1}'.format(
                        prefix,top_queries))
        #print("key:", chr(key)) # prints: 'key: 97' for 'a' pressed
                                        # '-1' on no presses
    time.sleep(.2)