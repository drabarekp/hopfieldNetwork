from Loader import *
from HopfieldNetwork import *

def sample():
    patterns = load('large-25x25.csv')
    hn = HopfieldNetwork(25*25, OJA)
    hn.load_patterns_and_learn(patterns)

    hn.draw_neurons(25, 25)

    hn.update_async(10)

    hn.draw_neurons(25, 25)
    hn.draw_weights()

    #for i in range(len(patterns)):
    #    hn.draw(patterns[i], 25, 25)

if __name__ == '__main__':
    sample()

