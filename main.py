import numpy as np
import connect_four
import model
import argparse
import os
import glob
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--play', action='store_true')
parser.add_argument('--assist', action='store_true')
parser.add_argument('--load', action='store_true')
args = parser.parse_args()

def main():
    if args.play:
        m = model.Model(7, 6)
        m.restore()
        g = connect_four.Board()
        i = 0
        while g.is_game_over() is None:
            policy, value = m.predict(np.expand_dims(g.training_state(), axis=0))
            print('value ', value)
            print('policy ', policy)
            if i % 2 == 0:
                action = int(input())
                g.make_move(action)
                g.print()
            else:
                g.play(m)
                g.print()
            i += 1
        print("Winner is ", g.is_game_over())
    else:
        if not args.train:
            m = model.Model(7, 6)
            for t in range(200):
                if not args.assist:
                    files = sorted(os.listdir('data'))
                    max_files = 10000
                    if len(files) > max_files:
                        remove = files[:-max_files]
                        for filename in remove:
                            try:
                                os.remove(os.path.join('data', filename))
                            except PermissionError:
                                pass
                for _ in range(10):
                    m.restore()
                    for game in range(100):
                        g = connect_four.Board()
                        g.save_game(m)
        else: 
            m = model.Model(7, 6)
            if args.load:
                m.restore()
            m.train()


if __name__ == "__main__":
    main()
