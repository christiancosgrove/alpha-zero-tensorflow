import connect_four
import model
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model1')
parser.add_argument('--model2')
args = parser.parse_args()
def main():
    
    board = connect_four.Board()
    m1 = model.Model(7,6)
    # m1.restore_from_file(args.model1)
    # sess = tf.Session()
    # m2 = model.Model(7,6, sess=sess)
    # m2.restore_from_file(args.model2)
    print(board.compare_models(m1, args.model1, args.model2))



if __name__ == "__main__":
    main()
