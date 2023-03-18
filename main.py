import argparse

from regression_model import *

def main():
    x = [[1, 3], [4, 2], [5, 1], [7, 4], [8, 9]]
    y = [1.002, 4.1, 4.96, 6.78, 8.2]

    adam = LR(x=x, y=y, epoch=1000, lr=1e-2, optimization="Adam")
    rms = LR(x=x, y=y, epoch=1000, lr=1e-2, optimization="RMSprop")
    none = LR(x=x, y=y, epoch=1000, lr=1e-2, optimization="None")
    m = LR(x=x, y=y, epoch=1000, lr=1e-2, optimization="Momentum")

    adam.fit()
    rms.fit()
    none.fit()
    m.fit()

    print("adam: ", adam.predict([[1, 3]]))
    print("rms: ", rms.predict([[1, 3]]))
    print("none: ", none.predict([[1, 3]]))
    print("m: ", m.predict([[1, 3]]))


main()
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', type=str, default='none',
                        help="Options are ['none', 'ports', 'ids', 'random', 'dropout']")
    parser.add_argument('--prob', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=4)  # 9 layers were used for skipcircles dataset
    parser.add_argument('--use_aux_loss', action='store_true', default=False)

    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--prob_ablation', action='store_true', default=False, help="Run probability ablation study")
    parser.add_argument('--num_runs_ablation', action='store_true', default=False,
                        help="Run number of runs ablation study")

    parser.add_argument('--dataset', type=str, default='limitsone',
                        help="Options are ['skipcircles', 'triangles', 'lcc', 'limitsone', 'limitstwo', 'fourcycles']")
    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)
    '''