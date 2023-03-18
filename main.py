import argparse

from regression_model import *
from data_preprocess import PCA, PR

def main(args):
    x = [[1, 3], [4, 2], [5, 1], [7, 4], [8, 9]]
    padada = [[1, 3, 8, 9, 2], [4, 2, 3, 3, 3], [5, 1, 8, 0, 1], [7, 4, 3, 3, 3], [8, 9, 2, 1, 7]]
    y = [1.002, 4.1, 4.96, 6.78, 8.2]

    xx = PCA(padada, 2)
    print(xx)
    print(x)

    reg = LR(x=x, y=y, epoch=args.epoch, lr=args.lr, alpha=args.alpha, gamma=args.gamma, beta=args.beta, batch=args.batch, optimizer=args.optimizer)
    '''
    adam = LR(x=x, y=y, epoch=1000, lr=1e-2, optimizer="Adam")
    rms = LR(x=x, y=y, epoch=1000, lr=1e-2, optimizer="RMSprop")
    none = LR(x=x, y=y, epoch=1000, lr=1e-2, optimizer="None")
    m = LR(x=x, y=y, epoch=1000, lr=1e-2, optimizer="Momentum")
    '''

    reg.fit()
    '''
    adam.fit()
    rms.fit()
    none.fit()
    m.fit()
    '''

    print("reg: ", reg.predict([[1, 3]]))
    '''
    print("adam: ", adam.predict([[1, 3]]))
    print("rms: ", rms.predict([[1, 3]]))
    print("none: ", none.predict([[1, 3]]))
    print("m: ", m.predict([[1, 3]]))
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--preprocess', type=str, default='None',
                        help="Options are ['None', 'PCA', 'PR']")
    parser.add_argument('--optimizer', type=str, default='None',
                        help="Options are ['None', 'Momentum', 'RMSprop', 'Adam']")

    args = parser.parse_args()
    main(args)

    print('Finished')