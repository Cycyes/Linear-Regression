import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataread import test_data_1, train_data_1
from data_preprocess import train_test_split, PR
from regression_model import LR


def main(args):
    x_train, y_train = train_data_1()
    x_test, y_test = test_data_1()
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
    x = [[1.0], [2.0], [3.0]]
    y = [1.0, 2.0, 3.0]
    print(np.shape(x_test))
    print(np.shape(y_test))

    reg = LR(x=x_train, y=y_train, epoch=args.epoch, lr=args.lr, alpha=args.alpha, gamma=args.gamma, beta=args.beta, batch=args.batch, optimizer=args.optimizer)

    reg.fit()

    # 19,female,27.9,0,yes,southwest,16884.924
    # print(reg.predict([[19, 27.9, 0, 100]]))
    # test_all(x_train, x_test, y_train, y_test)

    return 0


def test_all(x_train, x_test, y_train, y_test):
    print("test optimizer begin")

    optimizer_list = ["Adam", "RMSprop", "Momentum", "None"]

    optimizer_adam = LR(x=x_train, y=y_train, epoch=1000, lr=1e-2, optimizer="Adam")
    optimizer_rmsprop = LR(x=x_train, y=y_train, epoch=1000, lr=1e-2, optimizer="RMSprop")
    optimizer_momentum = LR(x=x_train, y=y_train, epoch=1000, lr=1e-2, optimizer="Momentum")
    optimizer_none = LR(x=x_train, y=y_train, epoch=1000, lr=1e-2, optimizer="None")

    optimizer_adam.fit()
    optimizer_rmsprop.fit()
    optimizer_none.fit()
    optimizer_momentum.fit()

    print("MSE")
    Optimizer_MSE_scores = [optimizer_adam.mse(x_test, y_test), optimizer_rmsprop.mse(x_test, y_test),
                  optimizer_momentum.mse(x_test, y_test), optimizer_none.mse(x_test, y_test)]
    for i in range(len(optimizer_list)):
        print(optimizer_list[i] + ": ", Optimizer_MSE_scores[i])

    print("RMSE")
    Optimizer_RMSE_scores = [optimizer_adam.rmse(x_test, y_test), optimizer_rmsprop.rmse(x_test, y_test),
                  optimizer_momentum.rmse(x_test, y_test), optimizer_none.rmse(x_test, y_test)]
    for i in range(len(optimizer_list)):
        print(optimizer_list[i] + ": ", Optimizer_RMSE_scores[i])

    print("MAE")
    Optimizer_MAE_scores = [optimizer_adam.mae(x_test, y_test), optimizer_rmsprop.mae(x_test, y_test),
                   optimizer_momentum.mae(x_test, y_test), optimizer_none.mae(x_test, y_test)]
    for i in range(len(optimizer_list)):
        print(optimizer_list[i] + ": ", Optimizer_MAE_scores[i])

    print("R_Squared")
    Optimizer_R_Squared_scores = [optimizer_adam.r_squared(x_test, y_test), optimizer_rmsprop.r_squared(x_test, y_test),
                   optimizer_momentum.r_squared(x_test, y_test), optimizer_none.r_squared(x_test, y_test)]
    for i in range(len(optimizer_list)):
        print(optimizer_list[i] + ": ", Optimizer_R_Squared_scores[i])

    print("test optimizer end")

    print("test preprocess bagin")



    print("test preprocess end")

    return optimizer_list, Optimizer_MSE_scores, Optimizer_RMSE_scores, Optimizer_MAE_scores, Optimizer_R_Squared_scores

def draw(optimizer_list, Optimizer_MSE_scores, Optimizer_RMSE_scores, Optimizer_MAE_scores, Optimizer_R_Squared_scores):
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    mse_width = range(0, len(optimizer_list))
    rmse_width = [i + 0.3 for i in mse_width]
    mae_width = [i + 0.3 for i in mse_width]
    r_squared_width = [i + 0.3 for i in mse_width]
    plt.bar(mse_width, Optimizer_MSE_scores, lw=0.5, fc="r", width=0.3, label="MSE")
    plt.bar(rmse_width, Optimizer_RMSE_scores, lw=0.5, fc="b", width=0.3, label="RMSE")
    plt.bar(mae_width, Optimizer_MAE_scores, lw=0.5, fc="g", width=0.3, label="MAE")
    plt.bar(r_squared_width, Optimizer_R_Squared_scores, lw=0.5, fc="y", width=0.3, label="R_Squared")
    plt.title("Eval scores of Optimizer")
    plt.xlabel("Optimizer")
    plt.ylabel("Scores")
    plt.show()



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