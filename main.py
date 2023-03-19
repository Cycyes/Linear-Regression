import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataread import test_data_1, train_data_1, housing_train_data
from data_preprocess import train_test_split, PCA, PR, FeatureScaling
from regression_model import LR


def main(args):
    x, y = housing_train_data()
    x_scaled = FeatureScaling(x)
    
    x_pca2 = PCA(x_scaled, 2)
    x_pca4 = PCA(x_scaled, 4)
    x_pca6 = PCA(x_scaled, 6)
    x_pca8 = PCA(x_scaled, 8)
    x_pca10 = PCA(x_scaled, 10)


    # x_train, y_train = train_data_1()
    # x_test, y_test = test_data_1()
    x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test = train_test_split(x, x_scaled, x_pca2, x_pca4, x_pca6, x_pca8, x_pca10, y, 0.2)

    reg = LR(x=x_train, y=y_train, epoch=args.epoch, lr=args.lr, alpha=args.alpha, gamma=args.gamma, beta=args.beta, batch=args.batch, optimizer=args.optimizer)

    reg.fit()
    
    # 77,79.77515201
    # print(reg.predict([[77.79]]))

    test_all(x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test)

    return 0


def test_all(x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test):

    #######################################################################################################
    ########################################### Feature Scaling ###########################################
    #######################################################################################################
    if False:
        print("test Feature Scaling begin")

        fs_list = ["none with lr=1e-5", "feature scaling with lr=1e-5", "none with lr=1e-6", "feature scaling with lr=1e-6"]
        
        fs_0 = LR(x=x_train, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        fs_1 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        fs_2 = LR(x=x_train, y=y_train, epoch=100, lr=1e-6, optimizer="None")
        fs_3 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-6, optimizer="None")

        fs_0_costs = fs_0.fit()
        fs_1_costs = fs_1.fit()
        fs_2_costs = fs_2.fit()
        fs_3_costs = fs_3.fit()

        print("MSE")
        Fs_MSE_scores = [fs_0.mse(x_test_scaled, y_test), fs_1.mse(x_test_scaled, y_test), fs_2.mse(x_test_scaled, y_test), fs_2.mse(x_test_scaled, y_test)]
        for i in range(len(fs_list)):
            print(fs_list[i] + ": ", Fs_MSE_scores[i])

        print("RMSE")
        Fs_RMSE_scores = [fs_0.rmse(x_test_scaled, y_test), fs_1.rmse(x_test_scaled, y_test), fs_2.rmse(x_test_scaled, y_test), fs_3.rmse(x_test_scaled, y_test)]
        for i in range(len(fs_list)):
            print(fs_list[i] + ": ", Fs_RMSE_scores[i])

        print("MAE")
        Fs_RAE_scores = [fs_0.mae(x_test_scaled, y_test), fs_1.mae(x_test_scaled, y_test), fs_2.mae(x_test_scaled, y_test), fs_3.mae(x_test_scaled, y_test)]
        for i in range(len(fs_list)):
            print(fs_list[i] + ": ", Fs_RAE_scores[i])

        print("R_Squared")
        Fs_R_Squared_scores = [fs_0.r_squared(x_test_scaled, y_test), fs_1.r_squared(x_test_scaled, y_test), fs_2.r_squared(x_test_scaled, y_test), fs_3.r_squared(x_test_scaled, y_test)]
        for i in range(len(fs_list)):
            print(fs_list[i] + ": ", Fs_R_Squared_scores[i])

        fs_0_pred = fs_0.predict(x_train)
        fs_1_pred = fs_1.predict(x_train_scaled)
        fs_2_pred = fs_2.predict(x_train)
        fs_3_pred = fs_3.predict(x_train_scaled)

        print("test Feature Scaling end")
        # x_train = x_train_scaled
        # x_test = x_test_scaled


    #######################################################################################################
    ############################################ Learning Rate ############################################
    #######################################################################################################
    if False:
        print("test Lrarning Rate begin")

        print("optimizer=None")

        lr_list = ["lr=1e-1", "lr=1e-2", "lr=1e-3", "lr=1e-4", "lr=1e-5"]

        lr_0_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-1, optimizer="None")
        lr_1_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-2, optimizer="None")
        lr_2_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-3, optimizer="None")
        lr_3_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-4, optimizer="None")
        lr_4_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, optimizer="None")

        lr_0_none_costs = lr_0_none.fit()
        lr_1_none_costs = lr_1_none.fit()
        lr_2_none_costs = lr_2_none.fit()
        lr_3_none_costs = lr_3_none.fit()
        lr_4_none_costs = lr_4_none.fit()

        print("MSE")
        Lr_None_MSE_scores = [lr_0_none.mse(x_test_scaled, y_test), lr_1_none.mse(x_test_scaled, y_test), lr_2_none.mse(x_test_scaled, y_test), lr_3_none.mse(x_test_scaled, y_test), lr_4_none.mse(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_None_MSE_scores[i])

        print("RMSE")
        Lr_None_RMSE_scores = [lr_0_none.rmse(x_test_scaled, y_test), lr_1_none.rmse(x_test_scaled, y_test),
                    lr_2_none.rmse(x_test_scaled, y_test), lr_3_none.rmse(x_test_scaled, y_test), lr_4_none.rmse(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_None_RMSE_scores[i])

        print("MAE")
        Lr_None_MAE_scores = [lr_0_none.mae(x_test_scaled, y_test), lr_1_none.mae(x_test_scaled, y_test),
                    lr_2_none.mae(x_test_scaled, y_test), lr_3_none.mae(x_test_scaled, y_test), lr_4_none.mae(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_None_MAE_scores[i])

        print("R_Squared")
        Lr_None_R_Squared_scores = [lr_0_none.r_squared(x_test_scaled, y_test), lr_1_none.r_squared(x_test_scaled, y_test),
                    lr_2_none.r_squared(x_test_scaled, y_test), lr_3_none.r_squared(x_test_scaled, y_test), lr_4_none.r_squared(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_None_R_Squared_scores[i])


        print("optimizer=Adam")

        lr_0_adam = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-1, optimizer="Adam")
        lr_1_adam = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-2, optimizer="Adam")
        lr_2_adam = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-3, optimizer="Adam")
        lr_3_adam = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-4, optimizer="Adam")
        lr_4_adam = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, optimizer="Adam")

        lr_0_adam_costs = lr_0_adam.fit()
        lr_1_adam_costs = lr_1_adam.fit()
        lr_2_adam_costs = lr_2_adam.fit()
        lr_3_adam_costs = lr_3_adam.fit()
        lr_4_adam_costs = lr_4_adam.fit()

        print("MSE")
        Lr_Adam_MSE_scores = [lr_0_adam.mse(x_test_scaled, y_test), lr_1_adam.mse(x_test_scaled, y_test), lr_2_adam.mse(x_test_scaled, y_test), lr_3_adam.mse(x_test_scaled, y_test), lr_4_adam.mse(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_Adam_MSE_scores[i])

        print("RMSE")
        Lr_Adam_RMSE_scores = [lr_0_adam.rmse(x_test_scaled, y_test), lr_1_adam.rmse(x_test_scaled, y_test),
                    lr_2_adam.rmse(x_test_scaled, y_test), lr_3_adam.rmse(x_test_scaled, y_test), lr_4_adam.rmse(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_Adam_RMSE_scores[i])

        print("MAE")
        Lr_Adam_MAE_scores = [lr_0_adam.mae(x_test_scaled, y_test), lr_1_adam.mae(x_test_scaled, y_test),
                    lr_2_adam.mae(x_test_scaled, y_test), lr_3_adam.mae(x_test_scaled, y_test), lr_4_adam.mae(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_Adam_MAE_scores[i])

        print("R_Squared")
        Lr_Adam_R_Squared_scores = [lr_0_adam.r_squared(x_test_scaled, y_test), lr_1_adam.r_squared(x_test_scaled, y_test),
                    lr_2_adam.r_squared(x_test_scaled, y_test), lr_3_adam.r_squared(x_test_scaled, y_test), lr_4_adam.r_squared(x_test_scaled, y_test)]
        for i in range(len(lr_list)):
            print(lr_list[i] + ": ", Lr_Adam_R_Squared_scores[i])

        print("test Lrarning Rate end")


    #######################################################################################################
    ##################################### Normalization and Preprocess ####################################
    #######################################################################################################
    if False:
        print("test Normalization and Preprocess begin")

        print("None Preprocess")
        norm_none_list = ["aplha=0", "alpha=0.001", "alpha=0.01", "alpha=0.1"]

        norm_none_0 = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, alpha=0, optimizer="None")
        norm_none_1 = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, alpha=0.001, optimizer="None")
        norm_none_2 = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, alpha=0.01, optimizer="None")
        norm_none_3 = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, alpha=0.1, optimizer="None")

        norm_none_0_costs = norm_none_0.fit()
        norm_none_1_costs = norm_none_1.fit()
        norm_none_2_costs = norm_none_2.fit()
        norm_none_3_costs = norm_none_3.fit()

        print("MSE")
        norm_none_MSE_scores = [norm_none_0.mse(x_test_scaled, y_test), norm_none_1.mse(x_test_scaled, y_test),
                    norm_none_2.mse(x_test_scaled, y_test), norm_none_3.mse(x_test_scaled, y_test)]
        for i in range(len(norm_none_list)):
            print(norm_none_list[i] + ": ", norm_none_MSE_scores[i])

        print("RMSE")
        norm_none_RMSE_scores = [norm_none_0.rmse(x_test_scaled, y_test), norm_none_1.rmse(x_test_scaled, y_test),
                    norm_none_2.rmse(x_test_scaled, y_test), norm_none_3.rmse(x_test_scaled, y_test)]
        for i in range(len(norm_none_list)):
            print(norm_none_list[i] + ": ", norm_none_RMSE_scores[i])

        print("MAE")
        norm_none_MAE_scores = [norm_none_0.mae(x_test_scaled, y_test), norm_none_1.mae(x_test_scaled, y_test),
                    norm_none_2.mae(x_test_scaled, y_test), norm_none_3.mae(x_test_scaled, y_test)]
        for i in range(len(norm_none_list)):
            print(norm_none_list[i] + ": ", norm_none_MAE_scores[i])

        print("R_Squared")
        norm_none_R_Squared_scores = [norm_none_0.r_squared(x_test_scaled, y_test), norm_none_1.r_squared(x_test_scaled, y_test),
                    norm_none_2.r_squared(x_test_scaled, y_test), norm_none_3.r_squared(x_test_scaled, y_test)]
        for i in range(len(norm_none_list)):
            print(norm_none_list[i] + ": ", norm_none_R_Squared_scores[i])


        print("Principal Component Analysis Preprocess (k = 6)")
        norm_pca_list = ["aplha=0", "alpha=0.001", "alpha=0.01", "alpha=0.1"]

        norm_pca_0 = LR(x=x_train_pca, y=y_train, epoch=10, lr=1e-5, alpha=0, optimizer="None")
        norm_pca_1 = LR(x=x_train_pca, y=y_train, epoch=10, lr=1e-5, alpha=0.001, optimizer="None")
        norm_pca_2 = LR(x=x_train_pca, y=y_train, epoch=10, lr=1e-5, alpha=0.01, optimizer="None")
        norm_pca_3 = LR(x=x_train_pca, y=y_train, epoch=10, lr=1e-5, alpha=0.1, optimizer="None")

        norm_pca_0_costs = norm_pca_0.fit()
        norm_pca_1_costs = norm_pca_1.fit()
        norm_pca_2_costs = norm_pca_2.fit()
        norm_pca_3_costs = norm_pca_3.fit()

        print("MSE")
        norm_pca_MSE_scores = [norm_pca_0.mse(x_test_pca, y_test), norm_pca_1.mse(x_test_pca, y_test),
                    norm_pca_2.mse(x_test_pca, y_test), norm_pca_3.mse(x_test_pca, y_test)]
        for i in range(len(norm_pca_list)):
            print(norm_pca_list[i] + ": ", norm_pca_MSE_scores[i])

        print("RMSE")
        norm_pca_RMSE_scores = [norm_pca_0.rmse(x_test_pca, y_test), norm_pca_1.rmse(x_test_pca, y_test),
                    norm_pca_2.rmse(x_test_pca, y_test), norm_pca_3.rmse(x_test_pca, y_test)]
        for i in range(len(norm_pca_list)):
            print(norm_pca_list[i] + ": ", norm_pca_RMSE_scores[i])

        print("MAE")
        norm_pca_MAE_scores = [norm_pca_0.mae(x_test_pca, y_test), norm_pca_1.mae(x_test_pca, y_test),
                    norm_pca_2.mae(x_test_pca, y_test), norm_pca_3.mae(x_test_pca, y_test)]
        for i in range(len(norm_pca_list)):
            print(norm_pca_list[i] + ": ", norm_pca_MAE_scores[i])

        print("R_Squared")
        norm_pca_R_Squared_scores = [norm_pca_0.r_squared(x_test_pca, y_test), norm_pca_1.r_squared(x_test_pca, y_test),
                    norm_pca_2.r_squared(x_test_pca, y_test), norm_pca_3.r_squared(x_test_pca, y_test)]
        for i in range(len(norm_pca_list)):
            print(norm_pca_list[i] + ": ", norm_pca_R_Squared_scores[i])


        print("Polynomial Regression (k = 2)")
        norm_pr_list = ["aplha=0", "alpha=0.001", "alpha=0.01", "alpha=0.1"]
        x_train_pr = PR(x_train_scaled, 2)
        x_test_pr = PR(x_test_scaled, 2)

        norm_pr_0 = LR(x=x_train_pr, y=y_train, epoch=10, lr=1e-5, alpha=0, optimizer="None")
        norm_pr_1 = LR(x=x_train_pr, y=y_train, epoch=10, lr=1e-5, alpha=0.001, optimizer="None")
        norm_pr_2 = LR(x=x_train_pr, y=y_train, epoch=10, lr=1e-5, alpha=0.01, optimizer="None")
        norm_pr_3 = LR(x=x_train_pr, y=y_train, epoch=10, lr=1e-5, alpha=0.1, optimizer="None")

        norm_pr_0_costs = norm_pr_0.fit()
        norm_pr_1_costs = norm_pr_1.fit()
        norm_pr_2_costs = norm_pr_2.fit()
        norm_pr_3_costs = norm_pr_3.fit()

        print("MSE")
        norm_pr_MSE_scores = [norm_pr_0.mse(x_test_pr, y_test), norm_pr_1.mse(x_test_pr, y_test),
                    norm_pr_2.mse(x_test_pr, y_test), norm_pr_3.mse(x_test_pr, y_test)]
        for i in range(len(norm_pr_list)):
            print(norm_pr_list[i] + ": ", norm_pr_MSE_scores[i])

        print("RMSE")
        norm_pr_RMSE_scores = [norm_pr_0.rmse(x_test_pr, y_test), norm_pr_1.rmse(x_test_pr, y_test),
                    norm_pr_2.rmse(x_test_pr, y_test), norm_pr_3.rmse(x_test_pr, y_test)]
        for i in range(len(norm_pr_list)):
            print(norm_pr_list[i] + ": ", norm_pr_RMSE_scores[i])

        print("MAE")
        norm_pr_MAE_scores = [norm_pr_0.mae(x_test_pr, y_test), norm_pr_1.mae(x_test_pr, y_test),
                    norm_pr_2.mae(x_test_pr, y_test), norm_pr_3.mae(x_test_pr, y_test)]
        for i in range(len(norm_pr_list)):
            print(norm_pr_list[i] + ": ", norm_pr_MAE_scores[i])

        print("R_Squared")
        norm_pr_R_Squared_scores = [norm_pr_0.r_squared(x_test_pr, y_test), norm_pr_1.r_squared(x_test_pr, y_test),
                    norm_pr_2.r_squared(x_test_pr, y_test), norm_pr_3.r_squared(x_test_pr, y_test)]
        for i in range(len(norm_pr_list)):
            print(norm_pr_list[i] + ": ", norm_pr_R_Squared_scores[i])

        print("test Normalization end")


    #######################################################################################################
    ############################### Principal Component Analysis Preprocess ###############################
    #######################################################################################################
    if True:
        print("test Principal Component Analysis Preprocess begin")

        pca_list = ["k=2", "k=4", "k=6", "k=8", "k=10", "none"]

        pca_k2 = LR(x=x_train_pca2, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        pca_k4 = LR(x=x_train_pca4, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        pca_k6 = LR(x=x_train_pca6, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        pca_k8 = LR(x=x_train_pca8, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        pca_k10 = LR(x=x_train_pca10, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        pca_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-5, optimizer="None")
        
        pca_k2_costs = pca_k2.fit()
        pca_k4_costs = pca_k4.fit()
        pca_k6_costs = pca_k6.fit()
        pca_k8_costs = pca_k8.fit()
        pca_k10_costs = pca_k10.fit()
        pca_none_costs = pca_none.fit()

        print("MSE")
        pca_MSE_scores = [pca_k2.mse(x_test_pca2, y_test), pca_k4.mse(x_test_pca4, y_test),
                    pca_k6.mse(x_test_pca6, y_test), pca_k8.mse(x_test_pca8, y_test), pca_k10.mse(x_test_pca10, y_test), pca_none.mse(x_test_scaled, y_test)]
        for i in range(len(pca_list)):
            print(pca_list[i] + ": ", pca_MSE_scores[i])

        print("RMSE")
        pca_RMSE_scores = [pca_k2.rmse(x_test_pca2, y_test), pca_k4.rmse(x_test_pca4, y_test),
                    pca_k6.rmse(x_test_pca6, y_test), pca_k8.rmse(x_test_pca8, y_test), pca_k10.rmse(x_test_pca10, y_test), pca_none.rmse(x_test_scaled, y_test)]
        for i in range(len(pca_list)):
            print(pca_list[i] + ": ", pca_RMSE_scores[i])

        print("MAE")
        pca_MAE_scores = [pca_k2.mae(x_test_pca2, y_test), pca_k4.mae(x_test_pca4, y_test),
                    pca_k6.mae(x_test_pca6, y_test), pca_k8.mae(x_test_pca8, y_test), pca_k10.mae(x_test_pca10, y_test), pca_none.mae(x_test_scaled, y_test)]
        for i in range(len(pca_list)):
            print(pca_list[i] + ": ", pca_MAE_scores[i])

        print("R_Squared")
        pca_R_Squared_scores = [pca_k2.r_squared(x_test_pca2, y_test), pca_k4.r_squared(x_test_pca4, y_test),
                    pca_k6.r_squared(x_test_pca6, y_test), pca_k8.r_squared(x_test_pca8, y_test), pca_k10.r_squared(x_test_pca10, y_test), pca_none.r_squared(x_test_scaled, y_test)]
        for i in range(len(pca_list)):
            print(pca_list[i] + ": ", pca_MAE_scores[i])

        print("test Principal Component Analysis Preprocess end")


    #######################################################################################################
    ############################################## Oprimizer ##############################################
    #######################################################################################################
    if False:
        print("test optimizer begin")

        optimizer_list = ["Adam", "RMSprop", "Momentum", "None"]

        optimizer_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-6, optimizer="Adam")
        optimizer_rmsprop = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-6, optimizer="RMSprop")
        optimizer_momentum = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-6, optimizer="Momentum")
        optimizer_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-6, optimizer="None")

        optimizer_adam_costs = optimizer_adam.fit()
        optimizer_rmsprop_costs = optimizer_rmsprop.fit()
        optimizer_momentum_costs = optimizer_momentum.fit()
        optimizer_none_costs = optimizer_none.fit()
        
        print("MSE")
        Optimizer_MSE_scores = [optimizer_adam.mse(x_test_scaled, y_test), optimizer_rmsprop.mse(x_test_scaled, y_test),
                    optimizer_momentum.mse(x_test_scaled, y_test), optimizer_none.mse(x_test_scaled, y_test)]
        for i in range(len(optimizer_list)):
            print(optimizer_list[i] + ": ", Optimizer_MSE_scores[i])

        print("RMSE")
        Optimizer_RMSE_scores = [optimizer_adam.rmse(x_test_scaled, y_test), optimizer_rmsprop.rmse(x_test_scaled, y_test),
                    optimizer_momentum.rmse(x_test_scaled, y_test), optimizer_none.rmse(x_test_scaled, y_test)]
        for i in range(len(optimizer_list)):
            print(optimizer_list[i] + ": ", Optimizer_RMSE_scores[i])

        print("MAE")
        Optimizer_MAE_scores = [optimizer_adam.mae(x_test_scaled, y_test), optimizer_rmsprop.mae(x_test_scaled, y_test),
                    optimizer_momentum.mae(x_test_scaled, y_test), optimizer_none.mae(x_test_scaled, y_test)]
        for i in range(len(optimizer_list)):
            print(optimizer_list[i] + ": ", Optimizer_MAE_scores[i])

        print("R_Squared")
        Optimizer_R_Squared_scores = [optimizer_adam.r_squared(x_test_scaled, y_test), optimizer_rmsprop.r_squared(x_test_scaled, y_test),
                    optimizer_momentum.r_squared(x_test_scaled, y_test), optimizer_none.r_squared(x_test_scaled, y_test)]
        for i in range(len(optimizer_list)):
            print(optimizer_list[i] + ": ", Optimizer_R_Squared_scores[i])

        print("test optimizer end")

    return "OK"
    # return optimizer_list, Optimizer_MSE_scores, Optimizer_RMSE_scores, Optimizer_MAE_scores, Optimizer_R_Squared_scores

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