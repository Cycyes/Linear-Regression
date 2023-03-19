import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataread import test_data_1, train_data_1, housing_train_data, get_data_2
from data_preprocess import train_test_split, PCA, PR, FeatureScaling
from regression_model import LR


def display_results(text_list, MSE_scores, RMSE_scores, MAE_scores, R_Squared_scores):
    print("MSE")
    for i in range(len(text_list)):
        print(text_list[i] + ": ", MSE_scores[i])
    print("RMSE")
    for i in range(len(text_list)):
        print(text_list[i] + ": ", RMSE_scores[i])
    print("MAE")
    for i in range(len(text_list)):
        print(text_list[i] + ": ", MAE_scores[i])
    print("R_Squared")
    for i in range(len(text_list)):
        print(text_list[i] + ": ", R_Squared_scores[i])

def main(args):
    x, y = housing_train_data()
    # x, y = get_data_2()
    x_scaled = FeatureScaling(x)
    
    x_pca2 = PCA(x_scaled, 2)
    x_pca4 = PCA(x_scaled, 4)
    x_pca6 = PCA(x_scaled, 6)
    x_pca8 = PCA(x_scaled, 8)
    x_pca10 = PCA(x_scaled, 10)


    # x_train, y_train = train_data_1()
    # x_test, y_test = test_data_1()
    x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test = train_test_split(x, x_scaled, x_pca2, x_pca4, x_pca6, x_pca8, x_pca10, y, 0.2)

    reg = LR(x=x_pca4, y=y_train, epoch=args.epoch, lr=args.lr, alpha=args.alpha, gamma=args.gamma, beta=args.beta, batch=args.batch, optimizer=args.optimizer)

    reg.fit()
    
    # 77,79.77515201
    # print(reg.predict([[77.79]]))

    test_all(x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test)

    return 0


def test_all(x_train, x_test, x_train_scaled, x_test_scaled, x_train_pca2, x_test_pca2, x_train_pca4, x_test_pca4, x_train_pca6, x_test_pca6, x_train_pca8, x_test_pca8, x_train_pca10, x_test_pca10, y_train, y_test):

    #######################################################################################################
    ########################################### Feature Scaling ###########################################
    #######################################################################################################
    if True:
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

        Fs_MSE_scores = [fs_0.mse(x_test, y_test), fs_1.mse(x_test_scaled, y_test), fs_2.mse(x_test, y_test), fs_3.mse(x_test_scaled, y_test)]
        Fs_RMSE_scores = [fs_0.rmse(x_test, y_test), fs_1.rmse(x_test_scaled, y_test), fs_2.rmse(x_test, y_test), fs_3.rmse(x_test_scaled, y_test)]
        Fs_MAE_scores = [fs_0.mae(x_test, y_test), fs_1.mae(x_test_scaled, y_test), fs_2.mae(x_test, y_test), fs_3.mae(x_test_scaled, y_test)]
        Fs_R_Squared_scores = [fs_0.r_squared(x_test, y_test), fs_1.r_squared(x_test_scaled, y_test), fs_2.r_squared(x_test, y_test), fs_3.r_squared(x_test_scaled, y_test)]

        display_results(fs_list, Fs_MSE_scores, Fs_RMSE_scores, Fs_MAE_scores, Fs_R_Squared_scores)

        fs_0_pred = fs_0.predict(x_test)
        fs_1_pred = fs_1.predict(x_test_scaled)
        fs_2_pred = fs_2.predict(x_test)
        fs_3_pred = fs_3.predict(x_test_scaled)

        print(fs_0_costs)

        print("test Feature Scaling end")


    #######################################################################################################
    ################################################ Batch ################################################
    #######################################################################################################
    if False:
        print("test Batch begin")

        batch_list = ["batch=1 (BGD)", "batch=4 (mini-BGD)", "batch=12 (mini-BGD)", "batch=num_item (SGD)"]
        
        batch_0 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, batch=1, optimizer="None")
        batch_1 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, batch=4, optimizer="None")
        batch_2 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, batch=12, optimizer="None")
        batch_3 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, batch=len(y_train), optimizer="None")

        batch_0_costs = batch_0.fit()
        batch_1_costs = batch_1.fit()
        batch_2_costs = batch_2.fit()
        batch_3_costs = batch_3.fit()

        Batch_MSE_scores = [batch_0.mse(x_test_scaled, y_test), batch_1.mse(x_test_scaled, y_test), batch_2.mse(x_test_scaled, y_test), batch_3.mse(x_test_scaled, y_test)]
        Batch_RMSE_scores = [batch_0.rmse(x_test_scaled, y_test), batch_1.rmse(x_test_scaled, y_test), batch_2.rmse(x_test_scaled, y_test), batch_3.rmse(x_test_scaled, y_test)]
        Batch_MAE_scores = [batch_0.mae(x_test_scaled, y_test), batch_1.mae(x_test_scaled, y_test), batch_2.mae(x_test_scaled, y_test), batch_3.mae(x_test_scaled, y_test)]
        Batch_R_Squared_scores = [fs_0.r_squared(x_test_scaled, y_test), fs_1.r_squared(x_test_scaled, y_test), fs_2.r_squared(x_test_scaled, y_test), fs_3.r_squared(x_test_scaled, y_test)]
        
        display_results(batch_list, Batch_MSE_scores, Batch_RMSE_scores, Batch_MAE_scores, Batch_R_Squared_scores)

        batch_0_pred = batch_0.predict(x_test_scaled)
        batch_1_pred = batch_1.predict(x_test_scaled)
        batch_2_pred = batch_2.predict(x_test_scaled)
        batch_3_pred = batch_3.predict(x_test_scaled)

        print("test Batch end")


    #######################################################################################################
    ############################################ Learning Rate ############################################
    #######################################################################################################
    if False:
        print("test Lrarning Rate begin")

        print("optimizer=None")

        lr_list = ["lr=1e-1", "lr=1e-2", "lr=1e-3", "lr=1e-4", "lr=1e-5"]

        lr_0_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-1, optimizer="None")
        lr_1_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-2, optimizer="None")
        lr_2_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-3, optimizer="None")
        lr_3_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-4, optimizer="None")
        lr_4_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, optimizer="None")

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

        lr_0_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-1, optimizer="Adam")
        lr_1_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-2, optimizer="Adam")
        lr_2_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-3, optimizer="Adam")
        lr_3_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-4, optimizer="Adam")
        lr_4_adam = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, optimizer="Adam")

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

        norm_none_0 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, alpha=0, batch=len(y_train), optimizer="None")
        norm_none_1 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, alpha=0.001, batch=len(y_train), optimizer="None")
        norm_none_2 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, alpha=0.01, batch=len(y_train), optimizer="None")
        norm_none_3 = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, alpha=0.1, batch=len(y_train), optimizer="None")

        norm_none_0_costs = norm_none_0.fit()
        norm_none_1_costs = norm_none_1.fit()
        norm_none_2_costs = norm_none_2.fit()
        norm_none_3_costs = norm_none_3.fit()

        norm_none_MSE_scores = [norm_none_0.mse(x_test_scaled, y_test), norm_none_1.mse(x_test_scaled, y_test),
                    norm_none_2.mse(x_test_scaled, y_test), norm_none_3.mse(x_test_scaled, y_test)]
        norm_none_RMSE_scores = [norm_none_0.rmse(x_test_scaled, y_test), norm_none_1.rmse(x_test_scaled, y_test),
                    norm_none_2.rmse(x_test_scaled, y_test), norm_none_3.rmse(x_test_scaled, y_test)]
        norm_none_MAE_scores = [norm_none_0.mae(x_test_scaled, y_test), norm_none_1.mae(x_test_scaled, y_test),
                    norm_none_2.mae(x_test_scaled, y_test), norm_none_3.mae(x_test_scaled, y_test)]
        norm_none_R_Squared_scores = [norm_none_0.r_squared(x_test_scaled, y_test), norm_none_1.r_squared(x_test_scaled, y_test),
                    norm_none_2.r_squared(x_test_scaled, y_test), norm_none_3.r_squared(x_test_scaled, y_test)]

        display_results(norm_none_list, norm_none_MSE_scores, norm_none_RMSE_scores, norm_none_MAE_scores, norm_none_R_Squared_scores)


        print("Principal Component Analysis Preprocess (k = 8)")
        norm_pca_list = ["aplha=0", "alpha=0.001", "alpha=0.01", "alpha=0.1"]

        norm_pca_0 = LR(x=x_train_pca8, y=y_train, epoch=100, lr=1e-5, alpha=0, batch=len(y_train), optimizer="None")
        norm_pca_1 = LR(x=x_train_pca8, y=y_train, epoch=100, lr=1e-5, alpha=0.001, batch=len(y_train), optimizer="None")
        norm_pca_2 = LR(x=x_train_pca8, y=y_train, epoch=100, lr=1e-5, alpha=0.01, batch=len(y_train), optimizer="None")
        norm_pca_3 = LR(x=x_train_pca8, y=y_train, epoch=100, lr=1e-5, alpha=0.1, batch=len(y_train), optimizer="None")

        norm_pca_0_costs = norm_pca_0.fit()
        norm_pca_1_costs = norm_pca_1.fit()
        norm_pca_2_costs = norm_pca_2.fit()
        norm_pca_3_costs = norm_pca_3.fit()

        norm_pca_MSE_scores = [norm_pca_0.mse(x_test_pca8, y_test), norm_pca_1.mse(x_test_pca8, y_test),
                    norm_pca_2.mse(x_test_pca8, y_test), norm_pca_3.mse(x_test_pca8, y_test)]
        norm_pca_RMSE_scores = [norm_pca_0.rmse(x_test_pca8, y_test), norm_pca_1.rmse(x_test_pca8, y_test),
                    norm_pca_2.rmse(x_test_pca8, y_test), norm_pca_3.rmse(x_test_pca8, y_test)]
        norm_pca_MAE_scores = [norm_pca_0.mae(x_test_pca8, y_test), norm_pca_1.mae(x_test_pca8, y_test),
                    norm_pca_2.mae(x_test_pca8, y_test), norm_pca_3.mae(x_test_pca8, y_test)]
        norm_pca_R_Squared_scores = [norm_pca_0.r_squared(x_test_pca8, y_test), norm_pca_1.r_squared(x_test_pca8, y_test),
                    norm_pca_2.r_squared(x_test_pca8, y_test), norm_pca_3.r_squared(x_test_pca8, y_test)]

        display_results(norm_pca_list, norm_pca_MSE_scores, norm_pca_RMSE_scores, norm_pca_MAE_scores, norm_pca_R_Squared_scores)

        print("Polynomial Regression (k = 2)")
        norm_pr_list = ["aplha=0", "alpha=0.001", "alpha=0.01", "alpha=0.1"]
        x_train_pr = PR(x_train_scaled, 2)
        x_test_pr = PR(x_test_scaled, 2)

        norm_pr_0 = LR(x=x_train_pr, y=y_train, epoch=100, lr=1e-5, alpha=0, batch=len(y_train), optimizer="None")
        norm_pr_1 = LR(x=x_train_pr, y=y_train, epoch=100, lr=1e-5, alpha=0.001, batch=len(y_train), optimizer="None")
        norm_pr_2 = LR(x=x_train_pr, y=y_train, epoch=100, lr=1e-5, alpha=0.01, batch=len(y_train), optimizer="None")
        norm_pr_3 = LR(x=x_train_pr, y=y_train, epoch=100, lr=1e-5, alpha=0.1, batch=len(y_train), optimizer="None")

        norm_pr_0_costs = norm_pr_0.fit()
        norm_pr_1_costs = norm_pr_1.fit()
        norm_pr_2_costs = norm_pr_2.fit()
        norm_pr_3_costs = norm_pr_3.fit()

        norm_pr_MSE_scores = [norm_pr_0.mse(x_test_pr, y_test), norm_pr_1.mse(x_test_pr, y_test),
                    norm_pr_2.mse(x_test_pr, y_test), norm_pr_3.mse(x_test_pr, y_test)]
        norm_pr_RMSE_scores = [norm_pr_0.rmse(x_test_pr, y_test), norm_pr_1.rmse(x_test_pr, y_test),
                    norm_pr_2.rmse(x_test_pr, y_test), norm_pr_3.rmse(x_test_pr, y_test)]
        norm_pr_MAE_scores = [norm_pr_0.mae(x_test_pr, y_test), norm_pr_1.mae(x_test_pr, y_test),
                    norm_pr_2.mae(x_test_pr, y_test), norm_pr_3.mae(x_test_pr, y_test)]
        norm_pr_R_Squared_scores = [norm_pr_0.r_squared(x_test_pr, y_test), norm_pr_1.r_squared(x_test_pr, y_test),
                    norm_pr_2.r_squared(x_test_pr, y_test), norm_pr_3.r_squared(x_test_pr, y_test)]

        display_results(norm_pr_list, norm_pr_MSE_scores, norm_pr_RMSE_scores, norm_pr_MAE_scores, norm_pr_R_Squared_scores)

        print("test Normalization end")


    #######################################################################################################
    ############################### Principal Component Analysis Preprocess ###############################
    #######################################################################################################
    if False:
        print("test Principal Component Analysis Preprocess begin")

        pca_list = ["k=2", "k=4", "k=6", "k=8", "k=10", "none"]

        pca_k2 = LR(x=x_train_pca2, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        pca_k4 = LR(x=x_train_pca4, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        pca_k6 = LR(x=x_train_pca6, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        pca_k8 = LR(x=x_train_pca8, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        pca_k10 = LR(x=x_train_pca10, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        pca_none = LR(x=x_train_scaled, y=y_train, epoch=100, lr=1e-5, optimizer="None")
        
        pca_k2_costs = pca_k2.fit()
        pca_k4_costs = pca_k4.fit()
        pca_k6_costs = pca_k6.fit()
        pca_k8_costs = pca_k8.fit()
        pca_k10_costs = pca_k10.fit()
        pca_none_costs = pca_none.fit()

        pca_MSE_scores = [pca_k2.mse(x_test_pca2, y_test), pca_k4.mse(x_test_pca4, y_test),
                    pca_k6.mse(x_test_pca6, y_test), pca_k8.mse(x_test_pca8, y_test), pca_k10.mse(x_test_pca10, y_test), pca_none.mse(x_test_scaled, y_test)]
        pca_RMSE_scores = [pca_k2.rmse(x_test_pca2, y_test), pca_k4.rmse(x_test_pca4, y_test),
                    pca_k6.rmse(x_test_pca6, y_test), pca_k8.rmse(x_test_pca8, y_test), pca_k10.rmse(x_test_pca10, y_test), pca_none.rmse(x_test_scaled, y_test)]
        pca_MAE_scores = [pca_k2.mae(x_test_pca2, y_test), pca_k4.mae(x_test_pca4, y_test),
                    pca_k6.mae(x_test_pca6, y_test), pca_k8.mae(x_test_pca8, y_test), pca_k10.mae(x_test_pca10, y_test), pca_none.mae(x_test_scaled, y_test)]
        pca_R_Squared_scores = [pca_k2.r_squared(x_test_pca2, y_test), pca_k4.r_squared(x_test_pca4, y_test),
                    pca_k6.r_squared(x_test_pca6, y_test), pca_k8.r_squared(x_test_pca8, y_test), pca_k10.r_squared(x_test_pca10, y_test), pca_none.r_squared(x_test_scaled, y_test)]

        display_results(pca_list, pca_MSE_scores, pca_RMSE_scores, pca_MAE_scores, pca_R_Squared_scores)

        pca_k2_pred = pca_k2.predict(x_test_pca2)
        pca_k4_pred = pca_k4.predict(x_test_pca4)
        pca_k6_pred = pca_k6.predict(x_test_pca6)
        pca_k8_pred = pca_k8.predict(x_test_pca8)
        pca_k10_pred = pca_k10.predict(x_test_pca10)
        pca_none_pred = pca_none.predict(x_test_scaled)

        print("test Principal Component Analysis Preprocess end")


    #######################################################################################################
    ############################################## Oprimizer ##############################################
    #######################################################################################################
    if False:
        print("test optimizer begin")

        optimizer_list = ["Adam", "RMSprop", "Momentum", "None"]

        optimizer_adam = LR(x=x_train_scaled, y=y_train, epoch=10000, lr=1e-3, batch=len(y_train), optimizer="Adam")
        optimizer_rmsprop = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-3, batch=len(y_train), optimizer="RMSprop")
        optimizer_momentum = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-3, batch=len(y_train), optimizer="Momentum")
        optimizer_none = LR(x=x_train_scaled, y=y_train, epoch=10, lr=1e-3, batch=len(y_train), optimizer="None")

        optimizer_adam_costs = optimizer_adam.fit(True)
        optimizer_rmsprop_costs = optimizer_rmsprop.fit()
        optimizer_momentum_costs = optimizer_momentum.fit()
        optimizer_none_costs = optimizer_none.fit()

        Optimizer_MSE_scores = [optimizer_adam.mse(x_test_scaled, y_test), optimizer_rmsprop.mse(x_test_scaled, y_test),
                    optimizer_momentum.mse(x_test_scaled, y_test), optimizer_none.mse(x_test_scaled, y_test)]
        Optimizer_RMSE_scores = [optimizer_adam.rmse(x_test_scaled, y_test), optimizer_rmsprop.rmse(x_test_scaled, y_test),
                    optimizer_momentum.rmse(x_test_scaled, y_test), optimizer_none.rmse(x_test_scaled, y_test)]
        Optimizer_MAE_scores = [optimizer_adam.mae(x_test_scaled, y_test), optimizer_rmsprop.mae(x_test_scaled, y_test),
                    optimizer_momentum.mae(x_test_scaled, y_test), optimizer_none.mae(x_test_scaled, y_test)]
        Optimizer_R_Squared_scores = [optimizer_adam.r_squared(x_test_scaled, y_test), optimizer_rmsprop.r_squared(x_test_scaled, y_test),
                    optimizer_momentum.r_squared(x_test_scaled, y_test), optimizer_none.r_squared(x_test_scaled, y_test)]
 
        display_results(optimizer_list, Optimizer_MSE_scores, Optimizer_RMSE_scores, Optimizer_MAE_scores, Optimizer_R_Squared_scores)

        optimizer_adam_pred = optimizer_adam.predict(x_test_scaled)
        optimizer_rmsprop_pred = optimizer_rmsprop.predict(x_test_scaled)
        optimizer_momentum_pred = optimizer_momentum.predict(x_test_scaled)
        optimizer_none_pred = optimizer_none.predict(x_test_scaled)
        
        print("test optimizer end")

    if True:
        from sklearn import linear_model
        demo = linear_model.LinearRegression()
        demo.fit(x_train, y_train)
        from sklearn.metrics import mean_squared_error
        print(mean_squared_error(y_test, demo.predict(x_test)))


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