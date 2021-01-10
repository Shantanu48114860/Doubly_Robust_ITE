from collections import OrderedDict
from datetime import date

import numpy as np

from Adversarial_Manager import Adversarial_Manager
from Constants import Constants
from DR_Net_Manager import DRNet_Manager
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    """
    This class specifies all the experimental details used in this research.
    """

    def __init__(self, running_mode):
        self.dL = DataLoader()
        self.running_mode = running_mode
        self.np_train = None
        self.np_test = None

    def run_all_experiments(self, train_path, test_path, iterations):
        """
        This function is the central point which controls the different experimental settings

        Input:
            train_path: path of the training dataset.
            train_path: path of the training dataset.
            iterations: Number of realizations of the dataset.
                        Default is 10 for Jobs dataset.
        """
        device = Utils.get_device()
        print(device)
        print("iterations", iterations)
        results_list = []

        run_parameters = self.__get_run_parameters()
        print(str(run_parameters["summary_file_name"]))
        file1 = open(run_parameters["summary_file_name"], "w")
        for iter_id in range(iterations):
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            np_train_X, np_train_T, np_train_e, np_train_yf, \
            np_test_X, np_test_T, np_test_e, np_test_yf = self.__load_data(train_path,
                                                                           test_path,
                                                                           iter_id)
            tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_e, np_train_yf)
            adv_manager = Adversarial_Manager(encoder_input_nodes=Constants.DRNET_INPUT_NODES,
                                              encoder_shared_nodes=Constants.Encoder_shared_nodes,
                                              encoder_x_out_nodes=Constants.Encoder_x_nodes,
                                              encoder_t_out_nodes=Constants.Encoder_t_nodes,
                                              encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                                              encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                                              decoder_in_nodes=Constants.Decoder_in_nodes,
                                              decoder_shared_nodes=Constants.Decoder_shared_nodes,
                                              decoder_out_nodes=Constants.Decoder_out_nodes,
                                              gen_in_nodes=Constants.Info_GAN_Gen_in_nodes,
                                              gen_shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                                              gen_out_nodes=Constants.Info_GAN_Gen_out_nodes,
                                              dis_in_nodes=Constants.Info_GAN_Dis_in_nodes,
                                              dis_shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                                              dis_out_nodes=Constants.Info_GAN_Dis_out_nodes,
                                              Q_in_nodes=Constants.Info_GAN_Q_in_nodes,
                                              Q_shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                                              Q_out_nodes=Constants.Info_GAN_Q_out_nodes,
                                              device=device)

            _train_parameters = {
                "epochs": Constants.Adversarial_epochs,
                "vae_lr": Constants.Adversarial_VAE_LR,
                "gan_G_lr": Constants.INFO_GAN_G_LR,
                "gan_D_lr": Constants.INFO_GAN_D_LR,
                "lambda": Constants.Adversarial_LAMBDA,
                "batch_size": Constants.Adversarial_BATCH_SIZE,
                "INFO_GAN_LAMBDA": Constants.INFO_GAN_LAMBDA,
                "INFO_GAN_ALPHA": Constants.INFO_GAN_ALPHA,
                "shuffle": True,
                "VAE_BETA": Constants.VAE_BETA,
                "train_dataset": tensor_train
            }
            print("Adversarial Model Training started....")
            adv_manager.train_adversarial_model(_train_parameters, device)
            np_y_cf = adv_manager.test_adversarial_model({"tensor_dataset": tensor_train}, device)
            print("Adversarial Model Training ended....")

            print("-----------> !! Supervised Training(DR_NET Models) !!<-----------")
            tensor_train_dr = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_y_cf)
            tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_e, np_test_yf)
            drnet_manager = DRNet_Manager(input_nodes=Constants.DRNET_INPUT_NODES,
                                          shared_nodes=Constants.DRNET_SHARED_NODES,
                                          outcome_nodes=Constants.DRNET_OUTPUT_NODES,
                                          device=device)

            _dr_train_parameters = {
                "epochs": Constants.DRNET_EPOCHS,
                "lr": Constants.DRNET_LR,
                "lambda": Constants.DRNET_LAMBDA,
                "batch_size": Constants.DRNET_BATCH_SIZE,
                "shuffle": True,
                "ALPHA": Constants.ALPHA,
                "BETA": Constants.BETA,
                "train_dataset": tensor_train_dr
            }
            drnet_manager.train_DR_NET(_dr_train_parameters, device)
            dr_eval_out = drnet_manager.test_DR_NET({"tensor_dataset": tensor_test}, device)
            dr_eval_in = drnet_manager.test_DR_NET({"tensor_dataset": tensor_train}, device)
            print("---" * 20)
            print("--> Model : DRNet Supervised Training Evaluation, Iter_id: {0}".format(iter_id))

            [RPol_out, ATT_out] = self.Perf_RPol_ATT(Utils.convert_to_col_vector(np.array(dr_eval_out["T_list"])),
                                                     Utils.convert_to_col_vector(np.array(dr_eval_out["yf_list"])),
                                                     Utils.convert_to_col_vector(np.array(dr_eval_out["y1_hat_list"])),
                                                     Utils.convert_to_col_vector(np.array(dr_eval_out["y0_hat_list"])))

            [RPol_in, ATT_in] = self.Perf_RPol_ATT(Utils.convert_to_col_vector(np.array(dr_eval_in["T_list"])),
                                                   Utils.convert_to_col_vector(np.array(dr_eval_in["yf_list"])),
                                                   Utils.convert_to_col_vector(np.array(dr_eval_in["y1_hat_list"])),
                                                   Utils.convert_to_col_vector(np.array(dr_eval_in["y0_hat_list"])))

            print("--------")
            print("RPol_out: ", RPol_out)
            print("ATT_out: ", ATT_out)
            print("RPol_in: ", RPol_in)
            print("ATT_in: ", ATT_in)

            print("---" * 20)

            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id

            result_dict["drnet_bias_att_out"] = ATT_out
            result_dict["drnet_policy_risk_out"] = RPol_out
            result_dict["drnet_bias_att_in"] = ATT_in
            result_dict["drnet_policy_risk_in"] = RPol_in

            file1.write("\nToday's date: {0}\n".format(date.today()))
            file1.write("Iter: {0}, drnet_bias_att_out: {1}, drnet_policy_risk_out: {2}, "
                        "drnet_bias_att_in: {3}, drnet_policy_risk_in: {4},\n"
                        .format(iter_id, ATT_out,
                                RPol_out, ATT_in, RPol_in))
            results_list.append(result_dict)

        drnet_policy_risk_set_out = []
        drnet_bias_att_set_out = []
        drnet_policy_risk_set_in = []
        drnet_bias_att_set_in = []

        for result in results_list:
            drnet_policy_risk_set_out.append(result["drnet_policy_risk_out"])
            drnet_bias_att_set_out.append(result["drnet_bias_att_out"])
            drnet_policy_risk_set_in.append(result["drnet_policy_risk_in"])
            drnet_bias_att_set_in.append(result["drnet_bias_att_in"])

        drnet_policy_risk_mean_out = np.mean(np.array(drnet_policy_risk_set_out))
        drnet_policy_risk_std_out = np.std(drnet_policy_risk_set_out)
        drnet_bias_att_mean_out = np.mean(np.array(drnet_bias_att_set_out))
        drnet_bias_att_std_out = np.std(drnet_bias_att_set_out)

        drnet_policy_risk_mean_in = np.mean(np.array(drnet_policy_risk_set_in))
        drnet_policy_risk_std_in = np.std(drnet_policy_risk_set_in)
        drnet_bias_att_mean_in = np.mean(np.array(drnet_bias_att_set_in))
        drnet_bias_att_std_in = np.std(drnet_bias_att_set_in)

        print("----------------- !!DR_Net Models(Results) !! ------------------------")
        print("--" * 20)
        print("DR_NET, policy_risk_out: {0}, SD: {1}"
              .format(drnet_policy_risk_mean_out, drnet_policy_risk_std_out))
        print("DR_NET, bias_att_out: {0}, SD: {1}"
              .format(drnet_bias_att_mean_out, drnet_bias_att_std_out))
        print("--" * 20)
        print("DR_NET, policy_risk_in: {0}, SD: {1}"
              .format(drnet_policy_risk_mean_in, drnet_policy_risk_std_in))
        print("DR_NET, bias_att_in: {0}, SD: {1}"
              .format(drnet_bias_att_mean_in, drnet_bias_att_std_in))
        print("--" * 20)

        file1.write("\n#####################")

        file1.write("\n---------------------")
        file1.write("\nDR_NET, policy_risk_out: {0}, SD: {1}"
                    .format(drnet_policy_risk_mean_out, drnet_policy_risk_std_out))
        file1.write("\nDR_NET, bias_att_out: {0}, SD: {1}"
                    .format(drnet_bias_att_mean_out, drnet_bias_att_std_out))

        file1.write("\nDR_NET, policy_risk_in: {0}, SD: {1}"
                    .format(drnet_policy_risk_mean_in, drnet_policy_risk_std_in))
        file1.write("\nDR_NET, bias_att_in: {0}, SD: {1}"
                    .format(drnet_bias_att_mean_in, drnet_bias_att_std_in))

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "MSE/Results_consolidated_final_2QHeads_lamda_10.csv"

            # NN
            run_parameters["nn_prop_file"] = "MSE/NN_Prop_score_{0}.csv"

            run_parameters["DCN_PD"] = "./MSE/ITE/ITE_DCN_PD_iter_{0}.csv"
            run_parameters["DCN_PD_02"] = "./MSE/ITE/ITE_DCN_PD_02_iter_{0}.csv"
            run_parameters["DCN_PD_05"] = "./MSE/ITE/ITE_DCN_PD_05_iter_{0}.csv"

            run_parameters["DCN_PM_GAN"] = "./MSE/ITE/ITE_DCN_PM_GAN_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_02"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_02_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_05"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_05_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_PD"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_PD_iter_{0}.csv"

            # model paths DCN
            run_parameters["Model_DCN_PD_shared"] = "./Models/DCN_PD/DCN_PD_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_y1"] = "./Models/DCN_PD/DCN_PD_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_y0"] = "./Models/DCN_PD/DCN_PD_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PD_02_shared"] = "./Models/DCN_PD_02/DCN_PD_02_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_02_y1"] = "./Models/DCN_PD_02/DCN_PD_02_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_02_y0"] = "./Models/DCN_PD_02/DCN_PD_02_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PD_05_shared"] = "./Models/DCN_PD_05/DCN_PD_05_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_05_y1"] = "./Models/DCN_PD_05/DCN_PD_05_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_05_y0"] = "./Models/DCN_PD_05/DCN_PD_05_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PM_GAN_shared"] = "./Models/PM_GAN/DCN_PM_GAN_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_y1"] = "./Models/PM_GAN/DCN_PM_GAN_iter_y1_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_y0"] = "./Models/PM_GAN/DCN_PM_GAN_iter_y0_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_02_shared"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_02_y1"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_02_y0"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_y0_iter_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_05_shared"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_05_y1"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_05_y0"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_y0_iter_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_PD_shared"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_PD_y1"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_PD_y0"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_y0_iter_{0}.pth"

            run_parameters["TARNET"] = "./MSE/ITE/ITE_TARNET_iter_{0}.csv"

            run_parameters["TARNET_PM_GAN"] = "./MSE/ITE/ITE_TARNET_PM_GAN_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Jobs_Stats.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            # run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated_final_2QHeads_lamda_10.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    def __load_data(self, train_path, test_path, iter_id):
        if self.running_mode == "original_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

        elif self.running_mode == "synthetic_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

    @staticmethod
    def Perf_RPol_ATT(Test_T, Test_Y, y1_hat, y0_hat):
        """
        This function is the central point which controls the different experimental settings

        Input:
            Test_T: Treatment variable of the test set
            Test_Y: True outcome variables of the test set.
                    Note that true y0 and y1 are concatnated in this single vector.

            y1_hat: Predicted y1 by the neural network
            y0_hat: Predicted y1 by the neural network

        Returns:
            RPol: Policy Risk
            ATT: Bias ATT

            Note: This part is taken from the implementation of GANITE (ICLR, 2018).
            Source code: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/68e4f7d13e4368eba655132a73ff9f278da5d3af/alg/ganite/ganite_ana.py
            Paper: https://openreview.net/pdf?id=ByKWUeWA-
        """

        # RPol
        # Decision of Output_Y
        hat_t = np.sign(y1_hat - y0_hat)
        hat_t = (0.5 * (hat_t + 1))
        new_hat_t = np.abs(1 - hat_t)

        # Intersection
        idx1 = hat_t * Test_T
        idx0 = new_hat_t * (1 - Test_T)

        # RPol Computation
        RPol1 = (np.sum(idx1 * Test_Y) / (np.sum(idx1) + 1e-8)) * np.mean(hat_t)
        RPol0 = (np.sum(idx0 * Test_Y) / (np.sum(idx0) + 1e-8)) * np.mean(new_hat_t)
        RPol = 1 - (RPol1 + RPol0)

        # ATT
        # Original ATT
        ATT_value = np.sum(Test_T * Test_Y) / (np.sum(Test_T) + 1e-8) - np.sum((1 - Test_T) * Test_Y) / (
                np.sum(1 - Test_T) + 1e-8)
        # Estimated ATT
        ATT_estimate = np.sum(Test_T * (y1_hat - y0_hat)) / (np.sum(Test_T) + 1e-8)
        # Final ATT
        ATT = np.abs(ATT_value - ATT_estimate)
        return [RPol, ATT]


    # def get_consolidated_file_name(self, ps_model_type):
    #     if ps_model_type == Constants.PS_MODEL_NN:
    #         return "./MSE/Results_consolidated_NN.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR:
    #         return "./MSE/Results_consolidated_LR.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
    #         return "./MSE/Results_consolidated_LR_LAsso.csv"
