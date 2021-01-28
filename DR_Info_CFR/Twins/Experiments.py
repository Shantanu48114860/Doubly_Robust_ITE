from collections import OrderedDict
from datetime import date

import numpy as np

from Adversarial_Manager import Adversarial_Manager
from Constants import Constants
from DR_Net_Manager import DRNet_Manager
from Metrics import Metrics
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def __init__(self, running_mode):
        self.dL = DataLoader()
        self.running_mode = running_mode
        self.np_train = None
        self.np_test = None

    def run_all_experiments(self, iterations):
        split_size = 0.8
        csv_path = "Dataset/Twin_data.csv"
        print("iterations: ", iterations)
        print("split_size: ", split_size)
        device = Utils.get_device()
        print(device)
        results_list = []
        run_parameters = self.__get_run_parameters()
        print(str(run_parameters["summary_file_name"]))
        file1 = open(run_parameters["summary_file_name"], "w")
        for iter_id in range(iterations):
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)

            np_train_X, np_train_T, np_train_yf, np_train_ycf, \
            np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total = \
                self.dL.load_train_test_twins_random(csv_path,
                                                     split_size)

            print("-----------> !! Supervised Training(DR_NET Models) !!<-----------")

            tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_train_ycf)

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

            tensor_train_dr = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_y_cf)
            tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_yf, np_test_ycf)
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
            drnet_manager = DRNet_Manager(input_nodes=Constants.DRNET_INPUT_NODES,
                                          shared_nodes=Constants.DRNET_SHARED_NODES,
                                          outcome_nodes=Constants.DRNET_OUTPUT_NODES,
                                          device=device)
            drnet_manager.train_DR_NET(_dr_train_parameters, device)
            dr_eval_out = drnet_manager.test_DR_NET({"tensor_dataset": tensor_test}, device)
            print("---" * 20)
            print("--> Model : DRNet Supervised Training Evaluation, Iter_id: {0}".format(iter_id))
            drnet_PEHE_out, drnet_ATE_metric_out = \
                self.__process_evaluated_metric(
                    dr_eval_out["y1_hat_list"],
                    dr_eval_out["y0_hat_list"],
                    dr_eval_out["y1_true_list"],
                    dr_eval_out["y0_true_list"])
            print("drnet_PEHE_out: ", drnet_PEHE_out)
            print("drnet_ATE_metric_out: ", drnet_ATE_metric_out)

            dr_eval_in = drnet_manager.test_DR_NET({"tensor_dataset": tensor_train}, device)
            print("---" * 20)
            drnet_PEHE_in, drnet_ATE_metric_in = \
                self.__process_evaluated_metric(
                    dr_eval_in["y1_hat_list"],
                    dr_eval_in["y0_hat_list"],
                    dr_eval_in["y1_true_list"],
                    dr_eval_in["y0_true_list"])
            print("drnet_PEHE_in: ", drnet_PEHE_in)
            print("drnet_ATE_metric_in: ", drnet_ATE_metric_in)

            print("---" * 20)

            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id

            result_dict["drnet_PEHE_out"] = drnet_PEHE_out
            result_dict["drnet_ATE_metric_out"] = drnet_ATE_metric_out
            result_dict["drnet_PEHE_in"] = drnet_PEHE_in
            result_dict["drnet_ATE_metric_in"] = drnet_ATE_metric_in

            file1.write("\nToday's date: {0}\n".format(date.today()))
            file1.write("Iter: {0}, PEHE_DR_NET_out: {1}, ATE_DR_NET_out: {2}, "
                        "PEHE_DR_NET_in: {3}, ATE_DR_NET_in: {4}, \n"
                        .format(iter_id, drnet_PEHE_out,
                                drnet_ATE_metric_out,
                                drnet_PEHE_in,
                                drnet_ATE_metric_in))
            results_list.append(result_dict)

        PEHE_set_drnet_out = []
        ATE_Metric_set_drnet_out = []
        PEHE_set_drnet_in = []
        ATE_Metric_set_drnet_in = []

        for result in results_list:
            PEHE_set_drnet_out.append(result["drnet_PEHE_out"])
            ATE_Metric_set_drnet_out.append(result["drnet_ATE_metric_out"])
            PEHE_set_drnet_in.append(result["drnet_PEHE_in"])
            ATE_Metric_set_drnet_in.append(result["drnet_ATE_metric_in"])

        PEHE_set_drnet_mean_out = np.mean(np.array(PEHE_set_drnet_out))
        PEHE_set_drnet_std_out = np.std(PEHE_set_drnet_out)
        ATE_Metric_set_drnet_mean_out = np.mean(np.array(ATE_Metric_set_drnet_out))
        ATE_Metric_set_drnet_std_out = np.std(ATE_Metric_set_drnet_out)

        PEHE_set_drnet_mean_in = np.mean(np.array(PEHE_set_drnet_in))
        PEHE_set_drnet_std_in = np.std(PEHE_set_drnet_in)
        ATE_Metric_set_drnet_mean_in = np.mean(np.array(ATE_Metric_set_drnet_in))
        ATE_Metric_set_drnet_std_in = np.std(ATE_Metric_set_drnet_in)

        print("----------------- !!DR_Net Models(Results) !! ------------------------")
        print("--" * 20)
        print("DR_NET, PEHE_out: {0}, SD: {1}"
              .format(PEHE_set_drnet_mean_out, PEHE_set_drnet_std_out))
        print("DR_NET, ATE Metric_out: {0}, SD: {1}"
              .format(ATE_Metric_set_drnet_mean_out, ATE_Metric_set_drnet_std_out))
        print("--" * 20)
        print("DR_NET, PEHE_in: {0}, SD: {1}"
              .format(PEHE_set_drnet_mean_in, PEHE_set_drnet_std_in))
        print("DR_NET, ATE Metric_in: {0}, SD: {1}"
              .format(ATE_Metric_set_drnet_mean_in, ATE_Metric_set_drnet_std_in))
        print("--" * 20)

        file1.write("\n#####################")

        file1.write("\n---------------------")
        file1.write("\nDR_NET, PEHE_out: {0}, SD: {1}"
                    .format(PEHE_set_drnet_mean_out, PEHE_set_drnet_std_out))
        file1.write("\nDR_NET, ATE Metric_out: {0}, SD: {1}"
                    .format(ATE_Metric_set_drnet_mean_out,
                            ATE_Metric_set_drnet_std_out))

        file1.write("\nDR_NET, PEHE_in: {0}, SD: {1}"
                    .format(PEHE_set_drnet_mean_in, PEHE_set_drnet_std_in))
        file1.write("\nDR_NET, ATE Metric_in: {0}, SD: {1}"
                    .format(ATE_Metric_set_drnet_mean_in,
                            ATE_Metric_set_drnet_std_in))

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"

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

            run_parameters["summary_file_name"] = "Twins_Stats.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            # run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def __process_evaluated_metric(y1_hat, y0_hat, y1_true, y0_true):
        y1_true_np = np.array(y1_true)
        y0_true_np = np.array(y0_true)
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)

        PEHE = Metrics.PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        ATE = Metrics.ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        print("PEHE: {0}".format(PEHE))
        print("ATE: {0}".format(ATE))
        # print(auc)

        # Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return PEHE, ATE

    # def get_consolidated_file_name(self, ps_model_type):
    #     if ps_model_type == Constants.PS_MODEL_NN:
    #         return "./MSE/Results_consolidated_NN.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR:
    #         return "./MSE/Results_consolidated_LR.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
    #         return "./MSE/Results_consolidated_LR_LAsso.csv"
