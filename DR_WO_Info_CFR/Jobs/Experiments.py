from collections import OrderedDict
from datetime import date

import numpy as np

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

    def run_all_experiments(self, train_path, test_path, iterations):
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
            input_nodes = run_parameters["input_nodes"]
            np_train_X, np_train_T, np_train_e, np_train_yf, \
            np_test_X, np_test_T, np_test_e, np_test_yf = self.__load_data(train_path,
                                                                           test_path,
                                                                           iter_id)
            print("-----------> !! Supervised Training(DR_NET Models) !!<-----------")
            drnet_manager = DRNet_Manager(input_nodes=Constants.DRNET_INPUT_NODES,
                                          shared_nodes=Constants.DRNET_SHARED_NODES,
                                          outcome_nodes=Constants.DRNET_OUTPUT_NODES,
                                          device=device)

            tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_e, np_train_yf)
            tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_e, np_test_yf)
            _train_parameters = {
                "epochs": Constants.DRNET_EPOCHS,
                "lr": Constants.DRNET_LR,
                "lambda": Constants.DRNET_LAMBDA,
                "batch_size": Constants.DRNET_BATCH_SIZE,
                "shuffle": True,
                "ALPHA": Constants.ALPHA,
                "BETA": Constants.BETA,
                "train_dataset": tensor_train
            }
            drnet_manager.train_DR_NET(_train_parameters, device)
            dr_eval = drnet_manager.test_DR_NET({"tensor_dataset": tensor_test}, device)
            print("---" * 20)
            print("--> Model : DRNet Supervised Training Evaluation, Iter_id: {0}".format(iter_id))
            drnet_ate_pred, drnet_att_pred, drnet_bias_att, drnet_atc_pred, \
            drnet_policy_value, drnet_policy_risk, drnet_err_fact = \
                self.__process_evaluated_metric(
                    dr_eval["y1_hat_list"],
                    dr_eval["y0_hat_list"],
                    dr_eval["yf_list"],
                    dr_eval["e_list"],
                    dr_eval["T_list"])

            print("drnet_bias_att: ", drnet_bias_att)
            print("drnet_policy_risk: ", drnet_policy_risk)

            print("---" * 20)

            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id

            result_dict["drnet_ate_pred"] = drnet_ate_pred
            result_dict["drnet_att_pred"] = drnet_att_pred
            result_dict["drnet_bias_att"] = drnet_bias_att
            result_dict["drnet_atc_pred"] = drnet_atc_pred
            result_dict["drnet_policy_value"] = drnet_policy_value
            result_dict["drnet_policy_risk"] = drnet_policy_risk
            result_dict["drnet_err_fact"] = drnet_err_fact

            file1.write("\nToday's date: {0}\n".format(date.today()))
            file1.write("Iter: {0}, drnet_bias_att: {1}, drnet_policy_risk: {2}, \n"
                        .format(iter_id, drnet_bias_att,
                                drnet_policy_risk))
            results_list.append(result_dict)

        drnet_policy_risk_set = []
        drnet_bias_att_set = []

        for result in results_list:
            drnet_policy_risk_set.append(result["drnet_policy_risk"])
            drnet_bias_att_set.append(result["drnet_bias_att"])

        drnet_policy_risk_mean = np.mean(np.array(drnet_policy_risk_set))
        drnet_policy_risk_std = np.std(drnet_policy_risk_set)
        drnet_bias_att_mean = np.mean(np.array(drnet_bias_att_set))
        drnet_bias_att_std = np.std(drnet_bias_att_set)

        print("----------------- !!DR_Net Models(Results) !! ------------------------")
        print("--" * 20)
        print("DR_NET, policy_risk: {0}, SD: {1}"
              .format(drnet_policy_risk_mean, drnet_policy_risk_std))
        print("DR_NET, bias_att: {0}, SD: {1}"
              .format(drnet_bias_att_mean, drnet_bias_att_std))
        print("--" * 20)

        file1.write("\n#####################")

        file1.write("\n---------------------")
        file1.write("\nDR_NET, policy_risk: {0}, SD: {1}"
                    .format(drnet_policy_risk_mean, drnet_policy_risk_std))
        file1.write("\nDR_NET, bias_att: {0}, SD: {1}"
                    .format(drnet_bias_att_mean, drnet_bias_att_std))

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "MSE/Results_consolidated.csv"

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

            run_parameters["summary_file_name"] = "DR_WO_Info_CFR_Jobs.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            # run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    def __load_data(self, train_path, test_path, iter_id):
        if self.running_mode == "original_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

        elif self.running_mode == "synthetic_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

    @staticmethod
    def cal_policy_val(t, yf, eff_pred):
        #  policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value

    def __process_evaluated_metric(self, y1_hat, y0_hat, y_f, e, T):
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)
        e_np = np.array(e)
        t_np = np.array(T)
        np_y_f = np.array(y_f)

        y1_hat_np_b = 1.0 * (y1_hat_np > 0.5)
        y0_hat_np_b = 1.0 * (y0_hat_np > 0.5)

        err_fact = np.mean(np.abs(y1_hat_np_b - np_y_f))
        att = np.mean(np_y_f[t_np > 0]) - np.mean(np_y_f[(1 - t_np + e_np) > 1])

        eff_pred = y0_hat_np - y1_hat_np
        eff_pred[t_np > 0] = -eff_pred[t_np > 0]

        ate_pred = np.mean(eff_pred[e_np > 0])
        atc_pred = np.mean(eff_pred[(1 - t_np + e_np) > 1])

        att_pred = np.mean(eff_pred[(t_np + e_np) > 1])
        bias_att = np.abs(att_pred - att)

        policy_value = self.cal_policy_val(t_np[e_np > 0], np_y_f[e_np > 0],
                                           eff_pred[e_np > 0])

        print("bias_att: " + str(bias_att))
        print("policy_value: " + str(policy_value))
        print("Risk: " + str(1 - policy_value))
        print("atc_pred: " + str(atc_pred))
        print("att_pred: " + str(att_pred))
        print("err_fact: " + str(err_fact))

        # Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, 1 - policy_value, err_fact

    # def get_consolidated_file_name(self, ps_model_type):
    #     if ps_model_type == Constants.PS_MODEL_NN:
    #         return "./MSE/Results_consolidated_NN.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR:
    #         return "./MSE/Results_consolidated_LR.csv"
    #     elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
    #         return "./MSE/Results_consolidated_LR_LAsso.csv"
