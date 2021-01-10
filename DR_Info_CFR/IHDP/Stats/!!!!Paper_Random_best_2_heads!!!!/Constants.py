class Constants:
    # DR_Net
    DRNET_EPOCHS = 50
    DRNET_SS_EPOCHS = 1
    DRNET_LR = 1e-3
    DRNET_LAMBDA = 0.001
    DRNET_BATCH_SIZE = 64
    DRNET_INPUT_NODES = 25
    DRNET_SHARED_NODES = 200
    DRNET_OUTPUT_NODES = 100
    ALPHA = 1
    BETA = 1

    # Adversarial VAE Info GAN
    Adversarial_epochs = 1000
    Adversarial_VAE_LR = 1e-3
    INFO_GAN_G_LR = 1e-4
    INFO_GAN_D_LR = 1e-4
    Adversarial_LAMBDA = 1e-5
    Adversarial_BATCH_SIZE = 64
    VAE_BETA = 1
    INFO_GAN_LAMBDA = 0.2
    INFO_GAN_ALPHA = 1

    Encoder_shared_nodes = 15
    Encoder_x_nodes = 5
    Encoder_t_nodes = 1
    Encoder_yf_nodes = 1
    Encoder_ycf_nodes = 1

    Decoder_in_nodes = Encoder_x_nodes + Encoder_t_nodes + Encoder_yf_nodes + Encoder_ycf_nodes
    Decoder_shared_nodes = 15
    Decoder_out_nodes = DRNET_INPUT_NODES

    Info_GAN_Gen_in_nodes = 100
    Info_GAN_Gen_shared_nodes = 30
    Info_GAN_Gen_out_nodes = 1

    # 25 -> covariates, 1 -> y0, 1-> y1
    Info_GAN_Dis_in_nodes = DRNET_INPUT_NODES + 2
    Info_GAN_Dis_shared_nodes = 30
    Info_GAN_Dis_out_nodes = 1

    Info_GAN_Q_in_nodes = 1
    Info_GAN_Q_shared_nodes = 30
    Info_GAN_Q_out_nodes = Decoder_in_nodes



