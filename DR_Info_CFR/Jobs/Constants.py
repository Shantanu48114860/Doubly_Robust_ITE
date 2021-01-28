class Constants:
    # DR_Net
    DRNET_EPOCHS = 75
    DRNET_SS_EPOCHS = 1
    DRNET_LR = 1e-4
    DRNET_LAMBDA = 0.0001
    DRNET_BATCH_SIZE = 64
    DRNET_INPUT_NODES = 17
    DRNET_SHARED_NODES = 200
    DRNET_OUTPUT_NODES = 100
    ALPHA = 1
    BETA = 1

    # Adversarial VAE Info GAN
    Adversarial_epochs = 1000
    Adversarial_VAE_LR = 1e-4
    INFO_GAN_G_LR = 1e-4
    INFO_GAN_D_LR = 1e-4
    Adversarial_LAMBDA = 1e-5
    Adversarial_BATCH_SIZE = 256
    VAE_BETA = 1
    # INFO_GAN_LAMBDA = 1
    INFO_GAN_LAMBDA = 10
    INFO_GAN_ALPHA = 1

    Encoder_x_shared_nodes = 200
    Encoder_t_shared_nodes = 200
    Encoder_yf_shared_nodes = 200
    Encoder_ycf_shared_nodes = 200

    Encoder_x_nodes = 10
    Encoder_t_nodes = 5
    Encoder_yf_nodes = 5
    Encoder_ycf_nodes = 5

    Decoder_in_nodes = Encoder_x_nodes + Encoder_t_nodes + Encoder_yf_nodes + Encoder_ycf_nodes
    Decoder_shared_nodes = Decoder_in_nodes
    Decoder_out_nodes = DRNET_INPUT_NODES

    Info_GAN_Gen_in_nodes = 200
    Info_GAN_Gen_shared_nodes = 200
    Info_GAN_Gen_out_nodes = 1

    # 25 -> covariates, 1 -> y0, 1-> y1
    Info_GAN_Dis_in_nodes = DRNET_INPUT_NODES + 2
    Info_GAN_Dis_shared_nodes = 200
    Info_GAN_Dis_out_nodes = 1

    Info_GAN_Q_in_nodes = 2
    Info_GAN_Q_shared_nodes = 200
    Info_GAN_Q_out_nodes = Decoder_in_nodes
