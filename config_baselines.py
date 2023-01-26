import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Saved_path related configs
    parser.add_argument("-d", "--dataset", type=str, default="gowalla", help="Dataset string")
    parser.add_argument("-bm", "--base_model", type=str, default="MGCCF")
    parser.add_argument("-alg", "--algorithm", type=str, default="Finetune")
    parser.add_argument("-graph_path", "--graph_path", type=str, default="./graph_log/",
                        help="data_path to save user_user or item_item self neighbors")
    parser.add_argument("-load_save_path_prefix", "--load_save_path_prefix", type=str, default='./model_log/',
                        help="upper-level folder for saved models")
    parser.add_argument("-log_folder", "--log_folder", type=str, default='.')
    parser.add_argument("-log", "--log_name", type=str, default='')
    parser.add_argument("-log_files", "--log_files", type=str,
                        default='recall_logs', help='saved path for result logging')


    parser.add_argument("-se", "--setting", type=str, default="no_adjust", help="Setting name of saved weights")

    #Data Processing
    #parser.add_argument("-data_split", "--data_split", default='[0.6,3,0.1]',
    #                    help='[a,b,c] a is the portion of first block, c is the portion of the last block, b is the numebr blocks in the middle')
    #parser.add_argument("-n_inc", "--n_inc", default=0, type=int, help='train n incremental blocks consecutively')
    parser.add_argument("-valid_test_ratio", "--valid_test_ratio", type=float, default=0.2)
    parser.add_argument("-test_ratio", "--test_ratio", type=float, default=0.5)
    parser.add_argument("-first_segment_time", "--first_segment_time", type=int, default=13)
    #Taobao2014: 18
    #Taobao2015: 2
    #Foursquare: 4
    #Netflix: 13 (novalid data when set as 11 or 12)
    #Alimama: 25
    parser.add_argument("-last_segment_time", "--last_segment_time", type=int, default=36)
    #Taobao2014: 48
    #Taobao2015: 31
    #Foursquare: 25
    #Netflix: 84
    #Alimama: 33
    # Running experiment
    parser.add_argument("-de", "--device", type=str, default='')
    parser.add_argument("-seed", "--seed", type=int, default=0)


    # model args
    parser.add_argument("-l2", "--l2", type=float, default=0.02, help="weight decay of embeddings")
    parser.add_argument("-dist_embed", "--dist_embed", type=float, default=0.002, help="weight decay of distance term")
    parser.add_argument("-e", "--num_epoch", type=int, default=100)
    parser.add_argument("-min_epoch","--min_epoch", type=int, default=8)
    # # optimizer args (adam by default)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    #Finetune: 1e-3
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    # # GCN hyper-parameters
    parser.add_argument("-act", "--activation", type=str, default='tf.nn.tanh',
                        choices=['tf.nn.relu', 'tf.nn.tanh'])
    parser.add_argument("-emb", "--embedded_dimension", type=str, default='[128, 128, 128]')
                      # [128, 128, 128]
    parser.add_argument("-wg_dimension", "--wg_dimension", type=int, default=64, help='hidden dimension for weight generator')
    parser.add_argument("-wg_act", "--wg_act", type=str, default='tf.nn.relu', help='activation function for weight generator')
    parser.add_argument("-ndrop", "--neighbor_dropout", type=float, default=0.1)
    parser.add_argument("-g", "--gcn_sample", default='[10, 5]', help='the number of sampled [1-hop, 2-hop] neighbours')
    parser.add_argument('-n_neigh', "--num_neigh", type=int, default=5,
                        help='the number of positive neighbors for contrastive loss')
                        #15
    parser.add_argument('-n_self_neigh', "--num_self_neigh", type=int, default=5,
                        help='the number of positive neighbors user_user of item_item_graph')
                        #15
    parser.add_argument('-con_positive', "--con_positive", type=int, default=5,
                        help='the number of positive neighbors used for contrastive loss for users')
    parser.add_argument('-con_negative', "--con_negative", type=int, default=5,
                        help='the number of negative neighbors used for contrastive loss for users')
    ##############################################################################################
    parser.add_argument('-ui_con_positive', "--ui_con_positive", type=int, default=5,
                        help='the number of positive neighbors used for contrastive loss for users')
    parser.add_argument('-con_ratios','--con_ratios',type=str,default='[1,1,1,1,1,1,1]',
                        help = 'the relative ratio of number of neighbor terms for contrastive loss')
    parser.add_argument('-ui_con_negative', "--ui_con_negative", type=int, default=5,
                        help='the number of negative neighbors used for contrastive loss for users')
    parser.add_argument('-iu_con_positive', "--iu_con_positive", type=int, default=5,
                        help='the number of positive neighbors used for contrastive loss for items')
    parser.add_argument('-iu_con_negative', "--iu_con_negative", type=int, default=5,
                        help='the number of negative neighbors used for contrastive loss for items')
    parser.add_argument('-uu_con_positive', "--uu_con_positive", type=int, default=5,
                        help='the number of positive neighbors used for contrastive loss for users')
    parser.add_argument('-uu_con_negative', "--uu_con_negative", type=int, default=5,
                        help='the number of negative neighbors used for contrastive loss for users')
    parser.add_argument('-ii_con_positive', "--ii_con_positive", type=int, default=5,
                        help='the number of positive neighbors used for contrastive loss for users')
    parser.add_argument('-ii_con_negative', "--ii_con_negative", type=int, default=5,
                        help='the number of negative neighbors used for contrastive loss for users')
    parser.add_argument('-include_self', "--include_self", type=int, default=0,
                        help='whether to sample self node as positive neighbors for contrastive ;pss')
    parser.add_argument('-trans_positive', "--trans_positive", type=int, default=10,
                        help='the number of positive neighbors used for calculate user weights using single transform')
    parser.add_argument('-n_cluster_points', "--n_cluster_points", type=int, default=60,
                       help='the number of points selected in each cluster')



    # training args
    parser.add_argument("-b", "--batch_pairs", type=int, default=1000,
                        help='the size of a mini-batch of positive pairs')
                        #5000
    parser.add_argument("-b_eval", "--batch_evaluate", type=int, default=1000)
                        #20000
    parser.add_argument("-neg", "--num_neg", type=int, default=10,
                        help='number of negative paris for each positve pair in bpr loss')
    parser.add_argument('-max', '--max_degree', default=128, type=int, choices=[64, 128, 256],
                        help='fixed number of neighbours for each node')
    parser.add_argument("-patience", "--patience", type=int, default=2)
    parser.add_argument("-nu", "--nu", type=float, default=1)

    # incremental args
    parser.add_argument("-load_cp", "--load_cp", type=str, default='')
    parser.add_argument("-save_cp", "--save_cp", type=str, default='b0_100e')
    parser.add_argument("-emb_a", "--emb_a", type=str, default='', help='used for embedding concat')
    parser.add_argument("-emb_b", "--emb_b", type=str, default='', help='used for embedding concat')
    parser.add_argument("-block", "--block", type=int, default=0)
    parser.add_argument("-train_mode", "--train_mode", type=str, default='sep', help='\'acc\' uses accumulated data until current block,\
                                                                                      \'sep\' uses the current block only for training')
    parser.add_argument("-full_batch", "--full_batch", action='store_true')
    parser.add_argument("-mse", "--lambda_mse", type=float, default=0)  # default=100
    parser.add_argument("-lambda_contrastive", '--lambda_contrastive', type=str, default='[0,0,0]')
    parser.add_argument("-lambda_layer_l2", '--lambda_layer_l2', type=str, default='[0,0,0]')
    parser.add_argument("-lambda_soft", '--lambda_soft', type=float, default=0)
    parser.add_argument("-local_distill", "--lambda_distillation", type=float, default=0)
    parser.add_argument("-local_mode", '--local_mode', type=str, default='local_anchor')
    parser.add_argument("-contrastive_mode", '--contrastive_mode', type=str, default='')
    parser.add_argument("-layer_l2_mode", '--layer_l2_mode', type=int, default=0)
    parser.add_argument("-adaptive_mode", '--adaptive_mode', type=str, default='')
    parser.add_argument("-layer_wise", '--layer_wise', type=int, default=0)
    parser.add_argument("-center_initialize", '--center_initialize', type=int, default=0)
    parser.add_argument("-soft_center", '--soft_center', type=int, default=0)
    parser.add_argument("-global_distill", "--lambda_global_distill", type=float, default=0)
    parser.add_argument("-global_k", "--k_centroids", type=str, default='[50,50]')  # [50, 50]
    parser.add_argument("-global_tau", "--tau", type=float, default=0.5)
    parser.add_argument("-alpha", "--alpha", type=float, default=0)  # [50, 50]
    parser.add_argument("-distill_mode", "--distill_mode", type=str, default='inner_product',
                        help='the distance metric used in graph distillation',
                        choices=['', 'euc', 'inner_product', 'rbf', 'poly'])
    parser.add_argument("-new_node_init", "--new_node_init", type=str, default='', choices=['', '2hop_mean'])

    # Reservoir
    parser.add_argument("-rs", "--reservoir_mode", type=str, default='',
                        choices=['full', 'sliding', 'reservoir_sampling'])
    parser.add_argument("-sliding_ratio", "--sliding_ratio", type=float, default=0)
    parser.add_argument("-replay_ratio", "--replay_ratio", type=float, default=0)
    parser.add_argument('-union_mode', '--union_mode', type=str, default='snu', choices=['uns', 'snu'])
    parser.add_argument('-sampling_mode', '--reservoir_selection', type=str, default='',
                        choices=['', 'uniform', 'boosting_inner_product', 'inverse_deg', 'prop_deg', 'latest',
                                 'boosting_wasserstein', 'mse_distill_score', 'adp_inverse_deg', 'item_embedding'])
    parser.add_argument("-inc_agg", '--inc_agg', type=int, default=0)
    parser.add_argument("-adaptive_reservoir", '--adaptive_reservoir', type=str, default='')

    # irrelevent args, do not need to touch for our experiment setup
    parser.add_argument('-test_split', action='store_true',
                        help='for each data block, split the block into test/valid sets')
    parser.add_argument('-shuffle', action='store_true', help='use shuffled dataset')
    parser.add_argument("-test_mode", "--test_mode", type=str, default='sep',
                        help='specify which block is used as test set, should be an integer')

    parser = parser.parse_args()

    return parser