import argparse

def generate_args():
    parser = argparse.ArgumentParser(description='Multi-omics translation')
    ################
    # for model
    parser.add_argument('--noise_rate', default=0.2, type=float)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    parser.add_argument('--n_source', default=3000, type=int)

    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    
    parser.add_argument('--knn_smooth', default=True, type=bool)
    parser.add_argument('--peak_threshold', default=0.05, type=float)
    parser.add_argument('--hvg_gene', default=1500, type=float)
    
    parser.add_argument('--adata_path', default='/home/wzk/ST_code/NicheTrans/2024_NicheTrans_upload_data/2023_nm_MISAR_seq', type=str)

    # Training options
    parser.add_argument('--max-epoch', default=20, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--stepsize', default=10, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")

    parser.add_argument('--train-batch', default=32, type=int)
    parser.add_argument('--test-batch', default=32, type=int)

    # Optimization options
    parser.add_argument('--optimizer', default='adam', type=str,
                        help="adam or SGD")
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        help="initial learning rate, use 0.0001")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    # Miscs
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--save-dir', type=str, default='./log', help="manual seed")
    parser.add_argument('--eval-step', type=int, default=1,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = generate_args()