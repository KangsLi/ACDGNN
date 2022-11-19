import argparse
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description='DSRADDI model')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                            help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="dev",
                        help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test",
                        help="Name of file containing validation triplets")
    parser.add_argument("--split", "-s", type=str, default="noinductive/data_10",
                        help="Choose one split dataset to train model.{data_0~data_9}")
    parser.add_argument("--dataset", "-d", type=str,default="drugbank",
                            help="Dataset string")
    parser.add_argument("--experiment_name", "-e", type=str, default="saved_model",
          help="A folder with this name would be created to dump saved models and log files")

    parser.add_argument("--emb_dim", "-e_dim", type=int, default=48,
                            help="Dimension of entity")
    parser.add_argument("--gpu_id", "-gpu", type=str, default='0',
      help="which gpu to use")
    parser.add_argument("--rel_dim", "-r_dim", type=int, default=48,
                            help="Dimension of relation")     
    parser.add_argument("--batch_size", "-bs", type=int, default=1024,
                            help="Size of batch") 
    parser.add_argument("--lr", "-lr", type=float, default=0.0005,
                            help="Learning rate") 
    parser.add_argument("--epoch", "-ep", type=int, default=1,
                            help="Training epoch") 
    parser.add_argument("--ffd_drop", "-fd", type=float, default=0.,
                            help="Training dropout")
    parser.add_argument("--attn_drop", "-ad", type=float, default=0.3,
                            help="Training dropout")
    parser.add_argument("--align_weight", "-aw", type=float, default=0.003,
                            help="the loss weight of ddi alignment.")
    parser.add_argument('--layers', help='Number of layers.', default=3, type=int)
    parser.add_argument('--hid',"-hd", help='Number of hidden units per head in each layer.',
                        nargs='*', default=[32, 32], type=int)
    parser.add_argument('--heads','-hs', help='Number of attention heads in each layer.',
                     default=2, type=int)
    parser.add_argument("-fsia", type=int, choices=[0,1], default=1, 
                        help='whether using feature-structure information aggregation')
    parser.add_argument("-cdt", type=int,choices=[0,1], default=1, 
                        help='whether using cross domain transformer')
    return parser.parse_known_args()[0]