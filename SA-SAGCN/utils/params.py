import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=16, type=int, metavar="Batch", help="Model Load Data Batch Size")
    parser.add_argument('--epochs', default=100, type=int, help="Model Epochs")
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('--adj-filename', default="./data/DataBase/adj_mx.csv", type=str)
    parser.add_argument('--id-filename', default="./data/DataBase/id_filename.txt", type=str)
    parser.add_argument('--num-of-vertices', default=330, type=int)
    parser.add_argument('--adj-steps', default=3, type=int)
    parser.add_argument('--num-of-history', default=6, type=int)
    parser.add_argument('--num-of-predict', default=3, type=int)
    parser.add_argument('--activation', default="relu", type=str)
    parser.add_argument('--use-mask', default=True, type=bool)
    parser.add_argument('--spatial-pos-embed', default=True, type=bool)
    parser.add_argument('--temporal-pos-embed', default=True, type=bool)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay (default: 0.05)')

    return parser.parse_args()
