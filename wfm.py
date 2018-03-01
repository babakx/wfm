import sys
sys.path.append('tffm')

import argparse
from data_utils import WfmData
from model import WfmModel

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Program options
    parser = argparse.ArgumentParser(description='Weighted Factorization Machines',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', metavar='ml1m/frappe/kassandra/msd/goodbooks', type=str, default='',
                    help='name of dataset', choices=['ml1m', 'frappe', 'kassandra', 'msd', 'goodbooks', 'msd20'])
    parser.add_argument('--model', metavar='bpr/fm/fmp/wfm/wfmp/wmf', type=str, default='fmp',
                    help='name of the model', choices=['bpr', 'fm', 'fmp', 'wfm', 'wfmp', 'wmf'])
    parser.add_argument('--load_model', metavar='FOLDER_PATH', type=str, help='the path to the model tf model folder')
    parser.add_argument('--save_model', metavar='FOLDER_PATH', type=str, help='folder path to save the tf model')
    parser.add_argument('--epochs', metavar='N', type=int, default=10, help='number of epochs')
    parser.add_argument('--eval_freq', metavar='N', type=int, default=10,
                        help='evaluate every N epochs')
    parser.add_argument('--eval_results', metavar='FILE_PATH', type=str, help='name of the file to save evaluation results to')
    parser.add_argument('--order', metavar='N', type=int, default=2, help='order of FM')
    parser.add_argument('--k', metavar='K', type=int, default=10, help='number of latent factors')
    parser.add_argument('--hyper_params', metavar='LR,REG,STD', type=str, default='0.01,0.01,0.01',
                        help='comma-separated list of hyper-parameters: LearningRate, Regularization, InitStd.')
    parser.add_argument('--batch_size', metavar='N', type=int, default=10000, help='number of samples in each mini-batch')
    parser.add_argument('--has_context', metavar='true/false', type=str2bool, default=True, help='Whether data has context or not')
    parser.add_argument('--implicit', metavar='true/false', type=str2bool, default=True, help='Whether dataset is implicit or explicit')
    parser.add_argument('--weights', metavar='all-one/all-diff/c-one/c-diff', type=str, default='c-dif',
                    help='how the weights should be initialized', choices=['all-one', 'all-diff', 'c-one', 'c-diff'])
    parser.add_argument('--all-conf', metavar='true/false', type=str2bool, default=False, help='Run experiments with pre-defined configurartions')

    args = parser.parse_args()

    if args.dataset == '':
        exit(1)

    print('loading data...')
    data = WfmData(args.dataset, args.weights, args.model, args.has_context, args.implicit)

    lr, reg, init_std = args.hyper_params.split(',')
    model = WfmModel(data, args.model, args.order, args.k, args.batch_size, float(lr), float(init_std), float(reg))

    print('\ntraining...')
    model.train_model(args.epochs, args.eval_freq, args.eval_results)
