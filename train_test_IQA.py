import os
import argparse
import random
import numpy as np
import logging
import datetime
from HyerIQASolver import HyperIQASolver, resHyperIQASolver


def setup_logger(dataset: str, model_name: str, loss_name: str) -> str:
    """Configure root logger to write INFO logs to console and a file in ./log.
    Log filename is formatted as <dataset>_YYYYMMDD_HHMMSS.log
    Returns the path to the logfile."""
    logdir = os.path.join('.', 'log')
    os.makedirs(logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(logdir, f"{model_name}_{loss_name}_{dataset}_{ts}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs when re-running
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.info('Logging started. Logfile: %s', logfile)
    return logfile




def main(config):

    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': '/root/datasets/dip/koniq_train/',
        'bid': '/home/ssl/Database/BID/',
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 7046)),
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    setup_logger(config.dataset, config.model_name, config.loss_type)
    os.makedirs(config.model_output_path, exist_ok=True)
    logging.info('Training and testing on %s dataset for %d rounds...', config.dataset, config.train_test_num)
    for i in range(config.train_test_num):
        logging.info('Round %d', (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = os.path.join(config.model_output_path, f'{config.model_name}_{config.loss_type}_{ts}.pkl')

        if config.model_type == 'residual':
            solver = resHyperIQASolver(config, folder_path[config.dataset], model_path, train_index, test_index)
        elif config.model_type == 'baseline':
            solver = HyperIQASolver(config, folder_path[config.dataset], model_path, train_index, test_index)
        else:
            logging.debug(f'model type {config.model_type} is not impelemented!')
            raise NotImplementedError(f'model type {config.model_type} is not impelemented!') 
        srcc_all[i], plcc_all[i] = solver.train()

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    logging.info('Testing median SRCC %4.4f,\tmedian PLCC %4.4f', srcc_med, plcc_med)

    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--model_name', dest='model_name', type=str, default='hyperIQA_baseline', help='Name of the model when saving its weights')
    parser.add_argument('--model_type', dest='model_type', type=str, default='baseline', help='Type of the model such as baseline | residual')
    parser.add_argument('--loss_type', dest='loss_type', type=str, default='l1', choices=['l1', 'l2', 'srcc', 'plcc', 'rank', 'pairwise'], help='Training loss to optimize')
    parser.add_argument('--soft_rank_tau', dest='soft_rank_tau', type=float, default=1.0, help='Temperature for soft ranking in SRCC loss')
    parser.add_argument('--rank_margin', dest='rank_margin', type=float, default=0.1, help='Margin used by rank loss')
    parser.add_argument('--pairwise_tau', dest='pairwise_tau', type=float, default=1.0, help='Sigma used by the rating-style pairwise fidelity loss')
    parser.add_argument('--model_output_path', dest='model_output_path', type=str, default='./checkpoints/', help='Folder where we save the best model weights')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=1, help='Batch size used during evaluation')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    config.loss_type = config.loss_type.lower()
    main(config)

