"""Standalone evaluation script for HyperIQA models."""

import argparse
import datetime
import glob
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

import data_loader
import models


def build_models(config) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], torch.device]:
    """Instantiate inference models once for reuse across checkpoints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyper_model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    residual_model = None
    if config.model_type == 'residual':
        layer_dims = [hyper_model.target_in_size, hyper_model.f1, hyper_model.f2, hyper_model.f3, hyper_model.f4, 1]
        residual_model = models.resTargetNet(layer_dims).to(device)

    hyper_model.eval()
    if residual_model:
        residual_model.eval()
    return hyper_model, residual_model, device


def load_checkpoint_weights(
    hyper_model: torch.nn.Module,
    residual_model: Optional[torch.nn.Module],
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load checkpoint state dicts into pre-built models."""
    logging.info('Loading checkpoint %s', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'hypernet' in checkpoint:
        hyper_model.load_state_dict(checkpoint['hypernet'])
        if residual_model:
            residual_model.load_state_dict(checkpoint['targetnet'])
    else:
        hyper_model.load_state_dict(checkpoint)

    hyper_model.eval()
    if residual_model:
        residual_model.eval()


def collect_checkpoints(model_path: str, prefix: Optional[str]) -> List[str]:
    """Return sorted checkpoint paths from file or directory, optionally filtered by prefix."""
    if os.path.isdir(model_path):
        checkpoints = sorted(
            p for p in glob.glob(os.path.join(model_path, '*.pkl')) if os.path.isfile(p)
        )
        if prefix:
            prefix_token = f'{prefix}_'
            checkpoints = [p for p in checkpoints if os.path.basename(p).startswith(prefix_token)]
    else:
        checkpoints = [model_path]

    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found at {model_path}')
    return checkpoints


def setup_logger(model_name: str, dataset: str) -> None:
    """Configure a dedicated logger for evaluation runs."""
    logdir = os.path.join('.', 'log')
    os.makedirs(logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(logdir, f"test_{model_name}_{dataset}_{ts}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.info('Evaluation logging started -> %s', logfile)


def evaluate_once(
    config,
    dataset_name: str,
    dataset_path: str,
    indices: List[int],
    hyper_model: torch.nn.Module,
    residual_model: Optional[torch.nn.Module],
    device: torch.device,
):
    """Run one evaluation pass over the full dataset and return SRCC/PLCC."""
    loader = data_loader.DataLoader(
        dataset_name,
        dataset_path,
        indices,
        config.patch_size,
        config.test_patch_num,
        batch_size=config.test_batch_size,
        istrain=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    ).get_data()

    pred_scores: List[float] = []
    gt_scores: List[float] = []
    hyper_model.train(False)
    if residual_model:
        residual_model.train(False)

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            paras = hyper_model(img)
            if residual_model:
                pred = residual_model(paras['target_in_vec'], paras)
            else:
                model_target = models.TargetNet(paras).to(device)
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

            pred_scores.extend(pred.detach().view(-1).cpu().tolist())
            gt_scores.extend(label.detach().view(-1).cpu().tolist())

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, config.test_patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, config.test_patch_num)), axis=1)
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    return float(test_srcc), float(test_plcc)


def test(config, datasets: List[str], folder_paths: Dict[str, str], img_indices: Dict[str, List[int]]):
    os.makedirs('./results', exist_ok=True)
    result_path = os.path.join('results', f'{config.model_name}.json')
    all_results = {}
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    checkpoints = collect_checkpoints(config.model_path, getattr(config, 'checkpoint_prefix', None))
    if config.max_checkpoints and config.max_checkpoints > 0:
        original_count = len(checkpoints)
        checkpoints = checkpoints[:config.max_checkpoints]
        if original_count > len(checkpoints):
            logging.info('Limiting checkpoints from %d to %d due to --max_checkpoints.', original_count, len(checkpoints))
    hyper_model, residual_model, device = build_models(config)

    for dataset in datasets:
        if dataset not in folder_paths:
            logging.warning('Dataset %s is not supported; skipping.', dataset)
            continue

        setup_logger(config.model_name, dataset)
        logging.info('Evaluating model %s on %s using %d checkpoints', config.model_name, dataset, len(checkpoints))

        srcc_scores = []
        plcc_scores = []
        checkpoint_records = []
        indices = img_indices[dataset]
        logging.info('Using entire dataset (%d samples) for testing.', len(indices))

        for idx, checkpoint_path in enumerate(checkpoints, start=1):
            load_checkpoint_weights(hyper_model, residual_model, checkpoint_path, device)
            srcc, plcc = evaluate_once(
                config,
                dataset,
                folder_paths[dataset],
                indices,
                hyper_model,
                residual_model,
                device,
            )
            srcc_scores.append(srcc)
            plcc_scores.append(plcc)
            checkpoint_records.append({
                'checkpoint': os.path.basename(checkpoint_path),
                'srcc': srcc,
                'plcc': plcc,
            })
            logging.info(
                'Checkpoint %d/%d (%s) -> SRCC %.4f, PLCC %.4f',
                idx,
                len(checkpoints),
                os.path.basename(checkpoint_path),
                srcc,
                plcc,
            )

        dataset_result = {
            'srcc': srcc_scores,
            'plcc': plcc_scores,
            'checkpoints': checkpoint_records,
            'srcc_mean': float(np.mean(srcc_scores)) if srcc_scores else None,
            'srcc_median': float(np.median(srcc_scores)) if srcc_scores else None,
            'plcc_mean': float(np.mean(plcc_scores)) if plcc_scores else None,
            'plcc_median': float(np.median(plcc_scores)) if plcc_scores else None
        }
        all_results[dataset] = dataset_result

        logging.info('Dataset %s summary -> mean SRCC %.4f, median SRCC %.4f, mean PLCC %.4f, median PLCC %.4f',
                     dataset,
                     dataset_result['srcc_mean'] or float('nan'),
                     dataset_result['srcc_median'] or float('nan'),
                     dataset_result['plcc_mean'] or float('nan'),
                     dataset_result['plcc_median'] or float('nan'))

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    logging.info('Saved evaluation results to %s', result_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate HyperIQA models on specified datasets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a checkpoint file or directory of checkpoints')
    parser.add_argument('--model_name', type=str, default=None, help='Identifier for logs/results filenames')
    parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'residual'])
    parser.add_argument('--datasets', nargs='+', default=['livec'], help='Datasets to evaluate: koniq-10k | spaq | kadid | agiqa')
    parser.add_argument('--max_checkpoints', '--test_runs', dest='max_checkpoints', type=int, default=0,
                        help='Maximum number of checkpoints to evaluate (0 means all found)')
    parser.add_argument('--train_patch_num', type=int, default=25)
    parser.add_argument('--test_patch_num', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_ratio', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=16)

    config = parser.parse_args()
    user_provided_model_name = bool(config.model_name)
    if not config.model_name:
        if os.path.isdir(config.model_path):
            config.model_name = os.path.basename(os.path.normpath(config.model_path))
        else:
            config.model_name = os.path.splitext(os.path.basename(config.model_path))[0]
    config.checkpoint_prefix = config.model_name if user_provided_model_name else None

    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': '/root/datasets/dip/koniq_test/',
        'bid': '/home/ssl/Database/BID/',
        'spaq': '/root/datasets/dip/spaq_test/',
        'kadid': '/root/datasets/dip/kadid_test/',
        'agiqa': '/root/datasets/dip/agiqa_test/'
    }

    img_num = {
        'live': list(range(29)),
        'csiq': list(range(30)),
        'tid2013': list(range(25)),
        'livec': list(range(1162)),
        'koniq-10k': list(range(2010)),
        'bid': list(range(586)),
        'spaq': list(range(2224)),
        'kadid': list(range(2000)),
        'agiqa': list(range(2982))
    }

    test(config, config.datasets, folder_path, img_num)


if __name__ == '__main__':
    main()

