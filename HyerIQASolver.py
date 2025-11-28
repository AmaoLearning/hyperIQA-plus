import torch
import torch.nn.functional as F
from scipy import stats
import numpy as np
import models
import data_loader
import logging
from tqdm.auto import tqdm


EPS = 1e-8


def _pearson_corr(pred, target):
    vx = pred - pred.mean()
    vy = target - target.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx.pow(2).sum() + EPS)) * torch.sqrt((vy.pow(2).sum() + EPS)))
    return corr.clamp(-1.0, 1.0)


def _soft_rank(x, tau):
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    ranks = torch.sigmoid(diff / max(tau, EPS)).sum(dim=1)
    return ranks + 0.5


def _srcc_loss(pred, target, tau):
    rank_pred = _soft_rank(pred, tau)
    rank_target = _soft_rank(target, tau)
    return 1.0 - _pearson_corr(rank_pred, rank_target)


def _plcc_loss(pred, target):
    return 1.0 - _pearson_corr(pred, target)


def _rank_loss(pred, target, margin):
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    target_sign = torch.sign(target_diff)
    mask = target_sign.ne(0)
    if not mask.any():
        return pred.new_tensor(0.0)
    pred_aligned = pred_diff[mask]
    sign = target_sign[mask]
    return F.relu(-pred_aligned * sign + margin).mean()


def _pairwise_rating_loss(pred_a, pred_b, target_a, target_b, sigma):
    pred_a = pred_a.view(-1)
    pred_b = pred_b.view(-1)
    target_a = target_a.view(-1)
    target_b = target_b.view(-1)

    if pred_a.numel() == 0 or pred_b.numel() == 0:
        return pred_a.new_tensor(0.0)

    sigma = max(sigma, EPS)
    sigma_tensor = pred_a.new_tensor(sigma)
    denom = torch.sqrt(pred_a.new_tensor(2.0)) * sigma_tensor

    pred_prob = 0.5 * (1.0 + torch.erf((pred_a - pred_b) / denom))
    target_prob = 0.5 * (1.0 + torch.erf((target_a - target_b) / denom))
    target_prob = target_prob.detach()

    term1 = torch.sqrt(pred_prob * target_prob + EPS)
    term2 = torch.sqrt((1.0 - pred_prob) * (1.0 - target_prob) + EPS)
    loss = 1.0 - term1 - term2
    return loss.mean()


def create_loss_function(config):
    loss_type = getattr(config, 'loss_type', 'l1').lower()
    tau = getattr(config, 'soft_rank_tau', 1.0)
    rank_margin = getattr(config, 'rank_margin', 0.1)
    pair_tau = getattr(config, 'pairwise_tau', 1.0)

    if loss_type == 'pairwise':
        return None

    def loss_fn(pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        if loss_type == 'l1':
            return F.l1_loss(pred_flat, target_flat)
        if loss_type == 'l2':
            return F.mse_loss(pred_flat, target_flat)
        if loss_type == 'plcc':
            return _plcc_loss(pred_flat, target_flat)
        if loss_type == 'srcc':
            return _srcc_loss(pred_flat, target_flat, tau)
        if loss_type == 'rank':
            return _rank_loss(pred_flat, target_flat, rank_margin)
        raise ValueError(f'Unsupported loss type: {loss_type}')

    return loss_fn

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, output_path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.tqdm_mininterval = 0.5
        self.loss_type = getattr(config, 'loss_type', 'l1').lower()
        self.uses_pairwise = self.loss_type == 'pairwise'
        self.pairwise_sigma = getattr(config, 'pairwise_tau', 1.0)

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.loss_fn = create_loss_function(config)

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.test_batch_size, istrain=False)
        self.train_data = train_loader.get_data()
        self.train_data_pair = None
        if self.uses_pairwise:
            pair_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
            self.train_data_pair = pair_loader.get_data()
        self.test_data = test_loader.get_data()

        self.output_path = output_path

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        logging.info('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        epoch_bar = tqdm(
            range(self.epochs),
            desc='Epochs',
            unit='epoch',
            mininterval=self.tqdm_mininterval,
            dynamic_ncols=True
        )
        for t in epoch_bar:
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            train_iter = zip(self.train_data, self.train_data_pair) if self.uses_pairwise else self.train_data
            total_batches = min(len(self.train_data), len(self.train_data_pair)) if self.uses_pairwise else len(self.train_data)

            batch_bar = tqdm(
                train_iter,
                desc=f'Epoch {t + 1} training',
                unit='batch',
                leave=False,
                mininterval=self.tqdm_mininterval,
                dynamic_ncols=True,
                total=total_batches
            )

            for batch in batch_bar:
                if self.uses_pairwise:
                    (img, label), (img_pair, label_pair) = batch
                    img_pair = img_pair.cuda(non_blocking=True)
                    label_pair = label_pair.cuda(non_blocking=True)
                else:
                    img, label = batch
                    img_pair = None
                    label_pair = None

                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred_scores = pred_scores + pred.detach().cpu().tolist()
                gt_scores = gt_scores + label.detach().cpu().tolist()

                if self.uses_pairwise and img_pair is not None:
                    paras_pair = self.model_hyper(img_pair)
                    model_target_pair = models.TargetNet(paras_pair).cuda()
                    for param in model_target_pair.parameters():
                        param.requires_grad = False
                    pred_pair = model_target_pair(paras_pair['target_in_vec'])
                    loss = _pairwise_rating_loss(
                        pred.squeeze(),
                        pred_pair.squeeze(),
                        label.float(),
                        label_pair.float(),
                        self.pairwise_sigma
                    )
                else:
                    loss = self.loss_fn(pred.squeeze(), label.float())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            batch_bar.close()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc       

            epoch_bar.set_postfix({
                'Train_Loss': f'{sum(epoch_loss) / len(epoch_loss):4.3f}',
                'Train_SRCC': f'{train_srcc:4.4f}' if train_srcc is not None else 'nan',
                'Test_SRCC': f'{test_srcc:4.4f}' if test_srcc is not None else 'nan',
                'Test_PLCC': f'{test_plcc:4.4f}' if test_plcc is not None else 'nan'
            })
            logging.info('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f',
                     t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc)

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        logging.info('Best test SRCC %f, PLCC %f', best_srcc, best_plcc)
        
        torch.save(self.model_hyper.state_dict(), self.output_path)
        logging.info(f'Weights of Epoch {t} is saved at: {self.output_path}')

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        data_bar = tqdm(
            data,
            desc='Testing',
            unit='batch',
            mininterval=self.tqdm_mininterval,
            dynamic_ncols=True
        )
        for img, label in data_bar:
            # Data.
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            pred_scores.extend(pred.detach().view(-1).cpu().tolist())
            gt_scores.extend(label.detach().view(-1).cpu().tolist())

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        data_bar.close()
        self.model_hyper.train(True)
        return test_srcc, test_plcc

class resHyperIQASolver(object):
    """Solver that trains HyperNet + residual TargetNet."""

    def __init__(self, config, path, output_path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.tqdm_mininterval = 0.5

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        layer_dims = [
            self.model_hyper.target_in_size,
            self.model_hyper.f1,
            self.model_hyper.f2,
            self.model_hyper.f3,
            self.model_hyper.f4,
            1
        ]
        self.model_res_target = models.resTargetNet(layer_dims).cuda()
        self.model_res_target.train(True)

        self.loss_type = getattr(config, 'loss_type', 'l1').lower()
        self.uses_pairwise = self.loss_type == 'pairwise'
        self.pairwise_sigma = getattr(config, 'pairwise_tau', 1.0)
        self.loss_fn = create_loss_function(config)

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay 
        paras = [
            {'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
            {'params': self.model_hyper.res.parameters(), 'lr': self.lr},
            {'params': self.model_res_target.parameters(), 'lr': self.lr * self.lrratio}
        ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        self.lr_step = 6
        self.lr_gamma = 0.1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, step_size=max(1, self.lr_step), gamma=self.lr_gamma)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.test_batch_size, istrain=False)
        self.train_data = train_loader.get_data()
        self.train_data_pair = None
        if self.uses_pairwise:
            pair_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
            self.train_data_pair = pair_loader.get_data()
        self.test_data = test_loader.get_data()

        self.output_path = output_path

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        logging.info('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        epoch_bar = tqdm(
            range(self.epochs),
            desc='Epochs',
            unit='epoch',
            mininterval=self.tqdm_mininterval,
            dynamic_ncols=True
        )
        for t in epoch_bar:
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            train_iter = zip(self.train_data, self.train_data_pair) if self.uses_pairwise else self.train_data
            total_batches = min(len(self.train_data), len(self.train_data_pair)) if self.uses_pairwise else len(self.train_data)

            batch_bar = tqdm(
                train_iter,
                desc=f'Epoch {t + 1} training',
                unit='batch',
                leave=False,
                mininterval=self.tqdm_mininterval,
                dynamic_ncols=True,
                total=total_batches
            )

            for batch in batch_bar:
                if self.uses_pairwise:
                    (img, label), (img_pair, label_pair) = batch
                    img_pair = img_pair.cuda(non_blocking=True)
                    label_pair = label_pair.cuda(non_blocking=True)
                else:
                    img, label = batch
                    img_pair = None
                    label_pair = None

                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Quality prediction
                pred = self.model_res_target(paras['target_in_vec'], paras)
                pred_scores = pred_scores + pred.detach().cpu().tolist()
                gt_scores = gt_scores + label.detach().cpu().tolist()

                if self.uses_pairwise and img_pair is not None:
                    paras_pair = self.model_hyper(img_pair)
                    pred_pair = self.model_res_target(paras_pair['target_in_vec'], paras_pair)
                    loss = _pairwise_rating_loss(
                        pred.squeeze(),
                        pred_pair.squeeze(),
                        label.float(),
                        label_pair.float(),
                        self.pairwise_sigma
                    )
                else:
                    loss = self.loss_fn(pred.squeeze(), label.float())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            batch_bar.close()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc

            epoch_bar.set_postfix({
                'Train_Loss': f'{sum(epoch_loss) / len(epoch_loss):4.3f}',
                'Train_SRCC': f'{train_srcc:4.4f}' if train_srcc is not None else 'nan',
                'Test_SRCC': f'{test_srcc:4.4f}' if test_srcc is not None else 'nan',
                'Test_PLCC': f'{test_plcc:4.4f}' if test_plcc is not None else 'nan'
            })
            logging.info('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f',
                     t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc)

            self.scheduler.step()

        logging.info('Best test SRCC %f, PLCC %f', best_srcc, best_plcc)
        
        torch.save({
            'hypernet': self.model_hyper.state_dict(),
            'targetnet': self.model_res_target.state_dict()
        }, self.output_path)
        logging.info(f'Weights of Epoch {t} saved at: {self.output_path}')

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        data_bar = tqdm(
            data,
            desc='Testing',
            unit='batch',
            mininterval=self.tqdm_mininterval,
            dynamic_ncols=True
        )
        self.model_res_target.train(False)
        for img, label in data_bar:
            # Data.
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            paras = self.model_hyper(img)
            pred = self.model_res_target(paras['target_in_vec'], paras)

            pred_scores.extend(pred.detach().view(-1).cpu().tolist())
            gt_scores.extend(label.detach().view(-1).cpu().tolist())

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        data_bar.close()
        self.model_res_target.train(True)
        self.model_hyper.train(True)
        return test_srcc, test_plcc