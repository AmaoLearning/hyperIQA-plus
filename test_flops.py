"""Utility to measure FLOPs, parameters, and latency for HyperIQA models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from thop import profile
from torchvision import transforms

import models


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class DynamicFC(nn.Module):
    """Applies a per-sample fully connected layer via grouped 1x1 conv."""

    def forward(self, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch = input_.shape[0]
        input_re = input_.view(1, batch * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = weight.view(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
        bias_re = bias.view(-1)
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=batch)
        return out.view(batch, weight.shape[1], input_.shape[2], input_.shape[3])


def count_dynamic_fc(module: DynamicFC, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], output: torch.Tensor) -> None:
    """Custom FLOPs counter for DynamicFC used by THOP."""

    input_ = inputs[0]
    weight = inputs[1]
    batch = input_.shape[0]
    in_channels = input_.shape[1]
    out_channels = weight.shape[1]
    kernel_ops = weight.shape[3] * weight.shape[4]
    total_ops = batch * in_channels * out_channels * kernel_ops * 2
    module.__flops__ += int(total_ops)


class BaselineTargetHead(nn.Module):
    """Target head that reuses hyper-network-generated weights each forward."""

    def __init__(self):
        super().__init__()
        self.fc1 = DynamicFC()
        self.fc2 = DynamicFC()
        self.fc3 = DynamicFC()
        self.fc4 = DynamicFC()
        self.fc5 = DynamicFC()
        self.activation = nn.Sigmoid()

    def forward(self, target_in_vec: torch.Tensor, paras: Dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.activation(self.fc1(target_in_vec, paras['target_fc1w'], paras['target_fc1b']))
        q = self.activation(self.fc2(q, paras['target_fc2w'], paras['target_fc2b']))
        q = self.activation(self.fc3(q, paras['target_fc3w'], paras['target_fc3b']))
        q = self.fc5(
            self.activation(self.fc4(q, paras['target_fc4w'], paras['target_fc4b'])),
            paras['target_fc5w'],
            paras['target_fc5b']
        )
        return q.squeeze()


class BaselineHyperIQAModel(nn.Module):
    def __init__(self, hyper_net: nn.Module):
        super().__init__()
        self.hyper = hyper_net
        self.target_head = BaselineTargetHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        paras = self.hyper(x)
        return self.target_head(paras['target_in_vec'], paras)


class ResidualHyperIQAModel(nn.Module):
    def __init__(self, hyper_net: nn.Module, res_target_net: nn.Module):
        super().__init__()
        self.hyper = hyper_net
        self.res_target = res_target_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        paras = self.hyper(x)
        return self.res_target(paras['target_in_vec'], paras)


def _unwrap_state_dict(checkpoint: object, key: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    if key is None:
        blob = checkpoint
    else:
        blob = checkpoint.get(key)
    if blob is None:
        return None
    if isinstance(blob, dict) and 'state_dict' in blob:
        return blob['state_dict']
    return blob


def _resolve_model_type(checkpoint: object, requested: str) -> str:
    if requested != 'auto':
        return requested
    if isinstance(checkpoint, dict) and 'hypernet' in checkpoint and 'targetnet' in checkpoint:
        return 'residual'
    return 'baseline'


def build_model(weights: Path, device: torch.device, model_type: str) -> nn.Module:
    checkpoint = torch.load(weights, map_location=device)
    resolved_type = _resolve_model_type(checkpoint, model_type)

    hyper_net = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)

    if resolved_type == 'baseline':
        state = _unwrap_state_dict(checkpoint, 'hypernet')
        if state is None:
            state = _unwrap_state_dict(checkpoint)
        if state is None:
            raise ValueError('Unable to locate HyperNet weights in checkpoint.')
        hyper_net.load_state_dict(state)
        model = BaselineHyperIQAModel(hyper_net)
    else:
        hyper_state = _unwrap_state_dict(checkpoint, 'hypernet')
        target_state = _unwrap_state_dict(checkpoint, 'targetnet')
        if hyper_state is None or target_state is None:
            raise ValueError('Residual checkpoints must contain "hypernet" and "targetnet" keys.')
        hyper_net.load_state_dict(hyper_state)
        layer_dims = [
            hyper_net.target_in_size,
            hyper_net.f1,
            hyper_net.f2,
            hyper_net.f3,
            hyper_net.f4,
            1,
        ]
        res_target = models.resTargetNet(layer_dims).to(device)
        res_target.load_state_dict(target_state)
        model = ResidualHyperIQAModel(hyper_net, res_target)

    return model.to(device).eval()


def load_image(image_path: Path, device: torch.device, resize: int, crop: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ]
    )
    tensor = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    return tensor.to(device)


def measure_latency(model: nn.Module, sample_input: torch.Tensor, warmup: int, runs: int) -> float:
    device_synchronize = torch.cuda.synchronize if sample_input.is_cuda else lambda: None
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        total = 0.0
        for _ in range(runs):
            device_synchronize()
            start = time.perf_counter()
            _ = model(sample_input)
            device_synchronize()
            total += time.perf_counter() - start
    return total / max(1, runs)


def main():
    parser = argparse.ArgumentParser(description='Compute FLOPs and latency for HyperIQA models.')
    parser.add_argument('--weights', type=Path, required=True, help='Path to model weights (.pkl or .pt).')
    parser.add_argument('--image', type=Path, default=Path('./data/reference.JPG'), help='Path to reference image.')
    parser.add_argument('--resize', type=int, default=512, help='Resize the shorter side to this value.')
    parser.add_argument('--crop', type=int, default=224, help='Center crop size before feeding the network.')
    parser.add_argument('--model-type', choices=['baseline', 'residual', 'auto'], default='auto', help='Type of checkpoint to load.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=5, help='Warm-up iterations for latency measurement.')
    parser.add_argument('--runs', type=int, default=20, help='Averaging iterations for latency measurement.')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args.weights, device, args.model_type)
    sample_input = load_image(args.image, device, args.resize, args.crop)

    custom_ops = {DynamicFC: count_dynamic_fc}
    with torch.no_grad():
        flops, params = profile(model, inputs=(sample_input,), custom_ops=custom_ops, verbose=False)

    avg_latency = measure_latency(model, sample_input, args.warmup, args.runs)

    with torch.no_grad():
        prediction = model(sample_input).mean().item()

    print(f'Model type      : {args.model_type}')
    print(f'Prediction      : {prediction:.4f}')
    print(f'FLOPs           : {flops / 1e9:.3f} GFLOPs')
    print(f'Parameters      : {params / 1e6:.3f} M')
    print(f'Average latency : {avg_latency * 1000:.3f} ms')


if __name__ == '__main__':
    main()