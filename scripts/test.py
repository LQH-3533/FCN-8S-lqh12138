import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from core.data.dataloader import get_segmentation_dataset

# Bayesian optimization (TPE) configuration -- identical for all architectures
BO_EPOCHS = 15          # fixed short-training epochs for each BO trial
FINAL_EPOCHS = 72       # full-training epochs for the best configuration
N_STARTUP_TRIALS = 5    # uniform random initialization trials before TPE
FINAL_VAL_EPOCH = 6     # validation interval during the final training
RESULTS_DIR = os.path.join(root_path, 'scripts', 'bo_results')
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # 模型和数据集参数
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3',
                                 'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet',
                                 'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet',
                                 'lednet', 'dfanet', 'swnet'],
                        help='模型名称 (默认: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                                 'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='骨干网络名称 (默认: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='数据集名称 (默认: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=530, help='基础图像大小')
    parser.add_argument('--crop-size', type=int, default=460, help='裁剪图像大小')
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N', help='数据加载线程数')
    # 训练超参数
    parser.add_argument('--jpu', action='store_true', default=False, help='使用JPU')
    parser.add_argument('--use-ohem', type=bool, default=False, help='使用OHEM损失')
    parser.add_argument('--aux', action='store_true', default=False, help='使用辅助损失')
    parser.add_argument('--aux-weight', type=float, default=0.8, help='辅助损失权重')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='训练批次大小 (默认: 8)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='起始训练轮数 (默认: 0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='训练总轮数 (默认: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='学习率 (默认: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='动量 (默认: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='权重衰减 (默认: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='预热迭代次数')
    parser.add_argument('--warmup-factor', type=float, default=0.1, help='预热学习率因子')
    parser.add_argument('--warmup-method', type=str, default='linear', help='预热方法')
    # CUDA设置
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA训练')
    parser.add_argument('--amp', action='store_true', default=False, help='启用自动混合精度训练')
    parser.add_argument('--local_rank', type=int, default=0)
    # 检查点和日志
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的文件路径')
    parser.add_argument('--save-dir', default='~/.torch/models', help='模型保存目录')
    parser.add_argument('--save-epoch', type=int, default=10, help='每隔多少轮保存一次模型')
    parser.add_argument('--log-dir', default='../runs/logs/', help='日志保存目录')
    parser.add_argument('--log-iter', type=int, default=10, help='每隔多少次迭代打印日志')
    # 验证设置
    parser.add_argument('--val-epoch', type=int, default=2, help='每隔多少轮验证一次')
    parser.add_argument('--skip-val', action='store_true', default=False, help='跳过验证')
    # Optuna 优化设置
    parser.add_argument('--optimize', action='store_true', default=False, help='启用 Optuna 超参数优化')
    parser.add_argument('--n-trials', type=int, default=50, help='Optuna 试验次数')
    args = parser.parse_args()

    # 默认设置
    if args.epochs is None:
        epoches = {'coco': 3, 'pascal_aug': 8, 'pascal_voc': 50, 'pcontext': 8, 'ade20k': 16, 'citys': 12, 'sbu': 16}
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {'coco': 0.004, 'pascal_aug': 0.001, 'pascal_voc': 0.0001, 'pcontext': 0.001, 'ade20k': 0.01, 'citys': 0.01, 'sbu': 0.001}
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


import pandas as pd  # 用于保存数据到 Excel

import pandas as pd  # 用于保存数据到 Excel

class Trainer:
    """训练器类"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 初始化记录指标的列表
        self.metrics_log = []  # 用于保存每次训练和验证的损失、准确率和 mIoU

        # 数据预处理
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch
        # default to a one-epoch linear warmup so that warmup_factor is effective
        if args.warmup_iters <= 0:
            args.warmup_iters = min(args.iters_per_epoch, max(1, args.max_iters // 4))

        # 数据加载器
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler,
                                            num_workers=args.workers, pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler,
                                          num_workers=args.workers, pin_memory=True)

        # 模型初始化
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'恢复训练，加载 {args.resume}...')
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # 损失函数
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # 优化器
        params_list = []
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # 学习率调度器
        self.lr_scheduler = WarmupPolyLR(self.optimizer, max_iters=args.max_iters, power=0.9,
                                         warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        # 分布式训练
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # 评估指标
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        self.best_pixAcc = 0.0
        self.best_mIoU = 0.0
        self.final_train_loss = 0.0

    def train(self):
        """训练模型"""
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info(f'开始训练，总轮数: {epochs} = 总迭代次数: {max_iters}')

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)
        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration += 1
            self.lr_scheduler.step()

            images, targets = images.to(self.device), targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())

            # 减少所有GPU的损失以用于日志记录
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.final_train_loss = losses_reduced.item()

            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 计算剩余时间
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # 记录训练损失
            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(f"Iter: {iteration}/{max_iters} || Lr: {self.optimizer.param_groups[0]['lr']:.6f} || "
                            f"Loss: {losses_reduced.item():.4f} || Cost Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))} || "
                            f"ETA: {eta_string}")
                # 保存训练损失到日志
                self.metrics_log.append({
                    "Epoch": iteration // self.args.iters_per_epoch,
                    "Iteration": iteration,
                    "Loss": losses_reduced.item(),
                    "pixAcc": None,  # 验证时更新
                    "mIoU": None     # 验证时更新
                })

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(iteration)
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(f"总训练时间: {total_training_str} ({total_training_time / max_iters:.4f}s / it)")

        # 保存指标到 Excel
        self.save_metrics_to_excel()

    def validation(self, iteration):
        """验证模型"""
        is_best = False
        self.metric.reset()
        model = self.model.module if self.args.distributed else self.model
        torch.cuda.empty_cache()
        model.eval()

        for i, (image, target, filename) in enumerate(self.val_loader):
            image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.args.amp):
                outputs = model(image)
            self.metric.update(outputs[0].float(), target)
            pixAcc, mIoU = self.metric.get()
            if (i + 1) % 100 == 0 or (i + 1) == len(self.val_loader):
                logger.info(f"样本: {i + 1}/{len(self.val_loader)}, 验证 pixAcc: {pixAcc:.3f}, mIoU: {mIoU:.3f}")

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.best_pixAcc = pixAcc
            self.best_mIoU = mIoU
        save_checkpoint(self.model, self.args, is_best)

        # 更新日志中的验证指标
        self.metrics_log[-1]["pixAcc"] = pixAcc
        self.metrics_log[-1]["mIoU"] = mIoU

        synchronize()

    def save_metrics_to_excel(self):
        """保存指标到 Excel 文件"""
        df = pd.DataFrame(self.metrics_log)  # 将日志转换为 DataFrame
        excel_path = os.path.join(self.args.save_dir, "training_metrics.xlsx")
        df.to_excel(excel_path, index=False)
        logger.info(f"训练指标已保存到 {excel_path}")


def save_checkpoint(model, args, is_best=False):
    """保存检查点"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{args.model}_{args.backbone}_{args.dataset}.pth'
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = f'{args.model}_{args.backbone}_{args.dataset}_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


def objective(trial, args):
    """Optuna objective: identical search space for every architecture."""
    # hyperparameters searched by the Tree-structured Parzen Estimator
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    args.momentum = trial.suggest_float("momentum", 0.5, 0.99)
    args.warmup_factor = trial.suggest_float("warmup_factor", 0.01, 1.0, log=True)

    # each trial is a fixed short training; validation runs only once at the end
    args.epochs = BO_EPOCHS
    args.val_epoch = BO_EPOCHS + 100
    trainer = Trainer(args)
    trainer.train()
    trainer.validation(args.epochs * args.iters_per_epoch)

    # objective to maximize: mean of pixel accuracy and mIoU on the validation set
    return trainer.best_pred


def optimize_hyperparameters(args):
    """Bayesian optimization with the TPE sampler, followed by full retraining."""
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    best = study.best_params
    print("Best hyperparameters:")
    for key, value in best.items():
        print(f"{key}: {value}")
    print(f"Best BO objective during short training: {study.best_value:.4f}")

    # full training with the best configuration
    # persist the complete trial history (optimization trajectory)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trials_csv = os.path.join(RESULTS_DIR, f"{args.model}_trials.csv")
    study.trials_dataframe().to_csv(trials_csv, index=False)

    args.lr = best["lr"]
    args.weight_decay = best["weight_decay"]
    args.momentum = best["momentum"]
    args.warmup_factor = best["warmup_factor"]
    args.epochs = FINAL_EPOCHS
    args.val_epoch = FINAL_VAL_EPOCH
    trainer = Trainer(args)
    trainer.train()
    trainer.validation(FINAL_EPOCHS * args.iters_per_epoch)

    # persist the result of this architecture
    import json
    result = {
        "model": args.model,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "n_startup_trials": N_STARTUP_TRIALS,
        "bo_epochs_per_trial": BO_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "trials_csv": trials_csv,
        "search_space": {"lr": [1e-5, 1e-2], "weight_decay": [1e-5, 1e-2],
                         "momentum": [0.5, 0.99], "warmup_factor": [0.01, 1.0]},
        "best_params": best,
        "bo_best_objective": study.best_value,
        "final_pixAcc": trainer.best_pixAcc,
        "final_mIoU": trainer.best_mIoU,
        "final_train_loss": trainer.final_train_loss,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{args.model}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Result saved to", result_path)
    return result


if __name__ == '__main__':
    args = parse_args()

    # 分布式设置
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    # 日志设置
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename=f'{args.model}_{args.backbone}_{args.dataset}_log.txt')
    logger.info(f"使用 {num_gpus} 个GPU")
    logger.info(args)

    # 如果启用优化，则运行 Optuna 优化
    if args.optimize:
        optimize_hyperparameters(args)
    else:
        # 否则直接训练
        trainer = Trainer(args)
        trainer.train()
    torch.cuda.empty_cache()



