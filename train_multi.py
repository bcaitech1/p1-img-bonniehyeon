import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from sklearn.metrics import f1_score
from adamp import AdamP

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5).astype('int64')
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: 'MaskSplitByProfileDataset'
    dataset = dataset_module(
        data_dir=data_dir,val_ratio=args.val_ratio
    )
    num_classes = dataset.num_classes  # 18
       

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )


    # -- model for multi
    model_module = getattr(import_module("model_multi"), args.model) 
    
    # gender 2, age 3, mask 3
    num_outdim = 3
    model = model_module(num_outdim).to(device)
    
    model = torch.nn.DataParallel(model)
   

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    #opt_module = AdamP
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
        #momentum=0.9
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1_score = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        interval_loss_value = 0
        interval_matches = 0

        total_labels_for_f1 = []
        total_preds_for_f1 = []
        interval_labels_for_f1 = []
        interval_preds_for_f1 = []

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.name == 'mask':
                mask_label, _, _ = MaskBaseDataset.decode_multi_class(labels)            
                labels = mask_label
            elif args.name == 'gender':
                _, gender_label, _ = MaskBaseDataset.decode_multi_class(labels)
                labels = gender_label
            elif args.name == 'age':
                _, _, age_label = MaskBaseDataset.decode_multi_class(labels)
                labels = age_label   

            optimizer.zero_grad()

            outputs = model(inputs)            
            preds = torch.argmax(outputs, dim=-1)                
                       
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()            
            total_labels_for_f1.append(labels.data.cpu())
            total_preds_for_f1.append(preds.cpu())

            interval_loss_value += loss.item()
            interval_matches += (preds == labels).sum().item()
            interval_labels_for_f1.append(labels.data.cpu())
            interval_preds_for_f1.append(preds.cpu())
            if (idx + 1) % args.log_interval == 0:              

                train_loss = interval_loss_value / args.log_interval
                train_acc = interval_matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                
                interval_labels_for_f1 = torch.cat(interval_labels_for_f1,0)
                interval_preds_for_f1 = torch.cat(interval_preds_for_f1,0)
                train_f1_score = f1_score(interval_labels_for_f1,interval_preds_for_f1,average='macro')

                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} ||training f1 score {train_f1_score:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/interval_loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/interval_accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/interval_f1_score", train_f1_score, epoch * len(train_loader) + idx)

                interval_loss_value = 0
                interval_matches = 0      
                interval_labels_for_f1 = []
                interval_preds_for_f1 = []          

        epoch_train_loss = loss_value / len(train_loader)
        epoch_train_acc = matches / len(train_loader.dataset)
        epoch_current_lr = get_lr(optimizer)

        total_labels_for_f1 = torch.cat(total_labels_for_f1,0)
        total_preds_for_f1 = torch.cat(total_preds_for_f1,0)
        epoch_train_f1_score = f1_score(total_labels_for_f1,total_preds_for_f1,average='macro')
        print("*"*18)
        print(
            f"Epoch[{epoch}/{args.epochs}] || "
            f"training loss {epoch_train_loss:4.4} || training accuracy {epoch_train_acc:4.2%} ||training f1 score {epoch_train_f1_score:4.4} || lr {epoch_current_lr}"
        )
        print("*"*18)
        logger.add_scalar("Train/epoch_loss", epoch_train_loss, epoch+1)
        logger.add_scalar("Train/epoch_accuracy", epoch_train_acc, epoch+1)
        logger.add_scalar("Train/epoch_f1_score", epoch_train_f1_score, epoch+1)

        loss_value = 0
        matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            total_labels_for_f1 = []
            total_preds_for_f1 = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if args.name == 'mask':
                    mask_label, _, _ = MaskBaseDataset.decode_multi_class(labels)            
                    labels = mask_label
                elif args.name == 'gender':
                    _, gender_label, _ = MaskBaseDataset.decode_multi_class(labels)
                    labels = gender_label
                elif args.name == 'age':
                    _, _, age_label = MaskBaseDataset.decode_multi_class(labels)
                    labels = age_label   

                outputs = model(inputs)            
                preds = torch.argmax(outputs, dim=-1)                
                       
                loss = criterion(outputs, labels)

                total_labels_for_f1.append(labels.data.cpu())
                total_preds_for_f1.append(preds.cpu())

                loss_item = loss.item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, preds, 16, args.dataset != "MaskSplitByProfileDataset")
                    
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            total_labels_for_f1 = torch.cat(total_labels_for_f1,0)
            total_preds_for_f1 = torch.cat(total_preds_for_f1,0)
            val_f1_score = f1_score(total_labels_for_f1,total_preds_for_f1,average='macro')
            
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_f1_score = val_f1_score
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1 score: {val_f1_score:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1 score: {best_val_f1_score:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch+1)
            logger.add_scalar("Val/accuracy", val_acc, epoch+1)
            logger.add_scalar("Val/f1_score", val_f1_score, epoch+1)
            logger.add_figure("results", figure, epoch+1)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='Album_Augmentations', help='data augmentation type (default: Album_Augmentations)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--model', type=str, default='MyModel_seperation_next', help='model type (default: MyModel_seperation_next)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='ratio for validaton (default: 0.3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='learning rate scheduler deacy step (default: 2)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)