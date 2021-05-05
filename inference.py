import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_class, device):
    model_cls = getattr(import_module("model_multi"), args.model)
    model = model_cls(num_class)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, name, output_dir, args):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # gender
    gender_model_dir = '/opt/ml/pstage1_seperate_train/model/gender_21_224_nofilter'
    gender_model = load_model(gender_model_dir, 2, device).to(device)
    gender_model.eval()
    # mask
    mask_model_dir = '/opt/ml/pstage1_seperate_train/model/mask_21_224_nofilter'
    mask_model = load_model(mask_model_dir, 3, device).to(device)
    mask_model.eval()
    # age
    age_model_dir = '/opt/ml/pstage1_seperate_train/model/age_21_224_nofilter'
    age_model = load_model(age_model_dir, 3, device).to(device)
    age_model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []

    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)

            gender = gender_model(images)
            mask = mask_model(images)
            age = age_model(images)

            mask_preds = torch.argmax(mask, dim=-1)
            gender_preds = torch.argmax(gender, dim=-1)
            age_preds = torch.argmax(age, dim=-1)

            pred = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)
            preds.extend(pred.cpu().numpy())   

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, '{}_output.csv'.format(name)), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='MyModel_seperation', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--name', type=str, default='combination')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir

    name = args.name
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, name, output_dir, args)
