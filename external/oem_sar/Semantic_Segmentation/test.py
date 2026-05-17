import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
import argparse
import warnings
from pathlib import Path
import segmentation_models_pytorch as smp
from Semantic_Segmentation import source


warnings.filterwarnings("ignore")


# class palette
class_rgb = {
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}


# class labels
class_gray = {
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}


def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    
    return out


def test_model(args, model, device):
    # path to save predictions
    os.makedirs(args.save_results, exist_ok=True)
    # load test data
    # changed
    test_fns = [f for f in Path(args.data_root).rglob("*.tif") if "/sar_images/" in str(f)]
    
    for fn_img in test_fns:
        img = source.dataset.load_grayscale(fn_img)
        h, w = img.shape[:2]
        power = math.ceil(np.log2(h) / np.log2(2))
        shape = (2 ** power, 2 ** power)
        img = cv2.resize(img, shape)

        # test time augmentation
        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, ::-1].copy())
        imgs.append(img[::-1, :].copy())
        imgs.append(img[::-1, ::-1].copy())

        input_ = torch.cat([TF.to_tensor(x).unsqueeze(0) for x in imgs], dim=0).float().to(device)
        pred = []
        with torch.no_grad():
            msk = model(input_)
            msk = torch.softmax(msk[:, :, ...], dim=1)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1])/4
        pred = pred.argmax(axis=0).astype("uint8")

        y_pr = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        # save image as png
        filename = os.path.splitext(os.path.basename(fn_img))[0]
        # y_pr_rgb = label2rgb(y_pr)
        Image.fromarray(y_pr).save(os.path.join(args.save_results, filename+'.png'))
        print('Processed file:', filename+'.png')
    print("Done!")
    print("Total files processed: ", len(test_fns))
    
    
def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_classes = len(args.classes)+1
    model = smp.Unet(
        classes=n_classes,
        # change channels for 
        in_channels = 1,
        activation=None,
        encoder_weights= "imagenet",
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    # no GPU, so added cpu param
    model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
    model.to(device).eval()
    
    # test model
    test_model(args, model, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', default="dataset/test/")
    parser.add_argument('--pretrained_model', default="pretrained/SAR_Pesudo_u-efficientnet-b4_s0_CELoss.pth") 
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()
    
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
    