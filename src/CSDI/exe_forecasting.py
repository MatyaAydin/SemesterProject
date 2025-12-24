import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)


if args.datatype == 'ewz_daily':
    target_dim = 59

elif args.datatype == 'electricity_benchmark':
    target_dim = 261

elif args.datatype == 'PEMS08':
    target_dim = 170

else:
    target_dim = 370

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))


foldername = "./save/forecasting_" + args.datatype + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)
print("number of trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
if args.modelfolder == "":
    print("training")
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    print("testing")
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1.,#scaler,
    mean_scaler=0.,#mean_scaler,
    foldername=foldername,
)
