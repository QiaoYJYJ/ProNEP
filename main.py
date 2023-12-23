from models import ProNEP
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import CNEDATA
from torch.utils.data import DataLoader
from train1 import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    train_path = os.path.join(dataFolder, 'seq_train10.csv')
    val_path = os.path.join(dataFolder, "seq_val10.csv")
    test_path = os.path.join(dataFolder, "seq_test10.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    max_length_NLR = 1279
    max_length_eff = 1759

    train_dataset = CNEDATA(df_train.index.values, df_train, max_length_NLR, max_length_eff)
    val_dataset = CNEDATA(df_val.index.values, df_val, max_length_NLR, max_length_eff)
    test_dataset = CNEDATA(df_test.index.values, df_test, max_length_NLR, max_length_eff)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
    
    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = ProNEP(**cfg).to(device)

    # if cfg.DA.USE:
    #     if cfg["DA"]["RANDOM_LAYER"]:
    #         domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
    #     else:
    #         domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
    #                                    n_class=cfg["DECODER"]["BINARY"]).to(device)
    #     opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    #     opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    # else:
    #     opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    #
    # torch.backends.cudnn.benchmark = True
    #
    # if not cfg.DA.USE:
    #     trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
    #                       discriminator=None,
    #                       experiment=experiment, **cfg)
    # else:
    #     trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
    #                       discriminator=domain_dmm,
    #                       experiment=experiment, **cfg)
    # result = trainer.train()

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    # 初始化 Trainer 对象
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                      discriminator=None, experiment=experiment, **cfg)

    # 执行训练
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
    result = trainer.train()
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
