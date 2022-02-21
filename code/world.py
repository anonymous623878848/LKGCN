import multiprocessing
import os
from os.path import join

import torch as t
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
args = parse_args()

ROOT_PATH = "../"
CODE_PATH = join(ROOT_PATH, "code")
DATA_PATH = join(ROOT_PATH, "dataset")
BOARD_PATH = join(ROOT_PATH, "runs")
FILE_PATH = join(ROOT_PATH, "checkpoints")

import sys

sys.path.append(join(CODE_PATH, "sources"))

# if os.path.exists(FILE_PATH):
#     os.mkdirs(FILE_PATH, exist_ok = True)


config = {}
all_datasets = ["lastfm", "gowalla", "yelp2018", "amazon-book", "amazon-cell", "mindreader", "mindreaderKG",
                "mindreaderMulti", "mindreaderPureEn", "ml100k", "ml100kMulti", "ml100kKG"]
all_models = ["mf", "lgn", "lgnKG", "lgnMulti", "pureEntity", "lgnMultiEntity", "lgnMultiAtt"]
config["batch_size"] = args.bpr_batch
config["latent_dim_rec"] = args.recdim
config["lgn_layers"] = args.layer
config["dropout"] = args.dropout
config["keep_prob"] = args.keepprob
config["n_fold"] = args.a_fold
config["test_bach_size"] = args.testbatch
config["multicore"] = args.multicore
config["lr"] = args.lr
config["decay"] = args.decay
config["pretrain"] = args.pretrain
config["a_split"] = False
config["bigdata"] = False
config["lambda"] = args.lbd
config["beta"] = args.beta

# print(config)

GPU = t.cuda.is_available()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.cuda.set_device(0)
# device = t.device("cpu")
CORES = multiprocessing.cpu_count() // 2
# print(CORES)
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_datasets:
    raise NotImplementedError(f"Have not supported dataset {dataset} yet! try: {all_datasets}")

if model_name not in all_models:
    raise NotImplementedError(f"Have not support model:{model_name}, please try: {all_models}")

train_epochs = args.epochs
load = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


logo = r"""

 █████╗ ███╗   ██╗ ██████╗ ███╗   ██╗██╗   ██╗███╗   ███╗ ██████╗ ██╗   ██╗███████╗
██╔══██╗████╗  ██║██╔═══██╗████╗  ██║╚██╗ ██╔╝████╗ ████║██╔═══██╗██║   ██║██╔════╝
███████║██╔██╗ ██║██║   ██║██╔██╗ ██║ ╚████╔╝ ██╔████╔██║██║   ██║██║   ██║███████╗
██╔══██║██║╚██╗██║██║   ██║██║╚██╗██║  ╚██╔╝  ██║╚██╔╝██║██║   ██║██║   ██║╚════██║
██║  ██║██║ ╚████║╚██████╔╝██║ ╚████║   ██║   ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝███████║
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
                                            
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)
#
cprint(f"device: {device}")
