import time
from os.path import join

import procedure as Procedure
import torch
import utils
import world
from tensorboardX import SummaryWriter
from world import cprint

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register as register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLossMultiAtt(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.load:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH,
             time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment + "-" + str(world.config["lambda"]))
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    #     S = utils.sample_1hop(dataset)
    max_recall = 0
    max_ndcg = 0
    for epoch in range(world.train_epochs):
        start = time.time()
        if epoch % 1 == 0:
            result_T = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            result_V = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'])
            cprint("[VALID]")
            print(result_V)
            cprint("[TEST]")
            print(result_T)
        output_information = Procedure.BPR_train_multiAtt(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.train_epochs}] {output_information}')
#     torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()

# python main.py --decay=1e-4 --lr=0.0001 --layer=3 --seed=2020 --dataset="mindreaderMulti" --topks="[20]" --recdim=64 --model="lgnMultiAtt" --testbatch=256 --bpr_batch=256 --epochs=100 --lbd=0.5 --beta=0.5 --comment="Att" --tensorboard=0
