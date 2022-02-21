import os
from time import time

import numpy as np
import torch as t
import world
from dataloader import BasicDataset
from model import PairWiseBasicModel
from sklearn.metrics import roc_auc_score
from torch import optim
from tqdm import tqdm
from world import cprint

try:
    from ccpimport import imp_from_filepath
    from os.path import join, dirname

    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss(object):
    def __init__(self, recmodel: PairWiseBasicModel, config: dict):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, regloss = self.model.bpr_loss(users, pos, neg)
        reg_loss = regloss * self.weight_decay
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


class BPRLossMulti(object):
    def __init__(self, recmodel: PairWiseBasicModel, config: dict):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, usersO, posO, negO):
        loss0, regloss0, loss1, regloss1 = self.model.bpr_loss(users, pos, neg, usersO, posO, negO)
        reg_loss = (regloss0 + regloss1) * self.weight_decay
        loss = loss0 + world.config["lambda"] * loss1
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item(), loss0.cpu().item(), loss1.cpu().item()


class BPRLossMultiEntity(object):
    def __init__(self, recmodel: PairWiseBasicModel, config: dict):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, usersO, posO, negO):
        loss0, regloss0, loss1, regloss1, loss2, regloss2 = self.model.bpr_loss(users, pos, neg, usersO, posO, negO)
        reg_loss = (regloss0 + regloss1 + regloss2) * self.weight_decay
        loss = loss0 + world.config["lambda"] * loss1 + world.config["beta"] * loss2
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item(), loss0.cpu().item(), loss1.cpu().item(), loss2.cpu().item()


class BPRLossMultiAtt(object):
    def __init__(self, recmodel: PairWiseBasicModel, config: dict):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, usersO, posO, negO, posEnti, negEnti):
        loss0, regloss0, loss1, regloss1, loss2, regloss2 = self.model.bpr_loss(users, pos, neg, usersO, posO, negO,
                                                                                posEnti, negEnti)
        reg_loss = (regloss0 + regloss1 + regloss2) * self.weight_decay
        loss = loss0 + world.config["lambda"] * loss1 + world.config["beta"] * loss2
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item(), loss0.cpu().item(), loss1.cpu().item(), loss2.cpu().item()


def UniformSampleOriginal(dataset, neg_ratio=1):
    dataset: BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items, dataset.trainDataSize, allPos, neg_ratio)
    else:
        print(f"python sampling")
        S = UniformSampleOriginal_python(dataset)
    return S


def UniformSampleOriginal_python(dataset):
    """
    the original implemetation of bpr sampling in lightGCN
    :param dataset:
    :return:
    """
    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if (len(posForUser) == 0):
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


def UniformSampleMulti(dataset):
    """
    The implementation of multi-task sampling function 
    :param dataset:
    :return:
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    usersO = np.random.choice(dataset.allOtherUsers, user_num)
    allPos = dataset.allPos
    allPosOthers = dataset.allPosOthers
    S0 = []
    S1 = []
    for i, user in enumerate(users):
        posForUser = allPos[user]

        if (len(posForUser) == 0):
            continue

        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S0.append([user, positem, negitem])

    for i, user in enumerate(usersO):
        posForUser = allPosOthers[user]
        if (len(posForUser) == 0):
            continue
        posindex = np.random.randint(0, len(posForUser))
        posother = posForUser[posindex]
        while True:
            negother = np.random.randint(0, dataset.n_others)
            if negother in posForUser:
                continue
            else:
                break
        S1.append([user, posother, negother])
    assert len(S0) == len(S1)

    return np.concatenate((np.array(S0), np.array(S1)), axis=1)


def sample_1hop(dataset):
    """
    This is different sample  strategy that not only sample one negative item, but also sample one hop negative entity
    """
    print(f"generating samplings ... ")
    dataset: BasicDataset
    trainDf = dataset.trainDf
    allPos = dataset.allPos
    allPosOthers = dataset.allPosOthers
    oidf = dataset.OIDf
    S0 = []
    S1 = []
    for uid, iid in tqdm(trainDf[["userId", "itemId"]].values, total=trainDf.shape[0]):
        posForUser = allPos[uid]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S0.append([uid, iid, negitem])
        if (len(allPosOthers[uid]) == 0):
            uoid = np.random.choice(dataset.allOtherUsers)
        else:
            uoid = uid
        while True:
            OthersSet = oidf[oidf["itemId"] == iid]["oId"].values
            posOtherSet = np.intersect1d(OthersSet, allPosOthers[uoid])
            if (len(posOtherSet) > 0):
                #                 print(f"strong sample")
                posOther = np.random.choice(posOtherSet)
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
            if (len(allPosOthers[uoid]) == 0):
                print(f"no sample")
                uoid = np.random.choice(dataset.allOtherUsers)
                continue
            else:
                #                 print(f"weak sample")
                posOther = np.random.choice(allPosOthers[uoid])
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
        S1.append([uoid, posOther, negOther])

    assert len(S0) == len(S1)
    return np.concatenate((np.array(S0), np.array(S1)), axis=1)


def sample4Att(dataset):
    """
    This is different sample  strategy that not only sample one negative item, but also sample one hop negative entity and the entities associated with sampled items
    """
    print(f"generating samplings ... ")
    dataset: BasicDataset
    trainDf = dataset.trainDf
    allPos = dataset.allPos
    allPosOthers = dataset.allPosOthers
    oidf = dataset.OIDf
    S0 = []
    S1 = []
    posEnt = []
    negEnt = []
    for uid, iid in tqdm(trainDf[["userId", "itemId"]].values, total=trainDf.shape[0]):
        posForUser = allPos[uid]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S0.append([uid, iid, negitem])
        negOtherSet = oidf[oidf["itemId"] == negitem]["oId"].values
        if (len(allPosOthers[uid]) == 0):
            uoid = np.random.choice(dataset.allOtherUsers)
        else:
            uoid = uid
        while True:
            OthersSet = oidf[oidf["itemId"] == iid]["oId"].values
            posOtherSet = np.intersect1d(OthersSet, allPosOthers[uoid])
            if (len(posOtherSet) > 0):
                #                 print(f"strong sample")
                posOther = np.random.choice(posOtherSet)
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
            if (len(allPosOthers[uoid]) == 0):
                print(f"no sample")
                uoid = np.random.choice(dataset.allOtherUsers)
                continue
            else:
                #                 print(f"weak sample")
                posOther = np.random.choice(allPosOthers[uoid])
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
        S1.append([uoid, posOther, negOther])
        posEnt.append(list(OthersSet))
        negEnt.append(list(negOtherSet))

    assert len(S0) == len(S1)
    assert len(S0) == len(posEnt)
    return np.concatenate((np.array(S0), np.array(S1)), axis=1), np.array(posEnt), np.array(negEnt)


def sample4AttML(dataset):
    """
    This is different sample  strategy that not only sample one negative item, but also sample one hop negative entity and the entities associated with sampled items(ML100K)
    """
    print(f"generating samplings ... ")
    dataset: BasicDataset
    trainDf = dataset.trainDf
    allPos = dataset.allPos
    allPosOthers = dataset.allPosOthers
    oidf = dataset.IODf
    S0 = []
    S1 = []
    posEnt = []
    negEnt = []
    for uid, iid in tqdm(trainDf[["userId", "itemId"]].values, total=trainDf.shape[0]):
        posForUser = allPos[uid]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S0.append([uid, iid, negitem])
        negOtherSet = oidf[oidf["entityId"] == negitem]["entityId"].values
        if (len(allPosOthers[uid]) == 0):
            uoid = np.random.choice(dataset.allOtherUsers)
        else:
            uoid = uid
        while True:
            OthersSet = oidf[oidf["itemId"] == iid]["entityId"].values
            posOtherSet = np.intersect1d(OthersSet, allPosOthers[uoid])
            if (len(posOtherSet) > 0):
                #                 print(f"strong sample")
                posOther = np.random.choice(posOtherSet)
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
            if (len(allPosOthers[uoid]) == 0):
                print(f"no sample")
                uoid = np.random.choice(dataset.allOtherUsers)
                continue
            else:
                #                 print(f"weak sample")
                posOther = np.random.choice(allPosOthers[uoid])
                negOther = np.random.randint(0, dataset.n_others)
                if negOther in allPosOthers[uoid]:
                    continue
                else:
                    break
        S1.append([uoid, posOther, negOther])
        posEnt.append(list(OthersSet))
        negEnt.append(list(negOtherSet))

    assert len(S0) == len(S1)
    assert len(S0) == len(posEnt)
    return np.concatenate((np.array(S0), np.array(S1)), axis=1), np.array(posEnt), np.array(negEnt)


def set_seed(seed):
    np.random.seed(seed)
    if (t.cuda.is_available()):
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    t.manual_seed(seed)


def getFileName():
    if world.model_name == "mf":
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "lgn":
        file = f"lgn-{world.dataset}-{world.config['lgn_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "lgnKG":
        file = f"lgnKG-{world.dataset}-{world.config['lgn_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "lgnMulti":
        file = f"lgnMulti-{world.dataset}-{world.config['lgn_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "pureEntity":
        file = f"pureEntity-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "lgnMultiEntity":
        file = f"lgnMultiEntity-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == "lgnMultiAtt":
        file = f"lgnMultiAtt-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", world.config["batch_size"])

    if (len(tensors) == 1):
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i: i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i: i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get("indices", False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All input to shuffle must have the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ===================Metrics===================

def RecallPrecision_ATK(test_data, r, k):
    """
    :param test_data: list since different test user should gave different length of pos items shape: (test_batch, k)
    :param r: right predict
    :param k: top - k
    :return:
    """
    right_predict = r[:, :k].sum(1)
    precision_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_predict / recall_n)
    precision = np.sum(right_predict) / precision_n
    return {"recall": recall, "precision": precision}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
