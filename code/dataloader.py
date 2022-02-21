import os
import warnings
from os.path import join
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import world
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from world import cprint

warnings.filterwarnings(action="ignore")


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch
    Incldue graph information
    LastFM dataset
    """

    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")

        # (users,users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
                                    shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data,
                                                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../dataset/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class AmazonLoader(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/amazon-cell", dataname="overlapCell.csv"):
        self.file_path = os.path.join(path, dataname)
        self.path = path
        cprint(f"loading [{self.file_path}]")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        df = pd.read_csv(self.file_path, index_col=0)
        df = df.reset_index().drop(columns=["index"])
        # reencode
        ule = preprocessing.LabelEncoder()
        ule.fit(df["reviewerID"].values)
        df["userId"] = ule.transform(df["reviewerID"].values)
        ile = preprocessing.LabelEncoder()
        ile.fit(df["asin"].values)
        df["itemId"] = ile.transform(df["asin"].values)
        simpleDf = df.drop(columns=["reviewerID", "asin"])
        # train test split
        self.num_user, self.num_item = simpleDf["userId"].nunique(), simpleDf["itemId"].nunique()
        self.trainDf, self.testDf = train_test_split(simpleDf, test_size=0.2)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
                adjMat = adjMat.tolil()
                R = self.UserItemNet.tolil()
                adjMat[:self.num_user, self.num_user:] = R
                adjMat[self.num_user:, :self.num_user] = R.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                sp.save_npz(self.path + "/s_pre_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class MindReader(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/mindreader"):
        self.rating_file = os.path.join(path, "ratings.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.validDataSize = self.validDf.shape[0]
        self.validUser = self.validDf["userId"].values
        self.validItem = self.validDf["itemId"].values
        self.validUniqueUser = self.validDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        ratingsDf = pd.read_csv(self.rating_file, index_col=0)
        ratingsDf = ratingsDf.dropna()
        uiDf = ratingsDf[ratingsDf["isItem"] == True]
        uiDf = uiDf.reset_index().drop(columns=["index"])
        uiDf["userId"] = uiDf["userId"].astype(str)
        uiDf["uri"] = uiDf["uri"].astype(str)
        # reencode
        ule = preprocessing.LabelEncoder()
        ule.fit(uiDf["userId"].values)
        uiDf["userId"] = ule.transform(uiDf["userId"].values)
        ile = preprocessing.LabelEncoder()
        ile.fit(uiDf["uri"].values)
        uiDf["itemId"] = ile.transform(uiDf["uri"].values)
        simpleDf = uiDf.drop(columns=["uri", "isItem"])
        # train test split
        self.num_user, self.num_item = simpleDf["userId"].nunique(), simpleDf["itemId"].nunique()
        self.trainDf, self.restDf = train_test_split(simpleDf, test_size=0.2, random_state=2020)
        self.validDf, self.testDf = train_test_split(self.restDf, test_size=0.5, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
                adjMat = adjMat.tolil()
                R = self.UserItemNet.tolil()
                adjMat[:self.num_user, self.num_user:] = R
                adjMat[self.num_user:, :self.num_user] = R.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                sp.save_npz(self.path + "/s_pre_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]

        return valid_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class ML100K(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/ml-100k"):
        self.rating_file = os.path.join(path, "ui.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.validDataSize = self.validDf.shape[0]
        self.validUser = self.validDf["userId"].values
        self.validItem = self.validDf["itemId"].values
        self.validUniqueUser = self.validDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        self.ratingsDf = pd.read_csv(self.rating_file, names=["userId", "itemId", "rating", "timestamp"])
        self.ratingsDf["userId"] = self.ratingsDf["userId"].apply(lambda x: x - 1)
        self.ratingsDf["itemId"] = self.ratingsDf["itemId"].apply(lambda x: x - 1)

        self.num_user = self.ratingsDf["userId"].nunique()
        self.num_item = self.ratingsDf["itemId"].nunique()

        # train test split
        self.trainDf, self.restDf = train_test_split(self.ratingsDf, test_size=0.2, random_state=2020)
        self.validDf, self.testDf = train_test_split(self.restDf, test_size=0.5, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
                adjMat = adjMat.tolil()
                R = self.UserItemNet.tolil()
                adjMat[:self.num_user, self.num_user:] = R
                adjMat[self.num_user:, :self.num_user] = R.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                #                 sp.save_npz(self.path + "/s_pre_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]

        return valid_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class MindReaderKG(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/mindreader"):
        self.rating_file = os.path.join(path, "ratings.csv")
        self.triple_file = os.path.join(path, "triples.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]...loading [{self.triple_file}]...")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")
        # UI net
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # UO net
        self.UserOtherNet = csr_matrix((np.ones(len(self.uoDf)), (self.uoDf["userId"], self.uoDf["oId"])),
                                       shape=(self.num_user, self.num_other))
        # IO net
        self.ItemOtherNet = csr_matrix((np.ones(len(self.IODf)), (self.OIDf["itemId"].values, self.OIDf["oId"].values)),
                                       shape=(self.num_item, self.num_other))
        # II net
        self.ItemItemNet = csr_matrix(
            (np.ones(len(self.IIDf)), (self.IIDf["itemId_head"].values, self.IIDf["itemId_tail"].values)),
            shape=(self.num_item, self.num_item))
        # OO net
        self.OtherOtherNet = csr_matrix(
            (np.ones(len(self.OODf)), (self.OODf["oId_head"].values, self.OODf["oId_tail"].values)),
            shape=(self.num_other, self.num_other))
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        ratingsDf = pd.read_csv(self.rating_file, index_col=0)
        tripleDf = pd.read_csv(self.triple_file, index_col=0)
        tripleDf = tripleDf.dropna()
        ratingsDf = ratingsDf.dropna()
        ratingsDf["userId"] = ratingsDf["userId"].astype(str)
        ratingsDf["uri"] = ratingsDf["uri"].astype(str)
        tripleDf["head_uri"] = tripleDf["head_uri"].astype(str)
        tripleDf["tail_uri"] = tripleDf["tail_uri"].astype(str)
        self.uiDf = ratingsDf[ratingsDf["isItem"] == True]
        self.uoDf = ratingsDf[ratingsDf["isItem"] == False]
        # make sure ther is no interaction between items id and other entities id
        assert len(np.intersect1d(self.uiDf["uri"].unique(), self.uoDf["uri"].unique())) == 0

        # encoding
        leU = LabelEncoder()
        leU.fit(ratingsDf["userId"].values)
        self.uiDf["userId"] = leU.transform(self.uiDf["userId"].values)
        leI = LabelEncoder()
        leI.fit(self.uiDf["uri"])
        self.uiDf["itemId"] = leI.transform(self.uiDf["uri"])
        leO = LabelEncoder()
        leO.fit(self.uoDf["uri"])
        self.uoDf["oId"] = leO.transform(self.uoDf["uri"])
        self.uoDf["userId"] = leU.transform(self.uoDf["userId"])

        unique_O = self.uoDf["uri"].unique()
        unique_I = self.uiDf["uri"].unique()

        self.OIDf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_I))]
        self.IODf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_O))]
        self.OODf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_O))]
        self.IIDf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_I))]
        print(f"the edge of UI: {self.uiDf.shape[0]} --- the edge of UO:{self.uoDf.shape[0]} --- the edge of OI: {
        self.OIDf.shape[0]} --- the edge of OO: {self.OODf.shape[0]} --- the edge pf II: {self.IIDf.shape[0]}")

        self.OIDf["oId"] = leO.transform(self.OIDf["head_uri"])
        self.OIDf["itemId"] = leI.transform(self.OIDf["tail_uri"])
        self.OODf["oId_head"] = leO.transform(self.OODf["head_uri"])
        self.OODf["oId_tail"] = leO.transform(self.OODf["tail_uri"])
        self.IIDf["itemId_head"] = leI.transform(self.IIDf["head_uri"])
        self.IIDf["itemId_tail"] = leI.transform(self.IIDf["tail_uri"])
        self.num_user = ratingsDf["userId"].nunique()
        self.num_item = self.uiDf["itemId"].nunique()
        self.num_other = self.uoDf["oId"].nunique()

        # train test split
        self.trainDf, self.testDf = train_test_split(self.uiDf, test_size=0.2)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def n_others(self):
        return self.num_other

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def otherUsers(self):
        return self.uoDf["userId"].unique()

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_KG_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix(
                    (self.num_user + self.num_item + self.num_other, self.num_user + self.num_item + self.num_other),
                    dtype=np.float32)
                adjMat = adjMat.tolil()
                Rui = self.UserItemNet.tolil()
                Ruo = self.UserOtherNet.tolil()
                Rio = self.ItemOtherNet.tolil()
                Rii = self.ItemItemNet.tolil()
                Roo = self.OtherOtherNet.tolil()
                adjMat[:self.num_user, self.num_user:self.num_user + self.num_item, ] = Rui
                adjMat[self.num_user: self.num_user + self.num_item, :self.num_user] = Rui.T
                adjMat[:self.num_user, self.num_user + self.num_item:] = Ruo
                adjMat[self.num_user + self.num_item:, :self.num_user] = Ruo.T
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user: self.num_user + self.num_item] = Rii
                adjMat[self.num_user + self.num_item:, self.num_user + self.num_item:] = Roo
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user + self.num_item:] = Rio
                adjMat[self.num_user + self.num_item:, self.num_user: self.num_user + self.num_item] = Rio.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                sp.save_npz(self.path + "/s_KG_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class ML100KG(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/ml-100k"):
        self.rating_file = os.path.join(path, "ui.csv")
        self.ue_file = os.path.join(path, "ue.csv")
        self.ie_file = os.path.join(path, "ie.csv")
        self.path = path
        #         cprint(f"loading [{self.rating_file}]...loading [{self.triple_file}]...")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.validItem = self.validDf["itemId"].values
        self.validUser = self.validDf["userId"].values
        self.validUniqueUser = self.validDf["userId"].unique
        self.testUniqueUser = self.testDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")
        # UI net
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # UO net
        self.UserOtherNet = csr_matrix((np.ones(len(self.uodf)), (self.uodf["userId"], self.uodf["entityId"])),
                                       shape=(self.num_user, self.num_other))
        # IO net
        print(f"num item:{self.num_item}, num entities: {self.num_other}")
        print(f"max item:{self.IODf['itemId'].max()}, max entity: {self.IODf['entityId'].max()}")
        self.ItemOtherNet = csr_matrix(
            (np.ones(len(self.IODf)), (self.IODf["itemId"].values, self.IODf["entityId"].values)),
            shape=(self.num_item, self.num_other))
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        self.ratingsDf = pd.read_csv(self.rating_file, names=["userId", "itemId"])
        self.uodf = pd.read_csv(self.ue_file, names=["userId", "entityId"])
        self.IODf = pd.read_csv(self.ie_file, names=["itemId", "entityId"])

        self.ratingsDf = self.ratingsDf.drop_duplicates()
        self.uodf = self.uodf.drop_duplicates()
        self.IODf = self.IODf.drop_duplicates()

        self.num_user = self.ratingsDf["userId"].nunique()
        self.num_item = self.ratingsDf["itemId"].nunique()
        self.num_other = self.IODf["entityId"].nunique()

        # train test split
        self.trainDf, self.restDf = train_test_split(self.ratingsDf, test_size=0.2, random_state=2020)
        self.validDf, self.testDf = train_test_split(self.restDf, test_size=0.5, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def n_others(self):
        return self.num_other

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def otherUsers(self):
        return self.uoDf["userId"].unique()

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_KG_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix(
                    (self.num_user + self.num_item + self.num_other, self.num_user + self.num_item + self.num_other),
                    dtype=np.float32)
                adjMat = adjMat.tolil()
                Rui = self.UserItemNet.tolil()
                Ruo = self.UserOtherNet.tolil()
                Rio = self.ItemOtherNet.tolil()
                adjMat[:self.num_user, self.num_user:self.num_user + self.num_item, ] = Rui
                adjMat[self.num_user: self.num_user + self.num_item, :self.num_user] = Rui.T
                adjMat[:self.num_user, self.num_user + self.num_item:] = Ruo
                adjMat[self.num_user + self.num_item:, :self.num_user] = Ruo.T
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user + self.num_item:] = Rio
                adjMat[self.num_user + self.num_item:, self.num_user: self.num_user + self.num_item] = Rio.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                #                 sp.save_npz(self.path + "/s_KG_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def __build_valid(self):
        test_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class MindreaderSimpleEntity(BasicDataset):
    """
    This dataloader is used for testing only using entity embedding
    """

    def __init__(self, config=world.config, path="../dataset/mindreader"):
        self.rating_file = os.path.join(path, "ratings.csv")
        self.triple_file = os.path.join(path, "triples.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]...loading [{self.triple_file}]...")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")
        # UI net
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # UO net
        self.UserOtherNet = csr_matrix((np.ones(len(self.uoDf)), (self.uoDf["userId"], self.uoDf["oId"])),
                                       shape=(self.num_user, self.num_other))
        # IO net
        self.ItemOtherNet = csr_matrix((np.ones(len(self.IODf)), (self.OIDf["itemId"].values, self.OIDf["oId"].values)),
                                       shape=(self.num_item, self.num_other))
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        #         self._allPosO = self.getUserPosOthers(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        ratingsDf = pd.read_csv(self.rating_file, index_col=0)
        tripleDf = pd.read_csv(self.triple_file, index_col=0)
        tripleDf = tripleDf.dropna()
        ratingsDf = ratingsDf.dropna()
        ratingsDf["userId"] = ratingsDf["userId"].astype(str)
        ratingsDf["uri"] = ratingsDf["uri"].astype(str)
        tripleDf["head_uri"] = tripleDf["head_uri"].astype(str)
        tripleDf["tail_uri"] = tripleDf["tail_uri"].astype(str)
        self.uiDf = ratingsDf[ratingsDf["isItem"] == True]
        self.uoDf = ratingsDf[ratingsDf["isItem"] == False]
        # make sure ther is no interaction between items id and other entities id
        assert len(np.intersect1d(self.uiDf["uri"].unique(), self.uoDf["uri"].unique())) == 0

        # encoding
        leU = LabelEncoder()
        leU.fit(ratingsDf["userId"].values)
        self.uiDf["userId"] = leU.transform(self.uiDf["userId"].values)
        leI = LabelEncoder()
        leI.fit(self.uiDf["uri"])
        self.uiDf["itemId"] = leI.transform(self.uiDf["uri"])
        leO = LabelEncoder()
        leO.fit(self.uoDf["uri"])
        self.uoDf["oId"] = leO.transform(self.uoDf["uri"])
        self.uoDf["userId"] = leU.transform(self.uoDf["userId"])

        unique_O = self.uoDf["uri"].unique()
        unique_I = self.uiDf["uri"].unique()

        self.OIDf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_I))]
        self.IODf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_O))]
        self.OODf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_O))]
        self.IIDf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_I))]
        print(f"the edge of UI: {self.uiDf.shape[0]} --- the edge of UO:{self.uoDf.shape[0]} --- the edge of OI: {
        self.OIDf.shape[0]} --- the edge of OO: {self.OODf.shape[0]} --- the edge pf II: {self.IIDf.shape[0]}")

        self.OIDf["oId"] = leO.transform(self.OIDf["head_uri"])
        self.OIDf["itemId"] = leI.transform(self.OIDf["tail_uri"])
        self.OODf["oId_head"] = leO.transform(self.OODf["head_uri"])
        self.OODf["oId_tail"] = leO.transform(self.OODf["tail_uri"])
        self.IIDf["itemId_head"] = leI.transform(self.IIDf["head_uri"])
        self.IIDf["itemId_tail"] = leI.transform(self.IIDf["tail_uri"])
        self.num_user = ratingsDf["userId"].nunique()
        self.num_item = self.uiDf["itemId"].nunique()
        self.num_other = self.uoDf["oId"].nunique()

        # train test split
        self.trainDf, self.testDf = train_test_split(self.uiDf, test_size=0.2, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def n_others(self):
        return self.num_other

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosOthers(self):
        return self._allPosO

    @property
    def allOtherUsers(self):
        return self.uoDf["userId"].unique()

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getEntityNet(self):
        print(f"loading user-entity and movie-entity net")
        UONet_tensor = self._convert_sp_mat_to_sp_tensor(self.UserOtherNet)
        IONet_tensor = self._convert_sp_mat_to_sp_tensor(self.ItemOtherNet)
        UONet_tensor = UONet_tensor.to(world.device)
        IONet_tensor = IONet_tensor.to(world.device)
        return UONet_tensor, IONet_tensor

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_KG_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix(
                    (self.num_user + self.num_item + self.num_other, self.num_user + self.num_item + self.num_other),
                    dtype=np.float32)
                adjMat = adjMat.tolil()
                Rui = self.UserItemNet.tolil()
                Ruo = self.UserOtherNet.tolil()
                Rio = self.ItemOtherNet.tolil()
                Rii = self.ItemItemNet.tolil()
                Roo = self.OtherOtherNet.tolil()
                adjMat[:self.num_user, self.num_user:self.num_user + self.num_item, ] = Rui
                adjMat[self.num_user: self.num_user + self.num_item, :self.num_user] = Rui.T
                adjMat[:self.num_user, self.num_user + self.num_item:] = Ruo
                adjMat[self.num_user + self.num_item:, :self.num_user] = Ruo.T
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user: self.num_user + self.num_item] = Rii
                adjMat[self.num_user + self.num_item:, self.num_user + self.num_item:] = Roo
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user + self.num_item:] = Rio
                adjMat[self.num_user + self.num_item:, self.num_user: self.num_user + self.num_item] = Rio.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                sp.save_npz(self.path + "/s_KG_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserPosOthers(self, users):
        posOthers = []
        for user in users:
            posOthers.append(self.UserOtherNet[user].nonzero()[1])
        return posOthers


class MindReaderMulti(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/mindreader"):
        self.rating_file = os.path.join(path, "ratings.csv")
        self.triple_file = os.path.join(path, "triples.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]...loading [{self.triple_file}]...")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.validDataSize = self.validDf.shape[0]
        self.validUser = self.validDf["userId"].values
        self.validItem = self.validDf["itemId"].values
        self.validUniqueUser = self.validDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")
        # UI net
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # UO net
        self.UserOtherNet = csr_matrix((np.ones(len(self.uoDf)), (self.uoDf["userId"], self.uoDf["oId"])),
                                       shape=(self.num_user, self.num_other))
        # IO net
        self.ItemOtherNet = csr_matrix((np.ones(len(self.IODf)), (self.OIDf["itemId"].values, self.OIDf["oId"].values)),
                                       shape=(self.num_item, self.num_other))
        # II net
        self.ItemItemNet = csr_matrix(
            (np.ones(len(self.IIDf)), (self.IIDf["itemId_head"].values, self.IIDf["itemId_tail"].values)),
            shape=(self.num_item, self.num_item))
        # OO net
        self.OtherOtherNet = csr_matrix(
            (np.ones(len(self.OODf)), (self.OODf["oId_head"].values, self.OODf["oId_tail"].values)),
            shape=(self.num_other, self.num_other))
        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self._allPosO = self.getUserPosOthers(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        ratingsDf = pd.read_csv(self.rating_file, index_col=0)
        tripleDf = pd.read_csv(self.triple_file, index_col=0)
        tripleDf = tripleDf.dropna()
        ratingsDf = ratingsDf.dropna()
        ratingsDf["userId"] = ratingsDf["userId"].astype(str)
        ratingsDf["uri"] = ratingsDf["uri"].astype(str)
        tripleDf["head_uri"] = tripleDf["head_uri"].astype(str)
        tripleDf["tail_uri"] = tripleDf["tail_uri"].astype(str)
        self.uiDf = ratingsDf[ratingsDf["isItem"] == True]
        self.uoDf = ratingsDf[ratingsDf["isItem"] == False]
        # make sure ther is no interaction between items id and other entities id
        assert len(np.intersect1d(self.uiDf["uri"].unique(), self.uoDf["uri"].unique())) == 0

        # encoding
        leU = LabelEncoder()
        leU.fit(ratingsDf["userId"].values)
        self.uiDf["userId"] = leU.transform(self.uiDf["userId"].values)
        leI = LabelEncoder()
        leI.fit(self.uiDf["uri"])
        self.uiDf["itemId"] = leI.transform(self.uiDf["uri"])
        leO = LabelEncoder()
        leO.fit(self.uoDf["uri"])
        self.uoDf["oId"] = leO.transform(self.uoDf["uri"])
        self.uoDf["userId"] = leU.transform(self.uoDf["userId"])

        unique_O = self.uoDf["uri"].unique()
        unique_I = self.uiDf["uri"].unique()

        self.OIDf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_I))]
        self.IODf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_O))]
        self.OODf = tripleDf[(tripleDf["head_uri"].isin(unique_O) & tripleDf["tail_uri"].isin(unique_O))]
        self.IIDf = tripleDf[(tripleDf["head_uri"].isin(unique_I) & tripleDf["tail_uri"].isin(unique_I))]
        print(f"the edge of UI: {self.uiDf.shape[0]} --- the edge of UO:{self.uoDf.shape[0]} --- the edge of OI: {
        self.OIDf.shape[0]} --- the edge of OO: {self.OODf.shape[0]} --- the edge pf II: {self.IIDf.shape[0]}")

        self.OIDf["oId"] = leO.transform(self.OIDf["head_uri"])
        self.OIDf["itemId"] = leI.transform(self.OIDf["tail_uri"])
        self.OODf["oId_head"] = leO.transform(self.OODf["head_uri"])
        self.OODf["oId_tail"] = leO.transform(self.OODf["tail_uri"])
        self.IIDf["itemId_head"] = leI.transform(self.IIDf["head_uri"])
        self.IIDf["itemId_tail"] = leI.transform(self.IIDf["tail_uri"])
        self.num_user = ratingsDf["userId"].nunique()
        self.num_item = self.uiDf["itemId"].nunique()
        self.num_other = self.uoDf["oId"].nunique()

        # train test split
        self.trainDf, self.restDf = train_test_split(self.uiDf, test_size=0.2, random_state=2020)
        self.validDf, self.testDf = train_test_split(self.restDf, test_size=0.5, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def n_others(self):
        return self.num_other

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosOthers(self):
        return self._allPosO

    @property
    def allOtherUsers(self):
        return self.uoDf["userId"].unique()

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getEntityNet(self):
        print(f"loading user-entity and movie-entity net")
        UONet_tensor = self._convert_sp_mat_to_sp_tensor(self.UserOtherNet)
        IONet_tensor = self._convert_sp_mat_to_sp_tensor(self.ItemOtherNet)
        UONet_tensor = UONet_tensor.to(world.device)
        IONet_tensor = IONet_tensor.to(world.device)
        return UONet_tensor, IONet_tensor

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/s_KG_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix(
                    (self.num_user + self.num_item + self.num_other, self.num_user + self.num_item + self.num_other),
                    dtype=np.float32)
                adjMat = adjMat.tolil()
                Rui = self.UserItemNet.tolil()
                Ruo = self.UserOtherNet.tolil()
                Rio = self.ItemOtherNet.tolil()
                Rii = self.ItemItemNet.tolil()
                Roo = self.OtherOtherNet.tolil()
                adjMat[:self.num_user, self.num_user:self.num_user + self.num_item, ] = Rui
                adjMat[self.num_user: self.num_user + self.num_item, :self.num_user] = Rui.T
                adjMat[:self.num_user, self.num_user + self.num_item:] = Ruo
                adjMat[self.num_user + self.num_item:, :self.num_user] = Ruo.T
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user: self.num_user + self.num_item] = Rii
                adjMat[self.num_user + self.num_item:, self.num_user + self.num_item:] = Roo
                adjMat[self.num_user: self.num_user + self.num_item, self.num_user + self.num_item:] = Rio
                adjMat[self.num_user + self.num_item:, self.num_user: self.num_user + self.num_item] = Rio.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                # sp.save_npz(self.path + "/s_KG_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]

        return valid_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserPosOthers(self, users):
        posOthers = []
        for user in users:
            posOthers.append(self.UserOtherNet[user].nonzero()[1])
        return posOthers


class ML100KMulti(BasicDataset):
    def __init__(self, config=world.config, path="../dataset/ml-100k"):
        self.rating_file = os.path.join(path, "ui.csv")
        self.ue_file = os.path.join(path, "ue.csv")
        self.ie_file = os.path.join(path, "ie.csv")
        self.path = path
        cprint(f"loading [{self.rating_file}]...loading [{self.ue_file}]...loading [{self.ie_file}]...")
        self.split = config['a_split']
        self.folds = config['n_fold']
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.__loading_data()

        self.trainUser = self.trainDf["userId"].values
        self.trainItem = self.trainDf["itemId"].values
        self.trainUniqueUser = self.trainDf["userId"].unique

        self.testDataSize = self.testDf.shape[0]
        self.testUser = self.testDf["userId"].values
        self.testItem = self.testDf["itemId"].values
        self.testUniqueUser = self.testDf["userId"].unique

        self.validDataSize = self.validDf.shape[0]
        self.validUser = self.validDf["userId"].values
        self.validItem = self.validDf["itemId"].values
        self.validUniqueUser = self.validDf["userId"].unique

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / (self.num_user * self.num_item)}")
        # UI net
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
                                      shape=(self.num_user, self.num_item))
        self.userD = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.userD[self.userD == 0] = 1
        self.itemD = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.itemD[self.itemD == 0] = 1
        # UO net
        self.UserOtherNet = csr_matrix((np.ones(len(self.uodf)), (self.uodf["userId"], self.uodf["entityId"])),
                                       shape=(self.num_user, self.num_other))
        # IO net
        self.ItemOtherNet = csr_matrix(
            (np.ones(len(self.IODf)), (self.IODf["itemId"].values, self.IODf["entityId"].values)),
            shape=(self.num_item, self.num_other))

        # precalculate
        self._allPos = self.getUserPosItems(list(range(self.num_user)))
        self._allPosO = self.getUserPosOthers(list(range(self.num_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        cprint(f"{world.dataset} is ready to go!")

    def __loading_data(self):
        # load dataset
        self.ratingsDf = pd.read_csv(self.rating_file, names=["userId", "itemId"])
        self.uodf = pd.read_csv(self.ue_file, names=["userId", "entityId"])
        self.IODf = pd.read_csv(self.ie_file, names=["itemId", "entityId"])

        self.ratingsDf = self.ratingsDf.drop_duplicates()
        self.uodf = self.uodf.drop_duplicates()
        self.IODf = self.IODf.drop_duplicates()

        self.num_user = self.ratingsDf["userId"].nunique()
        self.num_item = self.ratingsDf["itemId"].nunique()
        self.num_other = self.IODf["entityId"].nunique()

        # train test split
        self.trainDf, self.restDf = train_test_split(self.ratingsDf, test_size=0.2, random_state=2020)
        self.validDf, self.testDf = train_test_split(self.restDf, test_size=0.5, random_state=2020)

    @property
    def n_users(self):
        return self.num_user

    @property
    def m_items(self):
        return self.num_item

    @property
    def n_others(self):
        return self.num_other

    @property
    def trainDataSize(self):
        return self.trainDf.shape[0]

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosOthers(self):
        return self._allPosO

    @property
    def allOtherUsers(self):
        return self.uodf["userId"].unique()

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_user + self.num_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.num_user + self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getEntityNet(self):
        print(f"loading user-entity and movie-entity net")
        UONet_tensor = self._convert_sp_mat_to_sp_tensor(self.UserOtherNet)
        IONet_tensor = self._convert_sp_mat_to_sp_tensor(self.ItemOtherNet)
        UONet_tensor = UONet_tensor.to(world.device)
        IONet_tensor = IONet_tensor.to(world.device)
        return UONet_tensor, IONet_tensor

    def getSparseGraph(self):
        print(f"loading adjacency matrix...")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path + "/ml_adj_mat.npz")
                cprint("successfully loaded adj mat!")
                normAdj = preAdjMat

            except:
                cprint("generate adjacency matrix!")
                s = time()
                adjMat = sp.dok_matrix(
                    (self.num_user + self.num_item + self.num_other, self.num_user + self.num_item + self.num_other),
                    dtype=np.float32)
                adjMat = adjMat.tolil()
                Rui = self.UserItemNet.tolil()
                Ruo = self.UserOtherNet.tolil()
                Rio = self.ItemOtherNet.tolil()

                adjMat[:self.num_user, self.num_user:self.num_user + self.num_item, ] = Rui
                adjMat[self.num_user: self.num_user + self.num_item, :self.num_user] = Rui.T
                adjMat[:self.num_user, self.num_user + self.num_item:] = Ruo
                adjMat[self.num_user + self.num_item:, :self.num_user] = Ruo.T

                adjMat[self.num_user: self.num_user + self.num_item, self.num_user + self.num_item:] = Rio
                adjMat[self.num_user + self.num_item:, self.num_user: self.num_user + self.num_item] = Rio.T
                adjMat = adjMat.todok()

                rowSum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowSum, -1 / 2).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()

                print(f"costing %.2f s to generate adjacency graph, save norm mat..." % (end - s))
                #                 sp.save_npz(self.path + "/s_KG_adj_mat.npz", normAdj)

                if (self.split):
                    self.Graph = self._split_A_hat(normAdj)
                    print("done split matrix")

                else:
                    self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print("do not split the matrix")
                return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]

        return valid_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserPosOthers(self, users):
        posOthers = []
        for user in users:
            posOthers.append(self.UserOtherNet[user].nonzero()[1])
        return posOthers
