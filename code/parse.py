import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="start lightGCN")
    parser.add_argument("--bpr_batch", type=int, default=2048, help="the batch size for bpr loss trainning procedure")
    parser.add_argument("--recdim", type=int, default=64, help="the dimension of the embedding size")
    parser.add_argument("--layer", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument("--decay", type=float, default=1e-4, help="the weight decay for l2 norm")
    parser.add_argument("--dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--a_fold", type=int, default=100,
                        help="the fold number used to split large adj matrix like gowalla")
    parser.add_argument("--testbatch", type=int, default=100, help="the batch size of users for testing")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, amazon-cell, mindreader, mindreaderKG]")
    parser.add_argument("--path", type=str, default="./checkpoints", help="path for saving weight")
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--tensorboard", type=int, default=1, help="if use tensorboard to record")
    parser.add_argument("--comment", type=str, default="lightGCN")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument("--multicore", type=int, default=0, help="if use multicore process for evaluation")
    parser.add_argument("--pretrain", type=int, default=0, help="whether use pretrained weights")
    parser.add_argument("--seed", type=int, default=13, help="random seed")
    parser.add_argument("--model", type=str, default="lgn", help="available model: [lgn, MF, lgnKG]")
    parser.add_argument("--lbd", type=float, default=0.0, help="control the weigt of loss 2 in multi-task learning")
    parser.add_argument("--beta", type=float, default=0.0, help="control the weight of loss 3 in multi-task learning")
    return parser.parse_args()
