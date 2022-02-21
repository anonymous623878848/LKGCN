from pprint import pprint

import dataloader as dataloader
import model as model
import world as world

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../dataset/" + world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

elif world.dataset == "amazon-cell":
    dataset = dataloader.AmazonLoader()
elif world.dataset == "mindreader":
    dataset = dataloader.MindReader()
elif world.dataset == "mindreaderKG":
    dataset = dataloader.MindReaderKG()
elif world.dataset == "mindreaderMulti":
    dataset = dataloader.MindReaderMulti()
elif world.dataset == "mindreaderPureEn":
    dataset = dataloader.MindreaderSimpleEntity()
elif world.dataset == "ml100kMulti":
    dataset = dataloader.ML100KMulti()
elif world.dataset == "ml100k":
    dataset = dataloader.ML100K()
elif world.dataset == 'ml100kKG':
    dataset = dataloader.ML100KG()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.load)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    "lgnKG": model.LightKGGCN,
    "lgnMulti": model.LightGCNMulti,
    "pureEntity": model.PureEntity,
    "lgnMultiEntity": model.LightGCNMultiEntity,
    "lgnMultiAtt": model.LightGCNMultiAtt,
}
