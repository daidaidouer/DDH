# coding: utf-8
import sys
import warnings
import database as database
import frame as frame

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

lr = sys.argv[1]
alpha = sys.argv[2]
theta = sys.argv[3]
lamda = sys.argv[4]
max_epoch = sys.argv[5]
codes_dimension = sys.argv[6]
batch_size = sys.argv[7]


recall = 5000
num_epochs_per_decay = 45
# sys.argv[1]

model_ori = 'models/resnet_v2_50.ckpt'
img_tr = 'train.txt'
img_te = 'test.txt'
img_db = 'train.txt'
label_dim = 15

num_train = len(open(img_tr, 'r').readlines())
num_query = len(open(img_te, 'r').readlines())
num_database = len(open(img_db, 'r').readlines())
gpu = '0'


config = {
    'device': '/gpu:' + gpu,
    'label_dim': label_dim,
    'model_trained_dir': 'models/',
    'max_epoch': max_epoch,
    'theta': theta,
    'alpha': alpha,
    'lamda': lamda,
    'batch_size': batch_size,
    'num_epochs_per_decay': num_epochs_per_decay,
    'learning_rate_decay_factor': 0.5,
    'initial_learning_rate_img': lr,
    'lr': lr,
    'codes_dim': codes_dimension,
    'recall': recall,
    'model_ori': model_ori,
    'n_train': num_train,
    'n_database': num_database,
    'n_query': num_query,
    'img_tr': img_tr,
    'img_te': img_te,
    'img_db': img_db,
}

train_img = database.import_train(config)
model = frame.model(config)
model_out = model.train(train_img)
modelfile = model.save_dir
dataset_img, query = database.import_validation(config)
mAP_value = frame.validation(dataset_img, query, config, modelfile).get_mAP()
