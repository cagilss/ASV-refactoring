
from yacs.config import CfgNode as Config


''' 
    COnfiguration settings of the system

    ** Options ** 
    
    DATA.FEATURES: cqcc or mfcc 
    MODE: train, validate or infer 
    TRAIN.MODEL: LCNN, LSTM or ResNet
    DATA.LOADER: verification or recognition

 '''

############################################################
#  Configurations
############################################################

_C = Config()

# general settings 
_C.MODE = 'train'

# cuda related settings
_C.CUDNN = Config()
_C.CUDNN.BENCHMARK = False
_C.CUDNN.ENABLED = False
_C.CUDNN.DETERMINISTIC = True
_C.SEED = 4 

# data loader settings
_C.DATA = Config()
_C.DATA.TRAIN_X = 'train'
_C.DATA.TRAIN_Y = 'train.trn.txt'
_C.DATA.DEV_X = 'dev'
_C.DATA.DEV_Y = 'dev.trl.txt'
_C.DATA.EVAL_X = 'eval'
_C.DATA.EVAL_Y = 'eval.trl.txt'
_C.DATA.LOADER = 'varification'

# train settings 
_C.TRAIN = Config()
_C.TRAIN.MODEL = 'LCNN'
_C.TRAIN.NUM_CLASS = 2 

# input matrix dimensions [T,F]
_C.TRAIN.TIME = 90
_C.TRAIN.FETAURE = 280

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCH = 50

# loss function can be EDL 
# or ELBO or cross entropy 
# ELBO loss is use for the variational 
# dropout applied version of the networks 
_C.TRAIN.LOSS = 'CrossEntropy'

# dynamic paddding settings 
_C.TRAIN.BIN_SIZE = 1
_C.TRAIN.N_WORKER = 0


def update_config(cfg, args):
    ''' Update defaulth configuration settings '''
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()