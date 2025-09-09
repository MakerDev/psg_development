import platform
import numpy as np
import pickle as pk 
from os import path, listdir
from torch.utils import data
from sklearn.utils.class_weight import compute_sample_weight
"PATH CONFIG"
if platform.system() == 'Windows':
    SCH_BATCH_ROOT = "G:/STAGENET_R2/datas/sch_batch_50hz"
    SEMI_BATCH_ROOT = "G:/STAGENET_R2/datas/cr1r2/r2/"
else:
    SCH_BATCH_ROOT = "/mnt/DATASETS/STAGENET_R2/datas/sch_batch_50hz"
    SEMI_BATCH_ROOT = "/mnt/DATASETS/STAGENET_R2/datas/cr1r2/r2/"


class SCHPSGDataset(data.Dataset):

    def __init__(self, case, 
        set_tag='train', weight=True, return_key=False,
        root="/mnt/DATASETS/STAGENET_R2/datas/sch_batch_50hz", 
        extx='.npx', exty='.npy'):

        'Initialization'
        super(SCHPSGDataset, self).__init__()
        self.root = root 
        self.extx = extx
        self.exty = exty
        self.weight = weight
        self.return_key = return_key
        self.pathes = self.gen_pathes_as_case(case,set_tag=set_tag)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pathes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_path = self.pathes[index]

        # Load data and get label

        X = self.load(data_path + self.extx)    
        Y = self.load(data_path + self.exty)
        batch = (X, Y,)
        
        # Calculate weights
        if self.weight: 
            batch = batch+(self.calc_weights(Y),)
        
        if self.return_key:
            batch = (path.basename(data_path),) + batch

        return batch
    
    def gen_pathes(self, gender, severity, set_tag):

        "Define directroy path"
        dir_ = path.join(self.root, gender, severity, set_tag)

        "Extract KEY and PATH"
        pathes = list(set(self.onlykey(file_) for file_ in listdir(dir_)))
        pathes = [ path.join(dir_, file_) for file_ in pathes]

        return pathes

    def onlykey(self, file_):

        return file_.replace(self.extx, '').replace(self.exty, '')

    def gen_pathes_as_case(self, case, set_tag):
        
        if    case == 1:

            pathes = self.gen_pathes(
                gender="woman", severity="shallow", set_tag=set_tag)

        elif  case == 2:

            pathes = self.gen_pathes(
                gender="woman", severity="deeper", set_tag=set_tag)

        elif  case == 3:

            pathes = self.gen_pathes(
                gender="man", severity="shallow", set_tag=set_tag)

        elif  case == 4:

            pathes = self.gen_pathes(
                gender="man", severity="deeper", set_tag=set_tag)

        elif  case == 5:

            pathes = []
            for severity in ['shallow', 'deeper']:
                pathes = pathes + self.gen_pathes(
                    gender="woman", severity=severity, set_tag=set_tag)
       
        elif  case == 6:

            pathes = []
            for severity in ['shallow', 'deeper']:
                pathes = pathes + self.gen_pathes(
                    gender="man", severity=severity, set_tag=set_tag)

        elif  case == 7:

            pathes = []
            for gender in ['man', 'woman']:
                pathes = pathes + self.gen_pathes(
                    gender=gender, severity="shallow", set_tag=set_tag)

        elif  case == 8:

            pathes = []
            for gender in ['man', 'woman']:
                pathes = pathes + self.gen_pathes(
                    gender=gender, severity="deeper", set_tag=set_tag)
   

        elif  case == 9:

            pathes = []
            for severity in ['shallow', 'deeper']:
                for gender in ['man', 'woman']:
                    pathes = pathes + self.gen_pathes(
                        gender=gender, severity=severity, set_tag=set_tag)

        else:
            raise ValueError(f"Wrong case {case} - case code only could be between 0 to 9.")

        return pathes

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            ret = pk.load(f)
        return ret

    @staticmethod
    def calc_weights(Y):

        return compute_sample_weight('balanced', Y.reshape(-1)).reshape(Y.shape)


class SEMIDataset(data.Dataset):

    def __init__(self,
        set_tag='train', weight=True, return_key=False,
        root=SEMI_BATCH_ROOT, 
        extx='.npx', exty='.npy', dtype=np.float32):

        'Initialization'
        super(SEMIDataset, self).__init__()
        self.root = root 
        self.extx = extx
        self.exty = exty
        self.weight = weight
        self.dtype = dtype
        self.return_key = return_key
        self.pathes = self.gen_pathes(set_tag=set_tag)

    def __len__(self):

        'Denotes the total number of samples'
        return len(self.pathes)

    def __getitem__(self, index):
        
        'Generates one sample of data'
        # Select sample
        data_path = self.pathes[index]

        # Load data and get label

        X = self.load(data_path + self.extx).astype(self.dtype)
        Y = self.load(data_path + self.exty).astype(self.dtype)
        batch = (X, Y,)
        
        # Calculate weights
        if self.weight: 
            batch = batch+(self.calc_weights(Y).astype(self.dtype),)
        
        if self.return_key:
            batch = (path.basename(data_path),) + batch

        return batch
    
    def gen_pathes(self, set_tag):

        "Define directroy path"
        dir_ = path.join(self.root, set_tag)

        "Extract KEY and PATH"
        pathes = list(set(self.onlykey(file_) for file_ in listdir(dir_)))
        pathes = [ path.join(dir_, file_) for file_ in pathes]

        return pathes

    def onlykey(self, file_):

        return file_.replace(self.extx, '').replace(self.exty, '')

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            ret = pk.load(f)
        return ret

    @staticmethod
    def calc_weights(Y):

        mask = np.where(Y==-1)[0]
        W = compute_sample_weight('balanced', Y.reshape(-1)).reshape(Y.shape) 
        W[mask] = 0
        return W