import torch
import torch.nn.functional as F
import random

class MagScale(object):
    """Rescale the magnitude of all PSG channels with the same random scale factor"""
    
    def __init__(self, low = 0.8, high = 1.25):
        self.low = low
        self.high = high
    
    def __call__(self, recording, arousal):
        scale = self.low + torch.rand(1)*(self.high - self.low)
        recording = scale*recording

        return recording, arousal   

    
class MagScaleRandCh(object):
    """Rescale the magnitude of a randomly selected PSG channel with a random scale factor"""
    
    def __init__(self, n_channels = 13, low = 0.8, high = 1.25):
        self.n_channels = n_channels        
        self.low = low
        self.high = high
    
    def __call__(self, recording, arousal):
        scales = self.low + torch.rand(self.n_channels).view(-1,1)*(self.high - self.low)
        recording = scales*recording

        return recording, arousal       

    
class RandShuffle(object):
    """Randomly reshuffle a subset of related EEG channels"""
    
    def __init__(self):
        self.r2 = torch.LongTensor([6, 7, 8])
    
    def __call__(self, recording, arousal):    
        r1 = torch.randperm(6)     # shuffle EEG channels (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1)
        r = torch.cat((r1, self.r2)).type(torch.long) 

        return recording.index_select(0, r), arousal     


    
class AddRandGaussian2All(object):
    """Add zero-mean Gaussian noise to all PSG channels"""
    
    def __init__(self, z_norm = True):
        self.z_norm = z_norm
    
    def __call__(self, recording, arousal):
        if self.z_norm:
            std_dev = 0.1 
        else:
            std_dev = 0.1*torch.std(recording, 1, keepdim = True)
        recording = recording + std_dev*torch.randn(recording.shape)
        
        return recording, arousal


class NormaliseOnly(object):
    def __call__(self, recording, arousal):
        mean_val = torch.mean(recording, dim=1, keepdim=True)
        std_val  = torch.std(recording,  dim=1, keepdim=True)
        std_val[std_val < 1e-12] = 1e-12  # 0 나누기 방지
        recording = (recording - mean_val) / std_val

        return recording, arousal

class NormaliseAndAddRandNoise(object):
    def __init__(self, z_norm=True, noise_min=0.8, noise_max=1.3):
        """
        z_norm=True  -> 모든 채널 동일하게 std_dev=0.1
        z_norm=False -> 채널별 std_dev=0.1 * torch.std(채널)
        noise_min, noise_max: 노이즈 계수 범위
        """
        self.z_norm = z_norm
        self.noise_min = noise_min
        self.noise_max = noise_max

    def __call__(self, recording, arousal):
        mean_val = torch.mean(recording, dim=1, keepdim=True)
        std_val  = torch.std(recording,  dim=1, keepdim=True)
        std_val[std_val < 1e-12] = 1e-12  # 0 나누기 방지
        recording = (recording - mean_val) / std_val

        if self.z_norm:
            std_dev = 0.1  
        else:
            std_dev = 0.1 * torch.std(recording, dim=1, keepdim=True)

        factor = random.uniform(self.noise_min, self.noise_max)

        noise = factor * std_dev * torch.randn_like(recording)
        recording = recording + noise

        return recording, arousal
    
    
class InjectRandGaussian(object):
    """Replace a randomly selected PSG channel with a standard Gaussian noise sequence"""
    
    def __init__(self, n_channels = 6):
        self.n_channels = n_channels
    
    def __call__(self, recording, arousal):
        ri = torch.randint(0,self.n_channels,(1,)).type(torch.long)
        recording[ri] = torch.normal(mean = 0, std = 1, size = (1, recording.shape[1]))
        
        return recording, arousal
    

class TimeScale(object):
    """Stretch/shrink the recording and arousal signals with a random time scale while 
       maintaining the original lengths"""

    def __init__(self, interval, n_channels = 13):
        self.interval = interval
        self.n_channels = n_channels        

    def __call__(self, recording, arousal):
        scale = 1 + self.interval*(torch.rand(1) - 0.5)
        recording = F.interpolate(recording.reshape((1,self.n_channels,-1)), \
                                  scale_factor = scale, recompute_scale_factor = True)
        arousal = F.interpolate(arousal.reshape((1,1,-1)), scale_factor = scale, \
                                recompute_scale_factor = True)

        return recording, arousal

    
class SelectOne(object):
    """Select a single channel from each EEG, EOG, and EMG"""
    
    def __init__(self, n_channels = 9):
        self.n_channels = n_channels
    
    def __call__(self, recording, arousal):
        r_eeg = torch.randint(0,6,(1,)).type(torch.long)
        r_eog = torch.randint(6,8,(1,)).type(torch.long)
        r_emg = torch.LongTensor([8])
        
        return recording[[r_eeg, r_eog, r_emg],:], arousal

class SelectOneEval(object):
    """Select a single channel from each EEG, EOG, and EMG"""
    
    def __init__(self, n_channels = 9):
        self.n_channels = n_channels
    
    def __call__(self, recording, arousal):
        r_eeg = torch.LongTensor([0])
        r_eog = torch.LongTensor([6])
        r_emg = torch.LongTensor([8])
        
        return recording[[r_eeg, r_eog, r_emg],:], arousal

class ToTensor(object):
    """Convert the recording and arousal signals to Tensors"""

    def __call__(self, recording, arousal):
        return torch.Tensor(recording), torch.Tensor(arousal)     
    
    
class Compose:
    """Stack multiple transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, recording, arousal):
        for t in self.transforms:
            recording, arousal = t(recording, arousal)
            
        return recording, arousal   


def build_transforms(transforms = ["MagScale, RandShuffle"], n_channels = 9):
    transform_list = [ToTensor()]

    for transform in transforms:
        if transform == 'MagScale':
            transform_list.append(MagScale())
        if transform == 'MagScaleRandCh':
            transform_list.append(MagScaleRandCh(n_channels=n_channels))        
        if transform == 'TimeScale':
            transform_list.append(TimeScale(n_channels=n_channels))
        if transform == 'RandShuffle':
            transform_list.append(RandShuffle())
        if transform == 'AddRandGaussian2All':
            transform_list.append(AddRandGaussian2All())
        if transform == 'InjectRandGaussian':
            transform_list.append(InjectRandGaussian(n_channels=n_channels))
        if transform == 'NormaliseAndAddRandNoise':
            transform_list.append(NormaliseAndAddRandNoise())
        if transform == 'SelectOne':
            transform_list.append(SelectOne(n_channels=n_channels))
        if transform == 'SelectOneEval':
            transform_list.append(SelectOneEval(n_channels=n_channels))
        if transform == 'NormaliseOnly':
            transform_list.append(NormaliseOnly())
        
    transforms = Compose(transform_list)
    return transforms