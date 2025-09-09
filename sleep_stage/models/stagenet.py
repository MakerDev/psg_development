"[Library]"
import time
import numpy as np 
import pandas as pd
from tqdm import tqdm
from os import path, mkdir

"[Torch Library]"
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as Functional

from torch.autograd import Variable
import torch.optim as optim

from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, confusion_matrix)

########################################################################################################################
# Functional blocks
class Normalizer(nn.Module):
    def __init__(self, numChannels, momentum=0.985, channelNorm=True):
        super(Normalizer, self).__init__()
        self.momentum = momentum
        self.num_channels = numChannels
        self.channel_norm = channelNorm

        self.register_buffer('moving_average', torch.zeros(1, numChannels, 1))
        self.register_buffer('moving_variance', torch.ones(1, numChannels, 1))

        self.batch_norm_scale = nn.Parameter(torch.ones(1, numChannels, 1))
        self.batch_norm_bias = nn.Parameter(torch.zeros(1, numChannels, 1))

    def forward(self, x):
        if self.channel_norm:
            mean = torch.mean(x, dim=(0, 2), keepdim=True)
            std = torch.std(x, dim=(0, 2), keepdim=True) + 1e-5  
            x = (x - mean) / std

        if self.training:
            batch_mean = torch.mean(x, dim=(0, 2), keepdim=True)
            batch_var = torch.mean((x - batch_mean) ** 2, dim=(0, 2), keepdim=True) 

            self.moving_average = (self.momentum * self.moving_average + (1 - self.momentum) * batch_mean).detach()
            self.moving_variance = (self.momentum * self.moving_variance + (1 - self.momentum) * batch_var).detach()

        x = (x - self.moving_average) / (torch.sqrt(self.moving_variance) + 1e-5)
        x = (x * torch.abs(self.batch_norm_scale)) + self.batch_norm_bias

        return x


class SeperableDenseNetUnit(nn.Module):
    """
        Module that defines a sequence of two convolutional layers with selu activation on both. Channel Normalization
        and stochastic batch normalization with a per channel affine transform is applied before each non-linearity.
        """

    def __init__(self, in_channels, out_channels, kernelSize, dropout=0.2,
                 groups=1, dilation=1, channelNorm=True):
        super(SeperableDenseNetUnit, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelSize = kernelSize
        self.groups = groups
        self.dilation = dilation

        # Convolutional transforms
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=4*out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1)

        self.conv3 = nn.Conv1d(in_channels=4*out_channels, out_channels=4*out_channels, groups=4*out_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2, dilation=dilation)
        self.conv4 = nn.Conv1d(in_channels=4*out_channels, out_channels=out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1)

        self.norm1 = Normalizer(numChannels=4 * out_channels, channelNorm=channelNorm)
        self.norm2 = Normalizer(numChannels=out_channels, channelNorm=channelNorm)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Apply first convolution block
        y = self.conv2(self.conv1(x))
        y = self.norm1(y)
        y = Functional.selu(y)
        y = self.dropout(y)

        # Apply second convolution block
        y = self.conv4(self.conv3(y))
        y = self.norm2(y)
        y = Functional.selu(y)
        y = self.dropout(y)

        # Return densely connected feature map
        return torch.cat((y, x), dim=1)

########################################################################################################################
# Define the Sleep model

class SkipLSTM(nn.Module):
    """
    Module that defines a bidirectional LSTM model with a residual skip connection with transfer shape modulated with a
    mapping 1x1 linear convolution. The output results from a second 1x1 convolution after a tanh nonlinearity,
    critical to prevent divergence during training.
    """
    def __init__(self, in_channels, out_channels=4, hiddenSize=32, dropout=0.2, num_layers=1):
        super(SkipLSTM, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Bidirectional LSTM to apply temporally across input channels
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=hiddenSize, 
                           num_layers=num_layers, batch_first=True, dropout=dropout,
                           bidirectional=True)

        # Output convolution to map the LSTM hidden states from forward and backward pass to the output shape
        self.outputConv1 = nn.Conv1d(in_channels=hiddenSize*2, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)
        self.outputConv2 = nn.Conv1d(in_channels=hiddenSize, out_channels=out_channels, groups=1, kernel_size=1, padding=0)

        # Residual mapping
        self.identMap1 = nn.Conv1d(in_channels=in_channels, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        y = x
        
        # rnn input 
        # (batch, time_stpes, features_in) -> [rnn] -> (batch, time_stpes, features_out)
        
        y, z = self.rnn(y); z = None
        
        # Change shape for identMap1(it's origin input)!!
        # (batch, time_stpes, features_in) -> [permute] -> (batch, features_in, time_stpes)
        x = x.permute(0, 2, 1)

        # Change shape for RNN to CNN
        # (batch, time_stpes, features_in) -> [permute] -> (batch, features_out, time_stpes)
        y = y.permute(0, 2, 1)
        
        # outputConv1 input - ouput
        # (batch, features_in, time_stpes) -> [CNN] -> # (batch, features_out, time_stpes)

        # identMap1 input - output
        # (batch, features_in, time_stpes) -> [CNN] -> (batch, features_out, time_stpes)

        # outputConv1 ouput and identMap1 output is same shape!! 
        y = torch.tanh((self.outputConv1(y) + self.identMap1(x)) / 1.41421)

        # outputConv2 input - output
        # (batch, features_in, time_stpes) -> [CNN] -> (batch, features_out, time_stpes)
        y = self.outputConv2(y)
        # y = self.dropout(y)

        # reshape of softmax 
        # (batch, features, time_stpes) -> (batch, time_stpes, features)
        y = y.permute(0, 2, 1)


        return y

class StageNet_DCNN_SKIPLSTM(nn.Module):

    def __init__(self, num_signals=9):
        super(StageNet_DCNN_SKIPLSTM, self).__init__()
        self.channel_multiplier = 2
        self.kernelSize = 25
        self.num_signals = num_signals

        # Set up downsampling densenet blocks
        self.dsMod1 = SeperableDenseNetUnit(in_channels=self.num_signals, out_channels=self.channel_multiplier*self.num_signals,
                                kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)
        self.dsMod2 = SeperableDenseNetUnit(in_channels=(self.channel_multiplier+1)*self.num_signals, out_channels=self.channel_multiplier*self.num_signals,
                                 kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)
        self.dsMod3 = SeperableDenseNetUnit(in_channels=((2*self.channel_multiplier)+1)*self.num_signals, out_channels=self.channel_multiplier*self.num_signals,
                                 kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)

        # Set up densenet modules
        self.denseMod1 = SeperableDenseNetUnit(in_channels=((3 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=1, channelNorm=True)
        self.denseMod2 = SeperableDenseNetUnit(in_channels=((4 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=2, channelNorm=True)
        self.denseMod3 = SeperableDenseNetUnit(in_channels=((5 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=4, channelNorm=True)
        self.denseMod4 = SeperableDenseNetUnit(in_channels=((6 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=8, channelNorm=True)
        self.denseMod5 = SeperableDenseNetUnit(in_channels=((7 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=16, channelNorm=True)
        self.denseMod6 = SeperableDenseNetUnit(in_channels=((8 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=32, channelNorm=True)
        self.denseMod7 = SeperableDenseNetUnit(in_channels=((9 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                 kernelSize=self.kernelSize, groups=1, dilation=16, channelNorm=True)
        self.denseMod8 = SeperableDenseNetUnit(in_channels=((10 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                  kernelSize=self.kernelSize, groups=1, dilation=8, channelNorm=True)
        self.denseMod9 = SeperableDenseNetUnit(in_channels=((11 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                  kernelSize=self.kernelSize, groups=1, dilation=4, channelNorm=True)
        self.denseMod10 = SeperableDenseNetUnit(in_channels=((12 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                  kernelSize=self.kernelSize, groups=1, dilation=2, channelNorm=True)
        self.denseMod11 = SeperableDenseNetUnit(in_channels=((13 * self.channel_multiplier) + 1) * self.num_signals, out_channels=self.channel_multiplier * self.num_signals,
                                  kernelSize=self.kernelSize, groups=1, dilation=1, channelNorm=True)

        output_size = self.__calc_lstm_input_size()
        # self.skipLSTM = SkipLSTM(((14*self.channelMultiplier)+1)*self.numSignals, hiddenSize=self.channelMultiplier*64, out_channels=4)
        self.skipLSTM = SkipLSTM(output_size, #7830
                                 hiddenSize=self.channel_multiplier*128, 
                                 out_channels=5, dropout=0.1, num_layers=2)

        self.softmax = nn.Softmax(dim=1)

    def __calc_lstm_input_size(self):
        x = self.forward_cnn(torch.rand(1, 256, 1500, self.num_signals))

        # (Batch_T, Length, Channels) -> (Batch, Time_steps, (Length, Channels)->features)\
        # > (Batch, Time_steps, Features)
        # It's kind of flatten() -> "features" is embeded vector.
        length, channel = x.size()[1:]
        x = x.reshape((1, 256,)+(length * channel,))

        return x.size(-1)

    def to(self, device=None):
        self.skipLSTM.rnn = self.skipLSTM.rnn.to(device)
        return super(StageNet_DCNN_SKIPLSTM, self).to(device)

    def cuda(self, device=None):
        self.skipLSTM.rnn = self.skipLSTM.rnn.cuda(device)
        return super(StageNet_DCNN_SKIPLSTM, self).cuda(device)

    def forward_cnn(self, x):

        # INPUT SHAPE
        # (Batch, Time_steps, Length, Channels)

        # (Batch, Time_steps, Length, Channels) -> (Batch * Time_steps, Length, Channels)
        # > Batch * Time_steps => Batch_T, (Batch_T, Length, Channels)
        batch, time_steps, epoch_tick = x.size()[:3]
        x = x.contiguous().view((batch*time_steps, epoch_tick,)+x.size()[3:])

        # (Batch_T, Length, Channels) -> (Batch_T, Channels, Length)
        x = torch.transpose(x, 1, 2)

        # Downsampling to 1 entity per second
        x = self.dsMod1(x)
        x = Functional.max_pool1d(x, kernel_size=2)

        x = self.dsMod2(x)
        x = Functional.max_pool1d(x, kernel_size=5)

        x = self.dsMod3(x)
        x = Functional.max_pool1d(x, kernel_size=5)

        # Dilated Densenet
        x = self.denseMod1(x)
        x = self.denseMod2(x)
        x = self.denseMod3(x)
        x = self.denseMod4(x)
        x = self.denseMod5(x)
        # x = self.denseMod6(x)
        # x = self.denseMod7(x)
        # x = self.denseMod8(x)
        # x = self.denseMod9(x)
        # x = self.denseMod10(x)
        # x = self.denseMod11(x)

        return x 
    

    def forward(self, x):
        batch, time_steps, epoch_tick = x.size()[:3]

        x = self.forward_cnn(x)

        # (Batch_T, Length, Channels) -> (Batch, Time_steps, (Length, Channels)->features)\
        # > (Batch, Time_steps, Features)
        # It's kind of flatten() -> "features" is embeded vector.
        length, channel = x.size()[1:]
        x = x.reshape((batch, time_steps,)+(length * channel,))

        # Bidirectional skip LSTM and convert joint predictions to marginal predictions
        x = self.skipLSTM(x)
        
        # (Batch, Time_steps, OutFeatures) -> (Batch_T, OutFeatures)
        x = x.reshape((batch* time_steps,)+(x.size(-1),))

        # (Batch_T, OutFeatures) -> [softmax] -> (Batch_T, n_class)
        x = self.softmax(x)
  
        # (Batch_T, n_class) -> (Batch, Time_steps, n_class)
        x = x.reshape((batch, time_steps,)+(x.size(-1),))

        return x

    def represent(self, x):

        # INPUT SHAPE
        # (Batch, Time_steps, Length, Channels)

        # (Batch, Time_steps, Length, Channels) -> (Batch * Time_steps, Length, Channels)
        # > Batch * Time_steps => Batch_T, (Batch_T, Length, Channels)
        batch, time_steps, epoch_tick = x.size()[:3]
        x = x.contiguous().view((batch*time_steps, epoch_tick,)+x.size()[3:])

        # (Batch_T, Length, Channels) -> (Batch_T, Channels, Length)
        x = torch.transpose(x, 1, 2)

        # Downsampling to 1 entity per second
        x = self.dsMod1(x)
        x = Functional.max_pool1d(x, kernel_size=2)

        x = self.dsMod2(x)
        x = Functional.max_pool1d(x, kernel_size=5)

        x = self.dsMod3(x)
        x = Functional.max_pool1d(x, kernel_size=5)

        # Dilated Densenet
        x = self.denseMod1(x)
        x = self.denseMod2(x)
        x = self.denseMod3(x)
        x = self.denseMod4(x)
        x = self.denseMod5(x)
        x = self.denseMod6(x)
        x = self.denseMod7(x)
        x = self.denseMod8(x)
        x = self.denseMod9(x)
        x = self.denseMod10(x)
        x = self.denseMod11(x)

        # (Batch_T, Length, Channels) -> (Batch, Time_steps, (Length, Channels)->features)\
        # > (Batch, Time_steps, Features)
        # It's kind of flatten() -> "features" is embeded vector.
        length, channel = x.size()[1:]
        representation = x.reshape((batch, time_steps,)+(length * channel,))
            
        # Bidirectional skip LSTM and convert joint predictions to marginal predictions
        x = self.skipLSTM(representation)
        
        # (Batch, Time_steps, OutFeatures) -> (Batch_T, OutFeatures)
        x = x.reshape((batch* time_steps,)+(x.size(-1),))

        # (Batch_T, OutFeatures) -> [softmax] -> (Batch_T, n_class)
        x = self.softmax(x)
  
        # (Batch_T, n_class) -> (Batch, Time_steps, n_class)
        x = x.reshape((batch, time_steps,)+(x.size(-1),))

        return representation, x

class NoiseAware(nn.Module):

    def __init__(self, freeze_model, verbose=False, device='cpu'):
        super(NoiseAware, self).__init__()
        
        # default device 'cpu'
        self.device = device
        self.verbose = verbose
        self.channelMultiplier = 2
        self.len_repr_vec = 7835

        # external models
        self.freeze_model = freeze_model
        for param in self.freeze_model.parameters():  # model freesing
            param.requires_grad = False

        self.noise_classifier = SkipLSTM(self.len_repr_vec, hiddenSize=64, out_channels=32)
        self.logit   = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def to(self, device):

        self.device = device
        self.freeze_model.to(self.device)
        self.noise_classifier.rnn = self.noise_classifier.rnn.to(self.device)
        return super(NoiseAware, self).to(self.device)

    def print(self, line):
        if self.verbose: print(line)

    def forward(self, x, y):

        # x's shape should be (Batch, Time_steps, Length, Channels)
        # Batch * Time_steps should be 512
        if   (len(x.shape) == 4) and (len(y.shape) == 2): pass 
        elif (len(x.shape) == 3) and (len(y.shape) == 1):
            x = x.unsqueeze(0); y = y.unsqueeze(0)
        else: 
            raise ValueError(f"Invalid input data shape, x: {x.shape}, y: {y.shape}")
        
        self.print(f"[ Input layer part ]")
        self.print(f"x: {x.shape}\ny:{y.shape}\n")

        # from SEMIDataset with batch 2
        # x's shape is torch.Size([2, 256, 1500, 9])
        # y's shape is torch.Size([2, 256])

        # freeze_mode should have represent function 
        # which return representation, prob_y
        # representation (Batch, Time_steps, len_repr_vec)
        # representation's shape is torch.Size([2, 256, 7830])
        

        representation, prob_y = self.freeze_model.represent(x)
        prob_y = prob_y.detach()
        representation = representation.detach()
        _ , pred_y = prob_y.max(-1)
        
        self.print(f"[ Calc repr part ]")
        self.print(f"representation : {representation.shape}\nprob_y         : {prob_y.shape}\npred_y         : {pred_y.shape}\n")
                      

        # pred_y (Batch, Time_steps)
        # pred_y's shape is torch.Size([2, 256])
        # prob_y (Batch, Time_steps, n_class)
        # prob_y's shape is torch.Size([2, 256, 5])

        # Calc true_noise
        mask = torch.eq(y.flatten(), pred_y.flatten())
        true_noise = torch.zeros(y.flatten().shape).to(self.device)
        true_noise[~mask] = 0.9
        true_noise[mask] = 0.1
        self.print(f"[ Calc true_noise part ]")
        self.print(f"true_noise: {true_noise.shape}\n")

        # Clac pred_noise
        self.print(f"[ Calc pred_noise part ]")
        tensor_prob = prob_y.view(-1, 5).float()
        tensor_repr = representation.view(-1, 7830)
        self.print(f"input tensor_prob: {tensor_prob.shape}")
        self.print(f"input tensor_repr: {tensor_repr.shape}")
        noise_x = torch.cat([tensor_prob, tensor_repr], dim=1).unsqueeze(0)
        self.print(f"concated noise_x : {noise_x.shape}\n")
        # input tensor_prob: torch.Size([512, 5])
        # input tensor_repr: torch.Size([512, 7830])
        # concated noise_x : torch.Size([1, 512, 7835])
        
        # noise_classifier steps
        # noise_x's shape is torch.Size([1, 512, 7835])
        self.print(f"[ noise_classifier part ]")
        pred_noise = self.noise_classifier(noise_x).squeeze()
        self.print(f"noise_x   : {noise_x.shape}\npred_noise: {pred_noise.shape}\n")
        # noise_x   : torch.Size([1, 512, 7835])
        # pred_noise: torch.Size([512, 32])

        # logit steps
        pred_noise = self.logit(pred_noise)
        self.print(f"after logit pred_noise: {pred_noise.shape}")
        # after logit pred_noise: torch.Size([512, 1])

        # softmax steps
        pred_noise = self.sigmoid(pred_noise).squeeze()
        self.print(f"after softmax pred_noise: {pred_noise.shape}\n")
        # after softmax pred_noise: torch.Size([512])
        

        return true_noise, pred_noise, prob_y

    def predict(self, x):

        # x's shape should be (Batch, Time_steps, Length, Channels)
        # Batch * Time_steps should be 512
        if   len(x.shape) == 4: pass 
        elif len(x.shape) == 3:
            x = x.unsqueeze(0); y = y.unsqueeze(0)
        else: 
            raise ValueError(f"Invalid input data shape, x: {x.shape}, y: {y.shape}")


        representation, prob_y = self.freeze_model.represent(x)
        prob_y = prob_y.detach()
        representation = representation.detach()
        _, pred_y = prob_y.max(-1)

        # Clac pred_noise
        self.print(f"[ Calc pred_noise part ]")
        tensor_prob = prob_y.view(-1, 5).float()
        tensor_repr = representation.view(-1, 7830)
        self.print(f"input tensor_prob: {tensor_prob.shape}")
        self.print(f"input tensor_repr: {tensor_repr.shape}")
        noise_x = torch.cat([tensor_prob, tensor_repr], dim=1).unsqueeze(0)
        self.print(f"concated noise_x : {noise_x.shape}\n")
        # input tensor_prob: torch.Size([512, 5])
        # input tensor_repr: torch.Size([512, 7830])
        # concated noise_x : torch.Size([1, 512, 7835])

        # noise_classifier steps
        # noise_x's shape is torch.Size([1, 512, 7835])
        self.print(f"[ noise_classifier part ]")
        pred_noise = self.noise_classifier(noise_x).squeeze()
        self.print(f"noise_x   : {noise_x.shape}\npred_noise: {pred_noise.shape}\n")
        # noise_x   : torch.Size([1, 512, 7835])
        # pred_noise: torch.Size([512, 32])

        # logit steps
        pred_noise = self.logit(pred_noise)
        self.print(f"after logit pred_noise: {pred_noise.shape}")
        # after logit pred_noise: torch.Size([512, 1])

        # softmax steps
        pred_noise = self.sigmoid(pred_noise).squeeze()
        self.print(f"after softmax pred_noise: {pred_noise.shape}\n")
        # after softmax pred_noise: torch.Size([512])

        return pred_noise, prob_y


if __name__ == '__main__':
    num_channels = 9
    model = StageNet_DCNN_SKIPLSTM(num_signals=num_channels)
    x = torch.rand(2, 256, 1500, num_channels) # 
    y = torch.randint(0, 5, (2, 256))
    model(x)
    # model.represent(x)
    # model(x, y)
    # model.predict(x)
    print("Model test is done!!")