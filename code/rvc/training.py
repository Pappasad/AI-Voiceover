import torch
import torch.nn as nn
import torch.optim as optim
from dataset import VoiceDataset
from model import RVCNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainModel(model=RVCNetwork, epochs=100, )