# COde is going to be for the data loader.
# Need to turn the images into tensors
# Combine tensor with pickle file from xml extraction
# Output will be a dataset ready for model training
# Feed dataset into a dataloader
import os
import cv2
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import polars as pl

