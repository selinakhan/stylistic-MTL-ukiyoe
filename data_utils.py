from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utilities import SquarePad
from tqdm import tqdm 
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os
import json

def parse_data(filepath, img_dir, filename):

	''' Construct dataset containing all relevant information.
		Image data and metadata. '''

	data = pd.read_csv(filepath)

	columns = ['img', 'artist', 'date', 'era', 'source']
	metadata = pd.DataFrame(columns=columns)
	pbar = tqdm(sorted(os.listdir(img_dir)))

	for image in pbar:
		im = Image.open(img_dir + '/' + image)
		index = int(image.split('.')[0])
		info = data.loc[index]
		properties = [str(img_dir + '/' + image), info['full_name'], info['date'], info['era'], info['name']] 
		metadata.loc[index] = properties

	metadata.to_csv('data/' + filename)


class DataProcessor():
	def __init__(self, datapath, image_size, mode, prefix='', mappings=None):

		# Transform images
		self.transform_train = transforms.Compose([
			transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip()]),
			transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),

		])
		self.transform_test = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
		])

		self.prefix = prefix
		data = pd.read_csv(datapath)
		self.data = data
		self.mode = mode

		# Retrieve unique labels
		self.artists = data['artist'].unique()
		self.eras = data['era'].unique()

		self.artists_weights = data['artist'].value_counts()
		self.eras_weights = data['era'].value_counts()

		# Mapping from unique label to int label
		if mappings == None:
			self.artist_map = {self.artists[i]:i for i in range(0, len(self.artists))}
			self.era_map = {self.eras[i]:i for i in range(0, len(self.eras))}

			data = {'artist': self.artist_map, 'era': self.era_map}
			json.dump(data, open('data/mapping.json', 'w+'))
		else:
			self.artist_map = mappings['artist']
			self.era_map = mappings['era']
	
	def get_labels(self):
		return [len(self.artists), len(self.eras)]
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		img = Image.open(f'{self.prefix}/{row.img}').convert('RGB')

		if self.mode == 'train':
			img = self.transform_train(img)
		if self.mode == 'test':
			img = self.transform_test(img)


		artist = self.artist_map[row.artist]
		era = self.era_map[row.era]

		return img, artist, row.date, era

