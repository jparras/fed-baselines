# February 17th, 2021
# Reproduction - Weight Uncertainty in Neural Networks
# Based on https://github.com/saxena-mayur/Weight-Uncertainty-in-Neural-Networks/tree/

# Import relevant packages
import os
import csv
import torch
import urllib
import argparse

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from bnn_toolbox import *
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

torch.manual_seed(0)

def data_preparation(hyperparameters):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 255. / 126.),  # Divide as in paper
	])

	train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
	test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

	valid_size = 1 / 6
	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(valid_size * num_train)
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx) # Samples elements randomly from a given list of indices, without replacement.
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=hyperparameters.batch_size,
		sampler=train_sampler,
		num_workers=1)
	valid_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=hyperparameters.eval_batch_size,
		sampler=valid_sampler,
		num_workers=1)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=hyperparameters.eval_batch_size,
		num_workers=1)

	return train_loader, valid_loader, test_loader, 28*28, 10

def parse_arguments():
	parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with Variational Inference')
	parser.add_argument('--model', type=str, nargs='?', action='store', default='mixture_prior',
	                    help='Model to run.Default mixture prior. Options are \'gaussian_prior\', \'mixture_prior\'.')
	parser.add_argument('--hidd_units', type=int, nargs='?', action='store', default=1200,
	                    help='Neural network hidden units. Default 1200.')
	args = parser.parse_args()

	return args

if __name__ == '__main__':

	# Configuration
	print('[INFO] Environment configuration...')
	args = parse_arguments()
	mixture = True
	if args.model != 'mixture_prior':
		mixture = False
	hyperparameters = HyperparametersInitialization(hidden_units=args.hidd_units, mixture=mixture)

	# Data preparation
	print('[INFO] Preparing data...')
	train_loader, valid_loader, test_loader, n_input, n_ouput = data_preparation(hyperparameters)

	# Test parameters
	print('[INFO] Model hyperparameters:')
	print(hyperparameters.__dict__)

	# Initialize network
	print('[INFO] Initializing network...')
	model = BNN(n_input, n_ouput, hyperparameters).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters.lr, momentum=hyperparameters.momentum)

	train_losses = np.zeros(hyperparameters.max_epoch)
	valid_accs = np.zeros(hyperparameters.max_epoch)
	test_accs = np.zeros(hyperparameters.max_epoch)
	test_errs = np.zeros(hyperparameters.max_epoch)

	# Training
	print('[INFO] Training network for', hyperparameters.max_epoch, 'epochs...')
	for epoch in range(hyperparameters.max_epoch):
		train_loss = train(model, optimizer, train_loader, hyperparameters)
		valid_acc = evaluate(model, valid_loader, samples=hyperparameters.n_test_samples)
		test_acc = evaluate(model, test_loader, samples=hyperparameters.n_test_samples)

		print('Epoch', epoch + 1, 'Loss', float(train_loss),
			'Valid Error', round(100 * (1 - valid_acc / hyperparameters.eval_batch_size), 3), '%',
			'Test Error',  round(100 * (1 - test_acc / hyperparameters.eval_batch_size), 3), '%')

		valid_accs[epoch] = valid_acc
		test_accs[epoch] = test_acc
		train_losses[epoch] = train_loss
		test_errs[epoch] = round(100 * (1 - test_acc / hyperparameters.eval_batch_size), 3)

	# Save results
	if not os.path.exists('results'):
		os.makedirs('results')
	path = 'results/BBB_' + 'mnist' + '_' + str(hyperparameters.hidden_units) + '_' + str(hyperparameters.lr) + '_samples' + str(hyperparameters.n_samples) + '_' + str(args.model)
	wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
	wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])

	for i in range(hyperparameters.max_epoch):
		wr.writerow((i + 1, str(round(valid_accs[i] / hyperparameters.eval_batch_size * 100, 3)) + "%", str(round(test_accs[i] / hyperparameters.eval_batch_size * 100, 3)) + '_' + "%", train_losses[i]))

	torch.save(model.state_dict(), path + '.pth')

	# Plot test error
	plt.plot(test_errs)
	plt.xlabel('Epochs')
	plt.ylabel('Error (%)')
	plt.title('Test data error prediction')
	plt.grid(True)
	plt.savefig('results/BBB_' + 'mnist' + '_' + str(hyperparameters.hidden_units) + '_' + str(hyperparameters.lr) + '_samples' + str(hyperparameters.n_samples) + '_' + str(args.model), format='png')
	plt.show()

	print('[INFO] Done')
