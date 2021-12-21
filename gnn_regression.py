import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.io_utils import get_dataset
from libs.io_utils import MyDataset
from libs.io_utils import gnn_collate_fn

from libs.models import MyModel

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_regression


def main(args):
	# Set random seeds and device
	set_seed(seed=args.seed)
	device = set_device(
		use_gpu=args.use_gpu,
		gpu_idx=args.gpu_idx
	)

	# Prepare datasets and dataloaders
	train_set, valid_set, test_set = get_dataset(
		name=args.dataset_name,
		method=args.split_method,
		data_seed=args.data_seed,
	)
	
	train_ds = MyDataset(splitted_set=train_set)
	valid_ds = MyDataset(splitted_set=valid_set)
	test_ds = MyDataset(splitted_set=test_set)

	train_loader = DataLoader(
		dataset=train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		collate_fn=gnn_collate_fn
	)
	valid_loader = DataLoader(
		dataset=valid_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=gnn_collate_fn
	)
	test_loader = DataLoader(
		dataset=test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=gnn_collate_fn
	)

	# Construct model and load trained parameters if it is possible
	model = MyModel(
		model_type=args.model_type,
		num_layers=args.num_layers,
		hidden_dim=args.hidden_dim,
		readout=args.readout,
		dropout_prob=args.dropout_prob,
		norm_features=args.norm_features,
	)
	model = model.to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), 
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=40,
		gamma=0.1,
	)
	mse_loss = nn.MSELoss()
	for epoch in range(args.num_epoches):
		# Train
		model.train()
		num_batches = len(train_loader)
		train_loss = 0
		y_list = []
		pred_list = []
		for i, batch in enumerate(train_loader):
			st = time.time()
			optimizer.zero_grad()

			graph, y = batch[0], batch[1]
			graph = graph.to(device)
			y = y.to(device)
			y = y.float()

			pred, alpha = model(graph)
			y_list.append(y)
			pred_list.append(pred)

			loss = mse_loss(pred.squeeze(), y)
			loss.backward()
			optimizer.step()

			train_loss += loss.detach().cpu().numpy()

			et = time.time()
			print ("Train!!! Epoch:", epoch+1, \
				   "\t Batch:", i+1, '/', num_batches, \
				   "\t Loss:", loss.detach().cpu().numpy(), \
				   "\t Time spent:", round(et-st, 2), "(s)")
		scheduler.step()
		train_loss /= num_batches
		train_metrics = evaluate_regression(
			y_list=y_list,
			pred_list=pred_list
		)

		model.eval()
		with torch.no_grad():
			# Validation
			valid_loss = 0
			num_batches = len(valid_loader)
			y_list = []
			pred_list = []
			for i, batch in enumerate(valid_loader):
				st = time.time()

				graph, y = batch[0], batch[1]
				graph = graph.to(device)
				y = y.to(device)
				y = y.float()

				pred, alpha = model(graph)
				y_list.append(y)
				pred_list.append(pred)

				loss = mse_loss(pred.squeeze(), y)
				valid_loss += loss.detach().cpu().numpy()

				et = time.time()
				print ("Valid!!! Epoch:", epoch+1, \
					   "\t Batch:", i+1, '/', num_batches, \
					   "\t Loss:", loss.detach().cpu().numpy(), \
				   	   "\t Time spent:", round(et-st, 2), "(s)")
			valid_loss /= num_batches
			valid_metrics = evaluate_regression(
				y_list=y_list,
				pred_list=pred_list
			)

			# Test
			test_loss = 0
			num_batches = len(test_loader)
			y_list = []
			pred_list = []
			for i, batch in enumerate(test_loader):
				st = time.time()

				graph, y = batch[0], batch[1]
				graph = graph.to(device)
				y = y.to(device)
				y = y.float()

				pred, alpha = model(graph)
				y_list.append(y)
				pred_list.append(pred)

				loss = mse_loss(pred.squeeze(), y)
				test_loss += loss.detach().cpu().numpy()

				et = time.time()
				print ("Test!!! Epoch:", epoch+1, \
					   "\t Batch:", i+1, '/', num_batches, \
					   "\t Loss:", loss.detach().cpu().numpy(), \
				   	   "\t Time spent:", round(et-st, 2), "(s)")
			test_loss /= num_batches
			test_metrics = evaluate_regression(
				y_list=y_list,
				pred_list=pred_list
			)
		print ("End of ", epoch+1, "-th epoch", \
			   "MSE:", round(train_metrics[0], 3), "\t", round(valid_metrics[0], 3), "\t", round(test_metrics[0], 3), \
			   "RMSE:", round(train_metrics[1], 3), "\t", round(valid_metrics[1], 3), "\t", round(test_metrics[1], 3), \
			   "R2:", round(train_metrics[2], 3), "\t", round(valid_metrics[2], 3), "\t", round(test_metrics[2], 3))

		save_path = './save/' 
		save_path += str(args.job_title) + '_'
		save_path += str(args.model_type) + '_'
		save_path += str(args.hidden_dim) + '_'
		save_path += str(args.readout) + '_'
		save_path += str(args.split_method) + '_'
		save_path += str(args.data_seed) + '.pth'
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}, save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--job_title', type=str, default='GNN_vanila', 
						help='Job title of this execution')
	parser.add_argument('--use_gpu', type=str2bool, default=True, 
						help='whether to use GPU device')
	parser.add_argument('--gpu_idx', type=str, default='1', 
						help='index of gpu to use')
	parser.add_argument('--seed', type=int, default=999,
						help='Seed for all stochastic components')

	parser.add_argument('--dataset_name', type=str, default='Solubility', 
						help='What dataset to use for model development')
	parser.add_argument('--split_method', type=str, default='scaffold', 
						help='How to split dataset')
	parser.add_argument('--data_seed', type=int, default=999,
						help='Seed for dataset splitting')

	parser.add_argument('--model_type', type=str, default='gin', 
						help='Type of GNN model, Options: gcn, gin, gin, gat')
	parser.add_argument('--num_layers', type=int, default=4,
						help='Number of GIN layers for ligand featurization')
	parser.add_argument('--hidden_dim', type=int, default=128,
						help='Dimension of hidden features')
	parser.add_argument('--readout', type=str, default='mean', 
						help='Readout method, Options: sum, mean, ...')
	parser.add_argument('--dropout_prob', type=float, default=0.0, 
						help='Probability of dropout on node features')
	parser.add_argument('--norm_features', type=str2bool, default=False, 
						help='whether to normalize the node features at the PMA step')

	parser.add_argument('--optimizer', type=str, default='adam', 
						help='Options: adam, sgd, ...')
	parser.add_argument('--num_epoches', type=int, default=150,
						help='Number of training epoches')
	parser.add_argument('--num_workers', type=int, default=8,
						help='Number of workers to run dataloaders')
	parser.add_argument('--batch_size', type=int, default=64,
						help='Number of samples in a single batch')
	parser.add_argument('--lr', type=float, default=1e-3, 
						help='Initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-6, 
						help='Weight decay coefficient')

	parser.add_argument('--save_model', type=str2bool, default=True, 
						help='whether to save model')

	args = parser.parse_args()

	print ("Arguments")
	for k, v in vars(args).items():
		print (k, ": ", v)
	main(args)
