import time
import argparse

from functools import partial

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
from libs.utils import scatter_plot_regression
from libs.utils import heteroscedastic_loss
from libs.utils import evaluate_classification_multi2


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
		shuffle=False,
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
		out_dim=args.out_dim,
	)
	model = model.to(device)

	save_path = './save/' 
	save_path += str(args.job_title) + '_'
	save_path += str(args.model_type) + '_'
	save_path += str(args.hidden_dim) + '_'
	save_path += str(args.readout) + '_'
	save_path += str(args.split_method) + '_'
	save_path += str(args.data_seed) + '_mcdo.pth'
	ckpt = torch.load(save_path, map_location=device)
	model.load_state_dict(ckpt['model_state_dict'])

	model.eval()
	with torch.no_grad():
		# Train
		y_list = []
		pred_list = []
		ale_unc_list = []
		epi_unc_list = []
		for i, batch in enumerate(train_loader):
			st = time.time()

			tmp_list = []
			for _ in range(args.num_sampling):
				graph, y = batch[0], batch[1]
				graph = graph.to(device)
				y = y.to(device)
				y = y.float()
	
				pred, alpha = model(graph, training=True)
				pred = pred.unsqueeze(-1)
				tmp_list.append(pred)

			tmp_list = torch.cat(tmp_list, dim=-1)
			mean_list = torch.mean(tmp_list, dim=-1)
			std_list = torch.std(tmp_list, dim=-1)
			
			y_list.append(y)
			pred_list.append(mean_list[:,0])
			ale_unc_list.append(torch.exp(mean_list[:,1]))
			epi_unc_list.append(std_list[:,0])

		train_metrics = evaluate_regression(
			y_list=y_list,
			pred_list=pred_list
		)
		train_accuracy = evaluate_classification_multi2(
			y_list=y_list,
			pred_list=pred_list
		)
		figure_title = str(args.dataset_name) + '_'
		figure_title += str(args.job_title) + '_'
		figure_title += str(args.model_type) + '_'
		figure_title += str(args.data_seed) + '_mcdo_train'
		scatter_plot_regression(
			y_list=y_list,
			pred_list=pred_list,
			ale_unc_list=ale_unc_list,
			epi_unc_list=epi_unc_list,
			figure_title=figure_title,
			tot_unc_threshold=args.tot_unc_threshold,
		)

		# Validation
		y_list = []
		pred_list = []
		ale_unc_list = []
		epi_unc_list = []
		for i, batch in enumerate(valid_loader):
			st = time.time()

			tmp_list = []
			for _ in range(args.num_sampling):
				graph, y = batch[0], batch[1]
				graph = graph.to(device)
				y = y.to(device)
				y = y.float()
	
				pred, alpha = model(graph, training=True)
				pred = pred.unsqueeze(-1)
				tmp_list.append(pred)

			tmp_list = torch.cat(tmp_list, dim=-1)
			mean_list = torch.mean(tmp_list, dim=-1)
			std_list = torch.std(tmp_list, dim=-1)
			
			y_list.append(y)
			pred_list.append(mean_list[:,0])
			ale_unc_list.append(torch.exp(mean_list[:,1]))
			epi_unc_list.append(std_list[:,0])

		valid_metrics = evaluate_regression(
			y_list=y_list,
			pred_list=pred_list
		)
		valid_accuracy = evaluate_classification_multi2(
			y_list=y_list,
			pred_list=pred_list
		)

		figure_title = str(args.dataset_name) + '_'
		figure_title += str(args.job_title) + '_'
		figure_title += str(args.model_type) + '_'
		figure_title += str(args.data_seed) + '_mcdo_valid'
		scatter_plot_regression(
			y_list=y_list,
			pred_list=pred_list,
			ale_unc_list=ale_unc_list,
			epi_unc_list=epi_unc_list,
			figure_title=figure_title,
			tot_unc_threshold=args.tot_unc_threshold,
		)

		# Test
		y_list = []
		pred_list = []
		ale_unc_list = []
		epi_unc_list = []
		for i, batch in enumerate(test_loader):
			st = time.time()
	
			tmp_list = []
			for _ in range(args.num_sampling):
				graph, y = batch[0], batch[1]
				graph = graph.to(device)
				y = y.to(device)
				y = y.float()
	
				pred, alpha = model(graph, training=True)
				pred = pred.unsqueeze(-1)
				tmp_list.append(pred)

			tmp_list = torch.cat(tmp_list, dim=-1)
			mean_list = torch.mean(tmp_list, dim=-1)
			std_list = torch.std(tmp_list, dim=-1)
			
			y_list.append(y)
			pred_list.append(mean_list[:,0])
			ale_unc_list.append(torch.exp(mean_list[:,1]))
			epi_unc_list.append(std_list[:,0])

		test_metrics = evaluate_regression(
			y_list=y_list,
			pred_list=pred_list
		)
		test_accuracy = evaluate_classification_multi2(
			y_list=y_list,
			pred_list=pred_list
		)

		figure_title = str(args.dataset_name) + '_'
		figure_title += str(args.job_title) + '_'
		figure_title += str(args.model_type) + '_'
		figure_title += str(args.data_seed) + '_mcdo_test'
		scatter_plot_regression(
			y_list=y_list,
			pred_list=pred_list,
			ale_unc_list=ale_unc_list,
			epi_unc_list=epi_unc_list,
			figure_title=figure_title,
			tot_unc_threshold=args.tot_unc_threshold,
		)

	print ("RMSE:", round(train_metrics[1], 3), "\t", round(valid_metrics[1], 3), "\t", round(test_metrics[1], 3), \
		   "R2:", round(train_metrics[2], 3), "\t", round(valid_metrics[2], 3), "\t", round(test_metrics[2], 3), \
		   "Accuracy:", round(train_accuracy, 3), "\t", round(valid_accuracy, 3), "\t", round(test_accuracy, 3))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--job_title', type=str, default='Test', 
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

	parser.add_argument('--model_type', type=str, default='gcn', 
						help='Type of GNN model, Options: gcn, gin, gin_e, gat, ggnn')
	parser.add_argument('--num_layers', type=int, default=4,
						help='Number of GIN layers for ligand featurization')
	parser.add_argument('--hidden_dim', type=int, default=128,
						help='Dimension of hidden features')
	parser.add_argument('--out_dim', type=int, default=2,
						help='Dimension of final outputs')
	parser.add_argument('--readout', type=str, default='pma', 
						help='Readout method, Options: sum, mean, ...')
	parser.add_argument('--dropout_prob', type=float, default=0.2, 
						help='Probability of dropout on node features')

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

	parser.add_argument('--num_sampling', type=int, default=10,
						help='Number of MC-Sampling of output logits')
	parser.add_argument('--tot_unc_threshold', type=float, default=1.0, 
						help='Weight decay coefficient')

	parser.add_argument('--save_model', type=str2bool, default=True, 
						help='whether to save model')

	args = parser.parse_args()

	print ("Arguments")
	for k, v in vars(args).items():
		print (k, ": ", v)
	main(args)
