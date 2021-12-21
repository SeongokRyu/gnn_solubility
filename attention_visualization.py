import time
import argparse

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.io_inference import get_dataset
from libs.io_inference import MyDataset
from libs.io_inference import my_collate_fn

from libs.models import MyModel

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import plot_attention


def main(args):
	# Set random seeds and device
	set_seed(seed=args.seed)
	device = set_device(
		use_gpu=args.use_gpu,
		gpu_idx=args.gpu_idx
	)

	# Prepare datasets and dataloaders
	smi_list = get_dataset(
		path=args.csv_path,
		smi_column=args.smi_column,
		dropna=args.dropna
	)

	test_ds = MyDataset(smi_list=smi_list)
	test_loader = DataLoader(
		dataset=test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=my_collate_fn
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
		# Test
		pred_list = []
		ale_unc_list = []
		epi_unc_list = []
		smi_list = []
		for i, batch in enumerate(test_loader):
			st = time.time()
	
			tmp_list = []
			graph_tmp = batch[0]
			smi = batch[1]
			alpha_list = []
			for _ in range(args.num_sampling):
				graph = graph_tmp
				graph = graph.to(device)
				#pred, alpha = model(graph, training=True)
				pred, alpha = model(graph, training=False)
				pred = pred.unsqueeze(-1)
				tmp_list.append(pred)
				alpha_list.append(alpha.unsqueeze(-1))

			alpha_list = torch.cat(alpha_list, dim=-1)
			alpha_list = torch.mean(alpha_list, dim=-1)

			tmp_list = torch.cat(tmp_list, dim=-1)
			mean_list = torch.mean(tmp_list, dim=-1)
			std_list = torch.std(tmp_list, dim=-1)
			
			pred_list.append(mean_list[:,0])
			ale_unc_list.append(torch.exp(mean_list[:,1]))
			epi_unc_list.append(std_list[:,0])
			smi_list.extend(smi)

			plot_attention(
				smi_list=smi,
				attention_list=alpha,
				pred_list=mean_list[:,0],
				unc_list=torch.exp(mean_list[:,1])+std_list[:,0],
				prefix=args.title+'_'+str(i),
			)

			print (i, "/", len(test_loader))

	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
	ale_unc_list = torch.cat(ale_unc_list, dim=0).detach().cpu().numpy()
	epi_unc_list = torch.cat(epi_unc_list, dim=0).detach().cpu().numpy()
	tot_unc_list = ale_unc_list + epi_unc_list

	pred_list = list(np.around(pred_list, 3))
	ale_unc_list = list(np.around(ale_unc_list, 3))
	epi_unc_list = list(np.around(epi_unc_list, 3))
	tot_unc_list = list(np.around(tot_unc_list, 3))

	df = pd.DataFrame({})
	df['SMILES'] = smi_list
	df['Pred'] = pred_list
	df['Ale_unc'] = ale_unc_list
	df['Epi_unc'] = epi_unc_list
	df['Tot_unc'] = tot_unc_list

	df.to_csv('./results/'+args.title+'_solubility.csv', index=False)

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--title', type=str, default='test', 
						help='Title of this project')
	parser.add_argument('--csv_path', type=str, required=True, 
						help='Path of the csv file that provides the list of SMILES')
	parser.add_argument('--smi_column', type=str, default='SMILES', 
						help='Name of the column that provides the list of SMILES')
	parser.add_argument('--dropna', type=str2bool, default=True, 
						help='whether to drop N/A values')

	parser.add_argument('--job_title', type=str, default='MCDO', 
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

	parser.add_argument('--num_workers', type=int, default=8,
						help='Number of workers to run dataloaders')
	parser.add_argument('--batch_size', type=int, default=64,
						help='Number of samples in a single batch')

	parser.add_argument('--num_sampling', type=int, default=30,
						help='Number of MC-Sampling of output logits')

	args = parser.parse_args()

	print ("Arguments")
	for k, v in vars(args).items():
		print (k, ": ", v)
	main(args)
