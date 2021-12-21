import os
import sys
import numpy as np

import matplotlib.pyplot as plt



def main():
	method = 'Valina'
	readout = 'mean'
	seed_list = [
		999,
		888,
		777,
		666,
	]

	model_list = [
		'gcn',
		'gin',
		'gat',
	]
	idx_list = [
		6, 7, 8,
		10, 11, 12,
		14, 15, 16
	]

	rmsd_mean = []
	rmsd_std = []
	r2_mean = []
	r2_std = []
	for model in model_list:
		contents_list = []
		for seed in seed_list:
			name_ = './logs/' + method + '_' + model + '_' + readout + '_scaffold_' + str(seed) + '.log'
			command = 'grep \'End of \' ' + name_ + ' > tmp.txt'
			os.system(command)
			f = open('tmp.txt')
			lines = f.readlines()
			contents = []
			for l in lines:
				splitted = l.split()
				val_list = []
				for idx in idx_list:
					val_list.append(float(splitted[idx]))
				contents.append(val_list)

			idx = -1
			contents = np.asarray(contents)
			contents = contents[idx, :]
			contents_list.append(contents)
	
		mean_list = np.asarray(contents_list)
		mean_list = np.mean(contents_list, axis=0)
		mean_list = np.around(mean_list, 3)

		std_list = np.asarray(contents_list)
		std_list = np.std(contents_list, axis=0)
		std_list = np.around(std_list, 3)


		rmsd_mean.append(mean_list[3:6])
		rmsd_std.append(std_list[3:6])
		r2_mean.append(mean_list[6:])
		r2_std.append(std_list[6:])

	rmsd_mean = np.asarray(rmsd_mean)
	rmsd_std = np.asarray(rmsd_std)
	r2_mean = np.asarray(r2_mean)
	r2_std = np.asarray(r2_std)

	title = 'Vanila, Mean'
	legend = True

	x_ = np.arange(3)
	fig = plt.figure()
	plt.title(title, fontsize=18)
	plt.bar(x_+0.0, rmsd_mean[0,:], yerr=rmsd_std[0,:], color='b', width=0.25, alpha=0.5, label='GCN', capsize=5)
	plt.bar(x_+0.25, rmsd_mean[1,:], yerr=rmsd_std[0,:], color='g', width=0.25, alpha=0.5, label='GIN', capsize=5)
	plt.bar(x_+0.5, rmsd_mean[2,:], yerr=rmsd_std[0,:], color='r', width=0.25, alpha=0.5, label='GT', capsize=5)
	plt.xticks([0.25, 1.25, 2.25], ['Train', 'Valid', 'Test'], fontsize=15)
	plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], fontsize=18)
	plt.ylim(0.0, 1.65)
	plt.ylabel('RMSE', fontsize=18)
	if legend:
		plt.legend(fontsize=15)
	plt.tight_layout()
	plt.savefig('RMSE_' + method + '_' + readout + '.png')

	x_ = np.arange(3)
	fig = plt.figure()
	plt.title(title, fontsize=17)
	plt.bar(x_+0.0, r2_mean[0,:], yerr=r2_std[0,:], color='b', width=0.25, alpha=0.5, label='GCN', capsize=5)
	plt.bar(x_+0.25, r2_mean[1,:], yerr=r2_std[0,:], color='g', width=0.25, alpha=0.5, label='GIN', capsize=5)
	plt.bar(x_+0.5, r2_mean[2,:], yerr=r2_std[0,:], color='r', width=0.25, alpha=0.5, label='GT', capsize=5)
	plt.xticks([0.25, 1.25, 2.25], ['Train', 'Valid', 'Test'], fontsize=18)
	plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
	plt.ylim(0.0, 1.0)
	plt.ylabel('R2', fontsize=18)
	if legend:
		plt.legend(fontsize=15)
	plt.tight_layout()
	plt.savefig('R2_' + method + '_' + readout + '.png')



if __name__ == '__main__':
	main()
