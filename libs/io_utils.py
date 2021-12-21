import numpy as np
import torch
import dgl

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from tdc.single_pred import ADME
from tdc.single_pred import HTS
from tdc.single_pred import Tox


ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F', 
	'H', 'Si', 'P', 'Cl', 'Br',
	'Li', 'Na', 'K', 'Mg', 'Ca',
	'Fe', 'As', 'Al', 'I', 'B',
	'V', 'Tl', 'Sb', 'Sn', 'Ag', 
	'Pd', 'Co', 'Se', 'Ti', 'Zn',
	'Ge', 'Cu', 'Au', 'Ni', 'Cd',
	'Mn', 'Cr', 'Pt', 'Hg', 'Pb'
]

DATASET_DICT = {
	'Solubility': 'Solubility_AqSolDB',
	'Lipophilicity': 'Lipophilicity_AstraZeneca',
	'BBBP': 'BBB_martins',
	'CYP1A2': 'CYP1A2_Veith',
	'CYP2C9': 'CYP2C9_Veith',
	'CYP2C19': 'CYP2C19_Veith',
	'CYP2D6': 'CYP2D6_Veith',
	'CYP3A4': 'CYP3A4_Veith',
	'hERG': 'hERG',
	'AMES': 'AMES',
	'DILI': 'DILI',
}


def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1]
	return list(map(lambda s: float(x==s), vocab))


def get_atom_feature(atom):
	atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
	atom_feature += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
	atom_feature += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	atom_feature += [atom.GetIsAromatic()]
	return atom_feature
	

def get_bond_feature(bond):
	bt = bond.GetBondType()
	bond_feature = [
		bt == Chem.rdchem.BondType.SINGLE,
		bt == Chem.rdchem.BondType.DOUBLE,
		bt == Chem.rdchem.BondType.TRIPLE,
		bt == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	return bond_feature


def get_molecular_graph(smi):
	mol = Chem.MolFromSmiles(smi)
	graph = dgl.DGLGraph()

	atom_list = mol.GetAtoms()
	num_atoms = len(atom_list)
	graph.add_nodes(num_atoms)

	atom_feature_list = [get_atom_feature(atom) for atom in atom_list]
	atom_feature_list = torch.tensor(atom_feature_list, dtype=torch.float64)
	graph.ndata['h'] = atom_feature_list

	bond_list = mol.GetBonds()
	bond_feature_list = []
	for bond in bond_list:
		bond_feature = get_bond_feature(bond)

		src = bond.GetBeginAtom().GetIdx()
		dst = bond.GetEndAtom().GetIdx()

		# DGL graph is undirectional
		# Thus, we have to add edge pair of both (i,j) and (j, i)
		# i --> j
		graph.add_edges(src, dst)
		bond_feature_list.append(bond_feature)

		# j --> i
		graph.add_edges(dst, src)
		bond_feature_list.append(bond_feature)
	
	bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
	graph.edata['e_ij'] = bond_feature_list
	return graph


def get_smi_and_label(dataset):
	smi_list = list(dataset['Drug'])
	label_list = list(dataset['Y'])
	return smi_list, label_list


def gnn_collate_fn(batch):
	graph_list = []
	label_list = []
	for item in batch:
		smi = item[0]
		label = item[1]
		graph = get_molecular_graph(smi)
		graph_list.append(graph)
		label_list.append(label)
	graph_list = dgl.batch(graph_list)
	label_list = torch.tensor(label_list, dtype=torch.float64)
	return graph_list, label_list


def ecfp_collate_fn(batch):
	ecfp_list = []
	label_list = []
	for item in batch:
		smi = item[0]
		label = item[1]
		ecfp = convert_smi_to_ecfp(smi)
		ecfp_list.append(ecfp)
		label_list.append(label)
	ecfp_list = torch.tensor(ecfp_list, dtype=torch.float64)
	label_list = torch.tensor(label_list, dtype=torch.float64)
	return ecfp_list, label_list


def get_dataset(
		name,
		method='random',
		data_seed=999,
		frac=[0.7, 0.1, 0.2]
	):
	data = None
	adme_properties = [
		'Solubility',
		'Lipophilicity',
		'BBBP',
		'CYP1A2',
		'CYP2C9',
		'CYP2C19',
		'CYP2D6',
		'CYP3A4',
	]
	tox_properties = [
		'hERG',
		'AMES',
		'DILI',
	]
	if name in adme_properties:
		data = ADME(DATASET_DICT[name])
	elif name in tox_properties:
		data = Tox(DATASET_DICT[name])

	split = data.get_split(
		method=method,
		seed=data_seed,
		frac=frac,
	)
	train_set = split['train']
	valid_set = split['valid']
	test_set = split['test']
	return train_set, valid_set, test_set


def convert_smi_to_ecfp(smi, nBits=1024):
	mol = Chem.MolFromSmiles(smi)
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
	arr = np.zeros((0, ), dtype=np.int8)
	DataStructs.ConvertToNumpyArray(fp, arr)
	return arr


class MyDataset(torch.utils.data.Dataset):
	def __init__(
			self, 
			splitted_set
		):
		self.smi_list = list(splitted_set['Drug'])
		self.label_list = list(splitted_set['Y'])
	
	def __len__(self):
		return len(self.smi_list)
	
	def __getitem__(
			self, 
			idx
		):
		return self.smi_list[idx], self.label_list[idx]


class SolubilityClassificationDataset(torch.utils.data.Dataset):
	def __init__(
			self, 
			splitted_set
		):
		self.smi_list = list(splitted_set['Drug'])
		self.label_list = list(splitted_set['Y'])
		self.label_classification = []

		self.convert_to_classification()
	
	def __len__(self):
		return len(self.smi_list)

	def convert_to_classification(self):
		for label in self.label_list:
			out = 0
			if label > 0.0:
				out = 0
			elif label <= 0.0 and label > -2.0:
				out = 1
			elif label <= -2.0 and label > -4.0:
				out = 2
			elif label <= -4.0:
				out = 3
			self.label_classification.append(out)
	
	def __getitem__(
			self, 
			idx
		):
		return self.smi_list[idx], self.label_classification[idx]


def debugging():
	data = ADME(
		name='BBB_Martins'
	)
	split = data.get_split(
		method='random',
		seed=999,
		frac=[0.7, 0.1, 0.2],
	)
	train_set = split['train']
	valid_set = split['valid']
	test_set = split['test']
	
	smi_train, label_train = get_smi_and_label(train_set)
	graph = get_molecular_graph(smi_train[0])


if __name__ == '__main__':
	debugging()
