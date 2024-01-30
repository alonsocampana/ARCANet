import urllib
import polars as pl
import pandas as pd
import os
import urllib.request
import zipfile
from os.path import exists

def download_file(url, filename, root_folder = "/mnt/scratch/alonsocampana/data/"):
    urllib.request.urlretrieve(url, root_folder+filename)
    
def download_or_read(url, filename, root):
    if not exists(root+filename):
        download_file(url, filename, root)
    return pd.read_csv(root+filename)

import pandas as pd
import numpy as np
import base64
import multiprocessing as mp
import re
# Pytorch and Pytorch Geometric
from torch_geometric.data import Data, Batch
import torch_geometric
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import transforms as T
from torch_geometric.utils import coalesce
# RDkit
import rdkit
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdMolDescriptors
import warnings
from models_new import DeepSplineFusionModule, init_weights, CrossAttnPooling, LatentHillFusionModule, ConcatFusionModule, MLPEncoder, MultiModel, GatedAttnPooling, GATRes, MultiModelFP
import optuna
from torch_geometric import nn as gnn

RANDOM_SEED = 3558

def build_model(x,
                transform_log_conc,
                hidden_dim = 256,
                n_knots = 16,
                n_pooling_heads = 2,
                dropout_cattn = 0.1,
                dropout_genes = 0.4,
                dropout_fusion = 0.1,
                dropout_fc = 0.45,
                dropout_nodes_attn = 0.2,
                n_layers = 9,
                n_heads = 6,
                fc_hidden = 3898,
                n_transformers = 1,
                use_normalization = True,
                use_normalization_fc = True,
                use_normalization_fusion = True,
                activation_fn = "relu",
                fusion = "hill",
                linear_head = False,
                crossattn = "transformer"):
    if fusion == "spline":
        fusion_module = DeepSplineFusionModule(hidden_dim,hidden_dim, dropout_fusion, n_knots = n_knots, use_norm = use_normalization_fusion)
    elif fusion == "hill":
        fusion_module = LatentHillFusionModule(hidden_dim,hidden_dim, dropout_fusion, use_norm = use_normalization_fusion)
    elif fusion == "concat":
        fusion_module = ConcatFusionModule(hidden_dim,hidden_dim, dropout_fusion, use_norm = use_normalization_fusion)
    if crossattn == "transformer":
        graph_crossattention = CrossAttnPooling(hidden_dim, hidden_dim, n_pooling_heads, dropout_cattn)
    elif crossattn == "gated":
        graph_crossattention = GatedAttnPooling(hidden_dim,hidden_dim * 2, hidden_dim, dropout_cattn,dropout_nodes_attn,n_pooling_heads)
    return MultiModel(cell_encoder = MLPEncoder(x["expression"].size(1), hidden_dim, dropout_genes),
             drug_encoder = GATRes(init_dim = x["x"].size(1),
                                   edge_dim = x["edge_attr"].size(1),
                                   hidden_dim = hidden_dim,
                                   n_layers = n_layers,
                                   n_heads = n_heads,
                                   n_transformers = n_transformers),
             graph_crossattention = graph_crossattention,
             fusion_module = fusion_module,
             activation_fn = activation_fn,
             use_normalization = use_normalization,
             use_normalization2 = use_normalization_fc,
             dropout_fc = dropout_fc,
             fc_hidden  = fc_hidden,
             embed_dim = hidden_dim,
             linear_head = linear_head,
             transform_log_conc = transform_log_conc)

def build_model_fingerprint(x,
                transform_log_conc,
                hidden_dim = 256,
                n_knots = 16,
                dropout_genes = 0.4,
                dropout_drugs = 0.2,
                dropout_fusion = 0.1,
                dropout_fc = 0.45,
                fc_hidden = 3898,
                use_normalization = True,
                use_normalization_fc = True,
                use_normalization_fusion = True,
                activation_fn = "relu",
                fusion = "hill",
                linear_head = False,            
                **kwargs):
    if fusion == "spline":
        fusion_module = DeepSplineFusionModule(hidden_dim,hidden_dim, dropout_fusion, n_knots = n_knots, use_norm = use_normalization_fusion)
    elif fusion == "hill":
        fusion_module = LatentHillFusionModule(hidden_dim,hidden_dim, dropout_fusion, use_norm = use_normalization_fusion)
    elif fusion == "concat":
        fusion_module = ConcatFusionModule(hidden_dim,hidden_dim, dropout_fusion, use_norm = use_normalization_fusion)
    return MultiModelFP(cell_encoder = MLPEncoder(x["expression"].size(1), hidden_dim, dropout_genes),
             drug_encoder = MLPEncoder(2048, hidden_dim, 0.0, dropout_drugs),
             drug_cell_fusion = gnn.Sequential("x, y", 
                                               [(lambda x, y: torch.cat([x, y], -1), "x, y ->z"),
                                               nn.Linear(hidden_dim*2, fc_hidden),
                                               nn.ReLU(),
                                               nn.Linear(fc_hidden, hidden_dim)]),
             fusion_module = fusion_module,
             activation_fn = activation_fn,
             use_normalization = use_normalization,
             use_normalization2 = use_normalization_fc,
             dropout_fc = dropout_fc,
             fc_hidden  = fc_hidden,
             embed_dim = hidden_dim,
             linear_head = linear_head,
             transform_log_conc = transform_log_conc)

def print_epoch(epoch, train_dict, test_dict):
        print(epoch, train_dict, test_dict)
def metric_dict(metric_result, suffix = ""):
    return {it[0] + suffix:it[1].item() for it in metric_result.items()}

class CurveDataset(torch.utils.data.Dataset):
    def __init__(self, data,drugs,lines, shape_safe = True):
        self.data = data
        self.drugs = drugs
        self.lines = lines
        self.pairs = self.data.loc[:, ["drug", "cell"]].drop_duplicates().sort_index()
        self.data = self.data.sort_values(["drug", "cell"]).set_index("drug").sort_index()
        self.memo = []
        for idx in range(len(self.pairs)):
            pair = self.pairs.iloc[idx]
            drug = pair.loc["drug"]
            cell = pair.loc["cell"]
            entry = self.data.loc[[drug]].set_index("cell").loc[[cell]]
            concs = entry.loc[:, ["z"]].to_numpy()
            if idx == 0:
                n_concs = concs.shape[0]
            else:
                n_concs = min(n_concs, concs.shape[0])
            self.n_concs = n_concs
            self.memo += [[drug, cell, entry]]
    def __len__(self):
        return len(self.pairs)
    def infer_concs(self):
        for idx in range(len(self.pairs)):
            pair = self.pairs.iloc[idx]
            drug = pair.loc["drug"]
            cell = pair.loc["cell"]
            entry = self.data.loc[[cell]].set_index("drug").loc[[drug]]
            if idx == 0:
                n_concs = len(entry.loc[:, ["z"]].to_numpy())
            else:
                n_concs = min(len(entry.loc[:, ["z"]].to_numpy()), n_concs)
        return n_concs
    def __getitem__(self, idx):
        drug, cell, entry = self.memo[idx]
        concs = entry.loc[:, ["z"]].to_numpy()
        inhibs = entry.loc[:, ["y"]].to_numpy()
        g = self.drugs[drug].clone()
        g["z"] = torch.Tensor(concs).T
        g["y"] = torch.Tensor(inhibs).T
        g["expression"] = self.lines[cell].unsqueeze(0)
        concs = g["z"]
        inhibs = g["y"]
        if (concs.shape[1] > self.n_concs):
            diff = concs.shape[1] - self.n_concs
            indices = np.arange(concs.shape[1])
            to_drop = np.random.choice(indices, diff)
            ret_indices = np.delete(indices, to_drop)
            g["z"] = concs[:, ret_indices]
            g["y"] = inhibs[:, ret_indices]
        return g


class ExpCreator():
    def __init__(self):
        pass
    def __call__(self, x, feature_set = None):
        exp = x.set_index("cell")
        if feature_set is not None:
            exp = exp.loc[:, exp.columns.isin(feature_set)]
        exp_tensor = torch.Tensor(exp.to_numpy())
        cells = exp.index.to_numpy().squeeze()
        exp_dict = {}
        for i in range(len(cells)):
            exp_dict[cells[i]] = exp_tensor[i]
        return exp_dict

class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = torch.Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
        self.R = R
        self.fp_kwargs = fp_kwargs
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles)
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    def __str__(self):
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"
    
class GraphCreator():
    def __init__(self, use_supernode = False, 
                       add_linegraph = False,):
        self.use_supernode = use_supernode
        self.add_linegraph = add_linegraph
    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding


    def get_atom_features(self, atom, 
                          use_chirality = True, 
                          hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms

        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

        # compute atom features

        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc

        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)

    def get_bond_features(self, bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [int(bond.IsInRing())]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    def __call__(self, smiles_list, drugs = None, **kwargs):
        use_supernode = self.use_supernode
        add_linegraph = self.add_linegraph
        if drugs is None:
            drugs = np.arange(0, len(smiles_list))
        data_dict = {}

        for x, drug in enumerate(drugs):
            try:
                # convert SMILES to RDKit mol object
                smiles = smiles_list[x]
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2*mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = self.get_atom_features(atom)

                X = torch.tensor(X, dtype = torch.float)

                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim = 0)

                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))

                for (k, (i,j)) in enumerate(zip(rows, cols)):

                    EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

                EF = torch.tensor(EF, dtype = torch.float)
                add_f = {kwarg: torch.Tensor([kwargs[kwarg][x]]) for kwarg in kwargs.keys()}
                if use_supernode:
                    super_node = torch.zeros([1, X.shape[1]])
                    trgt_supernode = X.shape[0]
                    extra_indices = torch.cat([torch.arange(0, X.shape[0])[:, None], 
                                         torch.full([X.shape[0], 1],  trgt_supernode)], axis=1).T
                    extra_f = torch.zeros([extra_indices.shape[1], EF.shape[1]])
                    indicator_indices = torch.cat([torch.zeros([E.shape[1], 1]),
                                               torch.ones([extra_indices.shape[1], 1])], axis=0)
                    X = torch.cat([X, super_node], axis=0)
                    E = torch.cat([E, extra_indices], axis=1)
                    EF = torch.cat([indicator_indices, torch.cat([EF,
                                                        extra_f], axis=0)], axis=1)
                data_dict[drug] = Data(x = X, edge_index = E, edge_attr = EF, **add_f)
            except Exception as e:
                warnings.warn( f"{smiles_list[x]} could not be transformed into a graph", RuntimeWarning,)
        return data_dict
    def __str__(self):
        suffix = "graphs"
        if self.use_supernode:
            suffix += "_supernode"
        return suffix

class GDSC():
    def __init__(self, root = "/mnt/scratch/alonsocampana/data/",
                 dataset = "GDSC1",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        Dataset: Either GDSC1 or GDSC2
        target: Either LN_IC50 or AUC
        cell_lines: Data to represent the cell lines. Only expression is implemented.
        gene_subset: A numpy array containing the name of the genes to represent the cell-lines. If None, use all of them.
        """
        if not os.path.exists(root + dataset):
            os.mkdir(root + dataset)
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.gene_subset = gene_subset
        self.filter_missing_ids = filter_missing_ids
        self.dataset = dataset
        self.root = root
        if dataset == "GDSC1":
            if not os.path.exists(root + "data/raw/gdsc1raw.csv"):
                self.data = pd.read_csv("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_public_raw_data_24Jul22.csv.zip")
                self.data.to_csv(root + "data/raw/gdsc1raw.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc1raw.csv", index_col = 0)
        elif dataset == "GDSC2":
            if not os.path.exists(root + "data/raw/gdsc2raw.csv"):
                self.data = pd.read_csv("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_public_raw_data_24Jul22.csv.zip")
                self.data.to_csv(root + "data/raw/gdsc2raw.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc2raw.csv", index_col = 0)
        self.process_data()
        self.process_expression()
        self.process_drugs()
    def process_data(self):
        file = f"{self.root}{self.dataset}/inhibitions_processed.csv"
        if not os.path.exists(file):
            self.data_subset = self.data.copy()
            self.data_subset["INTENSITY"] = np.log(self.data_subset["INTENSITY"] + 1)
            identifier_col = self.data_subset.loc[:, "COSMIC_ID"].astype(str) + "&"+ self.data_subset.loc[:, "SCAN_ID"].astype(str) +   self.data_subset.loc[:, "DRUGSET_ID"].astype(str) +  self.data_subset.loc[:, "BARCODE"].astype(str) + self.data_subset.loc[:, "SEEDING_DENSITY"].astype(str)
            self.data_subset = self.data_subset.assign(identifier = identifier_col)
            data_viab = self.data_subset.groupby(["CONC", "identifier", "DRUG_ID", "COSMIC_ID"])["INTENSITY"].median()
            blank_vals = self.data_subset.query("TAG == 'NC-0'").groupby("identifier")["INTENSITY"].median()
            posblank_vals = self.data_subset.query("TAG == 'B'").groupby("identifier")["INTENSITY"].median()
            data_viab = data_viab.reset_index()
            max_vals = blank_vals.loc[data_viab.loc[:, "identifier"]]
            min_vals = posblank_vals.loc[data_viab.loc[:, "identifier"]]
            data_viab["INTENSITY"] = (data_viab["INTENSITY"].to_numpy() - min_vals.to_numpy())/(max_vals.to_numpy().squeeze() - min_vals.to_numpy().squeeze())
            data_viab = data_viab.groupby(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].mean().reset_index()
            data_viab.columns = ["drug", "cell", "z", "y"]
            data_viab.to_csv(file)
        drug_table = pd.read_csv("data/gdscidtoname.csv").rename (columns = {"DRUG_ID" : "drug"})
        self.data = pd.read_csv(file, index_col=0).merge(drug_table.loc[:, ["DRUG_NAME", "drug"]], on="drug")
        self.data  = self.data.drop("drug", axis=1).rename(columns = {"DRUG_NAME":"drug"})
    def process_expression(self):
        root = self.root
        if not os.path.exists(root + "data/processed/gdsc_expression.csv"):
            data = pd.read_csv("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip", compression = "zip", sep = "\t")
            data = data.set_index("GENE_SYMBOLS").iloc[:, 1:].T
            data.index = data.index.str.extract("DATA.([0-9]+)").to_numpy().squeeze()
            self.cell_lines = data.reset_index(drop=False).groupby("index").first()
            self.cell_lines.to_csv(root + "data/processed/gdsc_expression.csv")
        self.expression = pd.read_csv(root + "data/processed/gdsc_expression.csv")
        self.expression = self.expression.rename(columns = {"index":"cell"})
    def process_drugs(self):
        file = f"{self.root}{self.dataset}/drugs_processed.csv"
        if not os.path.exists(file):
            drugs = pd.read_csv("data/GDSC_smiles.csv")
            drugs.columns = ["drug", "smiles"]
            drugs.to_csv(file)
        self.drugs = pd.read_csv(file, index_col= 0)
    def __str__(self):
        return f"{self.dataset}_raw"
class NCI60():
    def __init__(self, ):
        root = "/mnt/mlshare/alonsocampana/data/"
        self.root = root
        if not os.path.isdir(root):
            os.mkdir(root)
        filename = "doseresp.zip"
        url = "https://wiki.nci.nih.gov/download/attachments/147193864/DOSERESP.zip?version=9&modificationDate=1696300958000&api=v2"
        if not os.path.exists(root+filename):
            download_file(url, filename, root)
            zip = zipfile.ZipFile(root+filename)
            zip.extract("DOSERESP.csv", root)
        self.process_data()
        self.process_expression()
        self.process_drugs()
    def process_data(self):
        root = self.root
        if not os.path.exists(root + "inhibition_processed.csv"):
            doseresp = pd.read_csv(self.root + "DOSERESP.csv")
            doseresp = doseresp.query("CONCENTRATION_UNIT == 'M'").loc[:, ["NSC", "CONCENTRATION", "CELL_NAME", "AVERAGE_GIPRCNT"]]
            doseresp.columns = ["drug", "z", "cell", "y"]
            doseresp.to_csv(root + "inhibition_processed.csv")
        self.data = pd.read_csv(root + "inhibition_processed.csv", index_col=0)
        self.data.loc[:,"z"] = 10**self.data.loc[:,"z"]
        self.data.loc[:,"y"] = self.data.loc[:,"y"]/100
    def process_expression(self):
        root = self.root
        file = self.root + "expression_processed.csv"
        if not os.path.exists(file):
            expression = pd.read_csv("data/expression_nsc.csv")
            expression.rename(columns = {"cellname":"cell"}).to_csv(root + "expression_processed.csv")
        self.expression = pd.read_csv(file, index_col=0)
    def process_drugs(self):
        root = self.root
        file = root + "drugs_processed.csv"
        if not os.path.exists(file):
            smiles_ori = pd.read_csv("data/smiles_ni.csv").dropna()
            smiles_ori["DRUG_ID"] = smiles_ori["DRUG_ID"].astype(int)
            smiles_ori.columns = ["nsc", "smiles"]
            smiles_nw = pd.read_csv("data/nscs_smiles.csv", index_col=0)
            smiles = smiles_nw.reset_index().merge(smiles_ori, how="outer").set_index("nsc").sort_index()
            is_missing_ori = smiles.loc[:, "smiles"].isna()
            all_smiles = pd.concat([smiles.loc[is_missing_ori].loc[:, ["1"]].rename(columns = {"1":"smiles"}), 
              smiles.loc[~is_missing_ori].loc[:, ["smiles"]]]).reset_index().rename(columns = {"nsc":"drug"}).to_csv(file)
        self.drugs = pd.read_csv(file, index_col=0)
    def __str__(self):
        return f"NCI60_raw"

class PRISM():
    def __init__(self):
        root = "/mnt/scratch/alonsocampana/data/PRISM/"
        self.root = root
        if not os.path.isdir(root):
            os.mkdir(root)
        self.logfold_collapsed = download_or_read("https://figshare.com/ndownloader/files/20237757", "PRISM_logfold_collapsed.csv", root)
        self.cell_info = download_or_read("https://figshare.com/ndownloader/files/20237769", "cell_line_info.csv", root)
        self.treatment = download_or_read("https://figshare.com/ndownloader/files/20237763", "treatment_info.csv", root)
        if not exists(root + "CCLE_exp.csv"):
            download_file("https://figshare.com/ndownloader/files/22897979", "CCLE_exp.csv", root)
        self.process_data()
        self.process_expression()
        self.process_drugs()
    def process_data(self):
        root = self.root
        if not exists(root + "processed_data.csv"):
            logfold_collapsed = self.logfold_collapsed.set_index("Unnamed: 0").stack().reset_index()
            logfold_collapsed.columns = ["depmap_id", "column_name", "y"]
            logfold_collapsed = logfold_collapsed.merge(self.treatment.loc[:, ["column_name", "dose", "name"]]).drop("column_name", axis=1)
            logfold_collapsed.columns  = ["cell", "y", "z", "drug"]
            logfold_collapsed.to_csv(root + "processed_data.csv")
        logfold_collapsed = pd.read_csv(root + "processed_data.csv", index_col=0)
        self.data = logfold_collapsed
    def process_expression(self):
        root = self.root
        if not exists(root + "expression_processed.csv"):
            expression = pd.read_csv(root +"CCLE_exp.csv", index_col = 0)
            expression.columns = expression.columns.str.extract("(.*) \(").squeeze()
            expression.reset_index().rename(columns = {"index":"cell"}).to_csv(root + "expression_processed.csv")
        else:
            expression = pd.read_csv(root + "expression_processed.csv", index_col=0)
        self.expression = expression
    def process_drugs(self):
        root = self.root
        if not exists(root + "drugs_processed.csv"):
            self.treatment.loc[:, ["name", "smiles"]].drop_duplicates().to_csv(root + "drugs_processed.csv")
        self.drugs = pd.read_csv(root + "drugs_processed.csv", index_col=0)
        self.drugs.columns = ["drug", "smiles"]
    def __str__(self):
        return f"PRISM_raw"
        
class CTRPv2():
    def __init__(self, root = "/mnt/scratch/alonsocampana/data/"):
        url = "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip"
        filename = "CTRPv2.0_2015_ctd2_ExpandedDataset.zip"
        self.root = root
        if not os.path.isdir(root):
            os.mkdir(root)
        if not os.path.isdir(root + "/CTRPv2/"):
            os.mkdir(root + "/CTRPv2/")
        if not exists(root+filename):
            download_file(url, filename, root)
            zip = zipfile.ZipFile(root+filename)
            zip.extractall(root + "/CTRPv2/")
        if not exists(root + "CCLE_exp.csv"):
            download_file("https://figshare.com/ndownloader/files/22897979", "CCLE_exp.csv", root)
        if not exists(root + "CCLE_info.csv"):
            download_file("https://figshare.com/ndownloader/files/22629137", "CCLE_info.csv", root)
        self.process_drugs()
        self.process_cells()
        self.process_data()
    def process_drugs(self):
        root = self.root
        if not exists(self.root + "/CTRPv2/" + "processed_drugs.csv"):
            drugs = pd.read_csv(root + "/CTRPv2/" + "v20.meta.per_compound.txt", sep = "\t").loc[:, ["master_cpd_id", "cpd_smiles"]]
            drugs.columns = ["drug", "smiles"]
            drugs.to_csv(self.root + "/CTRPv2/" + "processed_drugs.csv")
        else:
            drugs = pd.read_csv(self.root + "/CTRPv2/" + "processed_drugs.csv", index_col=0)
        self.drugs = drugs
    def process_cells(self):
        root = self.root
        if not exists(root + "/CTRPv2/" + "expression_processed.csv"):
            expression = pd.read_csv(root +"CCLE_exp.csv", index_col = 0)
            expression.columns = expression.columns.str.extract("(.*) \(").squeeze()
            expression.reset_index().rename(columns = {"index":"cell"}).to_csv(root + "/CTRPv2/" + "expression_processed.csv")
        else:
            expression = pd.read_csv(root + "/CTRPv2/" + "expression_processed.csv", index_col=0)
        self.expression = expression
    def process_data(self):
        root = self.root
        if not exists(root + "/CTRPv2/" + "processed_inhibitions.csv"):
            data = pl.read_csv(root + "/CTRPv2/" + "v20.data.per_cpd_pre_qc.txt", separator = "\t").to_pandas().loc[:, ["experiment_id","master_cpd_id",  "cpd_conc_umol", "cpd_avg_pv"]]
            CCLE_info = pd.read_csv(root +"CCLE_info.csv")
            cell_lines = pl.read_csv(root + "/CTRPv2/" + "v20.meta.per_cell_line.txt", separator = "\t").to_pandas().loc[:, ["master_ccl_id","ccl_name"]]
            experiments = pl.read_csv(root + "/CTRPv2/" + "v20.meta.per_experiment.txt", separator = "\t").to_pandas().loc[:, ["experiment_id", "baseline_signal", "master_ccl_id"]].merge(cell_lines).merge(CCLE_info, left_on = "ccl_name", right_on = "stripped_cell_line_name").loc[:, ["experiment_id", "DepMap_ID"]]
            valid_rows = (data.set_index(["experiment_id", "master_cpd_id", "cpd_conc_umol"]).unstack().isna().sum(axis=1) == 307)
            data.set_index(["experiment_id", "master_cpd_id", "cpd_conc_umol"]).unstack().iloc[valid_rows.to_numpy()].stack().to_csv(root + "/CTRPv2/" + "viabilities_processed.csv")
            data = pd.read_csv(root + "/CTRPv2/" + "viabilities_processed.csv")
            data = experiments.merge(data)
            data = data.loc[:, ["DepMap_ID", "master_cpd_id", "cpd_conc_umol", "cpd_avg_pv"]]
            data.columns = ["cell", "drug", "z", "y"]
            data.to_csv(root + "/CTRPv2/" + "processed_inhibitions.csv")
        else:
            data = pd.read_csv(root + "/CTRPv2/" + "processed_inhibitions.csv", index_col=0)
        self.data = data
    def __str__(self):
        return f"CTRPv2_raw"
    
class CurveSplitter():
    def __init__(self, data, folds = 9, random_state=RANDOM_SEED, shuffle = True, leave_out = None):
        self.leave_out = leave_out
        self.n = folds
        self.seed = random_state
        self.data = data
        self.shuffle = shuffle
        self.n_concs = self.infer_concs()
        self.get_folds()
    def infer_concs(self):
        data = self.data
        n_concs = data.groupby(["cell", "drug"])["z"].count()
        assert (n_concs.iloc[0] == n_concs).all(), "all concentrations must be equal"
        return n_concs.iloc[0]
    def pairwise_concs(self):
        return self.data.groupby(["drug", "cell"])["z"].unique()
    def drugwise_concs(self):
        return self.data.groupby(["drug"])["z"].unique()
    def cell_lines(self):
        return self.data.loc[:,"cell"].unique()
    def drugs(self):
        return self.data.loc[:,"drug"].unique()
    def __getitem__(self, idx):
        return pd.concat([self.folds[i] for i in range(self.n) if (i != idx)&(i != self.leave_out)]), self.folds[idx]
    
class PrecisionOncologySplitter(CurveSplitter):
    def __init__(self, data, folds = 10, random_state=RANDOM_SEED, shuffle = True, leave_out = None):
        super().__init__(data=data,
              folds=folds,
              random_state = random_state,
              shuffle = shuffle,
              leave_out = leave_out)
    def get_folds(self):
        np.random.seed(self.seed)
        lines = self.cell_lines()
        if self.shuffle:
            np.random.shuffle(lines)
        np.random.seed(self.seed)
        split_lines = np.array_split(lines, self.n)
        self.folds = []
        for i in range(self.n):
            lines_fold = split_lines[i]
            self.folds += [self.data.query(f"cell in @lines_fold")]
            
class DrugDiscoverySplitter(CurveSplitter):
    def __init__(self, data, folds = 10, random_state=RANDOM_SEED, shuffle = True, leave_out=None):
        super().__init__(data=data,
              folds=folds,
              random_state = random_state,
              shuffle = shuffle,
                        leave_out = leave_out)
    def get_folds(self):
        np.random.seed(self.seed)
        drugs = self.drugs()
        if self.shuffle:
            np.random.shuffle(drugs)
        np.random.seed(self.seed)
        split_drugs = np.array_split(drugs, self.n)
        self.folds = []
        for i in range(self.n):
            drugs_fold = split_drugs[i]
            self.folds += [self.data.query(f"drug in @drugs_fold")]
            
class SmoothingSplitter(CurveSplitter):
    def __init__(self, data, folds = None, random_state=RANDOM_SEED, shuffle = True, leave_out=None):
        self.data = data
        folds = self.infer_concs()
        super().__init__(data=data,
              folds=folds,
              random_state = random_state,
              shuffle = shuffle,
                        leave_out = leave_out)
    def get_folds(self):
        np.random.seed(self.seed)
        sampled_folds = self.pairwise_concs().apply(lambda x: np.random.choice(x, self.n, replace=False))
        self.folds = [sampled_folds.apply(lambda x: x[i]).reset_index().merge(self.data) for i in range(self.n)]
        
class ExtrapolationSplitter(CurveSplitter):
    def __init__(self, data, folds = 10, random_state=RANDOM_SEED, shuffle = True, leave_out=None):
        super().__init__(data=data,
              folds=folds,
              random_state = random_state,
              shuffle = shuffle,
                        leave_out = leave_out)
    def get_folds(self):
        np.random.seed(self.seed)
        drugs = self.drugs()
        if self.shuffle:
            np.random.shuffle(drugs)
        np.random.seed(self.seed)
        self.split_drugs = np.array_split(drugs, self.n)
    def __getitem__(self, idx):
        highest_conc = self.drugwise_concs().loc[self.split_drugs[idx]].apply(lambda x: x[-1]).reset_index()
        concat_concs = highest_conc.loc[:, "z"].astype(str) + highest_conc.loc[:, "drug"].astype(str)
        concat_data = self.data.loc[:, "z"].astype(str) + self.data.loc[:, "drug"].astype(str)
        test_set = concat_data.isin(concat_concs.to_numpy())
        return self.data.loc[~test_set], self.data.loc[test_set]
    
class InterpolationSplitter(CurveSplitter):
    def __init__(self, data, folds = None, random_state=RANDOM_SEED, shuffle = True, leave_out=None):
        self.data = data
        folds = self.infer_concs() - 2
        super().__init__(data=data,
              folds=folds,
              random_state = random_state,
              shuffle = shuffle,
                        leave_out = leave_out)
    def get_folds(self):
        np.random.seed(self.seed)
        self.folds = self.drugwise_concs().apply(lambda x: np.random.choice(x[1:-1], self.n))
    def __getitem__(self, idx):
        dropped_conc = self.folds.apply(lambda x: x[idx]).reset_index()
        concat_concs = dropped_conc.loc[:, "z"].astype(str) + dropped_conc.loc[:, "drug"].astype(str)
        concat_data = self.data.loc[:, "z"].astype(str) + self.data.loc[:, "drug"].astype(str)
        test_set = concat_data.isin(concat_concs.to_numpy())
        return self.data.loc[~test_set], self.data.loc[test_set]
    
def summarize_dataset(dataset, CUTOFF = None):
    n_points = dataset.data.groupby(["drug", "cell"])["z"].count().value_counts()
    if CUTOFF is not None:
        points_ = (n_points[n_points > CUTOFF]).index.to_numpy()
    else:
        points_ = (n_points.iloc[[0]]).index.to_numpy()
    filtered_pairs = dataset.data.groupby(["drug", "cell"])["z"].count().reset_index().query("z in @points_")
    concat_in = dataset.data["drug"].astype(str) + dataset.data["cell"].astype(str)
    concat_out = filtered_pairs["drug"].astype(str) + filtered_pairs["cell"].astype(str)
    is_in_selected = concat_in.isin(concat_out)
    filtered_data = dataset.data.loc[is_in_selected]
    drugs = dataset.drugs.loc[:, "drug"].unique()
    cells = dataset.expression.loc[:, "cell"].unique()
    filtered_data = filtered_data.query("drug in @drugs & cell in @cells")
    return {"Number of different concentrations": filtered_data["z"].unique().shape[0],
            "Number of points per curve" : points_,
            "Number of drugs": filtered_data["drug"].unique().shape[0],
            "Number of cell-lines": filtered_data["cell"].unique().shape[0],
            "Number of different points": filtered_data.shape[0]}

def process_dataset(dataset, CUTOFF = None, add_fingerprint = True):
    n_points = dataset.data.groupby(["drug", "cell"])["z"].count().value_counts()
    if CUTOFF is not None:
        points_ = (n_points[n_points > CUTOFF]).index.to_numpy()
    else:
        points_ = (n_points.iloc[[0]]).index.to_numpy()
    if str(dataset) == "NCI60_raw":
        filtered_pairs = dataset.data.groupby(["drug", "cell"])["z"].nunique().reset_index().query("z in @points_")
    else:
        filtered_pairs = dataset.data.groupby(["drug", "cell"])["z"].count().reset_index().query("z in @points_")
    concat_in = dataset.data["drug"].astype(str) + dataset.data["cell"].astype(str)
    concat_out = filtered_pairs["drug"].astype(str) + filtered_pairs["cell"].astype(str)
    is_in_selected = concat_in.isin(concat_out)
    filtered_data = dataset.data.loc[is_in_selected]
    if str(dataset) == "NCI60_raw":
        filtered_data = filtered_data.groupby(["drug", "cell", "z"]).median().reset_index()
    drugs = dataset.drugs.loc[:, "drug"].unique()
    cells = dataset.expression.loc[:, "cell"].unique()
    filtered_data = filtered_data.query("drug in @drugs & cell in @cells")
    drugs = filtered_data.loc[:, "drug"].unique()
    cells = filtered_data.loc[:, "cell"].unique()
    filtered_drugs = dataset.drugs.query("drug in @drugs")
    filtered_expression = dataset.expression.query("cell in @cells")
    graph_file = f"{dataset.root}/{str(dataset)}_graphs.pt"
    if not os.path.exists(graph_file):
        graphs = GraphCreator(use_supernode = True)(filtered_drugs.loc[:, "smiles"].to_numpy(), filtered_drugs.loc[:, "drug"].to_numpy())
        torch.save(graphs, graph_file)
    graphs = torch.load(graph_file)
    if add_fingerprint:
        fp_file = f"{dataset.root}/{str(dataset)}_fp.pt"
        if not os.path.exists(fp_file):
            fp = FingerprintFeaturizer()
            drugs_in_g = np.array(list(graphs.keys()))
            drugs_in_g = filtered_drugs.query("drug in @drugs_in_g")
            fps = fp(drugs_in_g.loc[:, "smiles"].to_numpy(), drugs_in_g.loc[:, "drug"].to_numpy())
            torch.save(fps, fp_file)
        fps = torch.load(fp_file)
        for dr_n in fps.keys():
            try:
                graphs[dr_n]["fingerprint"] = fps[dr_n].unsqueeze(0)
            except:
                pass
    exp_file = f"{dataset.root}/{str(dataset)}_exp.pt"
    if not os.path.exists(exp_file) & False:
        paccmann_list = pd.read_csv("https://raw.githubusercontent.com/prassepaul/mlmed_ranking/main/data/gdsc_data/paccmann_gene_list.txt", header=None).to_numpy().squeeze()
        exp = ExpCreator()(filtered_expression, paccmann_list)
        torch.save(exp, exp_file)
    exp = torch.load(exp_file)
    featurized_drugs = list(graphs.keys())
    featurized_cells = list(exp.keys())
    return filtered_data.query("drug in @featurized_drugs & cell in @featurized_cells"), exp, graphs

from functools import lru_cache
from torch_geometric.data import DataLoader

def get_train_test_data(dataset, setting, fold, drop_random = 0, drop_systematic = 0, leave_out = None):
    if dataset == "GDSC1":
        dataset = GDSC()
    if dataset == "GDSC2":
        dataset = GDSC(dataset = "GDSC2")
    if dataset == "CTRPv2":
        dataset = CTRPv2()
    if dataset == "PRISM":
        dataset = PRISM()
    if dataset == "NCI60":
        dataset = NCI60()
    data, exp, graphs = process_dataset(dataset)
    if setting == "precision_oncology":
        splitter = PrecisionOncologySplitter(data, leave_out=leave_out)
    elif setting == "drug_discovery":
        splitter = DrugDiscoverySplitter(data, leave_out=leave_out)
    elif setting == "interpolation":
        splitter = InterpolationSplitter(data, leave_out=leave_out)
    elif setting == "extrapolation":
        splitter = ExtrapolationSplitter(data, leave_out=leave_out)
    elif setting == "smoothing":
        splitter = SmoothingSplitter(data)
    train_data, test_data = splitter[fold]
    if drop_random:
        train_data, test_data, rescale_training = missing_at_random(train_data, test_data, ratio=drop_random)
    elif drop_systematic:
        train_data, test_data, rescale_training = missing_systematically(train_data, test_data, ratio=drop_systematic)
    return train_data, test_data, exp, graphs

@lru_cache(maxsize=None)
def get_dataloaders(dataset,
                    setting,
                    fold,
                    batchsize, 
                    only_test = False,
                    drop_random = 0,
                    drop_systematic = 0,
                   leave_out=None):
    rescale_training = 1  # dirty way to keep training length consistant
    train_data, test_data, exp, graphs = get_train_test_data(dataset, setting, fold, leave_out=leave_out)
    if drop_random:
        train_data, test_data, rescale_training = missing_at_random(train_data, test_data, ratio=drop_random)
    elif drop_systematic:
        train_data, test_data, rescale_training = missing_systematically(train_data, test_data, ratio=drop_systematic)
    test_loader = DataLoader(CurveDataset(test_data, graphs, exp), batch_size=batchsize, num_workers = 16)
    if only_test:
        return None, test_loader
    train_loader = DataLoader(CurveDataset(train_data, graphs, exp), batch_size=batchsize, num_workers = 16, shuffle = True, drop_last = True)
    return train_loader, test_loader, rescale_training

def get_config(dataset, setting):
    study_name = f"{dataset}_{setting}_0"
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.load_study(study_name, storage_name)
    return study.best_params
def serialize_config(dataset, setting):
    config = get_config(dataset, setting)
    config_copy = {"network":{"hidden_dim":config["hidden_dim"]*156,
                "n_knots": 16,
                "n_pooling_heads": config["n_pooling_heads"],
                "dropout_cattn" : config["dropout_cattn"],
                "dropout_genes" : config["dropout_genes"],
                "dropout_fusion" :config["dropout_fusion"],
                "dropout_fc" :config["dropout_fc"],
                "dropout_nodes_attn" :config["dropout_nodes_attn"],
                "n_layers" :config["n_layers"],
                "n_heads" : config["n_heads"],
                "use_normalization" : config["use_normalization"],
                "use_normalization_fc" : config["use_normalization_fc"],
                "use_normalization_fusion": config["use_normalization_fusion"],
                "fusion" :"hill",
                "n_transformers":config["n_transformers"],
                "activation_fn": config["activation_fn"],
                "crossattn" : config["crossattn"],
                "linear_head":False},
         
         "optimizer":{"batch_size":256,
                      "learning_rate":config["learning_rate"],
                      "gamma_factor":0.5,
                      "alpha":config["alpha"],
                      "clip_norm":config["clip_norm"]},
         "env":{"debug":True,
                "mixed_precision":True,
                "fingerprint": False,
                 "missing_systematically":0.0,
                 "missing_random":0.0,
                  "interpolation_augment":0.0}}
    try:
        config_copy["network"]["transform_log_conc"] = config["transform_log_conc"]
    except:
        config_copy["network"]["transform_log_conc"] = False
    try:
        config_copy["network"]["fc_hidden"] = config["fc_hidden"]
    except:
        config_copy["network"]["fc_hidden"] = 2048
    return config_copy

class Tabularizer():
    def __init__(self, dataset, splitter):
        self.dataset = dataset
        self.data, self.cells, self.drugs = process_dataset(dataset)
        self.splitter = splitter(self.data)

    def pivot_df(self, df):
        n = self.n_concs
        repeats = df.shape[0]//n
        idxs = np.tile(np.arange(n), repeats)
        with_ids = df.sort_values("z").groupby(["drug", "cell", "z"])["y"].median().reset_index().set_index(["drug", "cell"]).assign(z = idxs)
        y = with_ids.reset_index().pivot_table(index=["drug", "cell"], values= "y", columns = "z")
        return y

    def generate_X_y_arrays(self, df):
        n_concs = self.n_concs
        df_pivot = self.pivot_df(df)
        drugs_ids = df_pivot.reset_index().iloc[:, 0].to_numpy()
        cell_ids = df_pivot.reset_index().iloc[:, 1].to_numpy()
        all_fps = np.array([self.drugs[x]["fingerprint"].numpy() for x in drugs_ids]).squeeze()
        all_exp = np.array([self.cells[x].numpy() for x in cell_ids]).squeeze()
        X_ = np.concatenate([all_fps, all_exp], 1)
        y_ = df_pivot.to_numpy()
        return X_, y_
    
    def __getitem__(self, x, drop_random = 0, drop_systematic = 0):
        train, test = self.splitter[x]
        if drop_random:
            train, test, rescale_training = missing_at_random(train, test, ratio=drop_random)
        elif drop_systematic:
            train, test, rescale_training = missing_systematically(train, test, ratio=drop_systematic)
        self.n_concs = self.splitter.n_concs
        return self.generate_X_y_arrays(train), self.generate_X_y_arrays(test)

def get_tabular(dataset,
                setting,
                fold,
                drop_random = 0,
                drop_systematic = 0):
    if dataset == "GDSC1":
        dataset = GDSC()
    elif dataset == "GDSC2":
        dataset = GDSC(dataset = "GDSC2")
    elif dataset == "CTRPv2":
        dataset = CTRPv2()
    elif dataset == "PRISM":
        dataset = PRISM()
    elif dataset == "NCI60":
        dataset = NCI60()
    if setting == "precision_oncology":
        splitter = PrecisionOncologySplitter
    elif setting == "drug_discovery":
        splitter = DrugDiscoverySplitter
    elif setting == "interpolation":
        splitter = InterpolationSplitter
    elif setting == "extrapolation":
        splitter = ExtrapolationSplitter
    elif setting == "smoothing":
        splitter = SmoothingSplitter
    tabular = Tabularizer(dataset, splitter, drop_random, drop_systematic)
    return tabular[fold]

def interpolate_tensor(x, points = 1):
    concat = torch.cat([x.unsqueeze(-1)[:, :-1, :], x.unsqueeze(-1)[:, 1:, :]], -1)
    interp = torch.nn.functional.interpolate(concat, 2+points, mode="linear")
    interpolated = torch.cat([interp[:, 0, 0].unsqueeze(-1), interp[:, :, 1:-1].flatten(-2), interp[:, :, -1]], 1)
    return interpolated

def missing_systematically(train, test, ratio=0.5):
    np.random.seed(RANDOM_SEED)
    keep_ratio = 1- ratio
    unique_drugs = train.loc[:, "drug"].unique()
    unique_lines = train.loc[:, "cell"].unique()
    keep_drugs = np.random.choice(unique_drugs, int(len(unique_drugs)*keep_ratio), replace=False)
    keep_lines = np.random.choice(unique_lines, int(len(unique_lines)*keep_ratio), replace=False)
    train_ = train.query("drug in @keep_drugs & cell in @keep_lines")
    test_ = test.query("drug in @keep_drugs & cell in @keep_lines")
    return train_, test_, len(train)/len(train_)

def missing_at_random(train, test, ratio=0.5):
    np.random.seed(RANDOM_SEED)
    keep_ratio = 1- ratio
    concat_str = train.loc[:, "drug"] + train.loc[:, "cell"].astype(str)
    concat_str_test = test.loc[:, "drug"] + test.loc[:, "cell"].astype(str)
    unique_pairs = concat_str.unique()
    keep_pairs = np.random.choice(unique_pairs, int(len(unique_pairs)*keep_ratio), replace=False)
    train_ = train.loc[concat_str.isin(keep_pairs)]
    test_ = test.loc[concat_str_test.isin(keep_pairs)]
    return train_, test_, len(train)/len(train_)