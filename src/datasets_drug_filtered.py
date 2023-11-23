import numpy as np
import pandas as pd

import csv

from functools import cached_property
from itertools import product
# import logging
# import os


# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
# from torch_geometric import data as DATA
# from torch_geometric.data import InMemoryDataset
import itertools

DRUG_SMILES_PATH = "../data/raw/drug_smiles.csv"
DATA_BASE_DIR = "../"

RANDOM_STATE = 31

GENES_324 = pd.read_csv("../data/raw/gene2ind.txt", header=None)[0].tolist()
GENES_285 = pd.read_csv("../data/raw/gene2ind_285genes.txt", header=None)[0].tolist()
modified_GENES_285 = [c[0]+c[1] for c in list(itertools.product(GENES_285, ["_loss", "_", "_gain"]))]
clinvar_suffixes = [
    "_piu_max", "_piu_sum", "_piu_mean", "_piu_count",
    "_lu_max", "_lu_sum", "_lu_mean", "_lu_count",
    "_ncu_max", "_ncu_sum", "_ncu_mean", "_ncu_count",
    "_pathogenic_max", "_pathogenic_sum", "_pathogenic_mean", "_pathogenic_count",
    "_vus_max", "_vus_sum", "_vus_mean", "_vus_count",
    "_benign_max", "_benign_sum", "_benign_mean", "_benign_count",
]
modified_GENES_285_clinvar = [c[0]+c[1] for c in list(itertools.product(GENES_285, clinvar_suffixes))]

ALL_CCLE_GENES = pd.read_csv("../data/raw/gene2ind_allgenes.txt", header=None)[0].tolist()

cell_line_auc_df = pd.read_csv("../data/raw/cell_drug_auc_final_1111.csv")
cell_line_auc_df["depmap_id"] = cell_line_auc_df["ARXSPAN_ID"].astype("string")
cell_line_auc_df.drop("ARXSPAN_ID", axis=1, inplace=True)
cell_line_auc_df.set_index(["depmap_id"], inplace=True)
CELL_LINE_DRUGS_ALL = cell_line_auc_df.columns.tolist()


fname_drugs_cat = "../data/raw/druid-druglist.csv"
df_drugs_cat = pd.read_csv(fname_drugs_cat)
list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])]["drug_name"].tolist()

CELL_LINE_DRUGS_CAT_1 = list(
    set(CELL_LINE_DRUGS_ALL).intersection(set(list_drugs_cat1))
)

class CellLineDataset(Dataset):
    """
    Base class for datasets that hold cell line information
    """

    base_dir = DATA_BASE_DIR
    entity_identifier_name = None

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        """

    pass

class TcgaDataset(Dataset):
    """
    Base class for datasets that hold TCGA information
    """

    base_dir = DATA_BASE_DIR

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        #Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
        """

    pass

class Rad51Dataset(Dataset):
    """
    Base class for datasets that hold RAD51 information

    """

    base_dir = DATA_BASE_DIR

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        #Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
        """

    pass

class NuhDataset(Dataset):
    """
    Base class for datasets that hold NUH patient information
    """

    base_dir = DATA_BASE_DIR

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set - {self.response_column}
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        #Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
        """

    pass

class CategoricalAnnotatedCellLineDatasetFilteredByDrug(CellLineDataset):
    """
    Cell line data with categorical annotation features from annovar
    """

    entity_identifier_name = "depmap_id"

    def __init__(
        self,
        is_train=True,
        filter_for="rad51",
        xon17 = False,
        sample_id = 0
    ):
        """
        Parameters
        ----------
        is_train : bool
            Returns items from the train or test split (defaults to True)
        filter_for : str
            Filters for drugs in the dataset passed (defaults to "rad51"). Can also take in value "tcga" or "nuh_crc"
        xon17 : bool
            Uses the variant annotation that returns the score for each gene as 1 + x/17 (defaults to False)

        """
        self.is_train = is_train
        self.sample_id = sample_id
        if xon17 == False:
            self.df_reprn_mut = pd.read_csv("../data/processed/ccle_anno_features.csv",)
        else:
            raise("xon17 not implemented for ccle_anno_features.csv")
        self.df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in self.df_reprn_mut.columns:
            mask = mask & (self.df_reprn_mut[col] == 0)

        self.df_reprn_mut = self.df_reprn_mut[~mask].copy()

        df_auc = pd.read_csv("../data/raw/cell_drug_auc_final_1111.csv")
        df_auc["depmap_id"] = df_auc["ARXSPAN_ID"].astype("string")
        df_auc.drop("ARXSPAN_ID", axis=1, inplace=True)
        df_auc.set_index(["depmap_id"], inplace=True)

        # Filter for Rad51 drugs only
        self.filter_for = filter_for
        if self.filter_for == "rad51":
            list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOXORUBICIN"]
        elif self.filter_for == "tcga":
            list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOCETAXEL", "5-FLUOROURACIL", "CYCLOPHOSPHAMIDE"]
        elif self.filter_for == "nuh_crc":
            list_drugs = ["5-FLUOROURACIL", "OXALIPLATIN", "IRINOTECAN", "CETUXIMAB"]
        filtered_drugs = [col for col in df_auc.columns if col in list_drugs]
        df_auc = df_auc[filtered_drugs]

        train_cell_lines_ids = pd.read_csv(
            f"../data/raw/train_celllines_filtered4{filter_for}_drugs_sample{sample_id}.csv", header=None
        )[0].values

        test_cell_lines_ids = pd.read_csv(
            f"../data/raw/test_celllines_filtered4{filter_for}_drugs_sample{sample_id}.csv", header=None
        )[0].values

        if is_train is not None:
            if self.is_train:
                required_cell_line_ids = train_cell_lines_ids
            else:
                required_cell_line_ids = test_cell_lines_ids
        else:
            required_cell_line_ids = np.concatenate(
                [train_cell_lines_ids, test_cell_lines_ids]
            )

        
        y_df = df_auc[df_auc.index.isin(required_cell_line_ids)].copy()

        # The below filter is to remove those cellines for which there are no
        # annotation features available (likely due to the absence of point
        # mutations in such cases)
        #
        # TODO: Check how to represent those cases that do not have any point mutations
        y_df = y_df[y_df.index.isin(self.df_reprn_mut.index.get_level_values(0))].copy()

        y_df = y_df.reset_index().melt(
            id_vars=["depmap_id",], var_name="drug_name", value_name="auc",
        )

        # When scaling is not done, filter those entries with value -99999
        self.y_df = y_df[~(y_df.auc < 0)]

    def __len__(self):
        return len(self.y_df)

    def __getitem__(self, idx):
        record = self.y_df.iloc[idx]

        return {
            "depmap_id": record["depmap_id"],
            "drug_name": record["drug_name"],
            "auc": record["auc"],
        }

    @cached_property
    def mutations(self):
        return self.df_reprn_mut

    @cached_property
    def raw_mutations(self):
        df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
        return df_reprn_mut

    @cached_property
    def raw_mutations_all_genes(self):
        df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation_all_genes.csv",)
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut.reindex(columns=ALL_CCLE_GENES)
        return df_reprn_mut
    
    @cached_property
    def raw_mutations_285_genes(self):
        df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut[GENES_285]
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_285)
        return df_reprn_mut

    @cached_property
    def embedded_raw_mutations_all_genes(self):
        """
        324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder
        """
        df_reprn_mut_train = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_train.csv", index_col = 0)
        df_reprn_mut_test = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_test.csv", index_col = 0)
        return pd.concat([df_reprn_mut_test, df_reprn_mut_train])

    @cached_property
    def embedded_raw_mutations_all_genes_v2(self):
        """
        324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder with more training
        """
        df_reprn_mut_train = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_train_v2.csv", index_col = 0)
        df_reprn_mut_test = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_test_v2.csv", index_col = 0)
        return pd.concat([df_reprn_mut_test, df_reprn_mut_train])   

    @cached_property
    def gene_exp(self):
        """
        324 dimensional gene expression values for F1 genes
        """
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
        return gene_exp_df
    
    @cached_property
    def gene_exp_285(self):
        """
        285 dimensional intersecting gene set across Tempus, TruSight, F1
        """
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
        return gene_exp_df[GENES_285]

    @cached_property
    def gene_exp_1426(self):
        """
        CODE-AE genes' gene expression
        """
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression_code_ae_genes.csv", index_col=0)
        indices = list(pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0).index)
        return gene_exp_df.loc[indices]

    @cached_property
    def gene_exp_1426_codeae_sample(self):
        """
        gene expression for 1426 genes in CODE-AE alongwith all samples used in CODE-AE
        """
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression_code_ae_genes_and_samples_scaled.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["depmap_id"]).set_index("depmap_id", drop=True)
        return gene_exp_df

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)
        return cnv_df

    @cached_property
    def cnv_285(self):
        cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)
        return cnv_df[modified_GENES_285]

    @cached_property
    def concatenated_raw_mutation_cnv(self):
        # raw mutations 
        df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
        # cnv
        cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)

        return pd.concat([df_reprn_mut, cnv_df], axis = 1)

    @cached_property
    def concatenated_raw_mutation_cnv_285(self):
        # raw mutations 
        df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
        # cnv
        cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)

        return pd.concat([df_reprn_mut, cnv_df], axis = 1)[GENES_285 + modified_GENES_285]

    @cached_property
    def clinvar_gpd_annovar_annotated(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df
    
    @cached_property
    def clinvar_gpd_annovar_annotated_285(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df[modified_GENES_285_clinvar]
    
    @cached_property
    def concatenated_anno_mutation_gene_exp(self):
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv")
        clinvar_gpd_annovar_df.rename(columns={"DepMap_ID": "depmap_id"}, inplace=True)
        clinvar_gpd_annovar_df.set_index("depmap_id", drop=True, inplace=True)

        return pd.concat([gene_exp_df, clinvar_gpd_annovar_df], axis = 1)


class AggCategoricalAnnotatedCellLineDatasetFilteredByDrug(CategoricalAnnotatedCellLineDatasetFilteredByDrug):
    """
    Cell line data with categorical annotation features from annovar aggregated per gene
    """

    @cached_property
    def mutations(self):
        agg_results = {}
        for gene in GENES_324:
            filtered_df = self.df_reprn_mut.filter(regex=f"^{gene}_[a-z]*")
            # agg_results[gene] = filtered_df.mean(axis=1)
            # agg_results[gene] = filtered_df.sum(axis=1)
            curr_result = None
            for col in filtered_df.columns:
                if type(curr_result) == pd.Series:
                    curr_result = curr_result | (filtered_df[col] != 0)
                else:
                    curr_result = filtered_df[col] != 0

            agg_results[gene] = curr_result.astype(np.int32)

        agg_df = pd.DataFrame(agg_results)
        return agg_df

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df

class CategoricalAnnotatedTcgaDatasetFilteredByDrug(TcgaDataset):
    """
    TCGA data, used only for testing
    """

    entity_identifier_name = "submitter_id"

    def __init__(
        self,
        is_train=False,
        filter_for="tcga",
        xon17=False,
        sample_id = 0
    ):
        self.is_train = is_train
        self.sample_id = sample_id
        tcga_response = pd.read_csv("../data/processed/TCGA_drug_response_010222.csv")
        tcga_response.rename(
            columns={
                "patient.arr": self.entity_identifier_name,
                "drug": "drug_name",
                "response": "response_description",
                "response_cat": "response",
            },
            inplace=True,
        )

        if xon17 == False:
            tcga_mutation = pd.read_csv(
            "../data/processed/tcga_anno_features_only_categorical_agg.csv"
        )
        else:
            raise("xon17 not implemented for tcga_anno_features_only_categorical_agg.csv"
        )
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        self.tcga_mutation_filtered = tcga_mutation
        
        if filter_for=="tcga":
            list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOCETAXEL", "5-FLUOROURACIL", "CYCLOPHOSPHAMIDE"]
            tcga_response = tcga_response[tcga_response["drug_name"].isin(list_drugs)]
            tcga_response = tcga_response[
            tcga_response[self.entity_identifier_name].isin(
                tcga_mutation.index.get_level_values(0)
            )
            ]
            self.tcga_response = tcga_response[
                [self.entity_identifier_name, "drug_name", "response"]
            ].copy()
            self.tcga_response = self.tcga_response[
                self.tcga_response.drug_name.isin(list_drugs)
            ].reset_index(drop=True)
        elif filter_for=="nuh_crc":
            list_drugs = ["5-FLUOROURACIL", "OXALIPLATIN", "IRINOTECAN", "CETUXIMAB"]
            tcga_response = tcga_response[tcga_response["drug_name"].isin(list_drugs)]
            tcga_response = tcga_response[
            tcga_response[self.entity_identifier_name].isin(
                tcga_mutation.index.get_level_values(0)
            )
            ]
            self.tcga_response = tcga_response[
                [self.entity_identifier_name, "drug_name", "response"]
            ].copy()
            self.tcga_response = self.tcga_response[
                self.tcga_response.drug_name.isin(list_drugs)
            ].reset_index(drop=True)

        
        train_tcga_ids = pd.read_csv(
            f"../data/raw/train_tcga_filtered4tcga_drugs_sample{sample_id}.csv", header=None
        )[0].values

        test_tcga_ids = pd.read_csv(
            f"../data/raw/test_tcga_filtered4tcga_drugs_sample{sample_id}.csv", header=None
        )[0].values

        if is_train is not None:
            if self.is_train:
                required_tcga_ids = train_tcga_ids
            else:
                required_tcga_ids = test_tcga_ids
        else:
            required_tcga_ids = np.concatenate(
                [train_tcga_ids, test_tcga_ids]
            )

        
        self.tcga_response = self.tcga_response[self.tcga_response.submitter_id.isin(required_tcga_ids)].copy()
        
    def __len__(self):
        return len(self.tcga_response)

    def __getitem__(self, idx):
        record = self.tcga_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def mutations(self):
        return self.tcga_mutation_filtered

    @cached_property
    def raw_mutations(self):
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
        return tcga_mutation
    
    @cached_property
    def raw_mutations_all_genes(self):
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222_all_genes")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=ALL_CCLE_GENES)
        return tcga_mutation
    
    @cached_property
    def raw_mutations_285_genes(self):
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation[GENES_285].reindex(columns=GENES_285)
        return tcga_mutation
    
    @cached_property
    def embedded_raw_mutations_all_genes(self):
        """
        324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder
        """
        df_reprn_mut_train = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_train.csv", index_col = 0)
        df_reprn_mut_test = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_test.csv", index_col = 0)
        return pd.concat([df_reprn_mut_test, df_reprn_mut_train])     
    
    @cached_property
    def embedded_raw_mutations_all_genes_v2(self):
        """
        324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder with more training
        """
        df_reprn_mut_train = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_train_v2.csv", index_col = 0)
        df_reprn_mut_test = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_test_v2.csv", index_col = 0)
        return pd.concat([df_reprn_mut_test, df_reprn_mut_train])

    @cached_property
    def gene_exp(self):
        """
        324 dimensional gene expression values for F1 genes
        """
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
        return gene_exp_df
    
    @cached_property
    def gene_exp_285(self):
        """
        285 dimensional intersecting gene set across Tempus, TruSight, F1
        """
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
        return gene_exp_df[GENES_285]
    
    @cached_property
    def gene_exp_1426(self):
        """
        gene expression values for the 1426 genes used in CODE-AE for our samples
        """
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression_code_ae_genes.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
        return gene_exp_df
    
    @cached_property
    def gene_exp_1426_codeae_sample(self):
        """
        gene expression for 1426 genes in CODE-AE alongwith all samples used in CODE-AE
        """
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression_code_ae_genes_and_samples_scaled.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
        return gene_exp_df
    
    @cached_property
    def cnv(self):
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
        return tcga_cnv
    
    @cached_property
    def cnv_285(self):
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
        return tcga_cnv[modified_GENES_285]

    @cached_property
    def concatenated_raw_mutation_cnv(self):
        # raw mutations
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=GENES_324)

        # cnv
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
        return pd.concat([tcga_mutation, tcga_cnv], axis = 1)
    
    @cached_property
    def concatenated_raw_mutation_cnv_285(self):
        # raw mutations
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=GENES_324)

        # cnv
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
        return pd.concat([tcga_mutation, tcga_cnv], axis = 1)[GENES_285 + modified_GENES_285]
    
    @cached_property
    def raw_mutation_tcga_msk_impact(self):
        # raw mutations TCGA
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.rename(columns={self.entity_identifier_name: "patient_id"}, inplace=True)
        tcga_mutation.set_index("patient_id", drop=True, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=GENES_324)

        # raw mutations MSK Impact
        msk_impact_mutation = pd.read_csv("../data/processed/msk_impact_mutations.csv", index_col=0)
        return pd.concat([tcga_mutation, msk_impact_mutation], axis = 0)

    @cached_property
    def survival_info(self):
        survival_info_df = pd.read_csv("../data/processed/survival_rate_final_010222")
        survival_info_df.rename(
            columns={"demographic.days_to_death": "days"}, inplace=True
        )
        survival_info_df.drop(
            columns=["demographic.vital_status", "days_to_death_scaled"], inplace=True
        )
        return survival_info_df
    
    @cached_property
    def clinvar_gpd_annovar_annotated(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df

    @cached_property
    def clinvar_gpd_annovar_annotated_285(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df[modified_GENES_285_clinvar]
    
    @cached_property
    def concatenated_anno_mutation_gene_exp(self):
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
        gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).rename(columns={"tcga_id": "submitter_id"})
        gene_exp_df.set_index("submitter_id", drop=True, inplace=True)
        
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
        return pd.concat([gene_exp_df, clinvar_gpd_annovar_df], axis = 1).fillna(0)
    

class AggCategoricalAnnotatedTcgaDatasetFilteredByDrug(CategoricalAnnotatedTcgaDatasetFilteredByDrug):
    """
    Aggregated categorical annotations features for TCGA entities
    """

    @cached_property
    def mutations(self):
        agg_results = {}
        for gene in GENES_324:
            filtered_df = self.tcga_mutation_filtered.filter(regex=f"^{gene}_[a-z]*")
            # agg_results[gene] = filtered_df.mean(axis=1)
            # agg_results[gene] = filtered_df.sum(axis=1)
            curr_result = None
            for col in filtered_df.columns:
                if type(curr_result) == pd.Series:
                    curr_result = curr_result | (filtered_df[col] != 0)
                else:
                    curr_result = filtered_df[col] != 0

            agg_results[gene] = curr_result.astype(np.int32)

        agg_df = pd.DataFrame(agg_results)
        return agg_df

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df
    
class CategoricalAnnotatedRad51DatasetFilteredByDrug(Rad51Dataset):
    """
    Patient dataset shared by Robert

    """

    entity_identifier_name = "patient_id"
    response_column = "response"

    def __init__(
        self, is_train=False, filter_for="rad51", xon17 = False, sample_id = 0, only_first_line=False, non_surgery_first_line=False
    ):
        self.is_train = is_train

        self.rad51_mutations_df = pd.read_csv("../data/processed/rad51_mutations.csv")
        self.rad51_mutations_df.set_index("patient_id", inplace=True)
        if xon17 == False:
            self.rad51_anno_features_df = pd.read_csv(
            "../data/processed/rad51_anno_features.csv"
        )
        else:
            raise("xon17 not implemented for rad51_anno_features.csv")
        
        self.rad51_anno_features_df.set_index("patient_id", inplace=True)

        self.rad51_response = pd.read_csv("../data/processed/rad51_response.csv")

        # Filter for patients with mutation info
        self.rad51_response = self.rad51_response[
            self.rad51_response.patient_id.isin(
                self.rad51_anno_features_df.index.get_level_values(0)
            )
        ]
        self.rad51_response = self.rad51_response[
            self.rad51_response.patient_id.isin(
                self.rad51_mutations_df.index.get_level_values(0)
            )
        ]

        # Remove outliers in the dataset - detected from the response distribution (boxplot)
        self.rad51_response = self.rad51_response[
            ~(self.rad51_response.pfs_days > 1000)
        ]

        train_rad51_ids = pd.read_csv(
            f"../data/raw/train_rad51_filtered4rad51_drugs_sample{sample_id}.csv", header=None
        )[0].values

        test_rad51_ids = pd.read_csv(
            f"../data/raw/test_rad51_filtered4rad51_drugs_sample{sample_id}.csv", header=None
        )[0].values
        
        if filter_for=="rad51":
            list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOXORUBICIN"]
            self.rad51_response = self.rad51_response[
                self.rad51_response.drug_name.isin(list_drugs)
            ].reset_index(drop=True)

        if is_train is not None:
            if self.is_train:
                required_rad51_ids = train_rad51_ids
            else:
                required_rad51_ids = test_rad51_ids
        else:
            required_rad51_ids = np.concatenate(
                [train_rad51_ids, test_rad51_ids]
            )

        
        self.rad51_response = self.rad51_response[self.rad51_response.patient_id.isin(required_rad51_ids)].copy()
        if only_first_line == True: # retain only first line patients
            self.rad51_response = self.rad51_response[self.rad51_response.patient_id.str.contains("--1l")]

        imacgo_orig = pd.read_excel("../data/raw/IMACGO_09Nov2021.xlsx", skiprows=[0])
        ids2remove = imacgo_orig[(imacgo_orig["Neoadj (No 0 Yes 1) "] == 0)&(imacgo_orig["Surgical completeness (R0 0 , residual disease < 1cm: 1, residual disease > 1cm: 2, unknown 3, NA: no surgery) "].isin([0, 1]))]["ID"].values
        if non_surgery_first_line == True: # remove patients who underwent surgery before first line treatment - avoids surgery confounder
            self.rad51_response = self.rad51_response[self.rad51_response.patient_id.str.contains("--1l")]
            self.rad51_response["id"] = self.rad51_response["patient_id"].apply(lambda x: x.split("--")[0])
            self.rad51_response = self.rad51_response[~self.rad51_response.id.isin(ids2remove)][["patient_id", "drug_name", "response", "clinical_ben", "pfs_days", "did_progress"]].reset_index(drop=True)



    def __len__(self):
        return len(self.rad51_response)

    def __getitem__(self, idx):
        record = self.rad51_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def survival_info(self):
        survival_info_df = self.rad51_response.copy()
        survival_info_df.rename(columns={"pfs_days": "days"}, inplace=True)
        survival_info_df.drop(columns=["response", "clinical_ben"], inplace=True)
        return survival_info_df

    @cached_property
    def mutations(self):
        return self.rad51_anno_features_df

    @cached_property
    def raw_mutations(self):
        return self.rad51_mutations_df
    
    @cached_property
    def raw_mutations_285_genes(self):
        return self.rad51_mutations_df[GENES_285]

    @cached_property
    def cnv(self):
        rad51_cnv = pd.read_csv("../data/processed/rad51_cnv.csv")
        rad51_cnv.set_index(self.entity_identifier_name, inplace=True)
        rad51_cnv = rad51_cnv.reindex(columns=GENES_324)
        return rad51_cnv

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df
    
    @cached_property
    def clinvar_gpd_annovar_annotated(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_rad51_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df
    
    @cached_property
    def clinvar_gpd_annovar_annotated_285(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_rad51_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df[modified_GENES_285_clinvar]

class AggCategoricalAnnotatedRad51DatasetFilteredByDrug(CategoricalAnnotatedRad51DatasetFilteredByDrug):
    """
    Aggregated categorical annotations features for Rad51 First Line Patient data
    """

    @cached_property
    def mutations(self):
        agg_results = {}
        for gene in GENES_324:
            filtered_df = self.rad51_anno_features_df.filter(regex=f"^{gene}_[a-z]*")
            # agg_results[gene] = filtered_df.mean(axis=1)
            # agg_results[gene] = filtered_df.sum(axis=1)
            curr_result = None
            for col in filtered_df.columns:
                if type(curr_result) == pd.Series:
                    curr_result = curr_result | (filtered_df[col] != 0)
                else:
                    curr_result = filtered_df[col] != 0

            agg_results[gene] = curr_result.astype(np.int32)

        agg_df = pd.DataFrame(agg_results)
        return agg_df

class CategoricalAnnotatedNuhFirstLineDatasetFilteredByDrug(NuhDataset):
    """
    NUH First Line Patient data, used only for testing
    """

    entity_identifier_name = "patient_id"
    response_file_path = "../data/processed/nuh_survival_line1.csv"
    # response_column = "clinical_ben"
    response_column = "response"

    def __init__(
        self,
        is_train=False,
        filter_for="nuh_crc",
        sample_id = 0
    ):
        apply_train_test_filter=True
        only_cat_one_drugs=True
        include_all_cell_line_drugs=False

        nuh_mutation = pd.read_csv(
            "../data/processed/nuh_anno_features_only_categorical_agg.csv"
        )
        nuh_mutation.set_index(self.entity_identifier_name, inplace=True)
        self.nuh_mutation = nuh_mutation
        self.is_train = is_train

        self.nuh_response = pd.read_csv(self.response_file_path)
        self.nuh_response.drop(columns=["Unnamed: 0"], inplace=True)

        # TODO: Find smile strings for the missing drugs and update the csv
        drug_smiles = csv.reader(open(DRUG_SMILES_PATH))
        drug_names = [item[0] for item in drug_smiles]
        self.nuh_response = self.nuh_response[
            self.nuh_response["drug_name"].isin(drug_names)
        ]
        self.nuh_response = self.nuh_response[
            self.nuh_response["patient_id"].isin(
                self.nuh_mutation.index.get_level_values(0)
            )
        ]
        self.nuh_response["response"] = self.nuh_response[self.response_column]

        if include_all_cell_line_drugs:
            self.only_cat_one_drugs = only_cat_one_drugs
            if self.only_cat_one_drugs:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_CAT_1
            else:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_ALL

            nuh_response_with_required_drugs = pd.DataFrame(
                list(
                    product(
                        self.nuh_response[self.entity_identifier_name].unique(),
                        cell_line_drugs_to_use,
                    )
                ),
                columns=[self.entity_identifier_name, "drug_name"],
            )
            nuh_response_with_required_drugs = nuh_response_with_required_drugs.merge(
                self.nuh_response, how="left"
            )
            self.nuh_response = nuh_response_with_required_drugs

        train_nuh_crc_ids = pd.read_csv(
            f"../data/raw/train_nuh_crc_filtered4nuh_crc_drugs_sample{sample_id}.csv", header=None
        )[0].values

        test_nuh_crc_ids = pd.read_csv(
            f"../data/raw/test_nuh_crc_filtered4nuh_crc_drugs_sample{sample_id}.csv", header=None
        )[0].values
        

        if filter_for=="nuh_crc":
            list_drugs = ["5-FLUOROURACIL", "OXALIPLATIN", "IRINOTECAN", "CETUXIMAB"]
            self.nuh_response = self.nuh_response[
                self.nuh_response.drug_name.isin(list_drugs)
            ].reset_index(drop=True)

        if is_train is not None:
            if self.is_train:
                required_nuh_crc_ids = train_nuh_crc_ids
            else:
                required_nuh_crc_ids = test_nuh_crc_ids
        else:
            required_nuh_crc_ids = np.concatenate(
                [train_nuh_crc_ids, test_nuh_crc_ids]
            )

        
        self.nuh_response = self.nuh_response[self.nuh_response.patient_id.isin(required_nuh_crc_ids)].copy()
        # if apply_train_test_filter:
        #     uniq_patient_ids = self.nuh_response.patient_id.unique()
        #     train_ids, test_ids, _, _ = train_test_split(
        #         uniq_patient_ids,
        #         np.arange(len(uniq_patient_ids)),
        #         test_size=0.2,
        #         random_state=RANDOM_STATE,
        #     )
        #     filter_ids = train_ids if is_train else test_ids
        #     self.nuh_response = self.nuh_response[
        #         self.nuh_response.patient_id.isin(filter_ids)
        #     ].copy()

    def __len__(self):
        return len(self.nuh_response)

    def __getitem__(self, idx):
        record = self.nuh_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
            "days": record["days"],
        }

    @cached_property
    def mutations(self):
        return self.nuh_mutation

    @cached_property
    def raw_mutations(self):
        nuh_mutation = pd.read_csv("../data/processed/nuh_mutations.csv")
        nuh_mutation.set_index(self.entity_identifier_name, inplace=True)
        nuh_mutation = nuh_mutation.reindex(columns=GENES_324)
        return nuh_mutation

    @cached_property
    def survival_info(self):
        survival_info_df = pd.read_csv("../data/processed/nuh_survival_info.csv")
        return survival_info_df

    @cached_property
    def cnv(self):
        nuh_cnv = pd.read_csv("../data/processed/nuh_cnv.csv")
        nuh_cnv.set_index("patient_id", inplace=True)
        nuh_cnv = nuh_cnv.reindex(columns=GENES_324)
        return nuh_cnv

    @cached_property
    def clinvar_gpd_annovar_annotated(self):
        clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_nuh_feature_matrix.csv", index_col = 0)
        return clinvar_gpd_annovar_df
    
class AggCategoricalAnnotatedNuhFirstLineDatasetFilteredByDrug(CategoricalAnnotatedNuhFirstLineDatasetFilteredByDrug):
    """
    Aggregated categorical annotations features for NUH First Line Patient data
    """

    @cached_property
    def mutations(self):
        agg_results = {}
        for gene in GENES_324:
            filtered_df = self.nuh_mutation.filter(regex=f"^{gene}_[a-z]*")
            # agg_results[gene] = filtered_df.mean(axis=1)
            # agg_results[gene] = filtered_df.sum(axis=1)
            curr_result = None
            for col in filtered_df.columns:
                if type(curr_result) == pd.Series:
                    curr_result = curr_result | (filtered_df[col] != 0)
                else:
                    curr_result = filtered_df[col] != 0

            agg_results[gene] = curr_result.astype(np.int32)

        agg_df = pd.DataFrame(agg_results)
        return agg_df

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df