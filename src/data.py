import itertools
import json
import re
import logging
import subprocess
import tempfile

import numpy as np
import pandas as pd

from statsmodels.imputation.mice import MICEData

# Drop columns such as integrated_fitCons_score, GERP++_NR, GERP++_RS
# since they describe functional fitness rather than deleteriousnes/ they
# have no categorical column associated
REQUIRED_ANNOTATION_COLUMNS = [
    "SIFT_score",
    "SIFT_converted_rankscore",
    "SIFT_pred",
    "SIFT4G_score",
    "SIFT4G_converted_rankscore",
    "SIFT4G_pred",
    "LRT_score",
    "LRT_converted_rankscore",
    "LRT_pred",
    "MutationTaster_score",
    "MutationTaster_converted_rankscore",
    "MutationTaster_pred",
    "MutationAssessor_score",
    "MutationAssessor_rankscore",
    "MutationAssessor_pred",
    "FATHMM_score",
    "FATHMM_converted_rankscore",
    "FATHMM_pred",
    "PROVEAN_score",
    "PROVEAN_converted_rankscore",
    "PROVEAN_pred",
    "MetaSVM_pred",
    "M-CAP_score",
    "M-CAP_rankscore",
    "M-CAP_pred",
    "MVP_score",
    "MVP_rankscore",
    "MPC_score",
    "MPC_rankscore",
    "PrimateAI_score",
    "PrimateAI_rankscore",
    "PrimateAI_pred",
    "DEOGEN2_score",
    "DEOGEN2_rankscore",
    "DEOGEN2_pred",
    "BayesDel_addAF_score",
    "BayesDel_addAF_pred",
    "BayesDel_noAF_score",
    "BayesDel_noAF_rankscore",
    "BayesDel_noAF_pred",
    "ClinPred_score",
    "ClinPred_rankscore",
    "ClinPred_pred",
    "LIST-S2_score",
    "LIST-S2_rankscore",
    "LIST-S2_pred",
    "DANN_score",
    "DANN_rankscore",
    "fathmm-MKL_coding_score",
    "fathmm-MKL_coding_rankscore",
    "fathmm-MKL_coding_pred",
    "fathmm-XF_coding_score",
    "fathmm-XF_coding_rankscore",
    "fathmm-XF_coding_pred",
    "Eigen-raw_coding",
    "Eigen-raw_coding_rankscore",
    "Eigen-PC-raw_coding",
    "Eigen-PC-raw_coding_rankscore",
]
CATEGORICAL_COLUMNS = [
    "sift_pred",
    "sift4g_pred",
    "lrt_pred",
    "mutationtaster_pred",
    "mutationassessor_pred",
    "fathmm_pred",
    "provean_pred",
    "metasvm_pred",
    "m_cap_pred",
    "primateai_pred",
    "deogen2_pred",
    "bayesdel_addaf_pred",
    "bayesdel_noaf_pred",
    "clinpred_pred",
    "list_s2_pred",
    "fathmm_mkl_coding_pred",
    "fathmm_xf_coding_pred",
]

# The thresholds used in PREDICTOR_LAMBDA_MAP are taken from the corresponding
# technique's published paper/web page
PREDICTOR_LAMBDA_MAP = {
    "sift_pred": ("sift_score", lambda v: "D" if v <= 0.05 else "T"),
    "sift4g_pred": ("sift4g_score", lambda v: "D" if v <= 0.05 else "T"),
    "lrt_pred": ("lrt_score", lambda v: "D" if v <= 0.001 else "U"),
    "mutationtaster_pred": (
        "mutationtaster_score",
        lambda v: None,
    ),  # Threshold is not available and couldn't be derived from available values as well
    "mutationassessor_pred": (
        "mutationassessor_score",
        lambda v: "H" if v >= 3.5 else ("M" if v >= 1.94 else "L"),
    ),
    "fathmm_pred": ("fathmm_score", lambda v: "D" if v < 1.5 else "T"),
    "provean_pred": ("provean_score", lambda v: "D" if v <= 2.282 else "N"),
    "metasvm_pred": (
        "metasvm_pred",
        lambda v: None,
    ),  # No corresponding numeric score available for this method
    "m_cap_pred": ("m_cap_score", lambda v: "D" if v >= 0.025 else "T"),
    "primateai_pred": ("primateai_score", lambda v: "D" if v >= 0.803 else "T"),
    "deogen2_pred": ("deogen2_score", lambda v: "D" if v >= 0.45 else "T"),
    "bayesdel_addaf_pred": (
        "bayesdel_addaf_score",
        lambda v: "D" if v >= 0.0692 else "T",
    ),
    "bayesdel_noaf_pred": (
        "bayesdel_noaf_score",
        lambda v: "D" if v >= -0.0570 else "T",
    ),
    "clinpred_pred": ("clinpred_score", lambda v: "D" if v >= 0.5 else "T"),
    "list_s2_pred": ("list_s2_score", lambda v: "D" if v >= 0.85 else "T"),
    "fathmm_mkl_coding_pred": (
        "fathmm_mkl_coding_score",
        lambda v: "D" if v >= 0.5 else "N",
    ),
    "fathmm_xf_coding_pred": (
        "fathmm_xf_coding_score",
        lambda v: "D" if v >= 0.5 else "N",
    ),
}

DELETERIOUS_VALUES = ["D", "A", "H", "M"]

CNV_PATTERN = r"loss|amplification"

GENES_324 = pd.read_csv("../data/raw/gene2ind.txt", header=None)[0].tolist()

ANNOTATION_SCRIPT_PATH = "../script/goAAtoGv2.sh"

SPECIAL_CASES = r"rearrangement|truncation|fs|del|ins"


def get_alias_to_canonical_name_map():
    with open("../data/computed/gene_aliases.json", "r") as fp:
        aliases_on_disk = json.load(fp)

    alias_to_canonical_name_map = {}
    for canonical_name, aliases in aliases_on_disk.items():
        # Some canonical names have only one alias - convert those as list for consistency
        if type(aliases) != list:
            aliases = [aliases]

        for alias in aliases:

            # If an alias is one of the canonical names in GENES_324, do not add it to the map
            # Else, we'd be renaming a canonical named column into something else
            if alias in GENES_324:
                logging.info(f"Alias {alias} is a canonical_name, skipping")
                continue

            if alias in alias_to_canonical_name_map:
                logging.info(
                    f"Found multiple canonical names for alias - {alias} = {[canonical_name, alias_to_canonical_name_map[alias]]}"
                )
                # Drop aliases with conflicting canonical names per recommendation from clinicians
                alias_to_canonical_name_map.pop(alias)

            # Convert all aliases to be upper case for consistency
            alias_to_canonical_name_map[alias.upper()] = canonical_name.upper()

    return alias_to_canonical_name_map


ALIAS_TO_CANONICAL_NAME_MAP = get_alias_to_canonical_name_map()


def preprocess_annotation_features(input_df):
    """
    Applies preprocessing steps identified to the given DataFrame with annotation features

    Note: This preprocessing is done in a separate method so that this method can evolve
    independently from the feature generator method

    """

    modified_df = input_df.copy()
    for column in modified_df.columns:
        column_mask = modified_df[column] == "."
        if column_mask.sum():
            logging.info(f"Found {column_mask.sum()} rows with '.' in column {column}")
            modified_df.loc[column_mask, column] = None

        if not (("pred" in column) or ("DepMap-ID" in column) or ("input" in column)):
            modified_df[column] = modified_df[column].astype(np.float_)

    # Keep MICEData happy - it expects column names to be of lower case with no special characters such as "."
    modified_df.rename(columns={"GERP.._NR": "GERP_NR"}, inplace=True)
    modified_df.rename(
        columns={column: column.lower() for column in modified_df.columns}, inplace=True
    )
    modified_df.rename(
        columns={column: column.replace(".", "_") for column in modified_df.columns},
        inplace=True,
    )
    modified_df.rename(
        columns={column: column.replace("-", "_") for column in modified_df.columns},
        inplace=True,
    )
    modified_df.rename(
        columns={column: column.replace("+", "") for column in modified_df.columns},
        inplace=True,
    )
    return modified_df


def get_annotation_features(mutations):
    """
    For a given list of mutations, this method generates variant annotation features

    Sample Usage:
    >>> get_annotation_features(["RBM10:p.Q200*",])

    Note: the result from this function needs further preprocessing to be used in a
    model. Refer to preprocess_annotation_features for more information.

    """

    if (not mutations) or (not len(mutations)):
        raise ValueError(
            "mutations should be a list of non-empty mutations. Eg: ['ALK, R311H', ...]"
        )

    results = []
    for mutation in mutations:
        try:
            # Run annotation script within a temp file and extract features as DataFrame
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_file_path = tmpdirname + "anno_input.txt"
                with open(input_file_path, "w+") as input_file:
                    mutation_cleaned = [part for part in mutation.split(" ") if part]
                    input_file.write(":p.".join(mutation_cleaned))
                    input_file.write("\n")

                # Execute script
                cmd = "bash {0} {1}".format(ANNOTATION_SCRIPT_PATH, input_file_path)

                logging.info(f"Executing command {cmd}")
                subprocess.call(cmd, shell=True, executable="/bin/bash")
                out_file_path = f"{input_file_path}.annot.hg38_finalannot.txt"
                res = pd.read_table(out_file_path)
        # Some inputs lead to errors, such as "PTEN loss" - ignore and continue processing
        except Exception as e:
            logging.error(
                f"Encountered error while processing mutation {mutation} - {e}"
            )
            continue

        res["input"] = mutation
        res = res[~res.duplicated()]
        results.append(res)

    if not len(results):
        return None

    results_df = pd.concat(results)
    results_df.set_index(["input"], inplace=True)
    results_df.drop(columns=["Otherinfo1"], inplace=True)
    return results_df


def _is_valid_point_mutations(point_mutations):
    is_valid = True
    for mutation in point_mutations:
        # Capture cases where the input contains only the modified gene (Example: "TP53")
        if len([m for m in mutation.split(" ") if m]) < 2:
            print(f"Point mutation {mutation} is invalid")
            is_valid = False
    return is_valid


def canonicalize_mutations(patient_mutations):
    """
    Checks for aliases in input mutations and replaces the alias with
    canonical gene names

    """
    gene_aliases = list(ALIAS_TO_CANONICAL_NAME_MAP.keys())
    canonicalized_mutations = []
    for mutation in patient_mutations:
        # Standardize gene_part by taking upper - all aliases are also in upper case as
        # defined in get_alias_to_canonical_name_map
        gene_part = mutation.split(" ")[0].upper()
        # If the gene_part is already in its canonical state, no need to look for alias
        if (gene_part in GENES_324) or (gene_part not in gene_aliases):
            canonicalized_mutations.append(mutation)
        else:
            canonicalized_mutations.append(
                mutation.replace(
                    mutation.split(" ")[0], ALIAS_TO_CANONICAL_NAME_MAP[gene_part]
                )
            )

    return canonicalized_mutations


def construct_anno_features(patient_id, patient_mutations, agg_features=False):
    """
    TODO: Add support for other agg functions (mean, OR, etc) - as of 202209, only
    sum is supported
    """
    if agg_features:
        logging.warn(
            """
        Received agg_features=True -> As of now, construct_anno_features only supports sum aggregation.
        Please ensure that the agg used in dataset definition is sum - if it is not sum, please pass
        agg_features=False and perform agg in dataset definition
        """
        )

    if not _is_valid_point_mutations(patient_mutations):
        return None

    anno_features_combined_imputed_df = pd.read_csv(
        "../data/processed/anno_features_combined_imputed.csv"
    )
    logging.info(anno_features_combined_imputed_df.shape)
    anno_features_combined_imputed_df.set_index(["input"], inplace=True)
    anno_features_combined_imputed_df.head()

    canonical_mutations = canonicalize_mutations(patient_mutations)

    mutations_with_missing_annotations = []
    available_mutations = []
    for mutation in canonical_mutations:
        if mutation in anno_features_combined_imputed_df.index:
            available_mutations.append(mutation)
        elif not re.search(CNV_PATTERN, mutation, re.IGNORECASE):
            mutations_with_missing_annotations.append(mutation)

    if available_mutations:
        patient_anno_features = anno_features_combined_imputed_df.loc[
            available_mutations
        ]
        patient_anno_features = patient_anno_features[CATEGORICAL_COLUMNS].copy()
    else:
        patient_anno_features = None

    if len(mutations_with_missing_annotations) != 0:
        logging.info(
            f"Found mutations with missing annotations - {mutations_with_missing_annotations}"
        )
        missing_annotations = get_annotation_features(
            mutations_with_missing_annotations
        )
        if missing_annotations is not None:
            missing_annotations = missing_annotations[
                REQUIRED_ANNOTATION_COLUMNS
            ].copy()
            missing_annotations.reset_index(inplace=True)
            missing_annotations = preprocess_annotation_features(missing_annotations)
            missing_annotations = missing_annotations[~missing_annotations.duplicated()]
            missing_annotations.reset_index(drop=True, inplace=True)
            missing_annotations.set_index("input", inplace=True)

            numeric_columns = list(
                column
                for column in missing_annotations.columns
                if pd.api.types.is_numeric_dtype(missing_annotations[column])
            )
            # Prepare mask by identifying rows that have all na values for numeric_columns
            na_mask = None
            for col in numeric_columns:
                if type(na_mask) == pd.Series:
                    na_mask = na_mask & missing_annotations[col].isna()
                else:
                    na_mask = missing_annotations[col].isna()

            missing_annotations = missing_annotations[~na_mask]
            numeric_df = missing_annotations[numeric_columns].copy()
            logging.info(numeric_df.shape)
            numeric_df.head()
            numeric_df = pd.concat(
                [numeric_df, anno_features_combined_imputed_df[numeric_columns]],
            )

            categorical_columns = [
                column
                for column in missing_annotations.columns
                if column not in numeric_columns
            ]
            categorical_missing_annotations = missing_annotations[
                categorical_columns
            ].copy()
            logging.info(categorical_missing_annotations.shape)
            categorical_missing_annotations.head()

            imp = MICEData(numeric_df)
            # Impute missing values in numeric columns - Expensive!!
            imp.update_all()
            imputed_df = imp.data
            assert numeric_df.shape == imputed_df.shape
            imputed_df.index = numeric_df.index
            imputed_df = imputed_df[
                imputed_df.index.isin(mutations_with_missing_annotations)
            ].copy()
            numeric_imputed_df = pd.concat(
                [categorical_missing_annotations, imputed_df,], axis=1,
            )
            logging.info(numeric_imputed_df.shape)
            for column in CATEGORICAL_COLUMNS:
                logging.info(
                    column,
                    numeric_imputed_df[column].unique(),
                    len(numeric_imputed_df[numeric_imputed_df[column].isna()]),
                )
                col_na_mask = numeric_imputed_df[column].isna()
                numeric_imputed_df.loc[col_na_mask, column] = numeric_imputed_df[
                    col_na_mask
                ][PREDICTOR_LAMBDA_MAP[column][0]].apply(
                    PREDICTOR_LAMBDA_MAP[column][1]
                )

                # logging.info(
                #     column,
                #     numeric_imputed_df[column].unique(),
                #     len(numeric_imputed_df[numeric_imputed_df[column].isna()]),
                # )

            numeric_imputed_df = numeric_imputed_df.dropna()
            logging.info(numeric_imputed_df.shape)
            missing_anno_features_df = numeric_imputed_df[CATEGORICAL_COLUMNS].copy()

            patient_anno_features = pd.concat(
                [missing_anno_features_df, patient_anno_features]
            )

    # If there are no valid annotation features for any of the mutations, return all 0s
    if patient_anno_features is None:
        if agg_features:
            record = {gene: 0 for gene in GENES_324}
        else:

            record = {
                f"{gene}_{method}": 0
                for gene, method in itertools.product(GENES_324, CATEGORICAL_COLUMNS)
            }
    else:
        for col in patient_anno_features.columns:
            patient_anno_features[col] = patient_anno_features[col].apply(
                lambda v: 1 if v in DELETERIOUS_VALUES else 0
            )

        patient_anno_features.reset_index(inplace=True)
        patient_anno_features["gene"] = patient_anno_features.input.apply(
            lambda gene_mut: gene_mut.split(" ")[0]
        )

        if agg_features:
            # First, sum across all mutations for a given gene and then sum across
            # all 17 predictors (columns)
            agg_series = patient_anno_features.groupby(by="gene").sum().sum(axis=1)
            record = {
                gene: agg_series[gene] if gene in agg_series else 0
                for gene in GENES_324
            }
        else:
            record = {}
            for gene in GENES_324:
                filtered_df = patient_anno_features[
                    (patient_anno_features.gene == gene)
                ]
                record.update(
                    {
                        f"{gene}_{k}": v
                        for k, v in (
                            filtered_df[CATEGORICAL_COLUMNS]
                            .agg(["sum"],)
                            .iloc[0]
                            .to_dict()
                        ).items()
                    }
                )

    patient_mutations_df = pd.DataFrame([record])
    patient_mutations_df["patient_id"] = patient_id
    patient_mutations_df.set_index("patient_id", inplace=True)

    gene_to_mutations_dict = {}
    for mutation in canonical_mutations:
        gene = mutation.split(" ")[0]
        if gene not in GENES_324:
            continue

        mutation = mutation[len(gene) + 1 :]
        if gene in gene_to_mutations_dict:
            gene_to_mutations_dict[gene].append(mutation)
        else:
            gene_to_mutations_dict[gene] = [mutation]

    for gene, mutations in gene_to_mutations_dict.items():
        is_special_case_applicable = False
        for value in mutations:
            if re.search(SPECIAL_CASES, value, re.IGNORECASE):
                is_special_case_applicable = True

        if is_special_case_applicable:
            logging.info(
                f"Patient {patient_id} has special case in {gene} with mutations {mutations}"
            )
            if agg_features:
                if patient_mutations_df.loc[patient_id, gene] == 0:
                    patient_mutations_df.loc[patient_id, gene] = len(
                        CATEGORICAL_COLUMNS
                    )
            else:
                patient_features = patient_mutations_df.filter(regex=gene).loc[
                    patient_id
                ]
                if np.all(patient_features.values == 0):
                    patient_features = 1

    return patient_mutations_df


def construct_raw_mutation_features(patient_id, patient_mutations):

    canonical_mutations = canonicalize_mutations(patient_mutations)
    point_mutations = [
        mutation.split(" ")[0]
        for mutation in canonical_mutations
        if not re.search(CNV_PATTERN, mutation, re.IGNORECASE)
    ]

    mutation_barcode = [1 if gene in point_mutations else 0 for gene in GENES_324]
    patient_raw_mutations = pd.DataFrame.from_records(
        [mutation_barcode], columns=GENES_324
    )

    patient_raw_mutations["patient_id"] = patient_id
    patient_raw_mutations.set_index("patient_id", inplace=True)
    return patient_raw_mutations


def construct_raw_cnv_features(patient_id, patient_mutations):

    canonical_mutations = canonicalize_mutations(patient_mutations)
    cnv_mutations = [
        mutation.split(" ")[0]
        for mutation in canonical_mutations
        if re.search(CNV_PATTERN, mutation, re.IGNORECASE)
    ]

    mutation_barcode = [1 if gene in cnv_mutations else 0 for gene in GENES_324]

    patient_cnv = pd.DataFrame.from_records([mutation_barcode], columns=GENES_324)

    patient_cnv["patient_id"] = patient_id
    patient_cnv.set_index("patient_id", inplace=True)
    return patient_cnv


def construct_anno_features_xon17(patient_id, patient_mutations, agg_features=False):
    """
    TODO: Add support for other agg functions (mean, OR, etc) - as of 202209, only
    sum is supported
    Here, the aggregation is done as an average over all variants over all 17 algorithms.
    """
    if agg_features:
        logging.warn(
            """
        Received agg_features=True -> As of now, construct_anno_features only supports sum aggregation.
        Please ensure that the agg used in dataset definition is sum - if it is not sum, please pass
        agg_features=False and perform agg in dataset definition
        """
        )

    if not _is_valid_point_mutations(patient_mutations):
        return None

    anno_features_combined_imputed_df = pd.read_csv(
        "../data/processed/anno_features_combined_imputed.csv"
    )
    logging.info(anno_features_combined_imputed_df.shape)
    anno_features_combined_imputed_df.set_index(["input"], inplace=True)
    anno_features_combined_imputed_df.head()

    canonical_mutations = canonicalize_mutations(patient_mutations)

    mutations_with_missing_annotations = []
    available_mutations = []
    for mutation in canonical_mutations:
        if mutation in anno_features_combined_imputed_df.index:
            available_mutations.append(mutation)
        elif not re.search(CNV_PATTERN, mutation, re.IGNORECASE):
            mutations_with_missing_annotations.append(mutation)

    if available_mutations:
        patient_anno_features = anno_features_combined_imputed_df.loc[
            available_mutations
        ]
        patient_anno_features = patient_anno_features[CATEGORICAL_COLUMNS].copy()
    else:
        patient_anno_features = None

    if len(mutations_with_missing_annotations) != 0:
        logging.info(
            f"Found mutations with missing annotations - {mutations_with_missing_annotations}"
        )
        missing_annotations = get_annotation_features(
            mutations_with_missing_annotations
        )
        if missing_annotations is not None:
            missing_annotations = missing_annotations[
                REQUIRED_ANNOTATION_COLUMNS
            ].copy()
            missing_annotations.reset_index(inplace=True)
            missing_annotations = preprocess_annotation_features(missing_annotations)
            missing_annotations = missing_annotations[~missing_annotations.duplicated()]
            missing_annotations.reset_index(drop=True, inplace=True)
            missing_annotations.set_index("input", inplace=True)

            numeric_columns = list(
                column
                for column in missing_annotations.columns
                if pd.api.types.is_numeric_dtype(missing_annotations[column])
            )
            # Prepare mask by identifying rows that have all na values for numeric_columns
            na_mask = None
            for col in numeric_columns:
                if type(na_mask) == pd.Series:
                    na_mask = na_mask & missing_annotations[col].isna()
                else:
                    na_mask = missing_annotations[col].isna()

            missing_annotations = missing_annotations[~na_mask]
            numeric_df = missing_annotations[numeric_columns].copy()
            logging.info(numeric_df.shape)
            numeric_df.head()
            numeric_df = pd.concat(
                [numeric_df, anno_features_combined_imputed_df[numeric_columns]],
            )

            categorical_columns = [
                column
                for column in missing_annotations.columns
                if column not in numeric_columns
            ]
            categorical_missing_annotations = missing_annotations[
                categorical_columns
            ].copy()
            logging.info(categorical_missing_annotations.shape)
            categorical_missing_annotations.head()

            imp = MICEData(numeric_df)
            # Impute missing values in numeric columns - Expensive!!
            imp.update_all()
            imputed_df = imp.data
            assert numeric_df.shape == imputed_df.shape
            imputed_df.index = numeric_df.index
            imputed_df = imputed_df[
                imputed_df.index.isin(mutations_with_missing_annotations)
            ].copy()
            numeric_imputed_df = pd.concat(
                [categorical_missing_annotations, imputed_df,], axis=1,
            )
            logging.info(numeric_imputed_df.shape)
            for column in CATEGORICAL_COLUMNS:
                logging.info(
                    column,
                    numeric_imputed_df[column].unique(),
                    len(numeric_imputed_df[numeric_imputed_df[column].isna()]),
                )
                col_na_mask = numeric_imputed_df[column].isna()
                numeric_imputed_df.loc[col_na_mask, column] = numeric_imputed_df[
                    col_na_mask
                ][PREDICTOR_LAMBDA_MAP[column][0]].apply(
                    PREDICTOR_LAMBDA_MAP[column][1]
                )

                # logging.info(
                #     column,
                #     numeric_imputed_df[column].unique(),
                #     len(numeric_imputed_df[numeric_imputed_df[column].isna()]),
                # )

            numeric_imputed_df = numeric_imputed_df.dropna()
            logging.info(numeric_imputed_df.shape)
            missing_anno_features_df = numeric_imputed_df[CATEGORICAL_COLUMNS].copy()

            patient_anno_features = pd.concat(
                [missing_anno_features_df, patient_anno_features]
            )

    # If there are no valid annotation features for any of the mutations, return all 0s
    if patient_anno_features is None:
        if agg_features:
            record = {gene: 0 for gene in GENES_324}
        else:

            # record = {
            #     f"{gene}_{method}": 0
            #     for gene, method in itertools.product(GENES_324, CATEGORICAL_COLUMNS)
            # }
            record = {
                f"{gene}": 0
                for gene in GENES_324
            }
    else:
        for col in patient_anno_features.columns:
            patient_anno_features[col] = patient_anno_features[col].apply(
                lambda v: 1 if v in DELETERIOUS_VALUES else 0
            )

        patient_anno_features.reset_index(inplace=True)
        patient_anno_features["gene"] = patient_anno_features.input.apply(
            lambda gene_mut: gene_mut.split(" ")[0]
        )

        if agg_features:
            # First, sum across all 17 algorithms for each point mutation for a given gene and then average across
            # all variants (rows)
            agg_series = patient_anno_features.groupby(by="gene").sum().sum(axis=1)
            record = {
                gene: agg_series[gene] if gene in agg_series else 0
                for gene in GENES_324
            }
        else:
            record = {}
            for gene in GENES_324:
                filtered_df = patient_anno_features[
                    (patient_anno_features.gene == gene)
                ]
                if len(filtered_df) == 0:
                    record.update(
                    {
                        f"{gene}": 0
                    }
                    )
                else:
                    record.update(
                    {
                        f"{gene}": (filtered_df[CATEGORICAL_COLUMNS].sum(axis=1)/17).mean()
                    }
                    )
               # record.update(
               #     {
               #         f"{gene}_{k}": v
               #         for k, v in (
               #             filtered_df[CATEGORICAL_COLUMNS]
               #             .agg(["sum"],)
               #             .iloc[0]
               #             .to_dict()
               #         ).items()
               #     }
               # )

    patient_mutations_df = pd.DataFrame([record])
    patient_mutations_df["patient_id"] = patient_id
    patient_mutations_df.set_index("patient_id", inplace=True)

    gene_to_mutations_dict = {}
    for mutation in canonical_mutations:
        gene = mutation.split(" ")[0]
        if gene not in GENES_324:
            continue

        mutation = mutation[len(gene) + 1 :]
        if gene in gene_to_mutations_dict:
            gene_to_mutations_dict[gene].append(mutation)
        else:
            gene_to_mutations_dict[gene] = [mutation]

    for gene, mutations in gene_to_mutations_dict.items():
        is_special_case_applicable = False
        for value in mutations:
            if re.search(SPECIAL_CASES, value, re.IGNORECASE):
                is_special_case_applicable = True

        if is_special_case_applicable:
            logging.info(
                f"Patient {patient_id} has special case in {gene} with mutations {mutations}"
            )
            if agg_features:
                if patient_mutations_df.loc[patient_id, gene] == 0:
                    patient_mutations_df.loc[patient_id, gene] = len(
                        CATEGORICAL_COLUMNS
                    )
            else:
                patient_features = patient_mutations_df.filter(regex=gene).loc[
                    patient_id
                ]
                if np.all(patient_features.values == 0):
                    patient_features = 1

    return patient_mutations_df
