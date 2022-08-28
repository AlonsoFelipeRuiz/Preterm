from pickle import load
import pandas as pd
import numpy as np
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data-dir', nargs='?', default='/input')
parser.add_argument('-m', '--models-dir', nargs='?', default='/models')
parser.add_argument('-o', '--output-dir', nargs='?', default='/output')
args = parser.parse_args()

alpha=(
    pd.read_csv(args.data_dir + "/alpha_diversity/alpha_diversity.csv", 
                      index_col="specimen")
)
species=(
    pd.read_csv(args.data_dir + "/taxonomy/taxonomy_relabd.species.csv", 
                      index_col="specimen")
)
cst=(
    pd.read_csv(args.data_dir + "/community_state_types/cst_valencia.csv", 
                      index_col="specimen")
)
phylo=(
    pd.read_csv(args.data_dir + "/phylotypes/phylotype_relabd.1e0.csv", 
                      index_col="specimen")
)
genus=(
    pd.read_csv(args.data_dir + "/taxonomy/taxonomy_relabd.species.csv", 
                      index_col="specimen")
)
#important genus for preterm birth
genus_preterm=["Lactobacillus","Prevotella"]
genus=genus[genus_preterm]
metadata=(
    pd.read_csv(args.data_dir + "/metadata/metadata.csv", index_col="specimen", na_values="Unknown")
)

phylo_metadata=pd.merge(phylo, metadata, left_index=True, right_index=True)
phylo_cst_metadata=pd.merge(cst, phylo_metadata, left_index=True, right_index=True)
phylo_alpha_cst_metadata=pd.merge(alpha, phylo_cst_metadata, left_index=True, right_index=True)
phylo_alpha_cst_species_metadata=(
    pd.merge(species, phylo_alpha_cst_metadata, left_index=True, right_index=True)
)

complete_df=(
    pd.merge(phylo_alpha_cst_species_metadata, genus, suffixes=(None, "_genus"), 
             left_index=True, right_index=True)
)

complete_df.drop(["was_preterm", "was_term", "project", "participant_id", "delivery_wk", "was_early_preterm", "collect_wk"], axis=1, inplace=True)

X_train=complete_df.copy()

X_train_species=X_train[species.columns]
X_train_phylo=X_train[phylo.columns]

filename = args.models_dir + '/kmediods_species_final.sav'
with open(filename, 'rb') as f:
    km = load(f)

X_train["Kmediods_cluster"]=km.predict(X_train_species)

filename = args.models_dir + '/kmediods_phylo_final.sav'
with open(filename, 'rb') as f:
    km_phylo = load(f)

X_train["Kmediods_phylo"]=km_phylo.predict(X_train_phylo)

dis_variables = X_train.select_dtypes(exclude=np.number).columns.to_list()+["Kmediods_cluster","Kmediods_phylo"]
num_variables = (
    X_train.select_dtypes(include=np.number).columns
    .drop(["Kmediods_cluster","Kmediods_phylo"]).to_list()
)

filename = args.models_dir + '/RandomForest_preterm_final.sav'
with open(filename, 'rb') as f:
    term_pipe_RF = load(f)
filename = args.models_dir + '/RandomForest_early_preterm_final.sav'
with open(filename, 'rb') as f:
    early_term_pipe_RF = load(f)
preds = term_pipe_RF.predict(X_train)
early_preds = early_term_pipe_RF.predict(X_train)

output = pd.DataFrame(metadata['participant_id'])
output.rename(columns={'participant_id': 'participant'}, inplace=True)
output.loc[:, 'was_preterm'] = preds > 0.274311
output.loc[:, 'probability'] = preds
#output.loc[:, 'pred_proba_was_early_preterm'] = early_preds
#output.loc[:, 'pred_was_early_preterm'] = preds > 0.113617
to_return = output.groupby('participant').min()

filename = args.output_dir + '/predictions.csv'
to_return.to_csv(filename)
