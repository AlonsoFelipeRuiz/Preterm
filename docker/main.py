from joblib import load
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

alpha=(
    pd.read_csv("../training_data_2022-07-21/alpha_diversity/alpha_diversity.csv", 
                      index_col="specimen")
)
species=(
    pd.read_csv("../training_data_2022-07-21/taxonomy/taxonomy_relabd.species.csv", 
                      index_col="specimen")
)
cst=(
    pd.read_csv("../training_data_2022-07-21/community_state_types/cst_valencia.csv", 
                      index_col="specimen")
)
phylo=(
    pd.read_csv("../training_data_2022-07-21/phylotypes/phylotype_relabd.1e0.csv", 
                      index_col="specimen")
)
genus=(
    pd.read_csv("../training_data_2022-07-21/taxonomy/taxonomy_relabd.species.csv", 
                      index_col="specimen")
)
#important genus for preterm birth
genus_preterm=["Lactobacillus","Prevotella"]
genus=genus[genus_preterm]
metadata=(
    pd.read_csv("../training_data_2022-07-21/metadata/metadata.csv", index_col="specimen", na_values="Unknown")
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

X_train=complete_df.copy()

X_train_species=X_train[species.columns]
X_train_phylo=X_train[phylo.columns]

filename = 'Models/kmediods_species_final.sav'
km = load(filename)

X_train["Kmediods_cluster"]=km.predict(X_train_species)

filename = 'Models/kmediods_phylo_final.sav'
km_phylo = load(filename)

X_train["Kmediods_phylo"]=km_phylo.predict(X_train_phylo)

dis_variables = X_train.select_dtypes(exclude=np.number).columns.to_list()+["Kmediods_cluster","Kmediods_phylo"]
num_variables = (
    X_train.select_dtypes(include=np.number).columns
    .drop(["Kmediods_cluster","Kmediods_phylo"]).to_list()
)

filename = 'Models/RandomForest_preterm_final.sav'
term_pipe_RF = load(filename)
filename = 'Models/RandomForest_early_preterm_final.sav'
early_term_pipe_RF = load(filename)
preds = term_pipe_RF.predict(X_train)
early_preds = early_term_pipe_RF.predict(X_train)

output = pd.DataFrame(metadata['participant_id'])
output.loc[:, 'pred_proba_was_preterm'] = preds
output.loc[:, 'pred_was_preterm'] = preds > 0.274311
output.loc[:, 'pred_proba_was_early_preterm'] = early_preds
output.loc[:, 'pred_was_early_preterm'] = preds > 0.113617
to_return = output.groupby('participant_id').min()

filename = 'output/predictions.csv'
to_return.to_csv(filename)
