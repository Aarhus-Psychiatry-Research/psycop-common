from typing import Literal
import pandas as pd
from src.psycop_feature_generation.loaders.raw.load_text import load_text_sfis
from src.psycop_feature_generation.application_modules.save_dataset_to_disk import filter_by_split_ids,get_split_id_df

def load_text_split(
        text_sfi_names= str,
        split_name= Literal ["train","val"],
        ) -> pd.DataFrame:
    
    "Splits text data into training and validation set based on predefined splits."
    
    text_df = load_text_sfis(text_sfi_names=text_sfi_names)

    split_id_df = get_split_id_df(
        split_name=split_name)
    
    text_split_df = filter_by_split_ids(df_to_split=text_df, splid_id_df=split_id_df)

    return text_split_df

