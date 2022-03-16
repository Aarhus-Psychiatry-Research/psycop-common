"""
A script for conducting grid search using sklearn
"""

# https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines

#Hvordan pipeline kan understøtte kategorier
#https://stackoverflow.com/questions/57867974/one-pipeline-to-fit-both-text-and-categorical-features

import argparse

import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from utils import get_clf

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def grid_search(data="train_temp.csv",
                label_column="labels",
                sfi_column="SFI_Navn",
                text_column="text",
                Adiag_column="ADiagnoseKodeTekst",
                sex_column="KOEN",
                alder_column="ALDER_KOR",
                Kontakttype_column="KontakttypeEPJ",
                clfs=['nb', 'rf', 'en', 'ab', 'xg'],
                resampling=['under'],
                grid_search_clf=True,
                grid_seach_vectorization=False,
                cv=5,
                scoring="roc_auc",
                sampling_strategy=1,
                **kwargs
                ):
    """
    resampling (str | None) options are:
    over, under, smote, under_over, under_smote
    """
    df = pd.read_csv(data)

    # These are the variables checked for the grid search for vectorization
    # feel free to add to add or remove from this
    search_params_vect = {'ngram_range': [(1, 2), (1, 3), (1, 4)],
                          'lowercase': [True],
                          'max_df': [0.5],
                          'min_df': [2, 5],
                          'binary': [False],
                          }
    # These are the variables used for the grid search for classfiers
    # feel free to add or remove from this
    search_params_clf = \
        {'nb': {},
         'rf': {'max_depth': [None, 5, 10],
                'n_estimators': [50, 100, 200, 300],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]},
         'en': {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)},
         'ab': {'base_estimator': [DecisionTreeClassifier(max_depth=1),
                                   DecisionTreeClassifier(max_depth=2)]},
         'xg': {'learning_rate': [0.1, 0.2, 0.3],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 12, 18],
                'min_child_weight': [1, 2, 3],
                'gamma': [0.0, 0.1, 0.2]}}

    # Defining an inner function for grid search
    def __grid_search(df, c, sampling, **kwargs):
        clf = get_clf(c, **kwargs)
        
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
        preprocessor = ColumnTransformer(transformers=[
                ('sfi', categorical_transformer, ['SFI_Navn']),
                ('kontaktType', categorical_transformer, ['KontakttypeEPJ']),
                ('Adiag', categorical_transformer, ['ADiagnoseKodeTekst']),
                ('sex', categorical_transformer, ['KOEN']),
                # TfidfVectorizer accepts column name only between quotes
                ('vect', TfidfVectorizer(**kwargs), 'text')
        ])  
                
        if sampling is None:
            print("no sampling")
            pipe = Pipeline(steps=[
                 ('preprocessor', preprocessor),
                ('clf', clf(**kwargs))
            ])
        else:
            print("sampling")
            sampling = resampling_d[rs]
            pipe = Pipeline(steps=
                [('preprocessor', preprocessor),
                ('sampling', sampling(sampling_strategy, **kwargs)),
                ('clf', clf(**kwargs))
            ])
        # construct grid search parameters
        parameters = {}

        if grid_seach_vectorization:
            for k in search_params_vect:
                parameters["preprocessor__vect__"+k] = search_params_vect[k]
        if grid_search_clf:
            for k in search_params_clf[c]:
                parameters['clf__'+k] = search_params_clf[c][k]

        gs_clf = GridSearchCV(pipe, parameters, scoring=scoring, cv=cv,
                              verbose=True, n_jobs=-1)  # run on all cores
        
        """opdel datasæt i df_x og evt df_y """
        df_x=df.drop("labels", axis=1)
        df_x=df_x.drop("Unnamed: 0", axis=1)
        #df_x=df_x.drop("text", axis=1)
        for col in df_x.columns: 
            print(col) 
        fit = gs_clf.fit(df_x, df[label_column])
        print("done")
        return(fit.best_score_, fit.best_params_, fit)

    resampling_d = {"over": RandomOverSampler,
                    "under": RandomUnderSampler,
                    "smote": SMOTE,
                    None: None}

    results = {}
    for rs in resampling:
        res = {}
        for c in clfs:
            r = __grid_search(df, c, rs, **kwargs)
            res[c] = r
        results[rs] = res

    print("\n\nThe grid search is completed the results were:")
    for rs in results:
        print(f"\nUsing the resampling method: {rs}")
        for c in results[rs]:
            score, best_params, t = results[rs][c]
            print(f"\tThe best fit of the clf: {c}, " +
                  f"obtained a score of {round(score, 4)}, \
                      with the parameters:")
            for p in best_params:
                print(f"\t\t{p} = {best_params[p]}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="what data to use",
                        default="train.csv")
    parser.add_argument("-c", "--clfs",
                        help="What classifier should you use",
                        nargs='+', default=['nb'])
    parser.add_argument("-rs", "--resampling",
                        help="should you resample, and how. Can be multiple",
                        nargs='+', default=['under'])
    parser.add_argument("-gc", "--grid_search_clf",
                        help="should you grid search classifier?",
                        default=True, type=bool)
    parser.add_argument("-gv", "--grid_seach_vectorization",
                        help="should you grid search vectorizer?",
                        default=True, type=bool)
    parser.add_argument("-cv", "--cv",
                        help="number of cross validation folds",
                        default=5, type=int)

    parser.add_argument("-tc", "--text_column",
                        help="columns for text",
                        default="text")
    parser.add_argument("-lc", "--label_column",
                        help="columns for labels",
                        default="labels")
    parser.add_argument("-sc", "--scoring",
                        help="scoring function for grid search",
                        default="roc_auc")
    parser.add_argument("-ss", "--sampling_strategy",
                        help="the desired proportion of the minority to the \
                            majority",
                        default=1.0, type=float)

    # parse and clean args:
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    print("\n\nCalling grid search with the arguments:")
    for k in args:
        print(f"\t{k}: {args[k]}")
    grid_search(**args)
