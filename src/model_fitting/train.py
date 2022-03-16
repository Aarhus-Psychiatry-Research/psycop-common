# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:52:54 2020

@author: adminanddan
"""

# https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines

import numpy as np
import pandas as pd
import pickle
import sklearn.metrics as metrics

import os
import pickle

#Pakker til at tilgå data fra BI-server
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

import time
import urllib
import urllib.parse
import pandas as pd

def dan_dataframe(query, server, database):
    """Creates a pandas dataframe from an SQL connection"""
    tst = time.time()
    driver = "ODBC Driver 13 for SQL Server"
    params = urllib.parse.quote(f"DRIVER={driver};SERVER={server};DATABASE={databse};Trusted_Connection=yes"
    
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, poolclass= NullPool)

    df = pd.read_sql(query,engine)

    
    print("--- dan_dataframe: Read Rows:{0} Cols:{1} --- ".format(df.shape[0],df.shape[1]) )
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()

    #  Luk SQL-forbindelse
    session.close()

    tdif = time.time()-tst

    print("--- Tidsforbrug {0:.2f} sec ---".format(tdif))
    return df



os.chdir("E:\\Users\\adminanddan\\Desktop\\corona-psychopathology-master_v5")

import grid_search_text_and_metadata_v5_AND_v3 as gs

# mulige clfs 'nb','rf','en','ab','xg'
# mulige resampling 'over','under','smote','under_over','under_smote'

# argumenter result_gs(data, text, labels,classifier,resampling,grid_search_clf,grid_search_vectorization,cv_folds,scoring,sampling_strategy)
result_gs = gs.grid_search("\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\train_metadata.csv","labels","SFI_Navn","text","ADiagnoseKodeTekst","KOEN","ALDER_KOR","KontakttypeEPJ",['xg'],[None],True,True,5,"roc_auc",1)


os.chdir("\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening")
import pickle
f = open("result_gs_xg_meta_text_under.pkl","wb")
pickle.dump(result_gs,f)
f.close()


file = open("result_gs_xg_meta_text_noRS.pkl",'rb')
object_file = pickle.load(file)

print(object_file)


os.chdir("E:\\Users\\adminanddan\\Desktop\\corona-psychopathology-master_v5")
import classify_v6_AND as clas

os.chdir("\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening")
fpr_test, tpr_test, thresholds_test, roc_auc_test, model_fitted = clas.clf("train_metadata.csv","test_metadata.csv","labels","SFI_Navn","text","ADiagnoseKodeTekst","KOEN","ALDER_KOR","KontakttypeEPJ","xg",{'learning_rate': 0.3, 'n_estimators': 200, 'max_depth': 3, 'min_child_weight': 3, 'gamma': 0.1},"auc",None,(1, 4),True,0.5,2)

#evaluer modellen - https://heartbeat.fritz.ai/classification-model-evaluation-90d743883106
def evaluate_threshold(threshold, thresholds, tpr, fpr):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
evaluate_threshold(0.33,thresholds_test,tpr_test,fpr_test)

import matplotlib.pyplot as plt

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,6))
ax1.plot(fpr_test,tpr_test)
ax1.plot(fpr_test, tpr_test, color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc_test)
ax1.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Total - total')
ax1.legend(loc="lower right")


#gem den fittede model
#pickle.dump(model_fitted, open("model_fitted", 'wb'))

### Åben den gemte model og brug den på de 'nye' data
os.chdir("\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening")
model = pickle.load(open("model_fitted", 'rb'))

server   = "BI-DPA-PROD"
database = "USR_PS"

sql = "SELECT * FROM [dbo].[for_Corona_ML_projekt_Phyton]"
new_notes=dan_dataframe(sql,server,database)

new_notes.drop(new_notes.loc[new_notes['Alder']<18].index, inplace=True)
new_notes.reset_index(drop=True, inplace=True)
new_notes['ID'] = new_notes.index
#new_notes.to_csv('new_data_raw_24_03_2021.csv', sep=';', index=False)

columns_toDrop = ['DiagnoseGruppeStreng',\
              'afdelingstekst',\
              'Corona_COVID',\
              'virus_smitte_epidemi_pandemi',\
              'DW_EK_Borger',\
              'Afdelingskode']
new_notes = new_notes.drop(columns_toDrop, axis=1)
datotid_df = new_notes[['DATOTID']].copy()
new_notes_noDate = new_notes.drop("DATOTID", axis=1) 

#navngiv labels korrekt
new_notes_noDate = new_notes_noDate.rename({"Alder": "ALDER_KOR", "Fritekst_3": "text", "DiagnoseKodeTekst": "ADiagnoseKodeTekst"}, axis=1)

#Set rækkefølgen af kolonner korrekt
new_notes_noDate = new_notes_noDate[['SFI_Navn', 'text', 'ADiagnoseKodeTekst', 'KOEN', 'ALDER_KOR', 'KontakttypeEPJ']]


#kører modellen på det samlede 'nye' datasæt
test_prop_new_notes_noDate = model.predict_proba(new_notes_noDate)

#to resultater - ét resultat med features koblet på sandsynlighederne, ét resultat med score og datotid
result_full = pd.concat([new_notes, pd.DataFrame(test_prop_new_notes_noDate)], axis=1)
result_date_score = pd.concat([datotid_df, pd.DataFrame(test_prop_new_notes_noDate)], axis=1)

#Konverter datatid til dato
result_full['date'] = result_full['DATOTID'].dt.date
result_full = result_full.drop("DATOTID", axis=1)

result_date_score['date'] = result_date_score['DATOTID'].dt.date
result_date_score = result_date_score.drop("DATOTID", axis=1)

#Gør index til ID
result_full['ID'] = result_full.index
result_date_score['ID'] = result_date_score.index


#result_full.to_csv('new_data_results_full.csv', sep=';', index=False)
#result_date_score.to_csv('new_data_results_DateScore.csv', sep=';', index=False)

df_large = pd.read_csv('new_data_results_full.zip', sep=';')
df_small = pd.read_csv('new_data_results_DateScore.csv', sep=';')



## Kenneth har udvalgt 500 tilfældige fra det oprindelige 'nye' datasæt
#Henter det fulde datasæt og id på de 500 tilfældige
df_full = pd.read_csv('new_data_raw_24_03_2021.zip', sep=';')
df_samples = pd.read_csv('random_samples_500_prop.csv')
df_samples_ID = df_samples[['ID']]

#Udvælger sample på 500 fra det fulde datasæt som skal bruges til at finde guld standarden ved oskar og christopher
df_samples_500_text = pd.merge(df_samples_ID,df_full, on='ID')

df_samples_500_text.to_csv('df_samples_500_text.csv', sep=';', index=False)

#gemmer filen i et format som Excel kan læse
#df = pd.read_csv('df_samples_500_text.csv', sep=';')
#df.to_csv('500_samples_text.csv', sep=';',encoding='utf-8-sig', index=False)



#Henter det fulde datasæt og id på de 200 tilfældige som skal bruges til at Oskar og Christopher kan blive enige igen
df_full = pd.read_csv('new_data_raw_24_03_2021.zip', sep=';') #jeg har zippet det gemte datasæt
df_samples_200 = pd.read_csv('200_samples.csv')
df_samples_ID_200 = df_samples_200[['ID']]

#Udvælger sample på 500 fra det fulde datasæt (der var 500 i datasættet - Oskar og Christopher vælger blot de første 200)
df_samples_200_text = pd.merge(df_samples_ID_200,df_full, on='ID')

df_samples_200_text.to_csv('df_samples_200_text.csv', sep=';', index=False)


#Oskar og Christopher har nu gennemlæst notaterne og lavet en guldstandard
#udvælger dato, psykpato og prop status fra de 500 tilfældigt udvalgte
df_temp=pd.read_csv('df_samples_500_text.csv', sep=';;;', engine='python')
df_temp.columns=['temp']
df_temp['ID'] = df_temp['temp'].str[:5]
df_temp['ID'] = df_temp['ID'].str.replace("\t\d+"," ")
df_temp['ID'] = df_temp['ID'].str.strip()
df_temp['konklusion'] = df_temp['temp'].str.strip().str[-1]
df_temp = df_temp.drop('temp', axis=1)

df_temp["konklusion"] = df_temp["konklusion"].astype(str).astype(float)
df_temp.loc[df_temp['konklusion'] == 2, 'konklusion'] = 1

df_full = pd.read_csv('new_data_results_DateScore.csv', sep=';')
df_temp["ID"] = df_temp["ID"].astype(str).astype(float)
df_konklusion_500 = pd.merge(df_temp,df_full, on='ID')
df_konklusion_500 = df_konklusion_500.drop('ID', axis=1)

df_konklusion_500.to_csv('df_konklusion_500.csv', index=False)

test = pd.read_csv('df_konklusion_500.csv')


#udvælger dato og psykpato status fra oprindeligt datasæt
data="\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\Corona_projekt_psyk_samlet_data_efter_3runde_12_april_inkl_pato_V2.xlsx"
df_oprindeligt = pd.read_excel(data)

df_oprindeligt.drop(df_oprindeligt.loc[df_oprindeligt['ALDER_KOR']<18].index, inplace=True)

#konverterer Datotid til læsbart format (det er noget frygtelig rod)
df_oprindeligt['date'] = df_oprindeligt['Datotid'].str[:10]
df_oprindeligt["date"] = df_oprindeligt["date"].astype(str)
df_oprindeligt['date'] = pd.to_datetime(df_oprindeligt['date'], format='%d-%m-%Y')
df_oprindeligt['date'] = df_oprindeligt['date'].dt.date
df_oprindeligt = df_oprindeligt[['date', 'konklusion_efter_3_runde']]

df_oprindeligt["konklusion_efter_3_runde"] = df_oprindeligt["konklusion_efter_3_runde"].astype(str).astype(float)
df_oprindeligt.loc[df_oprindeligt['konklusion_efter_3_runde'] == 3, 'konklusion_efter_3_runde'] = 0
df_oprindeligt.loc[df_oprindeligt['konklusion_efter_3_runde'] == 2, 'konklusion_efter_3_runde'] = 1

#fjerner de datoer hvor der er mindre end 4 observationer
s=df_oprindeligt.groupby('date').date.count()
s=s[s>3]
s=s.to_frame()
s['date_2'] = s.index
s.rename(columns = {'date' : 'counts', 'date_2' : 'date'}, inplace = True)
s=s.reset_index(drop=True)

df_oprindeligt_noID = pd.merge(df_oprindeligt, s, on="date")
df_oprindeligt_noID = df_oprindeligt_noID.drop('counts', axis=1)

df_oprindeligt_noID.to_csv('df_oprindelig_konklusion.csv', index=False)






#Tester at jeg har gemt (og kan hente modellen) korrekt
test="test_metadata.csv"
test = pd.read_csv(test)
label_column="labels"

test_x=test.drop("labels", axis=1)
test_x=test_x.drop("Unnamed: 0", axis=1)


test_prop = model.predict_proba(test_x)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(test[label_column], test_prop[:,1], pos_label=1)    
roc_auc_test = metrics.auc(fpr_test, tpr_test)
print(roc_auc_test)




