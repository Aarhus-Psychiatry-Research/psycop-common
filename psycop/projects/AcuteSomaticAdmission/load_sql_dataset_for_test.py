import pandas as pd

from psycop.common.global_utils.sql.loader import sql_load


view = "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022] "
cols_to_keep = "datotid_indlaeggelse, datotid_udskrivning, dw_ek_borger, pattypetekst, akutindlaeggelse, shakKode_kontaktansvarlig"

#sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
 
sql = "SELECT * FROM [fct]." + view

#sql += " AND pattypetekst = 'Indlagt' AND akutindlaeggelse = 'true' AND SUBSTRING(shakKode_kontaktansvarlig, 1, 4) != '6600'"
#sql += " AND datotid_indlaeggelse IS NOT NULL AND datotid_udskrivning IS NOT NULL;"

df = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore

print(df)

print("arrrgh")