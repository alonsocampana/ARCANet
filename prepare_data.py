import pandas as pd
import numpy as np
from urllib.request import urlretrieve


url1 = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_public_raw_data_24Jul22.csv.zip"
file1 = "data/GDSC1_public_raw_data_24Jul22.csv.zip"
urlretrieve(url1, file1)
GDSC = pd.read_csv(file1)
GDSC["INTENSITY"] = np.log(GDSC["INTENSITY"] + 1)
identifier_col = GDSC.loc[:, "COSMIC_ID"].astype(str) + "&"+ GDSC.loc[:, "SCAN_ID"].astype(str) +   GDSC.loc[:, "DRUGSET_ID"].astype(str) +  GDSC.loc[:, "BARCODE"].astype(str) + GDSC.loc[:, "SEEDING_DENSITY"].astype(str)
GDSC = GDSC.assign(identifier = identifier_col)
data_viab = GDSC.groupby(["CONC", "identifier", "DRUG_ID", "COSMIC_ID"])["INTENSITY"].median()
blank_vals = GDSC.query("TAG == 'NC-0'").groupby("identifier")["INTENSITY"].median()
posblank_vals = GDSC.query("TAG == 'B'").groupby("identifier")["INTENSITY"].median()
data_viab = data_viab.reset_index()
max_vals = blank_vals.loc[data_viab.loc[:, "identifier"]]
min_vals = posblank_vals.loc[data_viab.loc[:, "identifier"]]
data_viab["INTENSITY"] = (data_viab["INTENSITY"].to_numpy() - min_vals.to_numpy())/(max_vals.to_numpy().squeeze() - min_vals.to_numpy().squeeze())
data_viab = data_viab.groupby(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].mean().reset_index()
matrix_viab = data_viab.set_index(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].reset_index().pivot(index=["DRUG_ID", "COSMIC_ID"], columns = "CONC", values= "INTENSITY")
matrix_viab = matrix_viab.loc[:, np.sort(matrix_viab.columns)]
matrix_viab.to_csv("data/matrix_corrected_GDSC1.csv")

url2 = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_public_raw_data_24Jul22.csv.zip"
file2 = "data/GDSC2_public_raw_data_24Jul22.csv.zip"
urlretrieve(url2, file2)
GDSC = pd.read_csv(file2)
GDSC["INTENSITY"] = np.log(GDSC["INTENSITY"] + 1)
identifier_col = GDSC.loc[:, "COSMIC_ID"].astype(str) + "&"+ GDSC.loc[:, "SCAN_ID"].astype(str) +   GDSC.loc[:, "DRUGSET_ID"].astype(str) +  GDSC.loc[:, "BARCODE"].astype(str) + GDSC.loc[:, "SEEDING_DENSITY"].astype(str)
GDSC = GDSC.assign(identifier = identifier_col)
data_viab = GDSC.groupby(["CONC", "identifier", "DRUG_ID", "COSMIC_ID"])["INTENSITY"].median()
blank_vals = GDSC.query("TAG == 'NC-0'").groupby("identifier")["INTENSITY"].median()
posblank_vals = GDSC.query("TAG == 'B'").groupby("identifier")["INTENSITY"].median()
data_viab = data_viab.reset_index()
max_vals = blank_vals.loc[data_viab.loc[:, "identifier"]]
min_vals = posblank_vals.loc[data_viab.loc[:, "identifier"]]
data_viab["INTENSITY"] = (data_viab["INTENSITY"].to_numpy() - min_vals.to_numpy())/(max_vals.to_numpy().squeeze() - min_vals.to_numpy().squeeze())
data_viab = data_viab.groupby(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].mean().reset_index()
matrix_viab = data_viab.set_index(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].reset_index().pivot(index=["DRUG_ID", "COSMIC_ID"], columns = "CONC", values= "INTENSITY")
matrix_viab = matrix_viab.loc[:, np.sort(matrix_viab.columns)]
matrix_viab.to_csv("data/matrix_corrected_GDSC2.csv")