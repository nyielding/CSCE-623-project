# %%
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# %%
testrun = True
PATHUP = '../'
TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt', 
            'salyut-7.txt', 'spektr.txt', 'zarya.txt']
# %%
if testrun:
    for i, txt in enumerate(TLE_FILES):
        TLE_FILES[i] = os.path.join(PATHUP, txt)


# %%
# Read the first 1000 TLEs from files
columns = ['rx','ry', 'rz', 'vx', 'vy', 'vz', 'error', 'sat_name']
dflist = []
for sat_name in TLE_FILES:
    TLEs = open(sat_name, 'r')
    sats = np.zeros([1000, 7])
    for i in range(1000):
        line1 = TLEs.readline()
        line2 = TLEs.readline()
        satellite = Satrec.twoline2rv(line1, line2)
        e, r, v = satellite.sgp4(satellite.jdsatepoch, satellite.jdsatepochF)
        sats[i, 0:3] = r
        sats[i, 3:6] = v
        sats[i, 6] = e
    if testrun:
        name_column = sat_name[3:-4]
    else:
        name_column = sat_name[:-4]
    df_temp = pd.DataFrame(data=sats, columns=columns[:-1])
    df_temp['sat_name'] = name_column
    dflist.append(df_temp)

df = pd.concat(dflist)

TLEs.close()

# %%
df.head()
df.describe()
# %%
df = df.dropna()
