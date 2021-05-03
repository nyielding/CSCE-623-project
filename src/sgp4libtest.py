# %%
from sgp4.api import Satrec
# %%

s = '1 25544U 98067A   98324.28472222 -.00003657  11563-4  00000+0 0    10'
t = '2 25544  51.5908 168.3788 0125362  86.4185 359.7454 16.05064833    05'

satellite = Satrec.twoline2rv(s, t)
# %%

print(satellite)
# %%
