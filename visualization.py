import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data4 = {'beams': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
         'rf': [148.1,
                151.7,
                150.5,
                153.7,
                150.9,
                150.1,
                152.0
                ],
         'displacement': [0.9822,
                          0.8976,
                          0.835,
                          0.7632,
                          0.7334,
                          0.7006,
                          0.5886
                          ]}
df4 = pd.DataFrame.from_dict(data4)
data6 = {'beams': [3, 4, 5, 6, 7, 8, 9, 10],
         'rf': [152.6,
                155.1,
                149.3,
                153,
                151.1,
                151.1,
                153.4,
                154.6,
                ],
         'displacement': [
             0.8591,
             0.7447,
             0.6521,
             0.6068,
             0.5504,
             0.5107,
             0.4711,
             0.4609
         ]}
df6 = pd.DataFrame.from_dict(data6)

data8 = {'beams': [3, 4, 5, 6, 7, 8, 9],
         'rf': [156.2,
                159.9,
                150.4,
                151.3,
                151.5,
                152.6,
                150.3
                ],
         'displacement':[
             0.6826,
             0.5608,
             0.4845,
             0.4536,
             0.3992,
             0.3549,
             0.3532
         ]}
df8 = pd.DataFrame.from_dict(data8)

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('colorblind')

fig, ax = plt.subplots()
sns.regplot(x='beams', y='rf', data=df4, ci=None, order=1, label="4 mm")
sns.regplot(x='beams', y='rf', data=df6, ci=None, order=2, label="6 mm")
sns.regplot(x='beams', y='rf', data=df8, ci=None, order=2, label="8 mm")
plt.legend()
plt.show()

fig, ax = plt.subplots()
sns.regplot(x='beams', y='displacement', data=df4, ci=None, order=2, label="4 mm")
sns.regplot(x='beams', y='displacement', data=df6, ci=None, order=2, label="6 mm")
sns.regplot(x='beams', y='displacement', data=df8, ci=None, order=2, label="8 mm")
plt.legend()
plt.show()
