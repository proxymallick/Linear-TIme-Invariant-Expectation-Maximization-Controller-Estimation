from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
""" # Get the data (csv file is hosted on the web)
url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)
 
# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes



X=np.random.randn(30,1).reshape(30,)
Y=np.random.randn(30,1).reshape(30,)
Z=-np.random.randn(50,1).reshape(50,)

fig_test = plt.figure()
ax = fig_test.gca(projection='3d')
#surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
surf = ax.plot_trisurf(X,Y,Z, cmap=plt.cm.viridis, linewidth=0.2)

fig_test.colorbar( surf, shrink=0.5, aspect=5)
plt.show()
  """

def multiple3(tpl_lst):
    mul = []
    for tpl in tpl_lst:
        calc = (.0001*tpl[0]) + (.017*tpl[1])+ 6.166
        mul.append(calc)
    return mul

fig = plt.figure()
ax = fig.gca(projection='3d')
'''some skipped code for the scatterplot'''
X = np.arange(0, 40000, 500)
Y = np.arange(0, 40, .5)
X, Y = np.meshgrid(X, Y)
Z = multiple3(zip(X,Y))

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.viridis,
                       linewidth=0.12, antialiased=False, alpha =.1)
ax.set_zlim(1.01, 11.01)
ax.set_xlabel(' x = IPP')
ax.set_ylabel('y = UNRP20')
ax.set_zlabel('z = DI')

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()