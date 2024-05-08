
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml


# Compute center of mass.
def com(c, r):
    return np.sum(np.multiply(c, r ** 3), axis=0) / np.sum(r ** 3, axis=0)


# Render an image of the aggregate. 
def render(c, r):

    u = np.linspace(0, 2*np.pi, 20)  # angles for plotting
    v = np.linspace(0,   np.pi, 20)

    x0 = np.outer(np.cos(u), np.sin(v))  # standard sphere coordinates
    y0 = np.outer(np.sin(u), np.sin(v))
    z0 = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(facecolor="black")
    ax = plt.axes(projection="3d")

    npart = r.size  # number of particles
    
    for ii in range(npart):  # loop over monomers
        x = r[ii] * x0 + c[ii,0]  #  x,y,z coordinates for each sphere
        y = r[ii] * y0 + c[ii,1]
        z = r[ii] * z0 + c[ii,2]

        ax.plot_surface(x, y, z, rstride=5, cstride=5)  # generate sphere
    
    ax.axis('equal')  # to avoid skew
    plt.axis('off')  # make axes invisible
    plt.show()


# Write data to an external xyz file.
def write_xyz(x, r, fname='collapsed.xyz'):
    df = pd.DataFrame(np.hstack((x, r)))

    with open(fname, 'w') as f:
        f.write(str(r.size) + '\nx \ty \tx \tr\n')

    df.to_csv(fname, sep = '\t', header=False, index=False, mode="a")



# To load options from YML configuration files.
def load_config(fname):
    with open(fname) as stream:
        out = yaml.safe_load(stream)

    return out
