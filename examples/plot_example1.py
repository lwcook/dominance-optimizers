import json
import sys
import os
import subprocess
import pdb
import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../domopt/')))

import utilities as utils

from populations import readArchive

def savefig(name='saved_fig', bSaveBase=False,
            base='/phd-thesis/Figs/', bSaveData=False, formatstr='pdf'):
    '''Function that saves the plot as well as the
    underlying data of the currently open figure:
    -name: string that the figure is saved as'''

    subprocess.call(["mkdir", "-p", "./figs/"])
    plt.savefig('./figs/'+str(name)+'.' + formatstr, format=formatstr)

def plot_points(alg, stype, **args):
    name = 'output/'+str(alg)+'_'+stype+'_points.txt'
    front = readArchive(name)

    xp = [p['design'][0] for p in front]
    yp = [p['design'][1] for p in front]
    plt.scatter(xp, yp, label=stype, **args)

def plot_moments(alg, stype, **args):
    name = 'output/'+str(alg)+'_'+stype+'_front.txt'
    front = readArchive(name)

    xp = [p['mean'] for p in front]
    yp = [p['std'] for p in front]
    plt.scatter(xp, yp, label=stype, **args)

def plot_CDFs(alg, stype, **args):

    name = 'output/'+str(alg)+'_'+stype+'_front.txt'
    front = readArchive(name)

    for ind in front:
        qs = [tup[0] for tup in ind['CDF']]
        hs = [tup[1] for tup in ind['CDF']]
        plt.plot(qs, hs, **args)
    plt.plot([], [], label=stype, **args)


def main():

    blue = utils.blue
    red = utils.red
    grey = utils.grey
    green = utils.green

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(ax1)
    plot_points('GA', 'fsd', c=utils.grey, lw=0, s=10)
    plot_points('GA', 'mv', c=utils.red, lw=0, s=10)
    plot_points('GA', 'mvfsd', c=utils.blue, lw=0, s=10)
    plot_points('GA', 'ssd', c=utils.green, lw=0, s=10)
    plt.legend(loc='upper right')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Genetic Algorithm')
    plt.sca(ax2)
    plot_points('TS', 'fsd', c=utils.grey, lw=0, s=10)
    plot_points('TS', 'mv', c=utils.red, lw=0, s=10)
    plot_points('TS', 'mvfsd', c=utils.blue, lw=0, s=10)
    plot_points('TS', 'ssd', c=utils.green, lw=0, s=10)
    plt.legend(loc='upper right')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Tabu Search')
    plt.tight_layout()
    savefig('GA_fronts')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(ax1)
    plot_moments('GA', 'fsd', c=utils.grey, lw=0, s=10)
    plot_moments('GA', 'mv', c=utils.red, lw=0, s=10)
    plot_moments('GA', 'mvfsd', c=utils.blue, lw=0, s=10)
    plot_moments('GA', 'ssd', c=utils.green, lw=0, s=10)
    plt.legend(loc='upper right')
    plt.xlim([8.5, 12.5])
    plt.ylim([0, 1.9])
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Genetic Algorithm')
    plt.sca(ax2)
    plot_moments('TS', 'fsd', c=utils.grey, lw=0, s=10)
    plot_moments('TS', 'mv', c=utils.red, lw=0, s=10)
    plot_moments('TS', 'mvfsd', c=utils.blue, lw=0, s=10)
    plot_moments('TS', 'ssd', c=utils.green, lw=0, s=10)
    plt.legend(loc='upper right')
    plt.xlim([8.5, 12.5])
    plt.ylim([0, 1.9])
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Tabu Search')
    plt.tight_layout()
    savefig('GA_fronts')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(ax1)
    plot_CDFs('GA', 'fsd', c=utils.grey, lw=0.8, linestyle='dotted')
    plot_CDFs('GA', 'mv', c=utils.red, lw=0.8, linestyle='dashed')
    plot_CDFs('GA', 'mvfsd', c=utils.blue, lw=0.8)
    plot_CDFs('GA', 'ssd', c=utils.green, lw=0.8)
    plt.legend(loc='upper left', handlelength=2)
    plt.xlim([5, 13])
    plt.ylim([0, 1])
    plt.xlabel('Quantity of Interest - q')
    plt.ylabel('CDF')
    plt.title('Genetic Algorithm')
    plt.sca(ax2)
    plot_CDFs('TS', 'fsd', c=utils.grey, lw=0.8, linestyle='dotted')
    plot_CDFs('TS', 'mv', c=utils.red, lw=0.8, linestyle='dashed')
    plot_CDFs('TS', 'mvfsd', c=utils.blue, lw=0.8)
    plot_CDFs('TS', 'ssd', c=utils.green, lw=0.8)
    plt.legend(loc='upper left', handlelength=2)
    plt.xlim([5, 13])
    plt.ylim([0, 1])
    plt.xlabel('Quantity of Interest - q')
    plt.ylabel('CDF')
    plt.title('Tabu Search')
    plt.tight_layout()
    savefig('GA_CDFs')



    plt.show()

if __name__ == "__main__":
    main()
