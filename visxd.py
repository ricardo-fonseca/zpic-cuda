import zdf

import sys
import os.path

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

def grid2d( filename, xlim = None, ylim = None, grid = False, cmap = None, vsim = False ):
    """Generates a colormap plot from a 2D grid zdf file

    Args:
        filename (str):
            Name of ZDF file to open
        xlim (tuple, optional):
            Lower and upper limits of x axis. Defaults to the x limits of the
            grid data.
        ylim (tuple, optional):
            Lower and upper limits of y axis. Defaults to the y limits of the
            grid data.
        grid (bool, optional):
            Display a grid on top of colormap. Defaults to False.
        cmap (str, optional):
            Name of the colormap to use. Defaults to the matplotlib imshow() 
            colormap.
        vsim:
            Setup a symmetric value scale [ -max(|val|), max(|val|) ]. Defaults to setting
            the value scale to [ min, max ]

    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (data, info) = zdf.read(filename)

    if ( info.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filename))
        return
    
    if ( info.grid.ndims != 2 ):
        print("(*error*) file {} is not a 2D grid file".format(filename))
        return

    range = [
        [info.grid.axis[0].min, info.grid.axis[0].max],
        [info.grid.axis[1].min, info.grid.axis[1].max]
    ]

    if ( vsim ):
        amax = np.amax( np.abs(data) )
        plt.imshow( data, interpolation = 'nearest', origin = 'lower',
                vmin = -amax, vmax = +amax,
                extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
                aspect = 'auto', cmap=cmap )
    else:
        plt.imshow( data, interpolation = 'nearest', origin = 'lower',
                extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
                aspect = 'auto', cmap=cmap )

    zlabel = "{}\,[{:s}]".format( info.grid.label, info.grid.units )

    plt.colorbar().set_label(r'$\sf{' + zlabel + r'}$')

    xlabel = "{}\,[{:s}]".format( info.grid.axis[0].label, info.grid.axis[0].units )
    ylabel = "{}\,[{:s}]".format( info.grid.axis[1].label, info.grid.axis[1].units )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    plt.title("$\sf {} $\nt = ${:g}$ [$\sf {}$]".format(
        info.grid.label.replace(" ","\;"),
        info.iteration.t,
        info.iteration.tunits))

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()

def part2D( filename, qx, qy, xlim = None, ylim = None, grid = True, 
    marker = '.', ms = 1, alpha = 1 ):
    """Generates an (x,y) scatter plot from a ZDF particle file.

    Args:
        filename (str):
            Name of ZDF file to open
        qx (str):
            X axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        qy (str): _description_
            Y axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        xlim (tuple, optional):
            Lower and upper limits of x axis. Defaults to the limits of the "qx" particle data.
        ylim (tuple, optional):
            Lower and upper limits of y axis. Defaults to the limits of the "qy" particle data.
        grid (bool, optional):
            Display a grid on top of scatter plot. Defaults to True.
        marker (str, optional)
            Plot marker to use for the scatter plot. Defaults to '.'.
        ms (int, optional):
            Marker size to use for the scatter plot. Defaults to 1.
        alpha (int, optional):
            Marker opacity to use for the scatter plot. Defaults to 1.
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (particles, info) = zdf.read(filename)

    if ( info.type != "particles" ):
        print("(*error*) file {} is not a particles file".format(filename))
        return
    
    if ( not qx in info.particles.quants ):
        print("(*error*) '{}' quantity (q1) is not present in file".format(qx) )
        return

    if ( not qy in info.particles.quants ):
        print("(*error*) '{}' quantity (q2) is not present in file".format(qy) )
        return

    x = particles[qx]
    y = particles[qy]

    plt.plot(x, y, marker, ms=ms, alpha = alpha)

    title = "{}/{}".format( info.particles.qlabels[qy], info.particles.qlabels[qx])
    timeLabel = "t = {:g}\,[{:s}]".format(info.iteration.t, info.iteration.tunits)

    plt.title(r'$\sf{' + title + r'}$' + '\n' + r'$\sf{' + timeLabel + r'}$')

    xlabel = "{}\,[{:s}]".format( info.particles.qlabels[qx], info.particles.qunits[qx] )
    ylabel = "{}\,[{:s}]".format( info.particles.qlabels[qy], info.particles.qunits[qy] )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()

def histogram( filename, q, bins = 128, range = None, density = True, log = False, color = None, histtype = 'bar' ):
    """Generates a histogram (frequency) plot from a ZDF particle file.

    Args:
        filename (str):
            Name of ZDF file to open
        q (str):
            Quantity to use, usually one of "x", "y", "ux", "uy", "uz", etc.
        bins (int, optional):
            Number of bins to use for the histogram. Defaults to 128.
        range (tuple, optional):
            Lower and upper limits of the histogram. Defaults to minimum and maximum values of the selected quantity.
        density (bool, optional):
            Plot a probability density (bin count divided by the total number of counts and the bin width) instead of 
            bin count. Defaults to True.
        log (bool, optional):
            Use log scale for histogram axis. Defaults to False.
        color (str, optional):
            Color for plot. Defaults to the matplotlib plot color.
        histtype (str, optional):
            Type of histogram to draw, check matplotlib histogram documentation for details. Defaults to 'bar'.
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (particles, info) = zdf.read(filename)

    if ( info.type != "particles" ):
        print("(*error*) file {} is not a particles file".format(filename))
        return
    
    if ( not q in info.particles.quants ):
        print("(*error*) '{}' quantity (q1) is not present in file".format(q) )
        return
    
    data = particles[q]

    plt.hist( data, bins = bins, range = range, density = density, log = log, color = color, histtype = histtype )
    title = "{} - {}".format( info.particles.label, info.particles.qlabels[q])
    timeLabel = "t = {:g}\,[{:s}]".format(info.iteration.t, info.iteration.tunits)

    plt.title(r'$\sf{' + title + r'}$' + '\n' + r'$\sf{' + timeLabel + r'}$')

    xlabel = "{}\,[{:s}]".format( info.particles.qlabels[q], info.particles.qunits[q] )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + "n" + r'}$')

    plt.show()