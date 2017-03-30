
"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class ICMplot contains generic plot functions to display S5p Tropomi data

-- generate figures --
 Public functions a page in the output PDF
 * draw_signal
 * draw_hist
 * draw_quality
 * draw_geolocation
 * draw_cmp_swir

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np

import matplotlib as mpl

#
# Suggestion for the name of the report/pdf-file
#    <identifier>_<yyyymmdd>_<orbit>.pdf
# where
#  identifier : name of L1B/ICM/OCM product or monitoring database
#  yyyymmdd   : coverage start-date or start-date of monitoring entry
#  orbit      : reference orbit
#
# Suggestion for info-structure to be displayed in figure
# * Versions used to generate the data:
#   algo_version : version of monitoring algorithm (SRON)
#   db_version   : version of the monitoring database (SRON)
#   l1b_version  : L1b processor version
#   icm_version  : ICM processor version
#   msm_date     : Date-time of first frame
# * Data in detector coordinates or column/row averaged as function of time
#   sign_median  : (biweight) median of signals
#   sign_spread  : (biweight) spread of signals
#   error_median : (biweight) median of errors
#   error_spread : (biweight) spread of errors
# * Detector quality data:
#   dpqf_01      : number of good pixels with threshold at 0.1
#   dpqf_08      : mumber of good pixels with threshold at 0.8
#
# To force the sequence of the displayed information it is advised to use
# collections.OrderedDict:
#
# > from collections import OrderedDict
# > dict_info = OrderedDict({key1 : value1})
# > dict_info.update({key2 : value2})
# etc.
#

#- local functions --------------------------------
def convert_units(units, vmin, vmax):
    dscale = 1.0
    if units is None:
        zunit = None
    elif units.find('electron') >= 0:
        max_value = max(abs(vmin), abs(vmax))

        if max_value > 1000000000:
            dscale = 1e9
            zunit = units.replace('electron', 'Ge')
        elif max_value > 1000000:
            dscale = 1e6
            zunit = units.replace('electron', 'Me')
        elif max_value > 1000:
            dscale = 1e3
            zunit = units.replace('electron', 'ke')
        else:
            zunit = units.replace('electron', 'e')
    elif units >= 'V':
        max_value = max(abs(vmin), abs(vmax))

        if max_value <= 1e-4:
            dscale = 1e-6
            zunit = units.replace('V', r'$\mu \mathrm{V}$')
        elif max_value <= 0.1:
            dscale = 1e-3
            zunit = units.replace('V', 'mV')
        else:
            zunit = units
    else:
        zunit = units

    return (zunit, dscale)

#- main function __--------------------------------
# pylint: disable=too-many-arguments, too-many-locals
class S5Pplot(object):
    """
    Generate figure(s) for the SRON Tropomi SWIR monitor website or MPC reports

    The PDF will have the following name:
        <dbname>_<startDateTime of monitor entry>_<orbit of monitor entry>.pdf
    """
    def __init__(self, figname, cmap="Rainbow", mode='frame'):
        """
        Initialize multipage PDF document for an SRON SWIR ICM report

        Parameters
        ----------
        figname   :  string
             name of PDF file
        cmap      :  string
             matplotlib color map
        mode      :  string
             data mode - 'frame' or 'dpqm'
        """
        from matplotlib.backends.backend_pdf import PdfPages

        self.__pdf  = PdfPages(figname)
        self.__cmap = cmap
        self.__mode = mode

    def __repr__(self):
        pass

    def __del__(self):
        """
        Close multipage PDF document
        """
        if self.__pdf is None:
            return

        doc = self.__pdf.infodict()
        doc['Title'] = 'Report on Tropomi SWIR diode-laser ISRF measurement'
        doc['Author'] = '(c) SRON, Netherlands Institute for Space Research'
        doc['Subject'] = 'Shown are ISRF parameters and goodness of fit'
        doc['Keywords'] = 'PdfPages multipage keywords author title subject'
        self.__pdf.close()

    # --------------------------------------------------
    @staticmethod
    def __fig_info(fig, dict_info, aspect=-1, fontsize='small'):
        """
        Add meta-information in the current figure

        Parameters
        ----------
        fig       :  Matplotlib figure instance
        dict_info :  dictionary or sortedDict
           legenda parameters to be displayed in the figure
        """
        from datetime import datetime

        info_str = ""
        if dict_info is not None:
            for key in dict_info:
                if isinstance(dict_info[key], float) \
                   or isinstance(dict_info[key], np.float32):
                    info_str += "{} : {:.5g}".format(key, dict_info[key])
                else:
                    info_str += "{} : {}".format(key, dict_info[key])
                info_str += '\n'
        info_str += 'created : ' \
                   + datetime.utcnow().isoformat(' ', timespec='minutes')

        if aspect == 4:
            fig.text(0.9, 0.975, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif aspect == 3:
            fig.text(0.9, 0.975, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif aspect == 2:
            fig.text(1, 1, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif aspect == 1:
            fig.text(0.3, 0.225, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        else:
            fig.text(0.9, 0.965, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})

    # --------------------------------------------------
    def draw_signal(self, msm, data_col=None, data_row=None,
                    title=None, sub_title=None, fig_info=None,
                    vperc=None, vrange=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        data_col   :  ndarray
           Numpy array (1D) column averaged values of data
           Default is calculated as biweight(data, axis=1)
        data_row   :  ndarray
           Numpy array (1D) row averaged values of data
           Default is calculated as biweight(data, axis=0)
        vrange     :  float in range of [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc will be ignored
        vperc      :  float in range of [0,100]
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.]
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title  :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data 
        median & spread.
        """
        import warnings

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .biweight import biweight
        from . import sron_colorschemes

        sron_colorschemes.register_cmap_rainbow()
        line_colors = sron_colorschemes.get_line_colors()

        # calculate column/row medians (if required)
        if data_col is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN slice encountered")
                data_col = np.nanmedian(msm.value, axis=1)

        if data_row is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN slice encountered")
                data_row = np.nanmedian(msm.value, axis=0)

        # determine aspect-ratio of data and set sizes of figure and sub-plots
        dims = msm.value.shape
        aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

        if aspect == 1:
            figsize = (10, 9)
        elif aspect == 2:
            figsize = (12, 7)
        elif aspect == 3:
            figsize = (14, 6.5)
        elif aspect == 4:
            figsize = (16, 6)
        else:
            print(__name__ + '.draw_signal', dims, aspect)
            print('*** FATAL: aspect ratio not implemented, exit')
            return

        # set label and range of X/Y axis
        print(msm.value.shape)
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]
        print(msm.coords._fields, extent)

        # scale data to keep reduce number of significant digits small to
        # the axis-label and tickmarks readable
        if vrange is None:
            if vperc is None:
                vperc = (1., 99.)
            else:
                assert len(vperc) == 2
            (vmin, vmax) = np.percentile(msm.value[np.isfinite(msm.value)],
                                         vperc)
        else:
            assert len(vrange) == 2
            (vmin, vmax) = vrange

        # convert units from electrons to ke, Me, ...
        print(msm.units)
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize=14,
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        img = ax_img.imshow(msm.value / dscale, cmap=self.__cmap,
                            extent=extent, origin='lower',
                            vmin=vmin / dscale, vmax=vmax / dscale,
                            aspect='equal', interpolation='none')
        xbins = len(ax_img.get_xticklabels())
        ybins = len(ax_img.get_yticklabels())
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_img.get_yticklabels():
            ytl.set_visible(False)
        if sub_title is not None:
            ax_img.set_title(sub_title)
        ax_img.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                    verticalalignment='bottom', rotation='vertical',
                    fontsize='xx-small', transform=ax_img.transAxes)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        plt.colorbar(img, cax=cax, label='{} [{}]'.format(msm.name, zunit))
        #
        ax_medx = divider.append_axes("bottom", 1.2, pad=0.25)
        ax_medx.plot(xdata, data_row / dscale,
                     lw=0.5, color=line_colors[0])
        xstep = (xdata[-1] - xdata[0]) // (xdata.size - 1)
        ax_medx.set_xlim([xdata[0], xdata[-1] + xstep])
        ax_medx.grid(True)
        ax_medx.set_xlabel(xlabel)
        #ax_medx.locator_params(axis='x', nbins=5)
        #ax_medx.locator_params(axis='y', nbins=4)
        #
        ax_medy = divider.append_axes("left", 1.1, pad=0.25)
        ax_medy.plot(data_col / dscale, ydata,
                     lw=0.5, color=line_colors[0])
        ystep = (ydata[-1] - ydata[0]) // (ydata.size - 1)
        ax_medy.set_ylim([ydata[0], ydata[-1] + ystep])
        ax_medy.locator_params(axis='y', nbins=ybins)
        print(ystep, [ydata[0], ydata[-1] + ystep])
        ax_medy.grid(True)
        ax_medy.set_ylabel(ylabel)
        #ax_medy.locator_params(axis='x', nbins=5)
        #ydata = np.append(ydata, ydata[-1]+1)
        #print(ydata[::ydata.size//4])
        #ax_medy.set_yticks(ydata[::ydata.size//4])

        # add annotation
        (median, spread) = biweight(msm.value, spread=True)
        if zunit is not None:
            median_str = '{:.5g} {}'.format(median / dscale, zunit)
            spread_str = '{:.5g} {}'.format(spread / dscale, zunit)
        else:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)

        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        self.__fig_info(fig, fig_info, aspect)
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_quality(self, qdata, low_thres=0.1, high_thres=0.8,
                     qlabels=None, title=None, sub_title=None, fig_info=None):
        """
        Display pixel quality data

        Parameters
        ----------
        qdata        :  ndarray
           Numpy array (2D) holding quality data, range [0, 1] or NaN.
           Zero is for dead pixels and one is for excellent pixels.
        low_thres   :  float
           threshold for usable pixels (with caution), below this threshold
           pixels are considered bad
        high_thres  :  float
           threshold for good pixels
        qlabels     :  list of strings
           quality ranking labels, default ['invalid', 'bad', 'poor', 'good']
            - 'invalid': value is negative or NaN
            - 'bad'    : 0 <= value < low_thres
            - 'poor'   : low_thres <= value < high_thres
            - 'good'   : value >= high_thres
        title       :  string
           Title of the figure (use attribute "title" of product)
        sub_title   :  string
           Sub-title of the figure (use attribute "comment" of product)
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date, 'thres_01', 
        'qmask_01', 'thres_08' and 'qmask_08'.
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if qlabels is None:
            qlabels = ["invalid", "bad", "poor", "good"]  ## "fair" or "poor"?

        # determine aspect-ratio of data and set sizes of figure and sub-plots
        dims = qdata.value.shape
        xdata = np.arange(dims[1], dtype=float)
        ydata = np.arange(dims[0], dtype=float)
        extent = [0, dims[1], 0, dims[0]]
        aspect = int(np.round(dims[1] / dims[0]))

        if aspect == 1:
            figsize = (10, 9)
        elif aspect == 2:
            figsize = (12, 7)
        elif aspect == 4:
            figsize = (16, 6)
        else:
            print(__name__, dims)
            print('*** FATAL: aspect ratio not implemented, exit')
            return

        # scale data to integers between [0, 10]
        thres_min = 10 * low_thres
        thres_max = 10 * high_thres
        qmask = (qdata.value * 10).astype(np.int8)

        # set columns and row with at least 75% dead-pixels to -1
        qmask[~np.isfinite(qdata.value)] = -1
        qmask[qmask < -1] = -1
        unused_cols = np.all(qmask <= 0, axis=0)
        if unused_cols.size > 0:
            qmask[:, unused_cols] = -1
        unused_rows = np.all(qmask <= 0, axis=1)
        if unused_rows.size > 0:
            qmask[unused_rows, :] = -1

        # define colormap with only 4 colors
        clist = ['#BBBBBB', '#EE6677','#CCBB44','#FFFFFF']
        cmap = mpl.colors.ListedColormap(clist)
        bounds = [-1, 0, thres_min, thres_max, 10]
        mbounds = [(bounds[1] + bounds[0]) / 2,
                   (bounds[2] + bounds[1]) / 2,
                   (bounds[3] + bounds[2]) / 2,
                   (bounds[4] + bounds[3]) / 2]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # set label and range of X/Y axis
        xlabel = 'column'
        ylabel = 'row'

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize=14,
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        img = ax_img.imshow(qmask, cmap=cmap, norm=norm, extent=extent,
                            vmin=-1, vmax=10, aspect='equal',
                            interpolation='none', origin='lower')
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_img.get_yticklabels():
            ytl.set_visible(False)
        if sub_title is not None:
            ax_img.set_title(sub_title)
        ax_img.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                    verticalalignment='bottom', rotation='vertical',
                    fontsize='xx-small', transform=ax_img.transAxes)

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(ax_img)
        #
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        plt.colorbar(img, cax=cax, ticks=mbounds, boundaries=bounds)
        cax.set_yticklabels(qlabels)
        #
        qmask_row_01 = np.sum(((qmask >= 0) & (qmask < thres_min)), axis=0)
        qmask_row_08 = np.sum(((qmask >= 0) & (qmask < thres_max)), axis=0)
        ax_medx = divider.append_axes("bottom", 1.2, pad=0.25, sharex=ax_img)
        ax_medx.step(xdata, qmask_row_08, lw=0.5, color=clist[2])
        ax_medx.step(xdata, qmask_row_01, lw=0.6, color=clist[1])
        ax_medx.set_xlim([0, dims[1]])
        ax_medx.grid(True)
        ax_medx.locator_params(axis='x', nbins=6)
        ax_medx.locator_params(axis='y', nbins=3)
        ax_medx.set_xlabel(xlabel)
        #
        qmask_col_01 = np.sum(((qmask >= 0) & (qmask < thres_min)), axis=1)
        qmask_col_08 = np.sum(((qmask >= 0) & (qmask < thres_max)), axis=1)
        ax_medy = divider.append_axes("left", 1.1, pad=0.25, sharey=ax_img)
        ax_medy.step(qmask_col_08, ydata, lw=0.5, color=clist[2])
        ax_medy.step(qmask_col_01, ydata, lw=0.6, color=clist[1])
        ax_medy.set_ylim([0, dims[0]])
        ax_medy.grid(True)
        ax_medy.locator_params(axis='x', nbins=3)
        ax_medy.locator_params(axis='y', nbins=4)
        ax_medy.set_ylabel(ylabel)

        # add annotation
        if fig_info is None:
            fig_info = OrderedDict({'thres_01': low_thres})
        else:
            fig_info.update({'thres_01': low_thres})
        fig_info.update({'qmask_01': np.sum(((qmask >= 0)
                                             & (qmask < thres_min)))})
        fig_info.update({'thres_08': high_thres})
        fig_info.update({'qmask_08': np.sum(((qmask >= 0)
                                             & (qmask < thres_max)))})

        self.__fig_info(fig, fig_info, aspect)
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_cmp_swir(self, msm, model_in, model_label='reference',
                      vrange=None, vperc=None,
                      title=None, sub_title=None, fig_info=None):
        """
        Display signal vs model (or CKD) comparison in three panels.
        Top panel shows data, middle panel shows residuals (data - model)
        and lower panel shows model.

        Optionally, two histograms are added, with the distribution of resp.
        the data values and residuals.

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        model       :  ndarray
           Numpy array (2D) holding the data to be displayed
        vrange     :  float in range of [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc wil be ignored
        vperc      :  float in range of [0,100]
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.]
        model_label :  string
           Name of dataset.  Default is 'reference'
        title       :  string
           Title of the figure (use attribute "title" of product)
        sub_title   :  string
           Sub-title of the figure (use attribute "comment" of product)
        """
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        from . import sron_colorschemes

        sron_colorschemes.register_cmap_rainbow()
        line_colors = sron_colorschemes.get_line_colors()

        # scale data to keep reduce number of significant digits small to
        # the axis-label and tickmarks readable
        if vperc is None:
            vperc = (1., 99.)
        else:
            assert len(vperc) == 2

        if vrange is None:
            (vmin, vmax) = np.percentile(msm.value[np.isfinite(msm.value)],
                                         vperc)
        else:
            assert len(vrange) == 2
            (vmin, vmax) = vrange

        # convert units from electrons to ke, Me, ...
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)
        if msm.units is None:
            zlabel = '{}'.format(msm.name)
        else:
            zlabel = '{} [{}]'.format(msm.name, zunit)

        # set label and range of X/Y axis
        print(msm.value.shape)
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

        # create residual image
        mask = np.isfinite(msm.value)
        signal = msm.value.copy()
        signal[mask] *= dscale
        mask = np.isfinite(model_in)
        model = model_in.copy()
        model[mask] *= dscale
        mask = np.isfinite(msm.value) & np.isfinite(model_in)
        residual = signal.copy()
        residual[~mask] = np.nan
        residual[mask] -= model[mask]

        # inititalize figure
        figsize = (9.2, 8.5)
        fig = plt.figure(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize=14,
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = GridSpec(4, 2)

        # create top-pannel with measurements
        ax1 = plt.subplot(gspec[0, :])
        for xtl in ax1.get_xticklabels():
            xtl.set_visible(False)
        if sub_title is not None:
            ax1.set_title(sub_title)
        img = ax1.imshow(signal, vmin=vmin, vmax=vmax, aspect='equal',
                         interpolation='none', origin='lower', extent=extent)
        ax1.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=ax1.transAxes)
        cbar = plt.colorbar(img)
        if zlabel is not None:
            cbar.set_label(zlabel)

        # create centre-pannel with residuals
        (rmin, rmax) = np.percentile(residual[np.isfinite(residual)], vperc)
        # convert units from electrons to ke, Me, ...
        (runit, rscale) = convert_units(msm.units, rmin, rmax)

        ax2 = plt.subplot(gspec[1, :], sharex=ax1)
        for xtl in ax2.get_xticklabels():
            xtl.set_visible(False)
        img = ax2.imshow(residual / rscale, aspect='equal', extent=extent,
                         vmin=rmin / rscale, vmax=rmax / rscale,
                         interpolation='none', origin='lower')
        ax2.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=ax2.transAxes)
        cbar = plt.colorbar(img)
        if runit is None:
            cbar.set_label('residuals')
        else:
            cbar.set_label('residuals [{}]'.format(runit))

        # create lower-pannel with reference (model, CKD, previous measurement)
        ax3 = plt.subplot(gspec[2, :], sharex=ax1)
        img = ax3.imshow(model, vmin=vmin, vmax=vmax, aspect='equal',
                         interpolation='none', origin='lower', extent=extent)
        ax3.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=ax3.transAxes)
        cbar = plt.colorbar(img)
        if zunit is None:
            cbar.set_label(model_label)
        else:
            cbar.set_label('{} [{}]'.format(model_label, zunit))

        # ignore NaN's and flatten the images for the histograms
        mask = np.isfinite(msm.value) & np.isfinite(model_in)
        signal = dscale * msm.value[mask]
        residual = dscale * (msm.value[mask] - model_in[mask])

        ax4 = plt.subplot(gspec[3, 0])
        ax4.hist(signal, range=[vmin, vmax], bins=15, normed=True,
                 color=line_colors[0])
        ax4.set_xlabel(zlabel)
        ax4.set_ylabel('fraction')
        ax4.grid(which='major', color='0.5', lw=0.5, ls='-')

        ax5 = plt.subplot(gspec[3, 1])
        ax5.hist(residual, range=[rmin, rmax], bins=15, normed=True,
                 color=line_colors[0])
        if zunit is None:
            ax5.set_xlabel('residual')
        else:
            ax5.set_xlabel('residual [{}]'.format(zunit))
        ax5.grid(which='major', color='0.5', lw=0.5, ls='-')

        # save and close figure
        plt.draw()
        self.__fig_info(fig, fig_info)
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_hist(self, msm, msm_err,
                  title=None, fig_info=None):
        """
        Display signal & its errors as histograms

        Parameters
        ----------
        msm         :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        msm_err     :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        title       :  string
           Title of the figure (use attribute "title" of product)
        fig_info    :  dictionary
           Dictionary holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date, signal 
        median & spread adn error meadian & spread.
        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from .biweight import biweight
        from .sron_colorschemes import get_line_colors

        line_colors = get_line_colors()

        if fig_info is None:
            fig_info = OrderedDict({'num_sigma' : 3})
        else:
            fig_info.update({'num_sigma' : 3})

        signal = msm.value[np.isfinite(msm.value)].reshape(-1)
        if 'sign_median' not in fig_info \
            or 'sign_spread' not in fig_info:
            (median, spread) = biweight(signal, spread=True)
            fig_info.update({'sign_median' : median})
            fig_info.update({'sign_spread' : spread})
        signal -= fig_info['sign_median']

        sign_err = msm_err.value[np.isfinite(msm_err.value)].reshape(-1)
        if 'error_median' not in fig_info \
            or 'error_spread' not in fig_info:
            (median, spread) = biweight(sign_err, spread=True)
            fig_info.update({'error_median' : median})
            fig_info.update({'error_spread' : spread})
        sign_err -= fig_info['error_median']

        if msm.units is not None:
            d_label = '{} [{}]'.format(msm.name, msm.units)
        else:
            d_label = msm.name

        if msm_err.units is not None:
            e_label = '{} [{}]'.format(msm_err.name, msm_err.units)
        else:
            e_label = msm_err.name

        fig = plt.figure(figsize=(12,7))
        if title is not None:
            fig.suptitle(title, fontsize=14,
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = gridspec.GridSpec(11,1)

        axx = plt.subplot(gspec[2:6,0])
        axx.hist(signal,
                 range=[-fig_info['num_sigma'] * fig_info['sign_spread'],
                        fig_info['num_sigma'] * fig_info['sign_spread']],
                 bins=15, color=line_colors[0])
        axx.set_title(r'Histogram is centered at the median with range of ' \
                      r'$\pm 3 \sigma$')
        axx.set_xlabel(d_label)
        axx.set_ylabel('count')
        axx.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=axx.transAxes)

        axx = plt.subplot(gspec[7:,0])
        axx.hist(sign_err,
                 range=[-fig_info['num_sigma'] * fig_info['error_spread'],
                        fig_info['num_sigma'] * fig_info['error_spread']],
                 bins=15, color=line_colors[0])
        axx.set_xlabel(e_label)
        axx.set_ylabel('count')
        axx.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=axx.transAxes)

        self.__fig_info(fig, fig_info)
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_geolocation(self, lats, lons, sequence=None,
                         subsatellite=False, title=None, fig_info=None):
        """
        Display footprint of sub-satellite coordinates project on the globe

        Parameters
        ----------
        lats         :  ndarray
           Latitude coordinates
        lons         :  ndarray
           Longitude coordinates
        sequence     :  list
           Indices to footprints to be drawn by polygons
        subsatellite :  boolean
           Coordinates are given for sub-satellite point. Default is False
        title        :  string
           Title of the figure (use attribute "title" of product)
        fig_info    :  dictionary
           Dictionary holding meta-data to be displayed in the figure
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        import shapely.geometry as sgeom

        watercolor = '#ddeeff'
        landcolor  = '#e1c999'
        gridcolor  = '#bbbbbb'
        s5p_color  = '#ee6677'

        class BetterTransverseMercator(ccrs.Projection):
            """
            Implement improved transverse mercator projection

            By Paul Tol (SRON)
            """
            def __init__(self, central_latitude=0.0, central_longitude=0.0,
                         orientation=0, scale_factor=1.0,
                         false_easting=0.0, false_northing=0.0, globe=None):
                centlon = np.round(central_longitude / 15.0) * 15.0 + 0.01
                gam = np.sign(orientation) * 89.99
                proj4_params = [('proj', 'omerc'), ('lat_0', central_latitude),
                                ('lonc', centlon), ('alpha', '0.01'),
                                ('gamma', gam), ('over', ''),
                                ('k_0', scale_factor),
                                ('x_0', false_easting), ('y_0', false_northing),
                                ('units', 'm')]
                super(BetterTransverseMercator, self).__init__(proj4_params, globe=globe)

            @property
            def threshold(self):
                return 1e4

            @property
            def boundary(self):
                xx0, xx1 = self.x_limits
                yy0, yy1 = self.y_limits
                return sgeom.LineString([(xx0, yy0), (xx0, yy1),
                                         (xx1, yy1), (xx1, yy0),
                                         (xx0, yy0)])
            @property
            def x_limits(self):
                return (-2e7, 2e7)

            @property
            def y_limits(self):
                return (-2e7, 2e7)

        # determine central longitude
        sphere_radius = 6370997.0
        parallel_half = 0.883 * sphere_radius
        meridian_half = 2.360 * sphere_radius
        if lons.max() - lons.min() > 180:
            if np.sum(lons > 0) > np.sum(lons < 0):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=-1)

        # inititalize figure
        fig = plt.figure(figsize=(15, 10))
        if title is not None:
            fig.suptitle(title, fontsize=14,
                         position=(0.5, 0.96), horizontalalignment='center')
        # draw worldmap
        axx = plt.axes(projection=BetterTransverseMercator(central_longitude=lon_0,
                                                           orientation=1,
                                                           globe=ccrs.Globe(ellipse='sphere')))
        axx.set_xlim(-meridian_half, meridian_half)
        axx.set_ylim(-parallel_half, parallel_half)
        axx.outline_patch.set_visible(False)
        axx.background_patch.set_facecolor(watercolor)
        axx.add_feature(cfeature.LAND, facecolor=landcolor, edgecolor='none')
        glx = axx.gridlines(linestyle='-', linewidth=0.5, color=gridcolor)
        glx.xlocator = mpl.ticker.FixedLocator(np.linspace(-180, 180, 13))
        glx.ylocator = mpl.ticker.FixedLocator(np.linspace(-90, 90, 13))
        glx.xformatter = LONGITUDE_FORMATTER
        glx.yformatter = LATITUDE_FORMATTER

        # draw footprint
        if not subsatellite:
            if sequence is None:
                lat = np.concatenate((lats[0, :], lats[1:-1, -1],
                                      lats[-1, ::-1], lats[1:-1:-1, 0]))
                lon = np.concatenate((lons[0, :], lons[1:-1, -1],
                                      lons[-1, ::-1], lons[1:-1:-1, 0]))

                poly = Polygon(xy=list(zip(lon, lat)), closed=True,
                               alpha=0.6, facecolor=s5p_color,
                               transform=ccrs.PlateCarree())
                axx.add_patch(poly)
            else:
                for ii in np.unique(sequence):
                    indx = np.unique(np.where(sequence == ii)[0])
                    indx_rev = indx[::-1]
                    lat = np.concatenate((lats[indx[0], :],
                                          lats[indx, -1],
                                          lats[indx[-1], ::-1],
                                          lats[indx_rev, 0]))
                    lon = np.concatenate((lons[indx[0], :],
                                          lons[indx, -1],
                                          lons[indx[-1], ::-1],
                                          lons[indx_rev, 0]))

                    poly = Polygon(xy=list(zip(lon, lat)), closed=True,
                                   alpha=0.6, facecolor=s5p_color,
                                   transform=ccrs.PlateCarree())
                    axx.add_patch(poly)

        # draw sub-satellite coordinates
        if subsatellite:
            axx.scatter(lons, lats, 4, transform=ccrs.PlateCarree(),
                        marker='o', color=s5p_color)

        axx.text(1, 0, r'$\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=axx.transAxes)
        if fig_info is None:
            fig_info = OrderedDict({'lon0': lon_0})

        self.__fig_info(fig, fig_info, aspect=1)
        plt.tight_layout()
        self.__pdf.savefig()
        plt.close()
