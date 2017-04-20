"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class ICMplot contains generic plot functions to display S5p Tropomi data

-- generate figures --
 Public functions a page in the output PDF
 * draw_frame
 * draw_signal
 * draw_quality
 * draw_cmp_swir
 * draw_hist
 * draw_qhist
 * draw_geolocation
 * draw_trend2d
 * draw_trend1d

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from collections import OrderedDict

import numpy as np
import matplotlib as mpl

from .s5p_msm import S5Pmsm

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
def add_copyright(axx):
    axx.text(1, 0, r' $\copyright$ SRON', horizontalalignment='right',
             verticalalignment='bottom', rotation='vertical',
             fontsize='xx-small', transform=axx.transAxes)

#-------------------------
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
        if zunit.find('.s-1') >= 0:
            zunit = zunit.replace('.s-1', ' s$^{-1}$')
    elif units >= 'V':
        max_value = max(abs(vmin), abs(vmax))

        if max_value <= 1e-4:
            dscale = 1e-6
            zunit = units.replace('V', u'\xb5V')
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
    def __init__(self, figname, mode='frame'):
        """
        Initialize multipage PDF document for an SRON SWIR ICM report

        Parameters
        ----------
        figname   :  string
             name of PDF or PNG file (extension required)
        cmap      :  string
             matplotlib color map
        mode      :  string
             data mode - 'frame' or 'dpqm'
        """
        self.__mode = mode
        self.filename = figname

        (_, ext) = os.path.splitext(figname)
        if ext.lower() == '.pdf':
            from matplotlib.backends.backend_pdf import PdfPages

            self.__pdf  = PdfPages(figname)
        else:
            self.__pdf = None

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
                if isinstance(dict_info[key], (float, np.float32)):
                    info_str += "{} : {:.5g}".format(key, dict_info[key])
                else:
                    info_str += "{} : {}".format(key, dict_info[key])
                info_str += '\n'
        info_str += 'created : {}'.format(
            datetime.utcnow().isoformat(timespec='seconds'))

        if aspect == 4:
            fig.text(0.9, 0.975, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif aspect == 3:
            fig.text(0.95, 0.975, info_str,
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
    def draw_frame(self, msm, *, vperc=None, vrange=None,
                   title=None, sub_title=None, fig_info=None):
        """
        Display 2D array data as image

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        vrange     :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc will be ignored
        vperc      :  list
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
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .biweight import biweight
        from .sron_colormaps import sron_cmap

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2

        # determine aspect-ratio of data and set sizes of figure and sub-plots
        dims = msm.value.shape
        aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

        if aspect == 1:
            figsize = (10, 7.5)
        elif aspect == 2:
            figsize = (12, 6)
        elif aspect == 3:
            figsize = (14, 5.5)
        elif aspect == 4:
            figsize = (15, 4.5)
        else:
            print(__name__ + '.draw_signal', dims, aspect)
            raise ValueError('*** FATAL: aspect ratio not implemented, exit')

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

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
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = ax_img.imshow(msm.value / dscale, interpolation='none',
                            vmin=vmin / dscale, vmax=vmax / dscale,
                            aspect='equal', origin='lower',
                            extent=extent,  cmap=sron_cmap('rainbow_PiRd'))
        add_copyright(ax_img)
        ax_img.set_xlabel(xlabel)
        ax_img.set_ylabel(ylabel)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        if zunit is None:
            plt.colorbar(img, cax=cax, label='{}'.format(msm.name))
        else:
            plt.colorbar(img, cax=cax, label=r'{} [{}]'.format(msm.name, zunit))

        # add annotation
        (median, spread) = biweight(msm.value, spread=True)
        if zunit is not None:
            median_str = r'{:.5g} {}'.format(median / dscale, zunit)
            spread_str = r'{:.5g} {}'.format(spread / dscale, zunit)
        else:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)

        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info, aspect)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_signal(self, msm, data_col=None, data_row=None,
                    *, vperc=None, vrange=None,
                    title=None, sub_title=None, fig_info=None):
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
        vrange     :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc will be ignored
        vperc      :  list
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
        from .sron_colormaps import sron_cmap, get_line_colors

        # define colors
        line_colors = get_line_colors()

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2

        # any valid values?
        if np.all(np.isnan(msm.value)):
            return

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
            figsize = (15, 6)
        else:
            print(__name__ + '.draw_signal', dims, aspect)
            raise ValueError('*** FATAL: aspect ratio not implemented, exit')

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

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
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = ax_img.imshow(msm.value / dscale, interpolation='none',
                            vmin=vmin / dscale, vmax=vmax / dscale,
                            aspect='equal', origin='lower',
                            extent=extent,  cmap=sron_cmap('rainbow_PiRd'))
        #xbins = len(ax_img.get_xticklabels())
        ybins = len(ax_img.get_yticklabels())
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_img.get_yticklabels():
            ytl.set_visible(False)
        add_copyright(ax_img)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        if zunit is None:
            plt.colorbar(img, cax=cax, label='{}'.format(msm.name))
        else:
            plt.colorbar(img, cax=cax, label=r'{} [{}]'.format(msm.name, zunit))
        #
        ax_medx = divider.append_axes("bottom", 1.2, pad=0.25)
        ax_medx.plot(xdata, data_row / dscale,
                     lw=0.75, color=line_colors[0])
        xstep = (xdata[-1] - xdata[0]) // (xdata.size - 1)
        ax_medx.set_xlim([xdata[0], xdata[-1] + xstep])
        ax_medx.grid(True)
        ax_medx.set_xlabel(xlabel)
        #ax_medx.locator_params(axis='x', nbins=5)
        #ax_medx.locator_params(axis='y', nbins=4)
        #
        ax_medy = divider.append_axes("left", 1.1, pad=0.25)
        ax_medy.plot(data_col / dscale, ydata,
                     lw=0.75, color=line_colors[0])
        ystep = (ydata[-1] - ydata[0]) // (ydata.size - 1)
        ax_medy.set_ylim([ydata[0], ydata[-1] + ystep])
        ax_medy.locator_params(axis='y', nbins=ybins)
        #print(ystep, [ydata[0], ydata[-1] + ystep])
        ax_medy.grid(True)
        ax_medy.set_ylabel(ylabel)
        #ax_medy.locator_params(axis='x', nbins=5)
        #ydata = np.append(ydata, ydata[-1]+1)
        #print(ydata[::ydata.size//4])
        #ax_medy.set_yticks(ydata[::ydata.size//4])

        # add annotation
        (median, spread) = biweight(msm.value, spread=True)
        if zunit is not None:
            median_str = r'{:.5g} {}'.format(median / dscale, zunit)
            spread_str = r'{:.5g} {}'.format(spread / dscale, zunit)
        else:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)

        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info, aspect)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_quality(self, qmsm, ckd_ref=None,
                     *, low_thres=0.1, high_thres=0.8, qlabels=None,
                     title=None, sub_title=None, fig_info=None):
        """
        Display pixel quality data

        Parameters
        ----------
        qmsm        :  S5Pmsm or ndarray
           Numpy array (2D) holding quality data, range [0, 1] or NaN.
           Zero is for dead pixels and one is for excellent pixels.
        ckd_ref     :  S5Pmsm, optional
           The difference with the CDK is shown, when present.
        low_thres   :  float
           Threshold for usable pixels (with caution), below this threshold
           pixels are considered worst, default=0.1
        high_thres  :  float
           Threshold for good pixels, default=0.8
        qlabels     :  list of strings
           Quality ranking labels, default ['invalid', 'worst', 'bad', 'good']
            - 'invalid': value is negative or NaN
            - 'worst'  : 0 <= value < low_thres
            - 'bad'    : low_thres <= value < high_thres
            - 'good'   : value >= high_thres
        title      :  string, optional
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product, default is None.
        sub_title  :  string, optional
           Sub-title of the figure. Default is None.
           Suggestion: use attribute "long_name" of dataset.
        fig_info   :  dictionary, optional
           OrderedDict holding meta-data to be displayed in the figure.

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date, 'thres_01',
        'qmask_01', 'thres_08' and 'qmask_08'.
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from . import swir_region

        # assert that we have some data to show
        if isinstance(qmsm, np.ndarray):
            qmsm = S5Pmsm(qmsm)
        assert isinstance(qmsm, S5Pmsm)
        assert qmsm is not None and qmsm.value.ndim == 2

        if qlabels is None:
            qlabels = ["invalid", "worst", "bad", "good"]

        # determine aspect-ratio of data and set sizes of figure and sub-plots
        dims = qmsm.value.shape
        aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

        if aspect == 1:
            figsize = (10, 9)
        elif aspect == 2:
            figsize = (12, 7)
        elif aspect == 3:
            figsize = (14, 6.5)
        elif aspect == 4:
            figsize = (15, 6)
        else:
            print(__name__, dims)
            raise ValueError('*** FATAL: aspect ratio not implemented, exit')

        # set label and range of X/Y axis
        (ylabel, xlabel) = qmsm.coords._fields
        ydata = qmsm.coords[0]
        xdata = qmsm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

        # scale data to integers between [0, 10]
        # set pixels outside illuminated area at -1
        thres_min = 10 * low_thres
        thres_max = 10 * high_thres
        qmask = (qmsm.value * 10).astype(np.int8)
        qmask[~np.isfinite(qmsm.value)] = -1
        qmask[~swir_region.mask()] = -1

        if ckd_ref is not None:
            qckd = (ckd_ref.value * 10).astype(np.int8)
            qckd[~np.isfinite(qmsm.value)] = -1
            qckd[~swir_region.mask()] = -1
            mm_10 = (qmask >= 0)
            mm_01 = ((qmask >= 0) & (qmask < thres_min)
                     & (qckd >= thres_min))
            mm_08 = ((qmask >= thres_min) & (qmask < thres_max)
                     & (qckd >= thres_max))
            qmask[mm_10] = 10    # this one first!
            qmask[mm_01] = 0
            qmask[mm_08] = 7

        # define colormap with only 4 colors
        clist = ['#BBBBBB', '#EE6677','#CCBB44','#FFFFFF']
        cmap = mpl.colors.ListedColormap(clist)
        bounds = [-1, 0, thres_min, thres_max, 10]
        mbounds = [(bounds[1] + bounds[0]) / 2,
                   (bounds[2] + bounds[1]) / 2,
                   (bounds[3] + bounds[2]) / 2,
                   (bounds[4] + bounds[3]) / 2]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=figsize)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = ax_img.imshow(qmask, vmin=-1, vmax=10, norm=norm,
                            interpolation='none', origin='lower',
                            aspect='equal', extent=extent, cmap=cmap)
        add_copyright(ax_img)
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_img.get_yticklabels():
            ytl.set_visible(False)

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
        ax_medx.step(xdata, qmask_row_08, lw=0.75, color=clist[2])
        ax_medx.step(xdata, qmask_row_01, lw=0.75, color=clist[1])
        ax_medx.set_xlim([0, dims[1]])
        ax_medx.grid(True)
        ax_medx.locator_params(axis='x', nbins=6)
        ax_medx.locator_params(axis='y', nbins=3)
        ax_medx.set_xlabel(xlabel)
        ax_medx.set_ylabel('count')
        #
        qmask_col_01 = np.sum(((qmask >= 0) & (qmask < thres_min)), axis=1)
        qmask_col_08 = np.sum(((qmask >= 0) & (qmask < thres_max)), axis=1)
        ax_medy = divider.append_axes("left", 1.1, pad=0.25, sharey=ax_img)
        ax_medy.step(qmask_col_08, ydata, lw=0.75, color=clist[2])
        ax_medy.step(qmask_col_01, ydata, lw=0.75, color=clist[1])
        ax_medy.set_ylim([0, dims[0]])
        ax_medy.grid(True)
        ax_medy.locator_params(axis='x', nbins=3)
        ax_medy.locator_params(axis='y', nbins=4)
        ax_medy.set_xlabel('count')
        ax_medy.set_ylabel(ylabel)

        # add annotation
        if fig_info is None:
            fig_info = OrderedDict({'bad  ':
                                    np.sum(((qmask >= 0)
                                            & (qmask < thres_max)))})
        else:
            fig_info.update({'bad  ' :  np.sum(((qmask >= 0)
                                                & (qmask < thres_max)))})
        fig_info.update({'worst' :  np.sum(((qmask >= 0)
                                            & (qmask < thres_min)))})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info, aspect)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_cmp_swir(self, msm, model_in, model_label='reference',
                      *, vrange=None, vperc=None,
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
        vrange     :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc will be ignored
        vperc      :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.]
        model_label :  string
           Name of reference dataset.  Default is 'reference'
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
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        from .sron_colormaps import sron_cmap, get_line_colors

        # refine colors
        line_colors = get_line_colors()

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)

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
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)
        if zunit is None:
            zlabel = '{}'.format(msm.name)
        else:
            zlabel = r'{} [{}]'.format(msm.name, zunit)

        # set label and range of X/Y axis
        #(ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

        # create value, model, residual datasets
        mask = np.isfinite(msm.value)
        value = msm.value.copy()
        value[mask] /= dscale

        mask = np.isfinite(model_in)
        model = model_in.copy()
        model[mask] /= dscale

        mask = np.isfinite(msm.value) & np.isfinite(model_in)
        residual = msm.value.copy()
        residual[~mask] = np.nan
        residual[mask] -= model_in[mask]
        (rmin, rmax) = np.percentile(residual[np.isfinite(residual)], vperc)
        # convert units from electrons to ke, Me, ...
        (runit, rscale) = convert_units(msm.units, rmin, rmax)
        residual[mask] /= rscale

        # inititalize figure
        fig = plt.figure(figsize=(10.8, 10))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = GridSpec(4, 2)

        # create top-pannel with measurements
        ax1 = plt.subplot(gspec[0, :])
        for xtl in ax1.get_xticklabels():
            xtl.set_visible(False)
        if sub_title is not None:
            ax1.set_title(sub_title, fontsize='large')
        img = ax1.imshow(value, vmin=vmin / dscale, vmax=vmax / dscale,
                         aspect='equal', interpolation='none', origin='lower',
                         extent=extent, cmap=sron_cmap('rainbow_PiRd'))
        add_copyright(ax1)
        #ax1.locator_params(axis='y', nbins=7)
        #ax1.yaxis.set_ticks([0,64,128,192,256])
        #ax1.set_ylabel(ylabel)

        cbar = plt.colorbar(img)
        if zlabel is not None:
            cbar.set_label(zlabel)

        # create centre-pannel with residuals
        ax2 = plt.subplot(gspec[1, :], sharex=ax1)
        for xtl in ax2.get_xticklabels():
            xtl.set_visible(False)
        img = ax2.imshow(residual, vmin=rmin / rscale, vmax=rmax / rscale,
                         aspect='equal', interpolation='none', origin='lower',
                         extent=extent, cmap=sron_cmap('diverging_BuRd'))
        add_copyright(ax2)
        #ax2.set_ylabel(ylabel)
        cbar = plt.colorbar(img)
        if runit is None:
            cbar.set_label('residual')
        else:
            cbar.set_label('residual [{}]'.format(runit))

        # create lower-pannel with reference (model, CKD, previous measurement)
        ax3 = plt.subplot(gspec[2, :], sharex=ax1)
        img = ax3.imshow(model, vmin=vmin / dscale, vmax=vmax / dscale,
                         aspect='equal', interpolation='none', origin='lower',
                         extent=extent, cmap=sron_cmap('rainbow_PiRd'))
        add_copyright(ax3)
        #ax3.set_xlabel(xlabel)
        #ax3.set_ylabel(ylabel)
        cbar = plt.colorbar(img)
        if zunit is None:
            cbar.set_label(model_label)
        else:
            cbar.set_label(r'{} [{}]'.format(model_label, zunit))

        # ignore NaN's and flatten the images for the histograms
        ax4 = plt.subplot(gspec[3, 0])
        ax4.hist(value[np.isfinite(value)],
                 range=[vmin / dscale, vmax / dscale], bins=15,
                 normed=True, color=line_colors[0])
        ax4.set_xlabel(zlabel)
        ax4.set_ylabel('fraction')
        ax4.grid(which='major', color='0.5', lw=0.75, ls='-')

        ax5 = plt.subplot(gspec[3, 1])

        ax5.hist(residual[np.isfinite(residual)],
                 range=[rmin / rscale, rmax / rscale], bins=15,
                 normed=True, color=line_colors[1])
        if runit is None:
            ax5.set_xlabel('residual')
        else:
            ax5.set_xlabel('residual [{}]'.format(runit))
        ax5.grid(which='major', color='0.5', lw=0.75, ls='-')

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_hist(self, msm, msm_err, sigma=3,
                  *, title=None, fig_info=None):
        """
        Display signal & its errors as histograms

        Parameters
        ----------
        msm         :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        msm_err     :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date, signal
        median & spread adn error meadian & spread.
        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from .biweight import biweight
        from .sron_colormaps import get_line_colors

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)

        if isinstance(msm_err, np.ndarray):
            msm_err = S5Pmsm(msm_err)
        assert isinstance(msm_err, S5Pmsm)

        line_colors = get_line_colors()
        fig = plt.figure(figsize=(10, 7))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = gridspec.GridSpec(11,1)

        #---------- create first histogram ----------
        values = msm.value[np.isfinite(msm.value)].reshape(-1)
        if values.size > 0:
            if 'val_median' not in fig_info or 'val_spread' not in fig_info:
                (median, spread) = biweight(values, spread=True)
                if fig_info is None:
                    fig_info = OrderedDict({'val_median' : median})
                else:
                    fig_info.update({'val_median' : median})
                fig_info.update({'val_spread' : spread})
            values -= fig_info['val_median']

            # convert units from electrons to ke, Me, ...
            zmin = -sigma * fig_info['val_spread']
            zmax = sigma * fig_info['val_spread']
            (zunit, zscale) = convert_units(msm.units, zmin, zmax)
            if zunit is None:
                d_label = msm.name
            else:
                d_label = r'{} [{}]'.format(msm.name, zunit)

            axx = plt.subplot(gspec[1:5, 0])
            axx.hist(values / zscale, range=[zmin / zscale, zmax / zscale],
                     bins=15, color=line_colors[0])
            axx.set_title(r'histogram centred at median'
                          r' (range $\pm {} \sigma$)'.format(sigma),
                          fontsize='large')
            add_copyright(axx)
            axx.set_xlabel(d_label)
            axx.set_ylabel('count')

        #---------- create second histogram ----------
        uncertainties = msm_err.value[np.isfinite(msm_err.value)].reshape(-1)
        if uncertainties.size > 0:
            if 'unc_median' not in fig_info or 'unc_spread' not in fig_info:
                (median, spread) = biweight(uncertainties, spread=True)
                fig_info.update({'unc_median' : median})
                fig_info.update({'unc_spread' : spread})
            uncertainties -= fig_info['unc_median']

            # convert units from electrons to ke, Me, ...
            umin = -sigma * fig_info['unc_spread']
            umax = sigma * fig_info['unc_spread']
            (uunits, uscale) = convert_units(msm_err.units, umin, umax)
            if msm_err.units is not None:
                u_label = '{} [{}]'.format(msm_err.name, uunits)
            else:
                u_label = msm_err.name

            axx = plt.subplot(gspec[7:-1, 0])
            axx.hist(uncertainties / uscale,
                     range=[umin / uscale, umax / uscale],
                     bins=15, color=line_colors[0])
            add_copyright(axx)
            axx.set_xlabel(u_label)
            axx.set_ylabel('count')

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_qhist(self, msm, msm_dark, msm_noise,
                   *, title=None, fig_info=None):
        """
        Display pixel quality as histograms

        Parameters
        ----------
        msm         :  pys5p.S5Pmsm
           Object holding pixel-quality data and attributes
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date, signal
        median & spread adn error meadian & spread.
        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from . import swir_region
        from .sron_colormaps import get_line_colors

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)

        if isinstance(msm_dark, np.ndarray):
            msm_dark = S5Pmsm(msm_dark)
        assert isinstance(msm_dark, S5Pmsm)

        if isinstance(msm_noise, np.ndarray):
            msm_noise = S5Pmsm(msm_noise)
        assert isinstance(msm_noise, S5Pmsm)

        line_colors = get_line_colors()
        fig = plt.figure(figsize=(10, 9))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = gridspec.GridSpec(15,1)

        axx = plt.subplot(gspec[1:5,0])
        axx.hist(msm.value[swir_region.mask()],
                 bins=11, range=[-.1, 1.], color=line_colors[0])
        add_copyright(axx)
        axx.set_title(r'histogram of {}'.format(msm.long_name),
                      fontsize='medium')
        axx.set_xlim([0, 1])
        axx.set_yscale('log', nonposy='clip')
        #axx.set_xlabel(d_label)
        axx.set_ylabel('count')

        axx = plt.subplot(gspec[6:10,0])
        axx.set_title(r'histogram of {}'.format(msm_dark.long_name),
                      fontsize='medium')
        axx.hist(msm_dark.value[swir_region.mask()],
                 bins=11, range=[-.1, 1.], color=line_colors[0])
        add_copyright(axx)
        axx.set_xlim([0, 1])
        axx.set_yscale('log', nonposy='clip')
        #axx.set_xlabel(u_label)
        axx.set_ylabel('count')

        axx = plt.subplot(gspec[11:,0])
        axx.set_title(r'histogram of {}'.format(msm_noise.long_name),
                      fontsize='medium')
        axx.hist(msm_noise.value[swir_region.mask()],
                 bins=11, range=[-.1, 1.], color=line_colors[0])
        add_copyright(axx)
        axx.set_xlim([0, 1])
        axx.set_yscale('log', nonposy='clip')
        axx.set_xlabel('value')
        axx.set_ylabel('count')

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_geolocation(self, lats, lons,
                         *, sequence=None, subsatellite=False,
                         title=None, fig_info=None):
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
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
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
                super(BetterTransverseMercator, self).__init__(proj4_params,
                                                               globe=globe)

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
            fig.suptitle(title, fontsize='x-large',
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
                print('Unique sequence: {}'.format(np.unique(sequence)))
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

        # draw sub-satellite spot(s)
        else:
            axx.scatter(lons, lats, 4, transform=ccrs.PlateCarree(),
                        marker='o', color=s5p_color)
        add_copyright(axx)
        if fig_info is None:
            fig_info = OrderedDict({'lon0': lon_0})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info, aspect=1)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_trend2d(self, msm_in, data_col=None, data_row=None,
                     *, time_axis=None, vperc=None, vrange=None,
                     title=None, sub_title=None, fig_info=None):
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
        from .sron_colormaps import sron_cmap, get_line_colors

        # define colors
        line_colors = get_line_colors()

        # assert that we have some data to show
        if isinstance(msm_in, np.ndarray):
            msm = S5Pmsm(msm_in)
        else:
            msm = msm_in.copy()
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2

        if time_axis is None:
            (ylabel, xlabel) = msm.coords._fields
            if ylabel == 'orbit' or ylabel == 'time':
                msm.transpose()
        elif time_axis == 0:
            msm.transpose()

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
        #dims = msm.value.shape
        #aspect = min(4, max(1, int(np.round(dims[0] / dims[1]))))
        #print('aspect[{}] {}'.format(dims, aspect))

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        xstep = np.diff(xdata).min()
        extent = [xdata[0]-xstep, xdata[-1], 0, len(ydata)]

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
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)

        # inititalize figure
        fig = plt.figure(figsize=(10, 9))
        ax_img = fig.add_subplot(111)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = plt.imshow(msm.value / dscale, interpolation='none',
                         vmin=vmin / dscale, vmax=vmax / dscale,
                         aspect='auto', origin='lower',
                         extent=extent, cmap=sron_cmap('rainbow_PiRd'))
        add_copyright(ax_img)
        #xbins = len(ax_img.get_xticklabels())
        ybins = len(ax_img.get_yticklabels())
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_img.get_yticklabels():
            ytl.set_visible(False)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        if msm.name == 'row_medians':
            zname = 'row value'
        elif msm.name == 'col_medians':
            zname = 'column value'
        else:
            zname = msm.name
        if zunit is None:
            plt.colorbar(img, cax=cax, label='{}'.format(zname))
        else:
            plt.colorbar(img, cax=cax, label=r'{} [{}]'.format(zname, zunit))
        #
        ax_medx = divider.append_axes("bottom", 1.2, pad=0.25)
        ax_medx.step(np.insert(xdata, 0, xdata[0] - xstep),
                     np.append(data_row / dscale, data_row[-1] / dscale),
                     where='post', lw=0.75, color=line_colors[0])
        ax_medx.set_xlim([xdata[0]-xstep, xdata[-1]])
        ax_medx.grid(True)
        ax_medx.set_xlabel(xlabel)
        #ax_medx.locator_params(axis='x', nbins=5)
        #ax_medx.locator_params(axis='y', nbins=4)
        #
        ax_medy = divider.append_axes("left", 1.1, pad=0.25)
        ax_medy.plot(data_col / dscale, ydata,
                     lw=0.75, color=line_colors[0])
        ystep = (ydata[-1] - ydata[0]) // (ydata.size - 1)
        ax_medy.set_ylim([ydata[0], ydata[-1] + ystep])
        ax_medy.locator_params(axis='y', nbins=ybins)
        #print(ystep, [ydata[0], ydata[-1] + ystep])
        ax_medy.grid(True)
        ax_medy.set_ylabel(ylabel)
        #ax_medy.locator_params(axis='x', nbins=5)
        #ydata = np.append(ydata, ydata[-1]+1)
        #print(ydata[::ydata.size//4])
        #ax_medy.set_yticks(ydata[::ydata.size//4])

        # add annotation
        (median, spread) = biweight(msm.value, spread=True)
        if zunit is not None:
            median_str = r'{:.5g} {}'.format(median / dscale, zunit)
            spread_str = r'{:.5g} {}'.format(spread / dscale, zunit)
        else:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)

        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info, aspect=3)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_trend1d(self, msm, hk_data=None, *, hk_keys=None,
                     title=None, sub_title=None, fig_info=None,
                     placeholder=False):
        """
        Display ...

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm, optional
           Object holding measurement data and its HDF5 attributes
        hk_data   :  pys5p.S5Pmsm, optional
           Object holding housekeeping data and its HDF5 attributes
        hk_keys    : list or tuple
           list of housekeeping parameters to be displayed
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
        from matplotlib import pyplot as plt

        from .sron_colormaps import get_line_colors

        assert msm is not None or hk_data is not None

        if self.__pdf is None:
            plt.rc('font', size=15)

        # define colors
        line_colors = get_line_colors()

        # how many histograms?
        nplots = 0
        if msm is not None:
            nplots += 1

        if hk_data is not None:
            if hk_keys is None:
                hk_keys = ('temp_det4', 'temp_obm_swir_grating')
            nplots += len(hk_keys)
        else:
            hk_keys = ()

        figsize = (10, 3.5 + nplots * 2)
        (fig, axarr) = plt.subplots(nplots, sharex=True, figsize=figsize)
        if nplots == 1:
            axarr = [axarr]
            caption = ''
            if title is not None:
                caption += title

            if sub_title is not None:
                if caption:
                    caption += '\n'
                caption += sub_title

            if caption:
                axarr[0].set_title(caption, fontsize='large')
        else:
            if title is not None:
                fig.suptitle(title, fontsize='x-large',
                             horizontalalignment='center')

            if sub_title is not None:
                axarr[0].set_title(sub_title, fontsize='large')

        i_ax = 0
        if msm is None:
            (xlabel,) = hk_data.coords._fields
            xdata = hk_data.coords[0][:]
            xstep = np.diff(xdata).min()
        elif (msm.value.dtype.names is not None
              and 'dpqf_08' in msm.value.dtype.names):
            (xlabel,) = msm.coords._fields
            xdata  = msm.coords[0][:]
            xstep = np.diff(xdata).min()
            axarr[i_ax].plot(xdata,
                             msm.value['dpqf_08'] - msm.value['dpqf_08'][0],
                             lw=1.5, color=line_colors[3],  # yellow
                             label='bad (quality < 0.8)')
            axarr[i_ax].plot(xdata,
                             msm.value['dpqf_01'] - msm.value['dpqf_01'][0],
                             lw=1.5, color=line_colors[4],  # red
                             label='worst (quality < 0.1)')
            axarr[i_ax].grid(True)
            axarr[i_ax].set_ylabel('{}'.format('count (relative)'))
            axarr[i_ax].legend(loc='upper left', fontsize='smaller')
            i_ax += 1
        else:
            (xlabel,) = msm.coords._fields
            xdata  = msm.coords[0][:]
            xstep = np.diff(xdata).min()

            # convert units from electrons to ke, Me, ...
            if msm.error is None:
                vmin = msm.value.min()
                vmax = msm.value.max()
            else:
                vmin = msm.error[0].min()
                vmax = msm.error[1].max()
            (zunit, dscale) = convert_units(msm.units, vmin, vmax)

            axarr[i_ax].step(np.insert(xdata, 0, xdata[0]-xstep),
                             np.append(msm.value / dscale,
                                       msm.value[-1] / dscale),
                             where='post', lw=1.5, color=line_colors[i_ax])
            if msm.error is not None:
                axarr[i_ax].fill_between(np.insert(xdata, 0, xdata[0]-xstep),
                                         np.append(msm.error[0] / dscale,
                                                   msm.error[0][-1] / dscale),
                                         np.append(msm.error[1] / dscale,
                                                   msm.error[1][-1] / dscale),
                                         step='post', facecolor='#dddddd')
            axarr[i_ax].set_xlim([xdata[0]-xstep, xdata[-1]])
            axarr[i_ax].grid(True)
            if zunit is None:
                axarr[i_ax].set_ylabel('median value')
            else:
                axarr[i_ax].set_ylabel(r'median value [{}]'.format(zunit))
            i_ax += 1

        if i_ax == 1:
            add_copyright(axarr[0])
            if placeholder:
                print('show placeholder')
                axarr[0].text(0.5, 0.5, 'PLACEHOLDER',
                              transform=axarr[0].transAxes, alpha=0.5,
                              fontsize=50, color='gray', rotation=45.,
                              ha='center', va='center')

        for key in hk_keys:
            if key in hk_data.value.dtype.names:
                indx = hk_data.value.dtype.names.index(key)
                if np.mean(hk_data.value[key]) < 150:
                    lcolor = line_colors[0]
                elif np.mean(hk_data.value[key]) < 190:
                    lcolor = line_colors[1]
                elif np.mean(hk_data.value[key]) < 220:
                    lcolor = line_colors[2]
                elif np.mean(hk_data.value[key]) < 250:
                    lcolor = line_colors[3]
                elif np.mean(hk_data.value[key]) < 270:
                    lcolor = line_colors[4]
                else:
                    lcolor = line_colors[5]
                axarr[i_ax].step(np.insert(xdata, 0, xdata[0]-xstep),
                                 np.append(hk_data.value[key],
                                           hk_data.value[key][-1]),
                                 where='post', lw=1.5, color=lcolor,
                                 label=hk_data.long_name[indx].decode('ascii'))
                axarr[i_ax].fill_between(np.insert(xdata, 0, xdata[0]-xstep),
                                         np.append(hk_data.error[key][:, 0],
                                                   hk_data.error[key][-1, 0]),
                                         np.append(hk_data.error[key][:, 1],
                                                   hk_data.error[key][-1, 1]),
                                         step='post', facecolor='#dddddd')
                axarr[i_ax].locator_params(axis='y', nbins=4)
                axarr[i_ax].grid(True)
                axarr[i_ax].set_ylabel('temperature [{}]'.format('K'))
                lg = axarr[i_ax].legend(loc='upper left')
                lg.draw_frame(False)
            i_ax += 1
        axarr[-1].set_xlabel(xlabel)

        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        # save and close figure
        if self.__pdf is None:
            plt.tight_layout()
            plt.savefig(self.filename, bbox_inches='tight', dpi=150)
        else:
            self.__fig_info(fig, fig_info, aspect=3)
            self.__pdf.savefig()
        plt.close()
