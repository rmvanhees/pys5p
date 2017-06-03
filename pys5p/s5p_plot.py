"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class ICMplot contains generic plot functions to display S5p Tropomi data

Suggestion for the name of the report/pdf-file
    <identifier>_<yyyymmdd>_<orbit>.pdf
  where
    identifier : name of L1B/ICM/OCM product or monitoring database
    yyyymmdd   : coverage start-date or start-date of monitoring entry
    orbit      : reference orbit

-- generate figures --
 Public functions a page in the output PDF
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

from .s5p_msm  import S5Pmsm
from .biweight import biweight

#- local functions --------------------------------
# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side
    from a prescribed midpoint value)

    e.g. im = ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,
                                                       vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        xx, yy = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, xx, yy), np.isnan(value))

#-------------------------
def convert_units(units, vmin, vmax):
    """
    Convert units electron or Volt to resp. 'e' and 'V'
    Scale data-range to [-1000, 0] or [0, 1000] based on data-range
    """
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

        if max_value <= 2e-4:
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
    def __init__(self, figname):
        """
        Initialize multipage PDF document for an SRON SWIR ICM report

        Parameters
        ----------
        figname   :  string
             name of PDF or PNG file (extension required)
        """
        self.data = None
        self.aspect = -1
        self.method = None

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
    def add_copyright(axx):
        """
        Display SRON copyright in current figure
        """
        axx.text(1, 0, r' $\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=axx.transAxes)

    def __fig_info(self, fig, dict_info, fontsize='small'):
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

        if self.aspect == 4:
            fig.text(0.9, 0.975, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif self.aspect == 3:
            fig.text(0.95, 0.975, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif self.aspect == 2:
            fig.text(1, 1, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        elif self.aspect == 1:
            fig.text(0.3, 0.225, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})
        else:
            fig.text(0.925, 0.965, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':5})

    #-------------------------
    def __fig_size(self):
        dims = self.data.shape
        self.aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

        if self.aspect == 1:
            figsize = (10, 9)
        elif self.aspect == 2:
            figsize = (12, 7)
        elif self.aspect == 3:
            figsize = (14, 6.5)
        elif self.aspect == 4:
            figsize = (15, 6)
        else:
            print(__name__ + '.draw_signal', dims, self.aspect)
            raise ValueError('*** FATAL: aspect ratio not implemented, exit')

        return figsize

    #-------------------------
    def __data_img(self, msm, ref_img):
        from . import swir_region

        if self.method == 'quality':
            iarr = (msm.value * 10).astype(np.int8)
            iarr[~np.isfinite(msm.value)] = -1
            iarr[~swir_region.mask()] = -1

            scale_dpqm = np.array([0, 1, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10],
                                  dtype=np.int8)
            self.data = scale_dpqm[iarr+1]
            if ref_img is not None:
                new_data = self.data.copy()
                
                iarr = (ref_img * 10).astype(np.int8)
                iarr[~np.isfinite(ref_img)] = -1
                iarr[~swir_region.mask()] = -1
                ref_data = scale_dpqm[iarr+1]
                print(np.sum(ref_data == 0), np.sum(ref_data == 1),
                      np.sum(ref_data == 8), np.sum(ref_data == 10)) 

                # all pixels are initialy unchanged
                self.data = np.zeros_like(iarr)
                # flag new invalid pixels
                self.data[(ref_data > 0)   & (new_data == 0)] = -1
                # flag new good pixels
                self.data[(ref_data != 10) & (new_data == 10)] = 10
                # flag new bad pixels
                self.data[(ref_data == 10) & (new_data == 8)]  = 8
                # new worst pixels
                self.data[(ref_data >= 8)  & (new_data == 1)] = 1
        else:
            self.data = msm.value.copy()
            if self.method == 'diff':
                assert ref_img is not None

                mask = np.isfinite(msm.value) & np.isfinite(ref_img)
                self.data[~mask] = np.nan
                self.data[mask] -= ref_img[mask]
            elif self.method == 'ratio':
                assert ref_img is not None

                mask = (np.isfinite(msm.value) & np.isfinite(ref_img)
                        & (ref_img != 0.))
                self.data[~mask] = np.nan
                self.data[mask] /= ref_img[mask]

    #-------------------------
    def __label_data(self, zunit=None):
        if self.method == 'diff':
            if zunit is None:
                zlabel = 'difference'
            else:
                zlabel = r'difference [{}]'.format(zunit)
        elif self.method == 'ratio':
            zunit  = None
            zlabel = 'ratio'
        else:
            if zunit is None:
                zlabel = 'value'
            else:
                zlabel = r'value [{}]'.format(zunit)

        return zlabel

    # --------------------------------------------------
    def draw_signal(self, msm, ref_data=None, method='data',
                    show_medians=True, *, vperc=None, vrange=None,
                    title=None, sub_title=None, fig_info=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        ref_data  :  ndarray, optional
           Numpy array holding reference data. Required for method equals
            'ratio' where measurement data is divided by the reference
            'diff'  where reference is subtracted from the measurement data
        method    : string
           Method of plot to be generated, default is 'data', optional are
           'ratio', 'diff'
        show_medians :  boolean
           show in side plots row and column medians. Default=True.

        vrange    :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
        vperc     :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.].
           keyword 'vperc' is ignored when vrange is given

        title     :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info  :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data
        median & spread.

        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .sron_colormaps import get_line_colors

        #++++++++++ assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2
        assert not np.all(np.isnan(msm.value))

        if ref_data is not None:
            assert msm.value.shape == ref_data.shape
            assert not np.all(np.isnan(ref_data))

        #++++++++++ set object attributes
        self.method = method

        #++++++++++
        self.__data_img(msm, ref_data)

        # define data-range
        if vrange is None:
            if vperc is None:
                vperc = (1., 99.)
            else:
                assert len(vperc) == 2
            (vmin, vmax) = np.nanpercentile(self.data, vperc)
        else:
            assert len(vrange) == 2
            (vmin, vmax) = vrange

        # convert units from electrons to ke, Me, ...
        (zunit, dscale) = convert_units(msm.units, vmin, vmax)
        vmin /= dscale
        vmax /= dscale
        self.data[np.isfinite(self.data)] /= dscale

        # define colrbar and its normalisation
        lcolor = get_line_colors()
        mid_val = (vmin + vmax) / 2
        if self.method == 'diff':
            cmap = sron_cmap('diverging_BuRd')
            if vmin < 0 and vmax > 0:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = -max(-tmp1, tmp2)
                vmax = max(-tmp1, tmp2)
                mid_val = 0.
        elif self.method == 'ratio':
            cmap = sron_cmap('diverging_BuRd')
            if vmin < 1 and vmax > 1:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = min(tmp1, 1 / tmp2)
                vmax = max(1 / tmp1, tmp2)
                mid_val = 1.
        else:
            cmap = sron_cmap('rainbow_PiRd')

        norm = MidpointNormalize(midpoint=mid_val, vmin=vmin, vmax=vmax)

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]
        zlabel = self.__label_data(zunit)

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=self.__fig_size())
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = ax_img.imshow(self.data, vmin=vmin, vmax=vmax, norm=norm,
                            interpolation='none', origin='lower',
                            aspect='equal', extent=extent, cmap=cmap)
        self.add_copyright(ax_img)
        if show_medians:
            for xtl in ax_img.get_xticklabels():
                xtl.set_visible(False)
            for ytl in ax_img.get_yticklabels():
                ytl.set_visible(False)
        else:
            ax_img.set_xlabel(xlabel)
            ax_img.set_ylabel(ylabel)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        plt.colorbar(img, cax=cax, label=zlabel)
        #
        if show_medians:
            ax_medx = divider.append_axes("bottom", 1.2, pad=0.25,
                                          sharex=ax_img)
            data_row = biweight(self.data, axis=0)
            if xdata.size < 250:
                ax_medx.plot(xdata, data_row, lw=0.75, color=lcolor[0])
            else:
                ax_medx.step(xdata, data_row, lw=0.75, color=lcolor[0])
            ax_medx.set_xlim([0, xdata.size])
            ax_medx.grid(True)
            ax_medx.set_xlabel(xlabel)

            ax_medy = divider.append_axes("left", 1.1, pad=0.25, sharey=ax_img)
            data_col = biweight(self.data, axis=1)
            if ydata.size < 500:
                ax_medy.step(data_col, ydata, lw=0.75, color=lcolor[0])
            else:
                ax_medy.plot(data_col, ydata, lw=0.75, color=lcolor[0])
            ax_medy.set_ylim([0, ydata.size])
            ax_medy.grid(True)
            ax_medy.set_ylabel(ylabel)

        # add annotation
        (median, spread) = biweight(self.data, spread=True)
        if zunit is None:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)
        else:
            median_str = r'{:.5g} {}'.format(median, zunit)
            spread_str = r'{:.5g} {}'.format(spread, zunit)

        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_quality(self, msm, ref_data=None, show_medians=True,
                     *, thres_worst=0.1, thres_bad=0.8,
                     title=None, sub_title=None, fig_info=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        ref_data  :  ndarray, optional
           Numpy array holding reference data, for example pixel quality
           reference map taken from the CKD. Shown are the changes with
           respect to the reference data. Default is None
        show_medians :  boolean
           show in side plots row and column medians. Default=True

        thres_worst  :  float
           Threshold to reject only the worst of the bad pixels, intended
           for CKD derivation. Default=0.1
        thres_bad    :  float
           Threshold for bad pixels. Default=0.8
        title     :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info  :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The quality ranking labels are ['invalid', 'worst', 'bad', 'good'],
        in case nor reference dataset is provided. Where:
        - 'invalid': value is negative or NaN
        - 'worst'  : 0 <= value < low_thres
        - 'bad'    : 0 <= value < high_thres
        - 'good'   : value >= high_thres
        Otherwise the labels for quality ranking indicate which pixels have 
        changed w.r.t. reference. The labels are:
        - 'invalid'   : from any rank to invalid
        - 'unchanged' : no change in rank
        - 'worst'     : from good or bad to worst
        - 'bad'       : from good to bad
        - 'good'      : from any rank to good

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data
        median & spread.

        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .sron_colormaps import get_qfour_colors, get_qfive_colors

        #++++++++++ assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2
        assert not np.all(np.isnan(msm.value))

        if ref_data is not None:
            assert msm.value.shape == ref_data.shape
            assert not np.all(np.isnan(ref_data))

        #++++++++++ set object attributes
        self.method = 'quality'

        # set data-values of central image
        self.__data_img(msm, ref_data)

        # define colors, data-range
        thres_worst = int(10 * thres_worst)
        thres_bad   = int(10 * thres_bad)
        if ref_data is None:
            qlabels = ["invalid", "worst", "bad", "good"]

            lcolor = get_qfour_colors()
            cmap = mpl.colors.ListedColormap(lcolor)
            bounds = [0, thres_worst, thres_bad, 10, 11]
            mbounds = [(bounds[1] + bounds[0]) / 2,
                       (bounds[2] + bounds[1]) / 2,
                       (bounds[3] + bounds[2]) / 2,
                       (bounds[4] + bounds[3]) / 2]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            qlabels = ["invalid", "unchanged", "worst", "bad", "good"]

            lcolor = get_qfive_colors()
            cmap = mpl.colors.ListedColormap(lcolor)
            bounds = [-1, 0, thres_worst, thres_bad, 10, 11]
            mbounds = [(bounds[1] + bounds[0]) / 2,
                       (bounds[2] + bounds[1]) / 2,
                       (bounds[3] + bounds[2]) / 2,
                       (bounds[4] + bounds[3]) / 2,
                       (bounds[5] + bounds[4]) / 2]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        (vmin, vmax) = (bounds[0], bounds[-1])

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

        # inititalize figure
        fig, ax_img = plt.subplots(figsize=self.__fig_size())
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = ax_img.imshow(self.data, vmin=vmin, vmax=vmax, norm=norm,
                            interpolation='none', origin='lower',
                            aspect='equal', extent=extent, cmap=cmap)
        self.add_copyright(ax_img)
        if show_medians:
            for xtl in ax_img.get_xticklabels():
                xtl.set_visible(False)
            for ytl in ax_img.get_yticklabels():
                ytl.set_visible(False)
        else:
            ax_img.set_xlabel(xlabel)
            ax_img.set_ylabel(ylabel)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        if ref_data is None:
            plt.colorbar(img, cax=cax, ticks=mbounds, boundaries=bounds)
        else:
            plt.colorbar(img, cax=cax, ticks=mbounds, boundaries=bounds,
                         label='difference w.r.t. reference')
        cax.set_yticklabels(qlabels)
        #
        if show_medians:
            ax_medx = divider.append_axes("bottom", 1.2, pad=0.25,
                                          sharex=ax_img)
            data_row = np.sum(((self.data == thres_worst)             ## bad
                               | (self.data == thres_bad)), axis=0)
            ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.bad)
            data_row = np.sum((self.data == thres_worst), axis=0)     ## worst
            ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.worst)
            if ref_data is not None:
                data_row = np.sum((self.data == -1), axis=0)          ## invalid
                ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.invalid)
                data_row    = np.sum((self.data == 10), axis=0)       ## good
                ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.good)
            ax_medx.set_xlim([0, xdata.size])
            ax_medx.grid(True)
            ax_medx.set_xlabel(xlabel)

            ax_medy = divider.append_axes("left", 1.1, pad=0.25, sharey=ax_img)
            data_col = np.sum(((self.data == thres_worst)             ## bad
                               | (self.data == thres_bad)), axis=1)
            ax_medy.step(data_col, ydata, lw=0.75, color=lcolor.bad)
            data_col = np.sum((self.data == thres_worst), axis=1)     ## worst
            ax_medy.step(data_col, ydata, lw=0.75, color=lcolor.worst)
            if ref_data is not None:
                data_col = np.sum((self.data == -1), axis=1)          ## invalid
                ax_medy.step(data_col, ydata, lw=0.75, color=lcolor.invalid)
                data_col = np.sum((self.data == 10), axis=1)          ## good
                ax_medy.step(data_col, ydata, lw=0.75, color=lcolor.good)
            ax_medy.set_ylim([0, ydata.size])
            ax_medy.grid(True)
            ax_medy.set_ylabel(ylabel)

        # add annotation
        if fig_info is None:
            fig_info = OrderedDict({'bad (quality < {})'.format(thres_bad / 10) :
                                    np.sum((self.data == thres_worst)
                                           | (self.data == thres_bad))})
        else:
            fig_info.update({'bad (quality < {})'.format(thres_bad / 10) :
                             np.sum((self.data == thres_worst)
                                    | (self.data == thres_bad))})
        fig_info.update({'worst (quality < {})'.format(thres_worst / 10) :
                         np.sum((self.data == thres_worst))})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
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
        self.add_copyright(ax1)
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
        self.add_copyright(ax2)
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
        self.add_copyright(ax3)
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
    def draw_hist(self, msm, msm_err, *, vperc=None, vrange=None,
                  clip=True, title=None, fig_info=None):
        """
        Display signal & its errors as histograms

        Parameters
        ----------
        msm         :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        msm_err     :  pys5p.S5Pmsm
           Object holding measurement data and attributes
        vrange     :  list [vmin, vmax]
           Range to normalize luminance data between vmin and vmax.
           Note that is you pass a vrange instance then vperc will be ignored
        vperc      :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.]
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

        from .sron_colormaps import get_line_colors

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)

        if isinstance(msm_err, np.ndarray):
            msm_err = S5Pmsm(msm_err)
        assert isinstance(msm_err, S5Pmsm)

        # scale data to keep reduce number of significant digits, and to keep
        # the axis-label and tickmarks readable
        if vrange is None:
            if vperc is None:
                vperc = (1., 99.)
            else:
                assert len(vperc) == 2
            (vmin, vmax) = np.percentile(msm.value[np.isfinite(msm.value)],
                                         vperc)
            (umin, umax) = np.percentile(
                msm_err.value[np.isfinite(msm_err.value)], vperc)
        else:
            assert len(vrange) == 2
            (vmin, vmax) = vrange
            indx = (np.isfinite(msm.value)
                    & (msm.value >= vmin) & (msm.value <= vmax))
            (umin, umax) = (msm_err.value[indx].min(),
                            msm_err.value[indx].max())

        line_colors = get_line_colors()
        fig = plt.figure(figsize=(10, 7))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.45, 0.925), horizontalalignment='center')
        gspec = gridspec.GridSpec(11,1)

        #---------- create first histogram ----------
        # convert units from electrons to ke, Me, ...
        (zunit, zscale) = convert_units(msm.units, vmin, vmax)

        # clip (limit) the values
        values = msm.value.reshape(-1)
        values[np.isnan(values)] = vmax
        if clip:
            values = np.clip(values, vmin, vmax)

        if values.size > 0:
            if zunit is None:
                d_label = msm.name
            else:
                d_label = r'{} [{}]'.format(msm.name, zunit)

            # create histogram
            axx = plt.subplot(gspec[1:5, 0])
            axx.hist(values / zscale,
                     range=[np.floor(vmin / zscale), np.ceil(vmax / zscale)],
                     bins=15, color=line_colors[0])
            axx.set_title(r'histogram of {}'.format(msm.long_name),
                          fontsize='large')
            self.add_copyright(axx)
            axx.set_xlim([np.floor(vmin / zscale), np.ceil(vmax / zscale)])
            axx.set_xlabel(d_label)
            axx.set_ylabel('count')

            # add annotation
            (median, spread) = biweight(msm.value, spread=True)
            if zunit is not None:
                median_str = r'{:.5g} {}'.format(median / zscale, zunit)
                spread_str = r'{:.5g} {}'.format(spread / zscale, zunit)
            else:
                median_str = '{:.5g}'.format(median)
                spread_str = '{:.5g}'.format(spread)

            if fig_info is None:
                fig_info = OrderedDict({'val_median' : median_str})
            else:
                fig_info.update({'val_median' : median_str})
            fig_info.update({'val_spread' : spread_str})

        #---------- create second histogram ----------
        # convert units from electrons to ke, Me, ...
        (uunit, uscale) = convert_units(msm_err.units, umin, umax)

        # clip (limit) the uncertainties
        uncertainties = msm_err.value.reshape(-1)
        uncertainties[np.isnan(uncertainties)] = umax
        if clip:
            uncertainties = np.clip(uncertainties, umin, umax)

        if uncertainties.size > 0:
            if uunit is None:
                d_label = msm_err.name
            else:
                d_label = r'{} [{}]'.format(msm_err.name, uunit)

            # create histogram
            axx = plt.subplot(gspec[6:-1, 0])
            axx.hist(uncertainties / uscale,
                     range=[np.floor(umin / uscale), np.ceil(umax / uscale)],
                     bins=15, color=line_colors[0])
            self.add_copyright(axx)
            axx.set_xlim([np.floor(umin / uscale), np.ceil(umax / uscale)])
            axx.set_xlabel(d_label)
            axx.set_ylabel('count')

            # add annotation
            (median, spread) = biweight(msm_err.value, spread=True)
            print(umin, umax, uscale, median, spread)
            if zunit is not None:
                median_str = r'{:.5g} {}'.format(median / uscale, uunit)
                spread_str = r'{:.5g} {}'.format(spread / uscale, uunit)
            else:
                median_str = '{:.5g}'.format(median)
                spread_str = '{:.5g}'.format(spread)

            fig_info.update({'unc_median' : median_str})
            fig_info.update({'unc_spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_qhist(self, msm_total, msm_dark, msm_noise,
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
        median & spread and error meadian & spread.
        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from . import swir_region
        from .sron_colormaps import get_line_colors

        line_colors = get_line_colors()

        # create figure
        fig = plt.figure(figsize=(10, 9))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        gspec = gridspec.GridSpec(15,1)

        # draw histograms
        ipos = 1
        for msm in [msm_total, msm_dark, msm_noise]:
            if isinstance(msm, np.ndarray):
                msm = S5Pmsm(msm)
            assert isinstance(msm, S5Pmsm)

            axx = plt.subplot(gspec[ipos:ipos+4, 0])
            axx.set_title(r'histogram of {}'.format(msm.long_name),
                          fontsize='medium')
            data = msm.value[swir_region.mask()]
            data[np.isnan(data)] = 0.
            axx.hist(data, bins=11, range=[-.1, 1.], color=line_colors[0])
            self.add_copyright(axx)
            axx.set_xlim([0, 1])
            axx.set_yscale('log', nonposy='clip')
            axx.set_ylabel('count')
            ipos += 5

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

        # define aspect for the location of fig_info
        self.aspect = 1

        # define colors
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
        self.add_copyright(axx)
        if fig_info is None:
            fig_info = OrderedDict({'lon0': lon_0})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
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

        from .sron_colormaps import sron_cmap, get_line_colors

        # define aspect for the location of fig_info
        self.aspect = 3

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
        self.add_copyright(ax_img)
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
        ax_medx.set_xlim([extent[0], extent[1]])
        ax_medx.grid(True)
        ax_medx.set_xlabel(xlabel)
        #
        ax_medy = divider.append_axes("left", 1.1, pad=0.25)
        ax_medy.step(data_col / dscale, ydata,
                     lw=0.75, color=line_colors[0])
        ystep = (ydata[-1] - ydata[0]) // (ydata.size - 1)
        ax_medy.set_ylim([ydata[0], ydata[-1] + ystep])
        ax_medy.locator_params(axis='y', nbins=ybins)
        ax_medy.grid(True)
        ax_medy.set_ylabel(ylabel)

        # add annotation
        (median, spread) = biweight(msm.value, spread=True)
        if zunit is not None:
            median_str = r'{:.5g} {}'.format(median / dscale, zunit)
            spread_str = r'{:.5g} {}'.format(spread / dscale, zunit)
        else:
            median_str = '{:.5g}'.format(median)
            spread_str = '{:.5g}'.format(spread)

        if 'orbit' in fig_info:
            fig_info.update({'orbit' : [extent[0], extent[1]]})
        if fig_info is None:
            fig_info = OrderedDict({'median' : median_str})
        else:
            fig_info.update({'median' : median_str})
        fig_info.update({'spread' : spread_str})

        # save and close figure
        if self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
        else:
            self.__fig_info(fig, fig_info)
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

        # define aspect for the location of fig_info
        self.aspect = 3

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
                fig.suptitle(title, fontsize='x-large', position=(0.5, 0.96),
                             horizontalalignment='center')

            if sub_title is not None:
                axarr[0].set_title(sub_title, fontsize='large')

        i_ax = 0
        if msm is None:
            (xlabel,) = hk_data.coords._fields
            xdata = hk_data.coords[0][:]
            xstep = np.diff(xdata).min()
        elif (msm.value.dtype.names is not None
              and 'bad' in msm.value.dtype.names):
            (xlabel,) = msm.coords._fields
            xdata  = msm.coords[0][:]
            xstep = np.diff(xdata).min()
            axarr[i_ax].step(np.insert(xdata, 0, xdata[0]-xstep),
                             np.append(
                                 msm.value['bad'] - msm.value['bad'][0],
                                 msm.value['bad'][-1] - msm.value['bad'][0]),
                             where='post', lw=1.5,
                             color=line_colors[3],  # yellow
                             label='bad (quality < 0.8)')
            axarr[i_ax].step(np.insert(xdata, 0, xdata[0]-xstep),
                             np.append(
                                 msm.value['worst'] - msm.value['worst'][0],
                                 msm.value['worst'][-1]-msm.value['worst'][0]),
                             where='post', lw=1.5,
                             color=line_colors[4],  # red
                             label='worst (quality < 0.1)')
            axarr[i_ax].set_xlim([xdata[0]-xstep, xdata[-1]])
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
                axarr[i_ax].set_ylabel('detector value')
            else:
                axarr[i_ax].set_ylabel(r'detector value [{}]'.format(zunit))
            i_ax += 1

        if i_ax == 1:
            self.add_copyright(axarr[0])
            if placeholder:
                print('*** show placeholder')
                axarr[0].text(0.5, 0.5, 'PLACEHOLDER',
                              transform=axarr[0].transAxes, alpha=0.5,
                              fontsize=50, color='gray', rotation=45.,
                              ha='center', va='center')

        def blank_legend_key():
            """
            Show only text in matplotlib legenda, no key
            """
            from matplotlib.patches import Rectangle

            return Rectangle((0,0), 0, 0, fill=False,
                             edgecolor='none', visible=False)

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
                                 where='post', lw=1.5, color=lcolor)
                axarr[i_ax].fill_between(np.insert(xdata, 0, xdata[0]-xstep),
                                         np.append(hk_data.error[key][:, 0],
                                                   hk_data.error[key][-1, 0]),
                                         np.append(hk_data.error[key][:, 1],
                                                   hk_data.error[key][-1, 1]),
                                         step='post', facecolor='#dddddd')
                axarr[i_ax].locator_params(axis='y', nbins=4)
                axarr[i_ax].grid(True)
                axarr[i_ax].set_ylabel('temperature [{}]'.format('K'))
                legenda = axarr[i_ax].legend(
                    [blank_legend_key()],
                    [hk_data.long_name[indx].decode('ascii')],
                    loc='upper left')
                legenda.draw_frame(False)
            i_ax += 1
        axarr[-1].set_xlabel(xlabel)

        fig.subplots_adjust(hspace=0.02)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        # save and close figure
        if self.__pdf is None:
            plt.tight_layout()
            plt.savefig(self.filename, bbox_inches='tight', dpi=150)
        else:
            self.__fig_info(fig, fig_info)
            self.__pdf.savefig()
        plt.close()
