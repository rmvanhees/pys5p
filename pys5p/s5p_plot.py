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

Copyright (c) 2017--2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import numpy as np

from .biweight import biweight
from .s5p_msm import S5Pmsm


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
    def __init__(self, figname, add_info=True):
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
        self.add_info = add_info

        self.filename = figname
        if Path(figname).suffix.lower() == '.pdf':
            from matplotlib.backends.backend_pdf import PdfPages

            self.__pdf  = PdfPages(figname)
        else:
            self.__pdf = None

    def __repr__(self):
        pass

    def close(self):
        """
        Close multipage PDF document
        """
        from matplotlib import pyplot as plt

        if self.__pdf is not None:
            doc = self.__pdf.infodict()
            doc['Title'] = 'Monitor report on Tropomi SWIR instrument'
            doc['Author'] = '(c) SRON, Netherlands Institute for Space Research'
            doc['Keywords'] = 'PdfPages multipage keywords author title'
            self.__pdf.close()
            plt.close('all')

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
            xpos = 0.9
            ypos = 0.975
        elif self.aspect == 3:
            xpos = 0.95
            ypos = 0.975
        elif self.aspect == 2:
            xpos = 1.
            ypos = 1.
        elif self.aspect == 1:
            xpos = 0.3
            ypos = 0.225
        else:
            xpos = 0.9
            ypos = 0.925

        fig.text(xpos, ypos, info_str,
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
        from . import error_propagation

        if self.method == 'quality':
            scale_dpqm = np.array([0, 1, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10],
                                  dtype=np.int8)

            iarr = (msm.value * 10).astype(np.int8)
            assert (iarr >= 0).all() and (iarr <= 10).all()

            self.data = scale_dpqm[iarr+1]
            if ref_img is not None:
                new_data = self.data.copy()

                iarr = (ref_img * 10).astype(np.int8)
                assert (iarr >= 0).all() and (iarr <= 10).all()
                ref_data = scale_dpqm[iarr+1]

                # all pixels are initialy unchanged
                self.data = np.full_like(iarr, 11)
                # flag new good pixels
                self.data[(ref_data != 10) & (new_data == 10)] = 10
                # flag new bad pixels
                self.data[(ref_data == 10) & (new_data == 8)]  = 8
                # new worst pixels
                self.data[(ref_data != 1)  & (new_data == 1)] = 1
                #
            self.data[~swir_region.mask()] = 0
        elif self.method == 'ratio_unc':
            assert ref_img is not None

            #mask = swir_region.mask() & (ref_img.value != 0.)
            mask = ref_img.value != 0.

            self.data = np.full_like(msm.value, np.nan)
            self.data[mask] = error_propagation.unc_div(
                msm.value[mask], msm.error[mask],
                ref_img.value[mask], ref_img.error[mask])
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

    # --------------------------------------------------
    def draw_signal(self, msm, ref_data=None, method='data',
                    add_medians=True, *, vperc=None, vrange=None,
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
            'diff', 'ratio', 'ratio_unc'
        add_medians :  boolean
           show in side plots row and column (biweight) medians. Default=True.

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
        (biweight) median & spread.

        """
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .sron_colormaps import sron_cmap, get_line_colors

        #++++++++++ assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)
        assert msm.value.ndim == 2
        assert not np.all(np.isnan(msm.value))

        if ref_data is not None:
            if isinstance(ref_data, np.ndarray):
                assert msm.value.shape == ref_data.shape
                assert not np.all(np.isnan(ref_data))
            elif method == 'ratio_unc':
                assert msm.value.shape == ref_data.value.shape
                assert not np.all(np.isnan(ref_data.value))
            else:
                raise TypeError

        #++++++++++ set object attributes
        self.method = method

        # set data to be displayed (based chosen method)
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
        if method == 'diff':
            cmap = sron_cmap('diverging_BuGnRd')
            if vmin < 0 and vmax > 0:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = -max(-tmp1, tmp2)
                vmax = max(-tmp1, tmp2)
                mid_val = 0.
        elif method == 'ratio':
            cmap = sron_cmap('diverging_BuGnRd')
            if vmin < 1 and vmax > 1:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = min(tmp1, 1 / tmp2)
                vmax = max(1 / tmp1, tmp2)
                mid_val = 1.
        elif method == 'data':
            cmap = sron_cmap('rainbow_PiRd')
        elif method == 'ratio_unc':
            cmap = sron_cmap('rainbow_PiRd')
        else:
            raise ValueError

        norm = MidpointNormalize(midpoint=mid_val, vmin=vmin, vmax=vmax)

        # set label and range of X/Y axis
        (ylabel, xlabel) = msm.coords._fields
        ydata = msm.coords[0]
        xdata = msm.coords[1]
        extent = [0, len(xdata), 0, len(ydata)]

        # inititalize figure (and its size & aspect-ratio)
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
        if add_medians:
            for xtl in ax_img.get_xticklabels():
                xtl.set_visible(False)
            for ytl in ax_img.get_yticklabels():
                ytl.set_visible(False)
        else:
            ax_img.set_xlabel(xlabel)
            ax_img.set_ylabel(ylabel)

        # define ticks locations for X & Y valid for most detectors
        if (len(xdata) % 10) == 0:
            minor_locator = MultipleLocator(len(xdata) / 20)
            major_locator = MultipleLocator(len(xdata) / 5)
            ax_img.xaxis.set_major_locator(major_locator)
            ax_img.xaxis.set_minor_locator(minor_locator)
        elif (len(xdata) % 8) == 0:
            minor_locator = MultipleLocator(len(xdata) / 16)
            major_locator = MultipleLocator(len(xdata) / 4)
            ax_img.xaxis.set_major_locator(major_locator)
            ax_img.xaxis.set_minor_locator(minor_locator)

        if (len(ydata) % 10) == 0:
            minor_locator = MultipleLocator(len(ydata) / 20)
            major_locator = MultipleLocator(len(ydata) / 5)
            ax_img.yaxis.set_major_locator(major_locator)
            ax_img.yaxis.set_minor_locator(minor_locator)
        elif (len(ydata) % 8) == 0:
            minor_locator = MultipleLocator(len(ydata) / 16)
            major_locator = MultipleLocator(len(ydata) / 4)
            ax_img.yaxis.set_major_locator(major_locator)
            ax_img.yaxis.set_minor_locator(minor_locator)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        if method == 'diff':
            if zunit is None:
                zlabel = 'difference'
            else:
                zlabel = r'difference [{}]'.format(zunit)
        elif method == 'ratio':
            zunit  = None
            zlabel = 'ratio'
        elif method == 'ratio_unc':
            zunit = None
            zlabel = 'uncertainty'
        elif msm.long_name.find('uncertainty') >= 0:
            if zunit is None:
                zlabel = 'uncertainty'
            else:
                zlabel = r'uncertainty [{}]'.format(zunit)
        else:
            if zunit is None:
                zlabel = 'value'
            else:
                zlabel = r'value [{}]'.format(zunit)
        plt.colorbar(img, cax=cax, label=zlabel)
        #
        if add_medians:
            ax_medx = divider.append_axes("bottom", 1.2, pad=0.25,
                                          sharex=ax_img)
            data_row = biweight(self.data, axis=0)
            if xdata.size > 250:
                ax_medx.plot(xdata, data_row, lw=0.75, color=lcolor[0])
            else:
                ax_medx.step(xdata, data_row, lw=0.75, color=lcolor[0])
            ax_medx.set_xlim([0, xdata.size])
            ax_medx.grid(True)
            ax_medx.set_xlabel(xlabel)

            ax_medy = divider.append_axes("left", 1.1, pad=0.25, sharey=ax_img)
            data_col = biweight(self.data, axis=1)
            if ydata.size > 250:
                ax_medy.plot(data_col, ydata, lw=0.75, color=lcolor[0])
            else:
                ax_medy.step(data_col, ydata, lw=0.75, color=lcolor[0])
            ax_medy.set_ylim([0, ydata.size])
            ax_medy.grid(True)
            ax_medy.set_ylabel(ylabel)

        # add annotation and save figure
        if self.add_info:
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
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

    # --------------------------------------------------
    def draw_quality(self, msm, ref_data=None, add_medians=True,
                     *, thres_worst=0.1, thres_bad=0.8, qlabels=None,
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
        add_medians :  boolean
           show in side plots number of bad and worst pixels. Default=True

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

        The quality ranking labels are ['unusable', 'worst', 'bad', 'good'],
        in case nor reference dataset is provided. Where:
        - 'unusable'  : pixels outside the illuminated region
        - 'worst'     : 0 <= value < thres_worst
        - 'bad'       : 0 <= value < thres_bad
        - 'good'      : thres_bad <= value <= 1
        Otherwise the labels for quality ranking indicate which pixels have
        changed w.r.t. reference. The labels are:
        - 'unusable'  : pixels outside the illuminated region
        - 'worst'     : from good or bad to worst
        - 'bad'       : from good to bad
        - 'good'      : from any rank to good
        - 'unchanged' : no change in rank

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and statistics
        on the number of bad and worst pixels.
        """
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MultipleLocator
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
            if qlabels is None:
                qlabels = ["unusable", "worst", "bad", "good"]
            else:
                assert len(qlabels) == 4, "*** Fatal, requires 4 labels"

            lcolor = get_qfour_colors()
            cmap = mpl.colors.ListedColormap(lcolor)
            bounds = [0, thres_worst, thres_bad, 10, 11]
            mbounds = [(bounds[1] + bounds[0]) / 2,
                       (bounds[2] + bounds[1]) / 2,
                       (bounds[3] + bounds[2]) / 2,
                       (bounds[4] + bounds[3]) / 2]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            if qlabels is None:
                qlabels = ["unusable", "to worst", "good to bad ", "to good",
                           "unchanged"]
            else:
                assert len(qlabels) == 5, "*** Fatal, requires 5 labels"

            lcolor = get_qfive_colors()
            cmap = mpl.colors.ListedColormap(lcolor)
            bounds = [0, thres_worst, thres_bad, 10, 11, 12]
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

        # inititalize figure (and its size & aspect-ratio)
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
        if add_medians:
            for xtl in ax_img.get_xticklabels():
                xtl.set_visible(False)
            for ytl in ax_img.get_yticklabels():
                ytl.set_visible(False)
        else:
            ax_img.set_xlabel(xlabel)
            ax_img.set_ylabel(ylabel)

        # define ticks locations for X & Y valid for most detectors
        if (len(xdata) % 10) == 0:
            minor_locator = MultipleLocator(len(xdata) / 20)
            major_locator = MultipleLocator(len(xdata) / 5)
            ax_img.xaxis.set_major_locator(major_locator)
            ax_img.xaxis.set_minor_locator(minor_locator)
        elif (len(xdata) % 8) == 0:
            minor_locator = MultipleLocator(len(xdata) / 16)
            major_locator = MultipleLocator(len(xdata) / 4)
            ax_img.xaxis.set_major_locator(major_locator)
            ax_img.xaxis.set_minor_locator(minor_locator)

        if (len(ydata) % 10) == 0:
            minor_locator = MultipleLocator(len(ydata) / 20)
            major_locator = MultipleLocator(len(ydata) / 5)
            ax_img.yaxis.set_major_locator(major_locator)
            ax_img.yaxis.set_minor_locator(minor_locator)
        elif (len(ydata) % 8) == 0:
            minor_locator = MultipleLocator(len(ydata) / 16)
            major_locator = MultipleLocator(len(ydata) / 4)
            ax_img.yaxis.set_major_locator(major_locator)
            ax_img.yaxis.set_minor_locator(minor_locator)

        # 'make_axes_locatable' returns an instance of the AxesLocator class,
        # derived from the Locator. It provides append_axes method that creates
        # a new axes on the given side of (“top”, “right”, “bottom” and “left”)
        # of the original axes.
        divider = make_axes_locatable(ax_img)

        # color bar
        cax = divider.append_axes("right", size=0.3, pad=0.05)
        plt.colorbar(img, cax=cax, ticks=mbounds, boundaries=bounds)
        cax.tick_params(axis='y', which='both', length=0)
        cax.set_yticklabels(qlabels)
        #
        if add_medians:
            ax_medx = divider.append_axes("bottom", 1.2, pad=0.25,
                                          sharex=ax_img)
            data_row = np.sum(((self.data == thres_worst)             ## bad
                               | (self.data == thres_bad)), axis=0)
            ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.bad)
            data_row = np.sum((self.data == thres_worst), axis=0)     ## worst
            ax_medx.step(xdata, data_row, lw=0.75, color=lcolor.worst)
            if ref_data is not None:
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
                data_col = np.sum((self.data == 10), axis=1)          ## good
                ax_medy.step(data_col, ydata, lw=0.75, color=lcolor.good)
            ax_medy.set_ylim([0, ydata.size])
            ax_medy.grid(True)
            ax_medy.set_ylabel(ylabel)

        # add annotation and save figure
        if ref_data is None:
            count = np.sum((self.data == thres_worst)
                           | (self.data == thres_bad))
            info_str = '{} (quality < {})'.format(qlabels[2], thres_bad/10)
            if fig_info is None:
                fig_info = OrderedDict({info_str : count})
            else:
                fig_info.update({info_str : count})

            count = np.sum(self.data == thres_worst)
            info_str = '{} (quality < {})'.format(qlabels[1], thres_worst/10)
            fig_info.update({info_str : count})
        else:
            if fig_info is None:
                fig_info = OrderedDict({'to good' : np.sum(self.data == 10)})
            else:
                fig_info.update({'to good' : np.sum(self.data == 10)})
            fig_info.update({'good to bad' : np.sum(self.data == 8)})
            fig_info.update({'to worst' : np.sum(self.data == 1)})

        if self.add_info:
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

    # --------------------------------------------------
    def draw_cmp_swir(self, msm, model_in, model_label='reference',
                      *, vrange=None, vperc=None,
                      add_hist=True, add_residual=True, add_model=True,
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
        from matplotlib.ticker import MultipleLocator
        from matplotlib.gridspec import GridSpec

        from .sron_colormaps import sron_cmap, get_line_colors

        # define aspect for the location of fig_info
        self.aspect = 4

        # define colors
        line_colors = get_line_colors()

        # assert that we have some data to show
        if isinstance(msm, np.ndarray):
            msm = S5Pmsm(msm)
        assert isinstance(msm, S5Pmsm)

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
        if zunit is None:
            zlabel = None ##'{}'.format(msm.name)
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
        if 1 == 1:
            (median, spread) = biweight(residual, spread=True)
            (rmin, rmax) = (median - 3 * spread, median + 3 * spread)
        else:
            (rmin, rmax) = np.percentile(residual[np.isfinite(residual)], vperc)
        # convert units from electrons to ke, Me, ...
        (runit, rscale) = convert_units(msm.units, rmin, rmax)
        residual[mask] /= rscale
        rmin /= dscale
        rmax /= dscale

        # inititalize figure
        nrow = 1
        if add_residual:
            nrow += 1
        if add_model:
            nrow += 1
        if add_hist:
            nrow += 1

        if nrow == 4:
            fig = plt.figure(figsize=(10.8, 10))
            if title is not None:
                fig.suptitle(title, fontsize='x-large',
                             position=(0.5, 0.96), horizontalalignment='center')
            gspec = GridSpec(4, 2)
        elif nrow == 3:
            fig = plt.figure(figsize=(10.8, 7.5))
            if title is not None:
                fig.suptitle(title, fontsize='x-large',
                             position=(0.5, 0.94), horizontalalignment='center')
            gspec = GridSpec(3, 2)
        elif nrow == 2:
            fig = plt.figure(figsize=(16.2, 7.5))
            if title is not None:
                fig.suptitle(title, fontsize='x-large',
                             position=(0.5, 0.94), horizontalalignment='center')
            gspec = GridSpec(2, 2)
        else:
            fig = plt.figure(figsize=(10.8, 3.0))
            if title is not None:
                fig.suptitle(title, fontsize='x-large',
                             position=(0.5, 0.94), horizontalalignment='center')
            gspec = GridSpec(1, 2)

        # create top-panel with measurements
        iplot = 0
        ax1 = plt.subplot(gspec[iplot, :])
        if sub_title is not None:
            ax1.set_title('(a) ' + sub_title)
        img = ax1.imshow(value, vmin=vmin / dscale, vmax=vmax / dscale,
                         aspect='equal', interpolation='none', origin='lower',
                         extent=extent, cmap=sron_cmap('rainbow_PiRd'))
        self.add_copyright(ax1)
        for xtl in ax1.get_xticklabels():
            xtl.set_visible(False)
        ax1.set_ylabel('row')

        # define ticks locations for X & Y valid for most detectors
        if (len(xdata) % 10) == 0:
            minor_locator = MultipleLocator(len(xdata) / 20)
            major_locator = MultipleLocator(len(xdata) / 5)
            ax1.xaxis.set_major_locator(major_locator)
            ax1.xaxis.set_minor_locator(minor_locator)
        elif (len(xdata) % 8) == 0:
            minor_locator = MultipleLocator(len(xdata) / 16)
            major_locator = MultipleLocator(len(xdata) / 4)
            ax1.xaxis.set_major_locator(major_locator)
            ax1.xaxis.set_minor_locator(minor_locator)

        cbar = plt.colorbar(img)
        if zlabel is not None:
            cbar.set_label(zlabel)

        # create centre-panel with residuals
        if add_residual:
            cmap = sron_cmap('diverging_BuGnRd')
            mid_val = (rmin + rmax) / 2
            (rmin_c, rmax_c) = (rmin, rmax)
            if rmin < 0 and rmax > 0:
                rmin_c = -max(-rmin, rmax)
                rmax_c = max(-rmin, rmax)
                mid_val = 0.
            norm = MidpointNormalize(midpoint=mid_val, vmin=rmin_c, vmax=rmax_c)

            iplot += 1
            ax2 = plt.subplot(gspec[iplot, :], sharex=ax1)
            if sub_title is not None:
                ax2.set_title('(b) residual')
            img = ax2.imshow(residual, vmin=rmin_c, vmax=rmax_c,
                             aspect='equal', interpolation='none',
                             origin='lower', extent=extent,
                             cmap=cmap, norm=norm)
            self.add_copyright(ax2)
            if add_model:
                for xtl in ax2.get_xticklabels():
                    xtl.set_visible(False)
            else:
                ax2.set_xlabel('column')
            ax2.set_ylabel('row')

            cbar = plt.colorbar(img)
            if zunit is not None:
                cbar.set_label(zlabel)

        # create lower-panel with reference (model, CKD, previous measurement)
        if add_model:
            iplot += 1
            ax3 = plt.subplot(gspec[iplot, :], sharex=ax1)
            if sub_title is not None:
                if add_residual:
                    ax3.set_title('(c) ' + model_label)
                else:
                    ax3.set_title('(b) ' + model_label)
            img = ax3.imshow(model, vmin=vmin / dscale, vmax=vmax / dscale,
                             aspect='equal', interpolation='none',
                             origin='lower', extent=extent,
                             cmap=sron_cmap('rainbow_PiRd'))
            self.add_copyright(ax3)
            if not add_hist:
                ax3.set_xlabel('column')
            ax3.set_ylabel('row')

            cbar = plt.colorbar(img)
            if runit is not None:
                cbar.set_label('value [{}]'.format(runit))

        if add_hist:
            iplot += 1

            # ignore NaN's and flatten the images for the histograms
            ax4 = plt.subplot(gspec[iplot, 0])
            ax4.hist(value[np.isfinite(value)].reshape(-1),
                     range=[vmin / dscale, vmax / dscale],
                     bins='auto', histtype='stepfilled',
                     density=True, color=line_colors[0])
            if zunit is None:
                ax4.set_xlabel('{}'.format(msm.name))
            else:
                ax4.set_xlabel( r'{} [{}]'.format(msm.name, zunit))
            ax4.set_ylabel('probability density')
            ax4.grid(which='major', color='0.5', lw=0.75, ls='-')

            ax5 = plt.subplot(gspec[iplot, 1])
            ax5.hist(residual[np.isfinite(residual)].reshape(-1),
                     range=[rmin / rscale, rmax / rscale],
                     bins='auto', histtype='stepfilled',
                     density=True, color=line_colors[1])
            if runit is None:
                ax5.set_xlabel('residual')
            else:
                ax5.set_xlabel('residual [{}]'.format(runit))
            ax5.grid(which='major', color='0.5', lw=0.75, ls='-')

        # add annotation and save figure
        if self.add_info:
            (median, spread) = biweight(residual, spread=True)
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
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

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

        # define aspect for the location of fig_info
        self.aspect = 4

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
            #print('vrange: ', umin, umax)

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
            axx.set_title('histograms', fontsize='large')
            res = axx.hist(values / zscale,
                           range=[np.floor(vmin / zscale),
                                  np.ceil(vmax / zscale)],
                           bins='auto', histtype='stepfilled',
                           color=line_colors[0])
            self.add_copyright(axx)
            indx = np.where(res[0] > 0)[0]
            if indx[0] == 0:
                xlim_mn = np.floor(vmin / zscale)
            else:
                xlim_mn = res[1][np.min(indx)]
            if indx[-1] == res[0].size-1:
                xlim_mx = np.ceil(vmax / zscale)
            else:
                xlim_mx = res[1][np.max(indx)+1]
            axx.set_xlim([xlim_mn, xlim_mx])
            axx.set_xlabel(d_label)
            axx.set_ylabel('count')

            # add annotation
            if self.add_info:
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
            res = axx.hist(uncertainties / uscale,
                           range=[np.floor(umin / uscale),
                                  np.ceil(umax / uscale)],
                           bins='auto', histtype='stepfilled',
                           color=line_colors[0])
            self.add_copyright(axx)
            indx = np.where(res[0] > 0)[0]
            if indx[0] == 0:
                xlim_mn = np.floor(umin / uscale)
            else:
                xlim_mn = res[1][np.min(indx)]
            if indx[-1] == res[0].size-1:
                xlim_mx = np.ceil(umax / uscale)
            else:
                xlim_mx = res[1][np.max(indx)+1]
            axx.set_xlim([xlim_mn, xlim_mx])
            axx.set_xlabel(d_label)
            axx.set_ylabel('count')

            # add annotation
            if self.add_info:
                (median, spread) = biweight(msm_err.value, spread=True)
                if zunit is not None:
                    median_str = r'{:.5g} {}'.format(median / uscale, uunit)
                    spread_str = r'{:.5g} {}'.format(spread / uscale, uunit)
                else:
                    median_str = '{:.5g}'.format(median)
                    spread_str = '{:.5g}'.format(spread)

                fig_info.update({'unc_median' : median_str})
                fig_info.update({'unc_spread' : spread_str})

        # add annotation and save figure
        if self.add_info:
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

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

        # define aspect for the location of fig_info
        self.aspect = -1

        # create figure
        line_colors = get_line_colors()
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
            axx.hist(data, bins=11, range=[-.1, 1.], histtype='stepfilled',
                     color=line_colors[0])
            self.add_copyright(axx)
            axx.set_xlim([0, 1])
            axx.set_yscale('log', nonposy='clip')
            axx.set_ylabel('count')
            ipos += 5

        # add annotation and save figure
        if self.add_info:
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

    # --------------------------------------------------
    def draw_trend2d(self, msm_in, *, time_axis=None, vperc=None, vrange=None,
                     title=None, sub_title=None, fig_info=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        msm       :  pys5p.S5Pmsm
           Object holding measurement data and attributes
           - Image data must have 2 dimensions, with regular gridded rows
           - Data coordinates must be increasing and of type integer
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

        # should be replaced by the numpy function (requires numpy v1.15+)
        from math import gcd

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

        # set X/Y label
        if isinstance(msm_in, np.ndarray):
            (ylabel, xlabel) = msm.coords._fields
            if time_axis == 0:
                msm.transpose()
                xlabel = 'time'
            else:
                ylabel = 'time'
        else:
            if time_axis is None:
                (ylabel, xlabel) = msm.coords._fields
                if ylabel == 'orbit' or ylabel == 'time':
                    msm.transpose()
            elif time_axis == 0:
                msm.transpose()
            (ylabel, xlabel) = msm.coords._fields

        # set range of X/Y axis
        # determine xstep as greatest common divisor (numpy v1.15+)
        xdata = msm.coords[1]
        assert np.issubdtype(xdata.dtype, np.integer) \
            and np.all(xdata[1:] > xdata[:-1])
        xsteps = np.unique(np.diff(xdata))
        xstep = xsteps.min()
        dx_mn = xstep
        for xx in xsteps:
            if gcd(dx_mn, xx) < xstep:
                xstep = gcd(dx_mn, xx)

        # determine ystep as greatest common divisor (numpy v1.15+)
        ydata = msm.coords[0]
        assert np.issubdtype(ydata.dtype, np.integer) \
            and np.all(ydata[1:] > ydata[:-1])
        ysteps = np.unique(np.diff(ydata))
        ystep = ysteps.min()
        dy_mn = ystep
        for yy in ysteps:
            if gcd(dy_mn, yy) < ystep:
                ystep = gcd(dy_mn, yy)

        # define extent of image
        extent = [xdata.min(), xdata.max()+xstep,
                  ydata.min(), ydata.max()+ystep]

        # generate data_full
        xdim = (extent[1] - extent[0]) // xstep
        ydim = (extent[3] - extent[2]) // ystep
        data_full = np.zeros((ydim, xdim), dtype=np.float)
        offs = 0
        for xx in range(msm.value.shape[1]):
            if xx < (msm.value.shape[1] - 1):
                ix = (xdata[xx+1] - xdata[xx]) // xstep
            else:
                ix = 1
            data_full[:, offs:offs+ix] = msm.value[:, [xx]]
            offs += ix

        # inititalize figure
        fig = plt.figure(figsize=(10, 9))
        ax_img = fig.add_subplot(111)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # the image plot:
        if sub_title is not None:
            ax_img.set_title(sub_title, fontsize='large')
        img = plt.imshow(data_full / dscale, interpolation='none',
                         vmin=vmin / dscale, vmax=vmax / dscale,
                         aspect='auto', origin='lower',
                         extent=extent, cmap=sron_cmap('rainbow_PiRd'))
        self.add_copyright(ax_img)
        xticks = len(ax_img.get_xticklabels())
        for xtl in ax_img.get_xticklabels():
            xtl.set_visible(False)
        yticks = len(ax_img.get_yticklabels())
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

        # draw lower-panel
        xaxis = np.arange(extent[0], extent[1]+xstep, xstep)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            xvals = np.nanmedian(data_full, axis=0)
        xvals = np.insert(xvals, 0, xvals[0])
        ax_medx = divider.append_axes("bottom", 1.2, pad=0.25)
        ax_medx.step(xaxis, xvals, where='pre',
                     lw=0.75, color=line_colors[0])
        ax_medx.set_xlim([extent[0], extent[1]])
        ax_medx.locator_params(axis='x', nbins=xticks)
        ax_medx.grid(True)
        ax_medx.set_xlabel(xlabel)

        # draw left-panel
        yaxis = np.arange(extent[2], extent[3]+ystep, ystep)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            yvals = np.nanmedian(data_full, axis=1)
        yvals = np.insert(yvals, 0, yvals[0])
        ax_medy = divider.append_axes("left", 1.1, pad=0.25)
        ax_medy.step(yvals, yaxis, where='pre',
                     lw=0.75, color=line_colors[0])
        ax_medy.set_ylim([extent[2], extent[3]])
        ax_medy.locator_params(axis='y', nbins=yticks)
        ax_medy.grid(True)
        ax_medy.set_ylabel(ylabel)

        # add annotation and save figure
        if self.add_info:
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
                if 'orbit' in fig_info:
                    fig_info.update({'orbit' : [extent[0], extent[1]]})
                fig_info.update({'median' : median_str})
            fig_info.update({'spread' : spread_str})
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

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
        # should be replaced by the numpy function (requires numpy v1.15+)
        from math import gcd

        from matplotlib import pyplot as plt

        from .sron_colormaps import get_qfour_colors, get_line_colors

        # we require mesurement data and/or house-keeping data
        assert msm is not None or hk_data is not None

        # ---------- local function ----------
        def blank_legend_key():
            """
            Show only text in matplotlib legenda, no key
            """
            from matplotlib.patches import Rectangle

            return Rectangle((0,0), 0, 0, fill=False,
                             edgecolor='none', visible=False)

        # make sure that we use 'large' fonts in the small plots
        if self.__pdf is None:
            plt.rc('font', size=15)

        # define aspect for the location of fig_info
        self.aspect = 3

        # define colors and number of panels
        lcolors = get_line_colors()

        if msm is None:
            plot_mode = 'house-keeping'
            npanels = 0
        elif (msm.value.dtype.names is not None
              and 'bad' in msm.value.dtype.names):
            plot_mode = 'quality'
            npanels = 2
        else:
            plot_mode = 'data'
            npanels = 1

        if hk_data is not None:
            if hk_keys is None:
                hk_keys = ('temp_det4', 'temp_obm_swir_grating')
            npanels += len(hk_keys)
        else:
            hk_keys = ()

        # initialize matplotlib using 'subplots'
        figsize = (10., (npanels + 1) * 1.8)
        margin = 1. / (1.8 * (npanels + 1))
        (fig, axarr) = plt.subplots(npanels, sharex=True, figsize=figsize)
        if npanels == 1:
            axarr = [axarr]
        fig.subplots_adjust(bottom=margin, top=1-margin, hspace=0.02)

        # draw titles (and put it at the same place)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 1 - margin / 3),
                         horizontalalignment='center')
        if sub_title is not None:
            axarr[0].set_title(sub_title, fontsize='large')

        # define x-axis and its label
        if plot_mode == 'quality' or plot_mode == 'data':
            (xlabel,) = msm.coords._fields
            xdata = msm.coords[0][:].copy()
            use_steps = xdata.size <= 256

            # define data gaps to avoid interpolation over missing data
            assert np.issubdtype(xdata.dtype, np.integer) \
                and np.all(xdata[1:] > xdata[:-1])
            xsteps = np.unique(np.diff(xdata))
            xstep = xsteps.min()
            dx_mn = xstep
            for xx in xsteps:
                if gcd(dx_mn, xx) < xstep:
                    xstep = gcd(dx_mn, xx)

            gap_list = 1 + np.where(np.diff(xdata) > xstep)[0]
            for indx in reversed(gap_list):
                xdata = np.insert(xdata, indx, xdata[indx])
                xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)
                xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)
            if use_steps:
                xdata = np.append(xdata, xdata[-1]+xstep)

        # Implemented 3 options
        # 1) only house-keeping data, no upper-panel with detector data
        # 2) draw pixel-quality data, displayed in the upper-panel
        # 3) draw measurement data, displayed in the upper-panel
        i_ax = 0
        if plot_mode == 'quality':
            qcolors = get_qfour_colors()
            qc_dict = {'bad'   : qcolors.bad,
                       'worst' : qcolors.worst}
            ql_dict = {'bad'   : 'bad (quality < 0.8)',
                       'worst' : 'worst (quality < 0.1)'}

            for key in ['bad', 'worst']:
                ydata = msm.value[key].copy().astype(float)
                for indx in reversed(gap_list):
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, ydata[indx-1])

                if use_steps:
                    ydata = np.append(ydata, ydata[-1])
                    axarr[i_ax].step(xdata, ydata, where='post',
                                     lw=1.5, color=qc_dict[key])
                else:
                    axarr[i_ax].plot(xdata, ydata,
                                     lw=1.5, color=qc_dict[key])

                axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
                axarr[i_ax].grid(True)
                axarr[i_ax].set_ylabel('{}'.format('count'))
                legenda = axarr[i_ax].legend(
                    [blank_legend_key()], [ql_dict[key]], loc='upper left')
                legenda.draw_frame(False)
                i_ax += 1
        elif plot_mode == 'data':
            # convert units from electrons to ke, Me, ...
            if msm.error is None:
                vmin = msm.value.min()
                vmax = msm.value.max()
            else:
                vmin = msm.error[0].min()
                vmax = msm.error[1].max()
            (zunit, dscale) = convert_units(msm.units, vmin, vmax)

            ydata = msm.value.copy() / dscale
            for indx in reversed(gap_list):
                ydata = np.insert(ydata, indx, np.nan)
                ydata = np.insert(ydata, indx, np.nan)
                ydata = np.insert(ydata, indx, ydata[indx-1])

            if use_steps:
                ydata = np.append(ydata, ydata[-1])
                axarr[i_ax].step(xdata, ydata, where='post',
                                 lw=1.5, color=lcolors.blue)
            else:
                axarr[i_ax].plot(xdata, ydata,
                                 lw=1.5, color=lcolors.blue)

            if msm.error is not None:
                yerr1 = msm.error[0].copy() / dscale
                yerr2 = msm.error[1].copy() / dscale
                for indx in reversed(gap_list):
                    yerr1 = np.insert(yerr1, indx, np.nan)
                    yerr2 = np.insert(yerr2, indx, np.nan)
                    yerr1 = np.insert(yerr1, indx, np.nan)
                    yerr2 = np.insert(yerr2, indx, np.nan)
                    yerr1 = np.insert(yerr1, indx, yerr1[indx-1])
                    yerr2 = np.insert(yerr2, indx, yerr2[indx-1])

                if use_steps:
                    yerr1 = np.append(yerr1, yerr1[-1])
                    yerr2 = np.append(yerr2, yerr2[-1])
                    axarr[i_ax].fill_between(xdata, yerr1, yerr2,
                                             step='post', facecolor='#BBCCEE')
                else:
                    axarr[i_ax].fill_between(xdata, yerr1, yerr2,
                                             facecolor='#BBCCEE')

            axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
            axarr[i_ax].grid(True)
            if zunit is None:
                axarr[i_ax].set_ylabel('detector value')
            else:
                axarr[i_ax].set_ylabel(r'detector value [{}]'.format(zunit))
            i_ax += 1

        # add figures with house-keeping data
        (xlabel,) = hk_data.coords._fields
        xdata = hk_data.coords[0][:].copy()
        use_steps = xdata.size <= 256

        # define data gaps to avoid interpolation over missing data
        assert np.issubdtype(xdata.dtype, np.integer),\
            "xdata must be of integer type"
        assert np.all(xdata[1:] > xdata[:-1]),\
            "xdata not increasing at {}".format(
                np.where(xdata[1:] <= xdata[:-1]))
        xsteps = np.unique(np.diff(xdata))
        xstep = xsteps.min()
        dx_mn = xstep
        for xx in xsteps:
            if gcd(dx_mn, xx) < xstep:
                xstep = gcd(dx_mn, xx)

        gap_list = 1 + np.where(np.diff(xdata) > xstep)[0]
        for indx in reversed(gap_list):
            xdata = np.insert(xdata, indx, xdata[indx])
            xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)
            xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)
        if use_steps:
            xdata = np.append(xdata, xdata[-1]+xstep)

        if xlabel == 'time':
            xdata = xdata.astype(np.float) / 3600
            xlabel += ' [hours]'

        for key in hk_keys:
            if key in hk_data.value.dtype.names:
                indx = hk_data.value.dtype.names.index(key)
                hk_unit = hk_data.units[indx].decode('ascii')
                full_string = hk_data.long_name[indx].decode('ascii')
                if hk_unit == 'K':
                    hk_name = full_string.rsplit(' ', 1)[0]
                    hk_label = 'temperature [{}]'.format(hk_unit)
                    lcolor = lcolors.blue
                    fcolor = '#BBCCEE'
                elif hk_unit == 'A' or hk_unit == 'mA':
                    hk_name = full_string.rsplit(' ', 1)[0]
                    hk_label = 'current [{}]'.format(hk_unit)
                    lcolor = lcolors.green
                    fcolor = '#CCDDAA'
                elif hk_unit == '%':
                    hk_name = full_string.rsplit(' ', 2)[0]
                    hk_label = 'duty cycle [{}]'.format(hk_unit)
                    lcolor = lcolors.red
                    fcolor = '#FFCCCC'
                else:
                    hk_name = full_string
                    hk_label = 'value [{}]'.format(hk_unit)
                    lcolor = lcolors.pink
                    fcolor = '#EEBBDD'

                ydata = hk_data.value[key].copy()
                yerr1 = hk_data.error[key][:, 0].copy()
                yerr2 = hk_data.error[key][:, 1].copy()
                for indx in reversed(gap_list):
                    ydata = np.insert(ydata, indx, np.nan)
                    yerr1 = np.insert(yerr1, indx, np.nan)
                    yerr2 = np.insert(yerr2, indx, np.nan)
                    ydata = np.insert(ydata, indx, np.nan)
                    yerr1 = np.insert(yerr1, indx, np.nan)
                    yerr2 = np.insert(yerr2, indx, np.nan)
                    ydata = np.insert(ydata, indx, ydata[indx-1])
                    yerr1 = np.insert(yerr1, indx, yerr1[indx-1])
                    yerr2 = np.insert(yerr2, indx, yerr2[indx-1])

                if np.all(np.isnan(ydata)):
                    ydata[:] = 0
                    yerr1[:] = 0
                    yerr2[:] = 0

                if use_steps:
                    ydata = np.append(ydata, ydata[-1])
                    yerr1 = np.append(yerr1, yerr1[-1])
                    yerr2 = np.append(yerr2, yerr2[-1])
                    axarr[i_ax].step(xdata, ydata,
                                     where='post', lw=1.5, color=lcolor)
                else:
                    axarr[i_ax].plot(xdata, ydata,
                                     lw=1.5, color=lcolor)

                if not (np.array_equal(ydata, yerr1)
                        and np.array_equal(ydata, yerr2)):
                    axarr[i_ax].fill_between(xdata, yerr1, yerr2,
                                             step='post', facecolor=fcolor)
                axarr[i_ax].locator_params(axis='y', nbins=4)
                axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
                axarr[i_ax].grid(True)
                axarr[i_ax].set_ylabel(hk_label)
                legenda = axarr[i_ax].legend(
                    [blank_legend_key()], [hk_name], loc='upper left')
                legenda.draw_frame(False)
            i_ax += 1
        axarr[-1].set_xlabel(xlabel)
        if npanels > 1:
            plt.setp([a.get_xticklabels()
                      for a in fig.axes[:-1]], visible=False)

        self.add_copyright(axarr[-1])
        if placeholder:
            print('*** show placeholder')
            axarr[0].text(0.5, 0.5, 'PLACEHOLDER',
                          transform=axarr[0].transAxes, alpha=0.5,
                          fontsize=50, color='gray', rotation=45.,
                          ha='center', va='center')

        # add annotation and save figure
        if self.add_info:
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')

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
        self.aspect = -1

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

        if self.add_info:
            if fig_info is None:
                fig_info = OrderedDict({'lon0': lon_0})
            self.__fig_info(fig, fig_info)

            if self.__pdf is None:
                plt.savefig(self.filename)
                plt.close(fig)
            else:
                self.__pdf.savefig()
        elif self.__pdf is None:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close(fig)
        else:
            self.__pdf.savefig(bbox_inches='tight')
