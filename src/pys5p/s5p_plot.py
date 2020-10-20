"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class S5Pplot contains generic plot functions

Generate Figures
----------------
- Creating an S5Pplot object will open multi-page PDF file or single-page PNG
- Each public function listed below can be used to create a (new) page
 * draw_signal
 * draw_quality
 * draw_cmp_swir
 * draw_hist
 * draw_qhist
 * draw_trend2d
 * draw_trend1d
 * draw_lines
 * draw_errorbar
 * draw_tracks
- Closing the S5Pplot object will write the report to disk

New: you can change the default color_maps by calling set_cmap(cmap)
     and return to the default behavior by calling unset_cmap()
     Currently, only implemented for draw_signal(...)

Suggestion for the name of a report/PDF-file:
    <identifier>_<yyyymmdd>_<orbit>.pdf
  where
    identifier : name of L1B/ICM/OCM product or monitoring database
    yyyymmdd   : coverage start-date or start-date of monitoring entry
    orbit      : reference orbit

Examples
--------
>>> from pys5p.s5p_plot import S5Pplot
>>> plot = S5Pplot('test_plot_class.pdf')
>>> plot.draw_signal(np.mean(signal, axis=0), title='signal of my measurement')
>>> plot.draw_trend2d(np.mean(signal, axis=2))
>>> plot.draw_trend1d(np.mean(signal, axis=(1, 2)), hk_data, hk_keys)
>>> plot.close()

Copyright (c) 2017-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
# pylint: disable=too-many-lines
from datetime import datetime
from pathlib import PurePath

try:
    from cartopy import crs as ccrs
except ModuleNotFoundError:
    FOUND_CARTOPY = False
else:
    FOUND_CARTOPY = True
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from . import error_propagation
from . import swir_region
from .biweight import biweight
from .ckd_io import CKDio
from .lib import plotlib
from .s5p_msm import S5Pmsm
from .tol_colors import tol_cmap, tol_cset


# - main function ----------------------------------
class S5Pplot():
    """
    Generate figure(s) for the SRON Tropomi SWIR monitor website or MPC reports

    The PDF will have the following name:
        <dbname>_<startDateTime of monitor entry>_<orbit of monitor entry>.pdf
    """
    def __init__(self, figname, pdf_title=None):
        """
        Initialize multi-page PDF document or a single-page PNG

        Parameters
        ----------
        figname   :  string
             Name of PDF or PNG file (extension required)
        pdf_title :  string
             Title of the PDF document (attribute of the PDF document)
             Default: 'Monitor report on Tropomi SWIR instrument'
        """
        self.filename = figname
        if PurePath(figname).suffix.lower() == '.pdf':
            self.__pdf = PdfPages(figname)
            doc = self.__pdf.infodict()
            if pdf_title is None:
                doc['Title'] = 'Monitor report on Tropomi SWIR instrument'
            else:
                doc['Title'] = pdf_title
            doc['Author'] = '(c) SRON Netherlands Institute for Space Research'
        else:
            self.__pdf = None

        self.__aspect = None
        self.__cmap = None
        self.__divider = None
        self.__zunit = None

        self.__mpl = None           # only used by draw_lines

    def __repr__(self):
        pass

    def __close_this_page(self, fig):
        """
        close current matplotlib figure or page in a PDF document
        """
        # add annotation and save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            # plt.savefig(self.filename, transparant=True)
            plt.close(fig)
        else:
            self.__pdf.savefig()
            # self.__pdf.savefig(transparant=True)
        self.__aspect = None
        self.__divider = None

    def close(self):
        """
        Close multipage PDF document
        """
        if self.__pdf is None:
            return

        self.__pdf.close()
        plt.close('all')

    # --------------------------------------------------
    @staticmethod
    def __add_copyright(axx):
        """
        Display SRON copyright in current figure
        """
        axx.text(1, 0, r' $\copyright$ SRON', horizontalalignment='right',
                 verticalalignment='bottom', rotation='vertical',
                 fontsize='xx-small', transform=axx.transAxes)

    def __set_aspect(self, dims):
        """
        set image aspect ratio
        """
        self.__aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

    def set_cmap(self, cmap):
        """
        Define alternative color-map to overrule the default

        Parameter
        ---------
         cmap :  matplotlib color-map
        """
        self.__cmap = cmap

    def unset_cmap(self):
        """
        Unset user supplied color-map, and use default color-map
        """
        self.__cmap = None

    @property
    def cmap(self):
        """
        Returns matplotlib colormap
        """
        return self.__cmap

    def set_zunit(self, units):
        """
        Provide units of data to be displayed
        """
        self.__zunit = units

    def unset_zunit(self):
        """
        Unset user supplied unit definition of data
        """
        self.__zunit = None

    @property
    def zunit(self):
        """
        Returns value of zunit
        """
        return self.__zunit

    # -------------------------
    def convert_zunit(self, vmin, vmax):
        """
        Convert units electron or Volt to resp. 'e' and 'V'
        Scale data-range to [-1000, 0] or [0, 1000] based on data-range

        Returns
        -------
        float: dscale
        """
        if self.zunit is None or self.zunit == '1':
            return 1.

        max_value = max(abs(vmin), abs(vmax))
        if self.zunit.find('electron') >= 0:
            zunit = self.zunit
            if zunit.find('.s-1') >= 0:
                zunit = zunit.replace('.s-1', ' s$^{-1}$')

            if max_value > 1000000000:
                dscale = 1e9
                zunit = zunit.replace('electron', 'Ge')
            elif max_value > 1000000:
                dscale = 1e6
                zunit = zunit.replace('electron', 'Me')
            elif max_value > 1000:
                dscale = 1e3
                zunit = zunit.replace('electron', 'ke')
            else:
                dscale = 1.
                zunit = zunit.replace('electron', 'e')

            self.set_zunit(zunit)
            return dscale

        if self.zunit[0] == 'V':
            zunit = self.zunit
            if max_value <= 2e-4:
                dscale = 1e-6
                zunit = zunit.replace('V', u'\xb5V')
            elif max_value <= 0.1:
                dscale = 1e-3
                zunit = zunit.replace('V', 'mV')
            else:
                dscale = 1.
                zunit = 'V'

            self.set_zunit(zunit)
            return dscale

        return 1.

    def __get_zlabel(self, method):
        """
        Return label of colorbar
        """
        if method == 'ratio':
            zlabel = 'ratio'
        elif method == 'ratio_unc':
            zlabel = 'uncertainty'
        elif method == 'diff':
            if self.zunit is None:
                zlabel = 'difference'
            else:
                zlabel = r'difference [{}]'.format(self.zunit)
        elif method == 'error':
            if self.zunit is None:
                zlabel = 'uncertainty'
            else:
                zlabel = r'uncertainty [{}]'.format(self.zunit)
        else:
            if self.zunit is None:
                zlabel = 'value'
            else:
                zlabel = r'value [{}]'.format(self.zunit)

        return zlabel

    def set_fig_data2d(self, method: str, data_in, ref_data=None):
        """
        Determine image data to be displayed
        """
        # check image data
        try:
            plotlib.check_data2d(method, data_in)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        # check reference data
        if ref_data is not None:
            try:
                plotlib.check_data2d(method, ref_data)
            except Exception as exc:
                raise RuntimeError('invalid reference-data provided') from exc

        img_data = None
        if isinstance(data_in, np.ndarray):
            img_data = data_in.copy()
        else:
            if method == 'error':
                img_data = data_in.error.copy()
            else:
                img_data = data_in.value.copy()

            if method == 'ratio_unc':
                self.unset_zunit()
                mask = ref_data.value != 0.
                img_data[~mask] = np.nan
                img_data[mask] = error_propagation.unc_div(
                    data_in.value[mask], data_in.error[mask],
                    ref_data.value[mask], ref_data.error[mask])
            elif method == 'diff':
                self.set_zunit(data_in.units)
                mask = np.isfinite(data_in.value) & np.isfinite(ref_data)
                img_data[~mask] = np.nan
                img_data[mask] -= ref_data[mask]
            elif method == 'ratio':
                self.unset_zunit()
                mask = (np.isfinite(data_in.value)
                        & np.isfinite(ref_data) & (ref_data != 0.))
                img_data[~mask] = np.nan
                img_data[mask] /= ref_data[mask]
            else:
                self.set_zunit(data_in.units)

        # set aspect ratio of image data
        self.__set_aspect(img_data.shape)

        return img_data

    def scale_data2d(self, method: str, img_data, vperc, vrange):
        """
        Determine range of image data and normalize colormap accordingly
        """
        cmap = tol_cmap('rainbow_PuRd')
        if method == 'diff':
            cmap = tol_cmap('sunset')
        elif method == 'ratio':
            cmap = tol_cmap('sunset')

        # set colormap
        if self.cmap is None:
            self.set_cmap(cmap)

        # define data-range
        if vrange is None:
            (vmin, vmax) = np.nanpercentile(img_data, vperc)
        else:
            (vmin, vmax) = vrange

        # convert units from electrons to ke, Me, ...
        dscale = self.convert_zunit(vmin, vmax)
        if not issubclass(img_data.dtype.type, np.integer):
            vmin /= dscale
            vmax /= dscale
            img_data[np.isfinite(img_data)] /= dscale

        mid_val = (vmin + vmax) / 2
        if method == 'diff':
            if vmin < 0 < vmax:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = -max(-tmp1, tmp2)
                vmax = max(-tmp1, tmp2)
                mid_val = 0.
        if method == 'ratio':
            if vmin < 1 < vmax:
                (tmp1, tmp2) = (vmin, vmax)
                vmin = min(tmp1, 1 / tmp2)
                vmax = max(1 / tmp1, tmp2)
                mid_val = 1.

        return plotlib.MidpointNormalize(midpoint=mid_val,
                                         vmin=vmin, vmax=vmax)

    def set_fig_quality(self, qthres, data, ref_data=None):
        """
        Check pixel-quality data and convert to quality classes

        Quality classes without reference data:
         [good]:   value=4
         [bad]:    value=2
         [worst]:  value=1
         [unused]: value=0

        Quality classes with reference data:
         [unchanged]:   value=8
         [to_good]:     value=4
         [good_to_bad]: value=2
         [to_worst]:    value=1
         [unused]:      value=0
        Note: ignored are [worst_to_bad]
        """
        def float_to_quality(qthres, arr):
            """
            Convert float value [0, 1] to quality classes
            """
            res = np.empty(arr.shape, dtype='i1')
            res[arr >= qthres['bad']] = 4
            res[(arr > qthres['worst']) & (arr < qthres['bad'])] = 2
            res[arr <= qthres['worst']] = 1
            res[~swir_region.mask()] = 0
            return res

        # check image data
        try:
            plotlib.check_data2d('quality', data)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        if isinstance(data, np.ndarray):
            qval = float_to_quality(qthres, data)
        else:
            qval = float_to_quality(qthres, data.value)

        # set aspect ratio of image data
        self.__set_aspect(qval.shape)

        if ref_data is None:
            return qval

        # check reference data
        try:
            plotlib.check_data2d('quality', ref_data)
        except Exception as exc:
            raise RuntimeError('invalid reference-data provided') from exc

        # return difference with reference
        qdiff = float_to_quality(qthres, ref_data) - qval
        qval = np.full_like(qdiff, 8)
        qval[(qdiff == -2) | (qdiff == -3)] = 4
        qval[qdiff == 2] = 2
        qval[(qdiff == 1) | (qdiff == 3)] = 1
        qval[~swir_region.mask()] = 0
        return qval

    @staticmethod
    def adjust_tickmarks(ax_fig, coords):
        """
        Define ticks locations for X & Y valid for most detectors
        """
        sz_xcoord = len(coords['X']['data'])
        sz_ycoord = len(coords['Y']['data'])
        if (sz_xcoord % 10) == 0:
            minor_locator = MultipleLocator(sz_xcoord / 20)
            major_locator = MultipleLocator(sz_xcoord / 5)
            ax_fig.xaxis.set_major_locator(major_locator)
            ax_fig.xaxis.set_minor_locator(minor_locator)
        elif (sz_xcoord % 8) == 0:
            minor_locator = MultipleLocator(sz_xcoord / 16)
            major_locator = MultipleLocator(sz_xcoord / 4)
            ax_fig.xaxis.set_major_locator(major_locator)
            ax_fig.xaxis.set_minor_locator(minor_locator)

        if (sz_ycoord % 10) == 0:
            minor_locator = MultipleLocator(sz_ycoord / 20)
            major_locator = MultipleLocator(sz_ycoord / 5)
            ax_fig.yaxis.set_major_locator(major_locator)
            ax_fig.yaxis.set_minor_locator(minor_locator)
        elif (sz_ycoord % 8) == 0:
            minor_locator = MultipleLocator(sz_ycoord / 16)
            major_locator = MultipleLocator(sz_ycoord / 4)
            ax_fig.yaxis.set_major_locator(major_locator)
            ax_fig.yaxis.set_minor_locator(minor_locator)

    def __add_data1d(self, plot_mode, axarr, msm_1, msm_2):
        """
        Implemented 3 options
         1) only house-keeping data, no upper-panel with detector data
         2) draw pixel-quality data, displayed in the upper-panel
         3) draw measurement data, displayed in the upper-panel
        """
        # define colors
        cset = tol_cset('bright')

        i_ax = 0
        use_steps = msm_1.value.size <= 256
        if plot_mode == 'quality':
            (xdata, gap_list) = plotlib.get_xdata(msm_1.coords[0][:].copy(),
                                                  use_steps)

            qc_dict = {'bad': cset.yellow,
                       'worst': cset.red}
            ql_dict = {'bad': 'bad (quality < 0.8)',
                       'worst': 'worst (quality < 0.1)'}

            for key in ['bad', 'worst']:
                ydata = msm_1.value[key].copy().astype(float)
                for indx in reversed(gap_list):
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, ydata[indx-1])

                if use_steps:
                    ydata = np.append(ydata, ydata[-1])
                    axarr[i_ax].step(xdata, ydata, where='post',
                                     linewidth=1.5, color=qc_dict[key])
                else:
                    axarr[i_ax].plot(xdata, ydata,
                                     linewidth=1.5, color=qc_dict[key])

                axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
                axarr[i_ax].grid(True)
                axarr[i_ax].set_ylabel('{}'.format('count'))
                legenda = axarr[i_ax].legend([plotlib.blank_legend_key()],
                                             [ql_dict[key]], loc='upper left')
                legenda.draw_frame(False)
                i_ax += 1

            return i_ax

        if plot_mode == 'data':
            for msm in (msm_1, msm_2):
                if msm is None:
                    continue

                # convert units from electrons to ke, Me, ...
                if msm.error is None:
                    vmin = msm.value.min()
                    vmax = msm.value.max()
                else:
                    vmin = msm.error[0].min()
                    vmax = msm.error[1].max()
                self.set_zunit(msm.units)
                dscale = self.convert_zunit(vmin, vmax)

                (xdata, gap_list) = plotlib.get_xdata(msm.coords[0][:].copy(),
                                                      use_steps)
                ydata = msm.value.copy() / dscale
                for indx in reversed(gap_list):
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, np.nan)
                    ydata = np.insert(ydata, indx, ydata[indx-1])

                if use_steps:
                    ydata = np.append(ydata, ydata[-1])
                    axarr[i_ax].step(xdata, ydata, where='post',
                                     linewidth=1.5, color=cset.blue)
                else:
                    axarr[i_ax].plot(xdata, ydata,
                                     linewidth=1.5, color=cset.blue)

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
                                                 step='post',
                                                 facecolor='#BBCCEE')
                    else:
                        axarr[i_ax].fill_between(xdata, yerr1, yerr2,
                                                 facecolor='#BBCCEE')

                axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
                axarr[i_ax].grid(True)
                if self.zunit is None:
                    axarr[i_ax].set_ylabel(msm.long_name)
                else:
                    axarr[i_ax].set_ylabel(r'{} [{}]'.format(
                        msm.long_name, self.zunit))
                i_ax += 1
            return i_ax

        return i_ax

    @staticmethod
    def __add_hkdata(i_ax, axarr, hk_data, hk_keys):
        """
        Add house-keeping information for method draw_trend1d
        """
        # define colors
        cset = tol_cset('bright')

        (xlabel,) = hk_data.coords._fields
        xdata = hk_data.coords[0][:].copy()
        use_steps = xdata.size <= 256
        (xdata, gap_list) = plotlib.get_xdata(xdata, use_steps)

        for key in hk_keys:
            if key not in hk_data.value.dtype.names:
                continue

            indx = hk_data.value.dtype.names.index(key)
            hk_unit = hk_data.units[indx]
            if isinstance(hk_unit, bytes):
                hk_unit = hk_unit.decode('ascii')
            full_string = hk_data.long_name[indx]
            if isinstance(full_string, bytes):
                full_string = full_string.decode('ascii')
            if hk_unit == 'K':
                hk_name = full_string.rsplit(' ', 1)[0]
                hk_label = 'temperature [{}]'.format(hk_unit)
                lcolor = cset.blue
                fcolor = '#BBCCEE'
            elif hk_unit in ('A', 'mA'):
                hk_name = full_string.rsplit(' ', 1)[0]
                hk_label = 'current [{}]'.format(hk_unit)
                lcolor = cset.green
                fcolor = '#CCDDAA'
            elif hk_unit == '%':
                hk_name = full_string.rsplit(' ', 2)[0]
                hk_label = 'duty cycle [{}]'.format(hk_unit)
                lcolor = cset.red
                fcolor = '#FFCCCC'
            else:
                hk_name = full_string
                hk_label = 'value [{}]'.format(hk_unit)
                lcolor = cset.purple
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
                axarr[i_ax].step(xdata, ydata, where='post',
                                 linewidth=1.5, color=lcolor)
            else:
                axarr[i_ax].plot(xdata, ydata,
                                 linewidth=1.5, color=lcolor)
            # we are interested to see the last 2 days of the data,
            # and any trend over the whole data, without outliers
            ylim = None
            ybuff = ydata[np.isfinite(ydata)]
            if xlabel == 'orbit' and ybuff.size > 5 * 15:
                ni = 2 * 15
                ylim = [min(ybuff[0:ni].min(), ybuff[-ni:].min()),
                        max(ybuff[0:ni].max(), ybuff[-ni:].max())]
            if not (np.array_equal(ydata, yerr1)
                    and np.array_equal(ydata, yerr2)):
                axarr[i_ax].fill_between(xdata, yerr1, yerr2,
                                         step='post', facecolor=fcolor)
                ybuff1 = yerr1[np.isfinite(yerr1)]
                ybuff2 = yerr2[np.isfinite(yerr2)]
                if xlabel == 'orbit' \
                   and ybuff1.size > 5 * 15 and ybuff2.size > 5 * 15:
                    ni = 2 * 15
                    ylim = [min(ybuff1[0:ni].min(), ybuff1[-ni:].min()),
                            max(ybuff2[0:ni].max(), ybuff2[-ni:].max())]
            axarr[i_ax].locator_params(axis='y', nbins=4)
            axarr[i_ax].set_xlim([xdata[0], xdata[-1]])
            if ylim is not None:
                delta = (ylim[1] - ylim[0]) / 5
                if delta == 0:
                    if ylim[0] == 0:
                        delta = 0.01
                    else:
                        delta = ylim[0] / 20
                axarr[i_ax].set_ylim([ylim[0] - delta, ylim[1] + delta])
            axarr[i_ax].grid(True)
            axarr[i_ax].set_ylabel(hk_label)
            legenda = axarr[i_ax].legend([plotlib.blank_legend_key()],
                                         [hk_name], loc='upper left')
            legenda.draw_frame(False)
            i_ax += 1

    def __add_colorbar(self, ax_img, labels, bounds=None):
        """
        Draw colorbar right of image panel
        """
        # define location of colorbar
        cax = self.__divider.append_axes("right", size=0.3, pad=0.05)

        # colorbar for image data
        if bounds is None:
            plt.colorbar(ax_img, cax=cax, label=labels)
            return

        # colorbar for pixel-quality data
        mbounds = [(bounds[ii+1] + bounds[ii]) / 2
                   for ii in range(len(bounds)-1)]
        plt.colorbar(ax_img, cax=cax, ticks=mbounds, boundaries=bounds)
        cax.tick_params(axis='y', which='both', length=0)
        cax.set_yticklabels(labels)

    def __add_side_panels(self, ax_fig, img_data, coords, quality=None):
        """
        Draw row and column medians left and below image panel
        """
        cset = tol_cset('bright')

        for xtl in ax_fig.get_xticklabels():
            xtl.set_visible(False)
        for ytl in ax_fig.get_yticklabels():
            ytl.set_visible(False)

        # ----- Panel bellow the image -----
        ax_medx = self.__divider.append_axes("bottom", 1.2, pad=0.25,
                                             sharex=ax_fig)
        if quality is None:
            data_row = biweight(img_data, axis=0)
            if len(coords['X']['data']) > 250:
                ax_medx.plot(coords['X']['data'], data_row, linewidth=0.75,
                             color=cset.blue)
            else:
                ax_medx.step(coords['X']['data'], data_row, linewidth=0.75,
                             color=cset.blue)
        else:
            data_row = np.sum(((img_data == 1) | (img_data == 2)), axis=0)
            ax_medx.step(coords['X']['data'], data_row, linewidth=0.75,
                         color=cset.yellow)
            data_row = np.sum((img_data == 1), axis=0)          # worst
            ax_medx.step(coords['X']['data'], data_row, linewidth=0.75,
                         color=cset.red)
            data_row = np.sum((img_data == 4), axis=0)          # good
            ax_medx.step(coords['X']['data'], data_row, linewidth=0.75,
                         color=cset.green)
        ax_medx.set_xlim([0, len(coords['X']['data'])])
        ax_medx.grid(True)
        ax_medx.set_xlabel(coords['X']['label'])

        # ----- Panel left of the image -----
        ax_medy = self.__divider.append_axes("left", 1.1, pad=0.25,
                                             sharey=ax_fig)
        if quality is None:
            data_col = biweight(img_data, axis=1)
            if len(coords['Y']['data']) > 250:
                ax_medy.plot(data_col, coords['Y']['data'], linewidth=0.75,
                             color=cset.blue)
            else:
                ax_medy.step(data_col, coords['Y']['data'], linewidth=0.75,
                             color=cset.blue)
        else:
            data_col = np.sum(((img_data == 1) | (img_data == 2)), axis=1)
            ax_medy.step(data_col, coords['Y']['data'], linewidth=0.75,
                         color=cset.yellow)
            data_col = np.sum(img_data == 1, axis=1)            # worst
            ax_medy.step(data_col, coords['Y']['data'], linewidth=0.75,
                         color=cset.red)
            data_col = np.sum(img_data == 4, axis=1)            # good
            ax_medy.step(data_col, coords['Y']['data'], linewidth=0.75,
                         color=cset.green)
        ax_medy.set_ylim([0, len(coords['Y']['data'])])
        ax_medy.grid(True)
        ax_medy.set_ylabel(coords['Y']['label'])

    def __add_fig_box_above(self, fig, dict_info):
        """
        Add meta-information in the current figure

        Parameters
        ----------
        fig :  Matplotlib figure instance
        dict_info :  dictionary or sortedDict
           Legend parameters to be displayed in the figure
        """
        info_str = plotlib.fig_info_as_str(dict_info)

        if self.__aspect == 1:
            xpos = 0.91
            ypos = 0.98
        elif self.__aspect == 2:
            xpos = 0.925
            ypos = 0.97
        elif self.__aspect == 3:
            xpos = 0.95
            ypos = 0.975
        elif self.__aspect == 4:
            xpos = 0.9
            ypos = 0.975
        else:
            xpos = 0.9
            ypos = 0.925

        fig.text(xpos, ypos, info_str,
                 fontsize='small', style='normal',
                 verticalalignment='top',
                 horizontalalignment='right',
                 multialignment='left',
                 bbox={'facecolor': 'white', 'pad': 5})

    def __add_fig_box_right(self, dict_info):
        """
        Add meta-information in the current figure

        Parameters
        ----------
        dict_info :  dictionary or sortedDict
           Legend parameters to be displayed in the figure
        """
        info_str, info_lines = plotlib.fig_info_as_str(dict_info, count=True)

        fontsize = 'x-small'
        xpos = 0.025
        ypos = 1.075
        if self.__aspect == 1:
            if info_lines > 51:
                fontsize = 5.25
        elif self.__aspect == 2:
            if info_lines > 40:
                ypos = 1.1
                fontsize = 'xx-small'
        else:
            if info_lines > 35:
                fontsize = 'xx-small'

        ax_info = self.__divider.append_axes("right", size=2.5, pad=.75)
        ax_info.set_xticks([])       # remove all X-axis tick locations
        ax_info.set_yticks([])       # remove all Y-axis tick locations
        for key in ('left', 'right', 'top', 'bottom'):
            ax_info.spines[key].set_color('white')
        ax_info.text(xpos, ypos, info_str,
                     fontsize=fontsize, style='normal',
                     verticalalignment='top',
                     horizontalalignment='left',
                     multialignment='left', linespacing=1.5)

    # -------------------------
    def fig_sz_data2d(self, info_pos='above'):
        """
        Define figure size depended on image aspect-ratio
        """
        fig_ext = 3.5 if info_pos == 'right' else 0

        if self.__aspect == 1:
            figsize = (10 + fig_ext, 9)
        elif self.__aspect == 2:
            figsize = (12 + fig_ext, 7)
        elif self.__aspect == 3:
            figsize = (14 + fig_ext, 6.5)
        elif self.__aspect == 4:
            figsize = (15 + fig_ext, 6)
        else:
            print(__name__ + '.draw_signal', self.__aspect)
            raise ValueError('*** FATAL: aspect ratio not implemented, exit')

        return figsize

    # --------------------------------------------------
    def draw_signal(self, data_in, ref_data=None, method='data', *,
                    add_medians=True, vperc=None, vrange=None,
                    title=None, sub_title=None,
                    extent=None, fig_info=None, info_pos=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        data :  numpy.ndarray or pys5p.S5Pmsm
           Object holding measurement data and attributes
        ref_data :  numpy.ndarray, optional
           Numpy array holding reference data. Required for method equals
            'ratio' where measurement data is divided by the reference
            'diff'  where reference is subtracted from the measurement data
           S5Pmsm object holding the reference data as value/error required
            by the method 'ratio_unc'
        method : string
           Method of plot to be generated, default is 'data', optional are
            'error', 'diff', 'ratio', 'ratio_unc'
        add_medians :  boolean
           show in side plots row and column (biweight) medians. Default=True.

        vperc :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.].
           keyword 'vperc' is ignored when vrange is given
        vrange :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.

        title :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info :  dictionary
           OrderedDict holding meta-data to be displayed in the figure
        info_pos :  ('above', 'right', 'none')
           Draw the info text-region right of the figure instead of above

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data
        (biweight) median & spread.
        """
        if method not in ('data', 'error', 'diff', 'ratio', 'ratio_unc'):
            raise RuntimeError('unknown method: {}'.format(method))

        if isinstance(data_in, S5Pmsm) \
           and data_in.long_name is not None \
           and data_in.long_name.find('uncertainty') >= 0:
            method = 'error'

        if info_pos is not None:
            if info_pos not in ('above', 'right', 'none'):
                raise KeyError('info should be above, right or none')
        else:
            info_pos = 'above'

        if vrange is None and vperc is None:
            vperc = (1., 99.)
        elif vrange is None:
            if len(vperc) != 2:
                raise TypeError('keyword vperc requires two values')
        else:
            if len(vrange) != 2:
                raise TypeError('keyword vrange requires two values')

        try:
            img_data = self.set_fig_data2d(method, data_in, ref_data)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        norm = self.scale_data2d(method, img_data, vperc, vrange)

        # inititalize figure (and its size & aspect-ratio)
        fig, ax_fig = plt.subplots(figsize=self.fig_sz_data2d(info_pos))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        if sub_title is not None:
            ax_fig.set_title(sub_title, fontsize='large')

        # draw image
        coords = plotlib.get_fig_coords(data_in)
        if extent is None:
            extent = [0, len(coords['X']['data']),
                      0, len(coords['Y']['data'])]
        ax_img = ax_fig.imshow(img_data, cmap=self.cmap, norm=norm,
                               interpolation='none', origin='lower',
                               aspect='equal', extent=extent)
        self.__add_copyright(ax_fig)

        # define ticks locations for X & Y valid for most detectors
        self.adjust_tickmarks(ax_fig, coords)

        self.__divider = make_axes_locatable(ax_fig)
        self.__add_colorbar(ax_img, self.__get_zlabel(method))
        if add_medians:
            self.__add_side_panels(ax_fig, img_data, coords)
        else:
            ax_fig.set_xlabel(coords['X']['label'])
            ax_fig.set_ylabel(coords['Y']['label'])

        # add annotation and save figure
        if info_pos != 'none':
            median, spread = biweight(img_data, spread=True)
            if self.zunit is None:
                median_str = '{:.5g}'.format(median)
                spread_str = '{:.5g}'.format(spread)
            else:
                median_str = r'{:.5g} {}'.format(median, self.zunit)
                spread_str = r'{:.5g} {}'.format(spread, self.zunit)

            fig_info = plotlib.fill_fig_info(fig_info, 'median', median_str)
            fig_info = plotlib.fill_fig_info(fig_info, 'spread', spread_str)

            if info_pos == 'above':
                self.__add_fig_box_above(fig, fig_info)
            elif info_pos == 'right':
                self.__add_fig_box_right(fig_info)

        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_quality(self, data_in, ref_data=None, *,
                     add_medians=True, thres_worst=0.1, thres_bad=0.8,
                     qlabels=None, title=None, sub_title=None,
                     extent=None, fig_info=None, info_pos=None):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        data :  numpy.ndarray or pys5p.S5Pmsm
           Object holding measurement data and attributes
        ref_data :  numpy.ndarray, optional
           Numpy array holding reference data, for example pixel quality
           reference map taken from the CKD. Shown are the changes with
           respect to the reference data. Default is None
        add_medians :  boolean
           show in side plots row and column (biweight) medians. Default=True.

        thres_worst :  float
           Threshold to reject only the worst of the bad pixels, intended
           for CKD derivation. Default=0.1
        thres_bad :  float
           Threshold for bad pixels. Default=0.8
        qlabel : list of strings
           Labels for the pixel-quality classes, see below

        title :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info :  dictionary
           OrderedDict holding meta-data to be displayed in the figure
        info_pos :  ('above', 'right', 'none')
           Draw the info text-region right of the figure instead of above

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
        in a small box. Where creation date and statistics on the number of
        bad and worst pixels are displayed.
        """
        if info_pos is not None:
            if info_pos not in ('above', 'right', 'none'):
                raise KeyError('info should be above, right or none')
        else:
            info_pos = 'above'

        qthres = {'worst': thres_worst, 'bad': thres_bad}
        try:
            qdata = self.set_fig_quality(qthres, data_in, ref_data)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        # define colors, data-range
        cset = tol_cset('bright')
        if ref_data is None:
            if qlabels is None:
                qlabels = ("unusable", "worst", "bad", "good")
            else:
                if len(qlabels) != 4:
                    raise TypeError('keyword qlabels requires four labels')

            # define colors for resp. unusable, worst, bad and good
            ctuple = (cset.grey, cset.red, cset.yellow, '#FFFFFF')
            bounds = [0, 1, 2, 4, 8]
        else:
            if qlabels is None:
                qlabels = ["unusable", "to worst", "good to bad ", "to good",
                           "unchanged"]
            else:
                if len(qlabels) != 5:
                    raise TypeError('keyword qlabels requires five labels')

            # define colors for resp. unusable, worst, bad, good and unchanged
            ctuple = (cset.grey, cset.red, cset.yellow, cset.green, '#FFFFFF')
            bounds = [0, 1, 2, 4, 8, 16]
        cmap = mpl.colors.ListedColormap(ctuple)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # inititalize figure (and its size & aspect-ratio)
        fig, ax_fig = plt.subplots(figsize=self.fig_sz_data2d(info_pos))
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')
        if sub_title is not None:
            ax_fig.set_title(sub_title, fontsize='large')

        # draw image
        coords = plotlib.get_fig_coords(data_in)
        if extent is None:
            extent = [0, len(coords['X']['data']),
                      0, len(coords['Y']['data'])]
        ax_img = ax_fig.imshow(qdata, cmap=cmap, norm=norm,
                               interpolation='none', origin='lower',
                               aspect='equal', extent=extent)
        self.__add_copyright(ax_fig)

        # define ticks locations for X & Y valid for most detectors
        self.adjust_tickmarks(ax_fig, coords)

        self.__divider = make_axes_locatable(ax_fig)
        self.__add_colorbar(ax_img, qlabels, bounds)

        if add_medians:
            self.__add_side_panels(ax_fig, qdata, coords, quality=qthres)
        else:
            ax_fig.set_xlabel(coords['X']['label'])
            ax_fig.set_ylabel(coords['Y']['label'])

        # add annotation and save figure
        if ref_data is None:
            count = np.sum((qdata == 1) | (qdata == 2))
            info_str = '{} (quality < {})'.format(qlabels[2], thres_bad)
            fig_info = plotlib.fill_fig_info(fig_info, info_str,
                                             '{}'.format(count))
            count = np.sum(qdata == 1)
            info_str = '{} (quality < {})'.format(qlabels[1], thres_worst)
            fig_info = plotlib.fill_fig_info(fig_info, info_str,
                                             '{}'.format(count))
        else:
            fig_info = plotlib.fill_fig_info(fig_info, 'to good',
                                             '{}'.format(np.sum(qdata == 4)))
            fig_info = plotlib.fill_fig_info(fig_info, 'good to bad',
                                             '{}'.format(np.sum(qdata == 2)))
            fig_info = plotlib.fill_fig_info(fig_info, 'to worst',
                                             '{}'.format(np.sum(qdata == 1)))

        if info_pos == 'above':
            self.__add_fig_box_above(fig, fig_info)
        elif info_pos == 'right':
            self.__add_fig_box_right(fig_info)

        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_cmp_swir(self, data_in, ref_data,
                      *, vperc=None, vrange=None,
                      model_label='reference',
                      add_residual=True, add_model=True,
                      title=None, sub_title=None,
                      extent=None, fig_info=None, info_pos=None):
        """
        Display signal vs model (or CKD) comparison in three panels.
        Top panel shows data, middle panel shows residuals (data - model)
        and lower panel shows model.

        Parameters
        ----------
        data :  numpy.ndarray or pys5p.S5Pmsm
           Object holding measurement data and attributes
        ref_data :  numpy.ndarray
           Numpy array holding reference data.

        vperc :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.].
           keyword 'vperc' is ignored when vrange is given
        vrange :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
        model_label :  string
           Name of reference dataset.  Default is 'reference'

        title :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info :  dictionary
           OrderedDict holding meta-data to be displayed in the figure
        info_pos :  ('above', 'right', 'none')
           Draw the info text-region right of the figure instead of above

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data
        median & spread.

        Note
        ----
        Current implementation only works for images with aspect-ration of 4
        (like Tropomi-SWIR).
        """
        if not (add_residual or add_model):
            raise KeyError('add_resudual or add_model should be true')

        if info_pos is not None:
            if info_pos not in ('above', 'right', 'none'):
                raise KeyError('info should be above, right or none')
        else:
            info_pos = 'above'

        if vrange is None and vperc is None:
            vperc = (1., 99.)
        elif vrange is None:
            if len(vperc) != 2:
                raise TypeError('keyword vperc requires two values')
        else:
            if len(vrange) != 2:
                raise TypeError('keyword vrange requires two values')

        try:
            img_diff = self.set_fig_data2d('diff', data_in, ref_data)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        (median, spread) = biweight(img_diff, spread=True)
        rrange = (median - 5 * spread, median + 5 * spread)
        rnorm = self.scale_data2d('diff', img_diff, None, rrange)

        try:
            img_data = self.set_fig_data2d('data', data_in, None)
        except Exception as exc:
            raise RuntimeError('invalid input-data provided') from exc

        vnorm = self.scale_data2d('data', img_data, vperc, vrange)

        # inititalize figure (and its size & aspect-ratio)
        npanel = True + add_residual + add_model
        if npanel == 3:
            fig = plt.figure(figsize=(10.8, 7.5))
        else:
            fig = plt.figure(figsize=(10.8, 6))

        gspec = mpl.gridspec.GridSpec(npanel, 2)
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(.5, .96), horizontalalignment='center')

        coords = plotlib.get_fig_coords(data_in)
        if extent is None:
            extent = [0, len(coords['X']['data']),
                      0, len(coords['Y']['data'])]

        # create image panel
        def draw_image(ipanel, data, sub_title, norm, extent):
            bullet = ['(a) ', '(b) ', '(c) ', None]

            ax_fig = plt.subplot(gspec[ipanel, :])
            if sub_title is not None:
                ax_fig.set_title(bullet[ipanel] + sub_title, fontsize='large')

            ax_img = ax_fig.imshow(data, cmap=self.cmap, norm=norm,
                                   interpolation='none', origin='lower',
                                   aspect='equal', extent=extent)
            self.__add_copyright(ax_fig)

            # define ticks locations for X & Y valid for most detectors
            self.adjust_tickmarks(ax_fig, coords)

            self.__divider = make_axes_locatable(ax_fig)
            if sub_title == 'residual':
                self.__add_colorbar(ax_img, self.__get_zlabel('diff'))
            else:
                self.__add_colorbar(ax_img, self.__get_zlabel('data'))
            if npanel == (ipanel + 1):
                ax_fig.set_xlabel(coords['X']['label'])
            else:
                for xtl in ax_fig.get_xticklabels():
                    xtl.set_visible(False)
            ax_fig.set_ylabel(coords['Y']['label'])

        ipanel = 0
        draw_image(ipanel, img_data, sub_title, vnorm, extent)
        if add_residual:
            ipanel += 1
            draw_image(ipanel, img_diff, 'residual', rnorm, extent)
        if add_model:
            ipanel += 1
            draw_image(ipanel, ref_data, model_label, vnorm, extent)

        # add annotation and save figure
        if info_pos != 'none':
            median, spread = biweight(img_diff, spread=True)
            if self.zunit is None:
                median_str = '{:.5g}'.format(median)
                spread_str = '{:.5g}'.format(spread)
            else:
                median_str = r'{:.5g} {}'.format(median, self.zunit)
                spread_str = r'{:.5g} {}'.format(spread, self.zunit)

            fig_info = plotlib.fill_fig_info(fig_info, 'median', median_str)
            fig_info = plotlib.fill_fig_info(fig_info, 'spread', spread_str)

            if info_pos == 'above':
                self.__add_fig_box_above(fig, fig_info)
            elif info_pos == 'right':
                self.__add_fig_box_right(fig_info)

        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_trend1d(self, msm1, hk_data=None, msm2=None, *, hk_keys=None,
                     title=None, sub_title=None,
                     fig_info=None, info_pos=None):
        """
        Display trends of measurement and house-keeping data

        Parameters
        ----------
        msm1      :  pys5p.S5Pmsm, optional
           Object with measurement data and its HDF5 attributes (first figure)
        msm2      :  pys5p.S5Pmsm, optional
           Object with measurement data and its HDF5 attributes (second figure)
        hk_data   :  pys5p.S5Pmsm, optional
           Object holding housekeeping data and its HDF5 attributes
        hk_keys    : list or tuple
           List of housekeeping parameters to be displayed

        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title  :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure
        info_pos :  ('above', 'right', 'none')
           Draw the info text-region right of the figure instead of above

        You have to provide a non-None value for parameter 'msm1' or 'hk_data'.
        Only house-keeping data will be shown when 'msm1' is None (parameter
        'msm2' is ignored when 'msm1' equals None.

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        # we require measurement data and/or house-keeping data
        if msm1 is None and hk_data is None:
            raise ValueError(
                'measurement data and/or house-keeping data are required')

        if info_pos is not None:
            if info_pos not in ('above', 'right', 'none'):
                raise KeyError('info should be above, right or none')
        else:
            info_pos = 'above'

        # ---------- local function ----------
        # make sure that we use 'large' fonts in the small plots
        if self.__pdf is None:
            plt.rc('font', size=10)

        # define number of panels for measurement data
        if msm1 is None:
            plot_mode = 'house-keeping'
            (xlabel,) = hk_data.coords._fields
            if xlabel == 'time':
                xdata = xdata.astype(np.float) / 3600
                xlabel += ' [hours]'
            npanels = 0
        elif (msm1.value.dtype.names is not None
              and 'bad' in msm1.value.dtype.names):
            plot_mode = 'quality'
            (xlabel,) = msm1.coords._fields
            npanels = 2
        else:
            plot_mode = 'data'
            (xlabel,) = msm1.coords._fields
            if msm2 is None:
                npanels = 1
            else:
                npanels = 2

        # add panels for housekeeping parameters
        if hk_data is not None:
            # default house-keeping parameters
            if hk_keys is None:
                if 'detector_temp' in hk_data.value.dtype.names:
                    hk_keys = ('detector_temp', 'grating_temp',
                               'imager_temp', 'obm_temp')
                elif 'temp_det4' in hk_data.value.dtype.names:
                    hk_keys = ('temp_det4', 'temp_obm_swir_grating')
                else:
                    hk_keys = tuple(hk_data.value.dtype.names)[0:4]
            npanels += len(hk_keys)

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

        # define aspect for the location of fig_info
        self.__aspect = 3

        # add figures with quality data or measurement data
        i_ax = self.__add_data1d(plot_mode, axarr, msm1, msm2)

        # add figures with house-keeping data
        if hk_data is not None:
            self.__add_hkdata(i_ax, axarr, hk_data, hk_keys)

        axarr[-1].set_xlabel(xlabel)
        if npanels > 1:
            plt.setp([a.get_xticklabels()
                      for a in fig.axes[:-1]], visible=False)

        self.__add_copyright(axarr[-1])

        # add annotation and save figure
        if info_pos == 'above':
            self.__add_fig_box_above(fig, fig_info)
        elif info_pos == 'right':
            self.__add_fig_box_right(fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_lines(self, xdata, ydata, *, color=0,
                   xlabel=None, ylabel=None, xlim=None, ylim=None,
                   title=None, sub_title=None, fig_info=None, **kwargs):
        """
        Parameters
        ----------
        xdata     :  ndarray
           x-axis data
           Special case if xdata is None then close figure
        ydata     :  ndarray
           y-axis data
        color     :  integer, optional
           index to color in tol_colors.tol_cset('bright'). Default is zero
        title     :  string, optional
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title :  string, optional
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info  :  dictionary, optional
           OrderedDict holding meta-data to be displayed in the figure
        **kwargs :   other keywords
           Pass all other keyword arguments to matplotlib.pyplot.plot()

        Returns
        -------
        Nothing

        Examples
        --------
        General example:
        >>> plot = S5Pplot(fig_name)
        >>> for ii, xx, yy in enumerate(data_of_each_line):
        >>>    plot.draw_lines(xx, yy, color=ii, label=mylabel[ii],
        >>>                    marker='o', linestyle='None')
        >>> plot.draw_lines(None, None, xlim=[0, 0.5], ylim=[-10, 10],
        >>>                 xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()

        Using a time-axis:
        >>> from datetime import datetime, timedelta
        >>> tt0 = (datetime(year=2020, month=10, day=1)
        >>>        + timedelta(seconds=sec_in_day))
        >>> tt = [tt0 + xx * t_step for xx in range(yy.size)]
        >>> plot = S5Pplot(fig_name)
        >>> plot.draw_lines(tt, yy, color=1, label=mylabel,
        >>>                 marker='o', linestyle='None')
        >>> plot.draw_line(None, None, ylim=[-10, 10],
        >>>                xlabel=my_xlabel, ylabel=my_ylabel)
        >>> plot.close()
        """
        # add annotation and close figure
        if xdata is None:
            if self.__mpl is None:
                raise ValueError('No plot defined and no data provided')

            # finalize figure
            if self.__mpl['time_axis']:
                plt.gcf().autofmt_xdate()
                my_fmt = mpl.dates.DateFormatter('%H:%M:%S')
                plt.gca().xaxis.set_major_formatter(my_fmt)

            self.__mpl['axarr'].grid(True)
            if xlabel is not None:
                self.__mpl['axarr'].set_xlabel(xlabel)
            if ylabel is not None:
                self.__mpl['axarr'].set_ylabel(ylabel)
            if xlim is not None:
                self.__mpl['axarr'].set_xlim(xlim)
            if ylim is not None:
                self.__mpl['axarr'].set_ylim(ylim)
            if 'xscale' in kwargs:
                self.__mpl['axarr'].set_xscale(kwargs['xscale'])
            if 'yscale' in kwargs:
                self.__mpl['axarr'].set_ylabel(kwargs['yscale'])

            # draw titles (and put it at the same place)
            if title is not None:
                self.__mpl['fig'].suptitle(title, fontsize='x-large',
                                           position=(0.5, 0.95),
                                           horizontalalignment='center')
            if sub_title is not None:
                self.__mpl['axarr'].set_title(sub_title, fontsize='large')

            # draw legenda in figure
            if self.__mpl['axarr'].get_legend_handles_labels()[1]:
                self.__mpl['axarr'].legend(fontsize='small', loc='best')

            # draw copyright
            self.__add_copyright(self.__mpl['axarr'])

            # close page
            self.__add_fig_box_above(self.__mpl['fig'], fig_info)
            self.__close_this_page(self.__mpl['fig'])
            self.__mpl = None
            return

        # define colors
        cset = tol_cset('bright')
        c_max = len(cset)

        # initialize matplotlib using 'subplots'
        # define aspect for the location of fig_info
        if self.__mpl is None:
            if len(xdata) <= 256:
                self.__aspect = 1
                figsize = (9, 9)
            elif 256 > len(xdata) <= 512:
                self.__aspect = 1
                figsize = (12, 9)
            elif 512 > len(xdata) <= 768:
                self.__aspect = 1
                figsize = (15, 9)
            else:
                self.__aspect = 2
                figsize = (18, 9)

            self.__mpl = dict(zip(('fig', 'axarr'),
                                  plt.subplots(1, figsize=figsize)))
            if isinstance(xdata[0], datetime):
                self.__mpl['time_axis'] = True
            else:
                self.__mpl['time_axis'] = False

        use_steps = False
        if use_steps:
            xx = np.append(xdata, xdata[-1])
            yy = np.append(ydata, ydata[-1])
            self.__mpl['axarr'].step(xx, yy, where='post',
                                     color=cset[color % c_max],
                                     **kwargs)
        else:
            self.__mpl['axarr'].plot(xdata, ydata,
                                     color=cset[color % c_max],
                                     **kwargs)

    # --------------------------------------------------
    def draw_tracks(self, lons, lats, icids, *, saa_region=None,
                    title=None, fig_info=None):
        """
        Display tracks of S5P on a world map using a Robinson projection

        Parameters
        ----------
        lats       :  ndarray [N, 2]
           Latitude coordinates at start and end of measurement
        lons       :  ndarray [N, 2]
           Longitude coordinates at start and end of measurement
        icids      :  ndarray [N]
           ICID of measurements per (lon, lat)
        saa_region :  'ckd' or ndarray
           Show SAA region. Its definition obtained from Tropomi Level-1B CKD,
           or as a matplotlib polygon patch. Default None
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if not FOUND_CARTOPY:
            raise RuntimeError('You need Cartopy to run this function')

        # define colors
        cset = tol_cset('bright')

        # define aspect for the location of fig_info
        self.__aspect = 4

        # define plot layout
        myproj = ccrs.Robinson(central_longitude=11.5)
        fig, axx = plt.subplots(figsize=(12.85, 6),
                                subplot_kw={'projection': myproj})
        axx.set_global()
        axx.coastlines(resolution='110m')
        axx.gridlines()
        axx.set_title('ground-tracks of Sentinel-5P')

        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.96), horizontalalignment='center')

        # draw SAA region
        if saa_region is not None:
            if saa_region in ('ckd', 'CKD'):
                with CKDio() as ckd:
                    res = ckd.saa()
                saa_region = list(zip(res['lon'], res['lat']))

            saa_poly = mpl.patches.Polygon(
                xy=saa_region, closed=True, alpha=1.0,
                facecolor=cset.grey, transform=ccrs.PlateCarree())
            axx.add_patch(saa_poly)

        # draw satellite position(s)
        icid_found = []
        for lon, lat, icid in zip(lons, lats, icids):
            if icid not in icid_found:
                indx_color = len(icid_found)
            else:
                indx_color = icid_found.index(icid)
            line, = plt.plot(lon, lat, linestyle='-', linewidth=3,
                             color=cset[indx_color % 6],
                             transform=ccrs.PlateCarree())
            if icid not in icid_found:
                line.set_label('ICID: {}'.format(icid))
                icid_found.append(icid)

        # finalize figure
        axx.legend(loc='lower left')
        self.__add_copyright(axx)
        self.__add_fig_box_above(fig, fig_info)
        self.__close_this_page(fig)
