"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class S5Pgeoplot displays footprints or observations projected
on the Earth surface.

Copyright (c) 2018-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import PurePath

import matplotlib as mpl
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                       LATITUDE_FORMATTER)
except Exception as exc:
    raise RuntimeError('This module require module Cartopy') from exc
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
import shapely.geometry as sgeom

from .lib.plotlib import FIGinfo
from .s5p_msm import S5Pmsm
from .tol_colors import tol_cmap


# - main function ----------------------------------
class BetterTransverseMercator(ccrs.Projection):
    """
    Implement improved transverse mercator projection

    By Paul Tol (SRON)
    """
    # pylint: disable=no-member,too-many-arguments
    def __init__(self, central_latitude=0.0, central_longitude=0.0,
                 orientation=0, scale_factor=1.0,
                 false_easting=0.0, false_northing=0.0, globe=None):
        """
        Parameters
        ----------
        * central_longitude - The true longitude of the central
            meridian in degrees. Defaults to 0.
        * central_latitude - The true latitude of the planar origin in
            degrees. Defaults to 0.
        * orientation - ...
            Defaults to 0.
        * scale_factor - Scale factor at the central meridian.
            Defaults to 1.
        * false_easting - X offset from the planar origin in metres.
            Defaults to 0.
        * false_northing - Y offset from the planar origin in metres.
            Defaults to 0.
        * globe - An instance of :class:`cartopy.crs.Globe`.
            If omitted, a default globe is created.
        """
        proj4_params = [
            ('proj', 'omerc'),
            ('lat_0', central_latitude),
            ('lonc', np.round(central_longitude / 15.0) * 15.0 + 0.01),
            ('alpha', '0.01'),
            ('gamma', np.sign(orientation) * 89.99),
            ('over', ''),
            ('k_0', scale_factor),
            ('x_0', false_easting),
            ('y_0', false_northing),
            ('units', 'm')]
        super().__init__(proj4_params, globe=globe)

    @property
    def threshold(self):
        return 1e4

    @property
    def boundary(self):
        xx0, xx1 = self.x_limits
        yy0, yy1 = self.y_limits
        return sgeom.LineString(
            [(xx0, yy0), (xx0, yy1), (xx1, yy1), (xx1, yy0), (xx0, yy0)])

    @property
    def x_limits(self):
        return (-2e7, 2e7)

    @property
    def y_limits(self):
        return (-2e7, 2e7)


# --------------------------------------------------
class S5Pgeoplot():
    """
    Generate figure(s) for the SRON Tropomi SWIR monitor website or MPC reports

    Attributes
    ----------
    add_info : bool
    aspect : int
    data : ndarray
    filename : string
    method : string

    Methods
    -------
    def close_page(fig, fig_info):
    def close():
    def add_copyright(axx):
    def draw_geo_tiles(lons, lats, sequence=None, title=None, fig_info=None)
    def draw_geo_subsat(lons, lats, title=None, fig_info=None)
    def draw_geo_msm(gridlon, gridlat, msm_in, vperc=None, vrange=None,
                     whole_globe=False, title=None, fig_info=None)

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, figname, pdf_title=None):
        """
        Initialize multi-page PDF document or a single-page PNG

        Parameters
        ----------
        figname   :  string
             name of PDF or PNG file (extension required)
        pdf_title :  string
             Title of the PDF document (attribute of the PDF document)
             Default: 'Monitor report on Tropomi SWIR instrument'
        """
        self.filename = figname

        if PurePath(figname).suffix.lower() == '.pdf':
            self.__pdf = PdfPages(figname)
            # add annotation
            doc = self.__pdf.infodict()
            if pdf_title is None:
                doc['Title'] = 'Monitor report on Tropomi SWIR instrument'
            else:
                doc['Title'] = pdf_title
            doc['Author'] = '(c) SRON Netherlands Institute for Space Research'
        else:
            self.__pdf = None

        # define colors
        self.cset = {
            'water': '#ddeeff',
            'land': '#e1c999',
            'grid': '#bbbbbb',
            's5p': '#ee6677'}

        #self.data = None
        #self.method = None

    def __repr__(self):
        pass

    def __close_this_page(self, fig):
        """
        Close current matplotlib figure or page in a PDF document
        """
        # add save figure
        if self.__pdf is None:
            plt.savefig(self.filename)
            # plt.savefig(self.filename, transparant=True)
            plt.close(fig)
        else:
            self.__pdf.savefig(transparant=True)

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

    @staticmethod
    def __add_fig_box(fig, fig_info) -> None:
        """
        Add meta-information in the current figure

        Parameters
        ----------
        fig :  Matplotlib figure instance
        fig_info :  FIGinfo
           instance of pys5p.lib.plotlib.FIGinfo to be displayed
        """
        if fig_info is None or fig_info.location == 'none':
            return

        xpos = 1 - 0.4 / fig.get_figwidth()
        ypos = 1 - 0.25 / fig.get_figheight()

        fig.text(xpos, ypos, fig_info.as_str(),
                 fontsize='small', style='normal',
                 verticalalignment='top',
                 horizontalalignment='right',
                 multialignment='left',
                 bbox={'facecolor': 'white', 'pad': 5})

    def __draw_worldmap(self, axx, whole_globe=True) -> None:
        """
        Draw worldmap
        """
        if whole_globe:
            sphere_radius = 6370997.0
            parallel_half = 0.883 * sphere_radius
            meridian_half = 2.360 * sphere_radius
            axx.set_xlim(-parallel_half, parallel_half)
            axx.set_ylim(-meridian_half, meridian_half)

        axx.outline_patch.set_visible(False)
        axx.background_patch.set_facecolor(self.cset['water'])
        axx.add_feature(cfeature.LAND, edgecolor='none',
                        facecolor=self.cset['land'])
        glx = axx.gridlines(linestyle='-', linewidth=0.5,
                            color=self.cset['grid'])
        glx.xlocator = mpl.ticker.FixedLocator(np.linspace(-180, 180, 13))
        glx.ylocator = mpl.ticker.FixedLocator(np.linspace(-90, 90, 13))
        glx.xformatter = LONGITUDE_FORMATTER
        glx.yformatter = LATITUDE_FORMATTER

    # --------------------------------------------------
    def draw_geo_tiles(self, lons, lats, *, sequence=None,
                       title=None, fig_info=None):
        """
        Display footprints projected with (beter) TransverseMercator

        Parameters
        ----------
        lats         :  ndarray
           Latitude coordinates
        lons         :  ndarray
           Longitude coordinates
        sequence     :  list
           Indices to footprints to be drawn by polygons
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # determine central longitude
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
        axx = plt.axes(projection=BetterTransverseMercator(
            central_longitude=lon_0, orientation=1,
            globe=ccrs.Globe(ellipse='sphere')))
        self.__draw_worldmap(axx)

        # draw footprint
        if sequence is None:
            lat = np.concatenate([lats[0, :], lats[1:-1, -1],
                                  lats[-1, ::-1], lats[1:-1:-1, 0]])
            lon = np.concatenate([lons[0, :], lons[1:-1, -1],
                                  lons[-1, ::-1], lons[1:-1:-1, 0]])

            poly = Polygon(xy=list(zip(lon, lat)), closed=True,
                           alpha=0.6, facecolor=self.cset['s5p'],
                           transform=ccrs.PlateCarree())
            axx.add_patch(poly)
        else:
            print('Unique sequence: {}'.format(np.unique(sequence)))
            for ii in np.unique(sequence):
                indx = np.unique(np.where(sequence == ii)[0])
                indx_rev = indx[::-1]
                lat = np.concatenate([lats[indx[0], :],
                                      lats[indx, -1],
                                      lats[indx[-1], ::-1],
                                      lats[indx_rev, 0]])
                lon = np.concatenate([lons[indx[0], :],
                                      lons[indx, -1],
                                      lons[indx[-1], ::-1],
                                      lons[indx_rev, 0]])

                poly = Polygon(xy=list(zip(lon, lat)), closed=True,
                               alpha=0.6, facecolor=self.cset['s5p'],
                               transform=ccrs.PlateCarree())
                axx.add_patch(poly)

        self.__add_copyright(axx)
        fig_info.add('lon0', lon_0)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_geo_subsat(self, lons, lats, *,
                        title=None, fig_info=None):
        """
        Display sub-satellite coordinates projected with TransverseMercator

        Parameters
        ----------
        lats         :  ndarray
           Latitude coordinates
        lons         :  ndarray
           Longitude coordinates
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box.
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # determine central longitude
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
        axx = plt.axes(projection=BetterTransverseMercator(
            central_longitude=lon_0, orientation=1,
            globe=ccrs.Globe(ellipse='sphere')))
        self.__draw_worldmap(axx)

        # draw sub-satellite spot(s)
        axx.scatter(lons, lats, 4, transform=ccrs.PlateCarree(),
                    marker='o', color=self.cset['s5p'])

        self.__add_copyright(axx)
        fig_info.add('lon0', lon_0)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)

    # --------------------------------------------------
    def draw_geo_msm(self, gridlon, gridlat, msm_in, *,
                     vperc=None, vrange=None,
                     whole_globe=False, title=None, fig_info=None):
        """
        Show measurement data projected with (better) TransverseMercator.

        Parameters
        ----------
        gridlon   :  array_like
        gridlat   :  array_like
           The coordinates of the quadrilateral corners, with dimnsions one
           larger then data
        data      :  array_like
           Measurement data as a scalar 2-D array. The values will be
           color-mapped.
        vrange    :  list [vmin,vmax]
           Range to normalize luminance data between vmin and vmax.
        vperc     :  list
           Range to normalize luminance data between percentiles min and max of
           array data. Default is [1., 99.].
           keyword 'vperc' is ignored when vrange is given
        title     :  string
           Title of the figure. Default is None
           [Suggestion] use attribute "title" of data-product
        sub_title :  string
           Sub-title of the figure. Default is None
           [Suggestion] use attribute "comment" of data-product
        fig_info :  FIGinfo, optional
           OrderedDict holding meta-data to be displayed in the figure

        The information provided in the parameter 'fig_info' will be displayed
        in a small box. In addition, we display the creation date and the data
        (biweight) median & spread.
        """
        if fig_info is None:
            fig_info = FIGinfo()

        # assert that we have some data to show
        if isinstance(msm_in, np.ndarray):
            msm = S5Pmsm(msm_in)
        else:
            if not isinstance(msm_in, S5Pmsm):
                raise TypeError('msm not an numpy.ndarray or pys5p.S5Pmsm')
            msm = msm_in

        if msm.value.ndim != 2:
            raise ValueError('input data must be two dimensional')
        if np.all(np.isnan(msm.value)):
            raise ValueError('input data must contain valid data')
        data = msm.value.copy()
        if msm.units is None:
            zunit = None
        elif msm.units == '1':
            zunit = None
        else:
            zunit = msm.units

        # define data-range
        if vrange is None:
            if vperc is None:
                vperc = (1., 99.)
            else:
                if len(vperc) != 2:
                    raise TypeError('keyword vperc requires two values')
            vmin, vmax = np.nanpercentile(data, vperc)
        else:
            if len(vrange) != 2:
                raise TypeError('keyword vrange requires two values')
            vmin, vmax = vrange

        # determine central longitude
        if gridlon.max() - gridlon.min() > 180:
            if np.sum(gridlon > 0) > np.sum(gridlon < 0):
                gridlon[gridlon < 0] += 360
            else:
                gridlon[gridlon > 0] -= 360
        lon_0 = np.around(np.mean(gridlon), decimals=-1)

        # inititalize figure
        myproj = BetterTransverseMercator(central_longitude=lon_0,
                                          orientation=0,
                                          globe=ccrs.Globe(ellipse='sphere'))

        fig, axx = plt.subplots(1, 1, figsize=(10, 12.5),
                                subplot_kw={'projection': myproj})
        if title is not None:
            fig.suptitle(title, fontsize='x-large',
                         position=(0.5, 0.9), horizontalalignment='center')

        # Add the colorbar axes anywhere in the figure.
        # Its position will be re-calculated at each figure resize.
        cax = fig.add_axes([0, 0, 0.1, 0.1])
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.85)

        # define worldmap
        self.__draw_worldmap(axx, whole_globe)

        # draw image and colorbar
        img = axx.pcolormesh(gridlon, gridlat, data,
                             vmin=vmin, vmax=vmax,
                             cmap=tol_cmap('sunset'),
                             transform=ccrs.PlateCarree())
        plt.draw()
        posn = axx.get_position()
        cax.set_position([posn.x0 + posn.width + 0.01,
                          posn.y0, 0.04, posn.height])
        if zunit is None:
            zlabel = 'value'
        else:
            zlabel = r'value [{}]'.format(zunit)
        plt.colorbar(img, cax=cax, label=zlabel)

        # finalize figure
        self.__add_copyright(axx)
        fig_info.add('lon0', lon_0)
        self.__add_fig_box(fig, fig_info)
        self.__close_this_page(fig)
