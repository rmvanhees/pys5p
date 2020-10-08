"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Definition of colour schemes for lines and maps that also work for colour-blind
people. See https://personal.sron.nl/~pault/ for background information and
best usage of the schemes.

Reference
---------
   https://personal.sron.nl/~pault/

Copyright (c) 2019-2020, Paul Tol (SRON)
All rights reserved.

License:  Standard 3-clause BSD
"""
from collections import namedtuple

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array


def discretemap(colormap, hexclrs):
    """
    Produce a colormap from a list of discrete colors without interpolation.
    """
    clrs = to_rgba_array(hexclrs)
    clrs = np.vstack([clrs[0], clrs, clrs[-1]])
    cdict = {}
    for ii, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(jj / (len(clrs)-2.), clrs[jj, ii], clrs[jj+1, ii])
                      for jj in range(len(clrs)-1)]
    return LinearSegmentedColormap(colormap, cdict)


# pylint: disable=invalid-name
class TOLcmaps():
    """
    Class TOLcmaps definition.
    """
    def __init__(self):
        """
        """
        self.cmap = None
        self.cname = None
        self.namelist = (
            'sunset_discrete', 'sunset', 'BuRd_discrete', 'BuRd',
            'PRGn_discrete', 'PRGn', 'YlOrBr_discrete', 'YlOrBr', 'WhOrBr',
            'iridescent', 'rainbow_PuRd', 'rainbow_PuBr', 'rainbow_WhRd',
            'rainbow_WhBr', 'rainbow_WhBr_condense', 'rainbow_discrete')

        self.funcdict = dict(
            zip(self.namelist,
                (self.__sunset_discrete, self.__sunset, self.__BuRd_discrete,
                 self.__BuRd, self.__PRGn_discrete, self.__PRGn,
                 self.__YlOrBr_discrete, self.__YlOrBr, self.__WhOrBr,
                 self.__iridescent, self.__rainbow_PuRd, self.__rainbow_PuBr,
                 self.__rainbow_WhRd, self.__rainbow_WhBr,
                 self.__rainbow_WhBr_condense, self.__rainbow_discrete)))

    def __sunset_discrete(self):
        """
        Define colormap 'sunset_discrete'.
        """
        clrs = ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
                '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D',
                '#A50026']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __sunset(self):
        """
        Define colormap 'sunset'.
        """
        clrs = ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
                '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D',
                '#A50026']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __BuRd_discrete(self):
        """
        Define colormap 'BuRd_discrete'.
        """
        clrs = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFEE99')

    def __BuRd(self):
        """
        Define colormap 'BuRd'.
        """
        clrs = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFEE99')

    def __PRGn_discrete(self):
        """
        Define colormap 'PRGn_discrete'.
        """
        clrs = ['#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                '#D9F0D3', '#ACD39E', '#5AAE61', '#1B7837']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFEE99')

    def __PRGn(self):
        """
        Define colormap 'PRGn'.
        """
        clrs = ['#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                '#D9F0D3', '#ACD39E', '#5AAE61', '#1B7837']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFEE99')

    def __YlOrBr_discrete(self):
        """
        Define colormap 'YlOrBr_discrete'.
        """
        clrs = ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
                '#EC7014', '#CC4C02', '#993404', '#662506']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#888888')

    def __YlOrBr(self):
        """
        Define colormap 'YlOrBr'.
        """
        clrs = ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
                '#EC7014', '#CC4C02', '#993404', '#662506']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#888888')

    def __WhOrBr(self):
        """
        Define colormap 'WhOrBr'.
        """
        clrs = ['#FFFFFF', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
                '#EC7014', '#CC4C02', '#993404', '#662506']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#888888')

    def __iridescent(self):
        """
        Define colormap 'iridescent'.
        """
        clrs = ['#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF',
                '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1',
                '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD',
                '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388',
                '#805770', '#684957', '#46353A']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#999999')

    def __rainbow_PuRd(self):
        """
        Define colormap 'rainbow_PuRd'.
        """
        clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __rainbow_PuBr(self):
        """
        Define colormap 'rainbow_PuBr'.
        """
        clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
                '#521A13']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __rainbow_WhRd(self):
        """
        Define colormap 'rainbow_WhRd'.
        """
        clrs = ['#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
                '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
                '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
                '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
                '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
                '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#666666')

    def __rainbow_WhBr(self):
        """
        Define colormap 'rainbow_WhBr'.
        """
        clrs = ['#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2',
                '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9',
                '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3',
                '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68',
                '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39',
                '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222',
                '#B8221E', '#95211B', '#721E17', '#521A13']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#666666')

    def __rainbow_WhBr_condense(self):
        """
        Define colormap 'rainbow_WhBr_condense'.
        """
        clrs = ['#E8ECFB', '#E4E5F7', '#E1DFF3', '#DDD8EF', '#D9D0EA',
                '#D5C8E5', '#D1C1E1', '#CCB8DB', '#C7B0D6', '#C3A8D1',
                '#BEA0CC', '#BA97C7', '#B58FC2', '#B087BD', '#AC80B8',
                '#A778B4', '#A371AF', '#9F6AAB', '#9B62A7', '#965CA2',
                '#91559E', '#8C4E99', '#824D9A', '#794C9A', '#6F4C9B',
                '#6A509F', '#6555A4', '#6059A9', '#5C5EAE', '#5963B3',
                '#5568B8', '#526EBC', '#5073C0', '#4E79C5', '#4E7FC5',
                '#4D84C6', '#4D8AC6', '#4D8EC3', '#4D92C0', '#4E96BC',
                '#5099B9', '#529CB6', '#549EB3', '#56A1AF', '#58A3AC',
                '#59A5A9', '#5CA7A5', '#5EA9A1', '#60AB9E', '#63AD99',
                '#66AF94', '#69B190', '#6EB38A', '#73B584', '#77B77D',
                '#7EB976', '#85BA6F', '#8CBC68', '#95BD61', '#9DBD5B',
                '#A6BE54', '#AEBD50', '#B6BD4C', '#BEBC48', '#C4BA45',
                '#CAB843', '#D1B541', '#D5B23F', '#D9AE3E', '#DDAA3C',
                '#DFA63B', '#E1A13A', '#E49C39', '#E59738', '#E69137',
                '#E78C35', '#E68534', '#E67F33', '#E67932', '#E57130',
                '#E56A2F', '#E4632D', '#E25A2C', '#E1512A', '#DF4828',
                '#DE3B26', '#DC2F24', '#DA2222', '#CF2221', '#C42220',
                '#B8221E', '#AC221D', '#A1211C', '#95211B', '#892019',
                '#7E1F18', '#721E17', '#681C15', '#5D1B14', '#521A13']
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#666666')

    def __rainbow_discrete(self, lut=None):
        """
        Define colormap 'rainbow_discrete'.
        """
        clrs = ['#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4',
                '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0',
                '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265',
                '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141',
                '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518',
                '#DC050C', '#A5170E', '#72190E', '#42150A']
        indexes = [[9], [9, 25], [9, 17, 25], [9, 14, 17, 25],
                   [9, 13, 14, 17, 25], [9, 13, 14, 16, 17, 25],
                   [8, 9, 13, 14, 16, 17, 25], [8, 9, 13, 14, 16, 17, 22, 25],
                   [8, 9, 13, 14, 16, 17, 22, 25, 27],
                   [8, 9, 13, 14, 16, 17, 20, 23, 25, 27],
                   [8, 9, 11, 13, 14, 16, 17, 20, 23, 25, 27],
                   [2, 5, 8, 9, 11, 13, 14, 16, 17, 20, 23, 25],
                   [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 20, 23, 25],
                   [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25],
                   [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
                   [2, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
                   [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25,
                    27],
                   [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25,
                    26, 27],
                   [1, 3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23,
                    25, 26, 27],
                   [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 21,
                    23, 25, 26, 27],
                   [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20,
                    22, 24, 25, 26, 27],
                   [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20,
                    22, 24, 25, 26, 27, 28],
                   [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18,
                    20, 22, 24, 25, 26, 27, 28]]
        if lut is None or lut < 1 or lut > 23:
            lut = 22
        self.cmap = discretemap(self.cname, [clrs[i] for i in indexes[lut-1]])
        if lut == 23:
            self.cmap.set_bad('#777777')
        else:
            self.cmap.set_bad('#FFFFFF')

    def show(self):
        """
        List names of defined colormaps.
        """
        print(' '.join(repr(n) for n in self.namelist))

    def get(self, cname='rainbow_PuRd', lut=None):
        """
        Return requested colormap, default is 'rainbow_PuRd'.
        """
        self.cname = cname
        if cname == 'rainbow_discrete':
            self.__rainbow_discrete(lut)
        else:
            self.funcdict[cname]()
        return self.cmap


def tol_cmap(colormap=None, lut=None):
    """
    Continuous and discrete color sets for ordered data.

    Return a matplotlib colormap.
    Parameter lut is ignored for all colormaps except 'rainbow_discrete'.
    """
    obj = TOLcmaps()
    if colormap is None:
        return obj.namelist
    if colormap not in obj.namelist:
        colormap = 'rainbow_PuRd'
        print('*** Warning: requested colormap not defined,',
              'known colormaps are {}.'.format(obj.namelist),
              'Using {}.'.format(colormap))
    return obj.get(colormap, lut)


def tol_cset(colorset=None):
    """
    Discrete color sets for qualitative data.

    Define a namedtuple instance with the colors.
    Examples for: cset = tol_cset(<scheme>)
      - cset.red and cset[1] give the same color (in default 'bright' colorset)
      - cset._fields gives a tuple with all color names
      - list(cset) gives a list with all colors
    """
    namelist = ('bright', 'high-contrast', 'vibrant', 'muted', 'light')
    if colorset is None:
        return namelist

    if colorset not in namelist:
        print('*** Warning: requested colorset not defined,',
              'known colorsets are {}.'.format(namelist),
              'Using {}.'.format(colorset))

    if colorset == 'high-contrast':
        cset = namedtuple('Hcset',
                          'blue yellow red black')
        return cset('#004488', '#DDAA33', '#BB5566', '#000000')

    if colorset == 'vibrant':
        cset = namedtuple('Vcset',
                          'orange blue cyan magenta red teal grey black')
        return cset('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
                    '#009988', '#BBBBBB', '#000000')

    if colorset == 'muted':
        cset = namedtuple('Mcset',
                          ('rose indigo sand green cyan wine teal'
                           ' olive purple pale_grey black'))
        return cset('#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                    '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD',
                    '#000000')

    if colorset == 'light':
        cset = namedtuple('Lcset',
                          ('light_blue orange light_yellow pink light_cyan'
                           ' mint pear olive pale_grey black'))
        return cset('#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
                    '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#000000')

    # Default: return colorset is 'bright'
    cset = namedtuple('Bcset',
                      'blue red green yellow cyan purple grey black')
    return cset('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                '#AA3377', '#BBBBBB', '#000000')


def main():
    """
    Show all available colormaps and colorsets
    """
    # Change default colorset (for lines) and colormap (for maps).
#    plt.rc('axes', prop_cycle=plt.cycler('color', list(tol_cset('bright'))))
#    plt.cm.register_cmap('rainbow_PuRd', tol_cmap('rainbow_PuRd'))
#    plt.rc('image', cmap='rainbow_PuRd')

    # Show colorsets tol_cset(<scheme>).
    schemes = tol_cset()
    fig, axes = plt.subplots(ncols=len(schemes))
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.0, right=0.91)
    for ax, scheme in zip(axes, schemes):
        cset = tol_cset(scheme)
        names = cset._fields
        colors = list(cset)
        for name, color in zip(names, colors):
            ax.scatter([], [], c=color, s=80, label=name)
        ax.set_axis_off()
        ax.legend(loc=2)
        ax.set_title(scheme)
    plt.show()

    # Show colormaps tol_cmap(<scheme>).
    schemes = tol_cmap()
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, axes = plt.subplots(nrows=len(schemes))
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.29, right=0.99)
    for ax, scheme in zip(axes, schemes):
        pos = list(ax.get_position().bounds)
        ax.set_axis_off()
        ax.imshow(gradient, aspect=4, cmap=tol_cmap(scheme))
        fig.text(pos[0] - 0.01, pos[1] + pos[3]/2.,
                 scheme, va='center', ha='right', fontsize=10)
    plt.show()

    # Show colormaps tol_cmap('rainbow_discrete', <lut>).
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, axes = plt.subplots(nrows=23)
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.25, right=0.99)
    for lut, ax in enumerate(axes, start=1):
        pos = list(ax.get_position().bounds)
        ax.set_axis_off()
        ax.imshow(gradient, aspect=4, cmap=tol_cmap('rainbow_discrete', lut))
        fig.text(pos[0] - 0.01, pos[1] + pos[3]/2.,
                 'rainbow_discrete, ' + str(lut),
                 va='center', ha='right', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
