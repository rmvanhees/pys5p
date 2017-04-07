"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Definition of alternative color schemes.

These schemes are developed by Paul Tol (SRON) to make graphics with your
scientific results as clear as possible, it is handy to have a palette of
colours that are:
 - distinct for all people, including colour-blind readers;
 - distinct from black and white;
 - distinct on screen and paper;
 - still match well together.

Reference
---------
   https://personal.sron.nl/~pault/

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from matplotlib.colors import LinearSegmentedColormap

class SRONcmaps(object):
    """
    Class SRONcmaps definition

    Defined color maps:
     'qualitative_6'     :  qualitative colormap
     'sequential_YlOrBr' :  sequential colormap
     'diverging_BuRd'    :  diverging colormap
     'diverging_PiGn'    :  diverging colormap
     'rainbow_PiRd'      :  rainbow colormap (default)
     'rainbow_PiBr'      :  rainbow colormap
     'rainbow_WhBr'      :  rainbow colormap (shows details for low values)
     'rainbow_WhRd'      :  rainbow colormap
    """
    def __init__(self):
        """
        """
        self.cmap = None
        self.cname = None
        self.namelist = (
            'qualitative_6', 'sequential_YlOrBr',
            'diverging_BuRd', 'diverging_PiGn',
            'rainbow_PiRd', 'rainbow_PiBr',
            'rainbow_WhBr', 'rainbow_WhRd')

        self.funcdict = dict(
            zip(self.namelist,
                (self.__qualitative_6, self.__sequential_YlOrBr,
                 self.__diverging_BuRd, self.__diverging_PiGn,
                 self.__rainbow_PiRd, self.__rainbow_PiBr,
                 self.__rainbow_WhBr, self.__rainbow_WhRd)))

    def __qualitative_6(self):
        """
        Define colormap "qualitative_6"
        """
        cmap_def = [[68, 119, 170], [102, 204, 238], [34, 136, 51],
                    [204, 187, 68], [238, 102, 119], [170, 51, 119],
                    [187, 187, 187], [0, 0, 0]]
        bad_def = [187, 187, 187]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __sequential_YlOrBr(self):
        """
        Define colormap "sequential_YlOrBr"
        """
        cmap_def = [[255, 255, 229], [255, 247, 188], [254, 227, 145],
                    [254, 196, 79], [251, 154, 41], [236, 112, 20],
                    [204, 76, 2], [153, 52, 4], [102, 37, 6]]
        bad_def = [136, 136, 136]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __diverging_BuRd(self):
        """
        Define colormap "diverging_BuRd"
        """
        cmap_def = [[33, 102, 172], [67, 147, 195], [146, 197, 222],
                    [209, 229, 240], [247, 247, 247], [253, 219, 199],
                    [244, 165, 130], [214, 96, 77], [178, 24, 43]]
        bad_def = [255, 238, 153]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __diverging_PiGn(self):
        """
        Define colormap "diverging_PiGn"
        """
        cmap_def = [[118, 42, 131], [153, 112, 171], [194, 165, 207],
                    [231, 212, 232], [247, 247, 247], [217, 240, 211],
                    [172, 211, 158], [90, 174, 97], [27, 120, 55]]
        bad_def = [255, 238, 153]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __rainbow_PiRd(self):
        """
        Define colormap "rainbow_PiRd"
        """
        #part_1 = [[232, 236, 251], [221, 216, 239], [209, 193, 225],
        #          [195, 168, 209], [181, 143, 194], [167, 120, 180],
        #          [155, 98, 167], [140, 78, 153]]
        part_2 = [[111, 76, 155], [96, 89, 169], [85, 104, 184], [78, 121, 197],
                  [77, 138, 198], [78, 150, 188], [84, 158, 179],
                  [89, 165, 169], [96, 171, 158], [105, 177, 144],
                  [119, 183, 125], [140, 188, 104], [166, 190, 84],
                  [190, 188, 72], [209, 181, 65], [221, 170, 60],
                  [228, 156, 57], [231, 140, 53], [230, 121, 50],
                  [228, 99, 45], [223, 72, 40], [218, 34, 34]]
        #part_3 = [[184, 34, 30], [149, 33, 27], [114, 30, 23], [82, 26, 19]]
        cmap_def = part_2
        bad_def = [255, 255, 255]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __rainbow_PiBr(self):
        """
        Define colormap "rainbow_PiBr"
        """
        #part_1 = [[232, 236, 251], [221, 216, 239], [209, 193, 225],
        #          [195, 168, 209], [181, 143, 194], [167, 120, 180],
        #          [155, 98, 167], [140, 78, 153]]
        part_2 = [[111, 76, 155], [96, 89, 169], [85, 104, 184],
                  [78, 121, 197], [77, 138, 198], [78, 150, 188],
                  [84, 158, 179], [89, 165, 169], [96, 171, 158],
                  [105, 177, 144], [119, 183, 125], [140, 188, 104],
                  [166, 190, 84], [190, 188, 72], [209, 181, 65],
                  [221, 170, 60], [228, 156, 57], [231, 140, 53],
                  [230, 121, 50], [228, 99, 45], [223, 72, 40], [218, 34, 34]]
        part_3 = [[184, 34, 30], [149, 33, 27], [114, 30, 23], [82, 26, 19]]
        cmap_def = part_2 + part_3
        bad_def = [255, 255, 255]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __rainbow_WhBr(self):
        """
        Define colormap "rainbow_WhBr"
        """
        part_1 = [[232, 236, 251], [221, 216, 239], [209, 193, 225],
                  [195, 168, 209], [181, 143, 194], [167, 120, 180],
                  [155, 98, 167], [140, 78, 153]]
        part_2 = [[111, 76, 155], [96, 89, 169], [85, 104, 184],
                  [78, 121, 197], [77, 138, 198], [78, 150, 188],
                  [84, 158, 179], [89, 165, 169], [96, 171, 158],
                  [105, 177, 144], [119, 183, 125], [140, 188, 104],
                  [166, 190, 84], [190, 188, 72], [209, 181, 65],
                  [221, 170, 60], [228, 156, 57], [231, 140, 53],
                  [230, 121, 50], [228, 99, 45], [223, 72, 40], [218, 34, 34]]
        part_3 = [[184, 34, 30], [149, 33, 27], [114, 30, 23], [82, 26, 19]]
        cmap_def = part_1 + part_2 + part_3
        bad_def = [119, 119, 119]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def __rainbow_WhRd(self):
        """
        Define colormap "rainbow_WhRd"
        """
        part_1 = [[232, 236, 251], [221, 216, 239], [209, 193, 225],
                  [195, 168, 209], [181, 143, 194], [167, 120, 180],
                  [155, 98, 167], [140, 78, 153]]
        part_2 = [[111, 76, 155], [96, 89, 169], [85, 104, 184],
                  [78, 121, 197], [77, 138, 198], [78, 150, 188],
                  [84, 158, 179], [89, 165, 169], [96, 171, 158],
                  [105, 177, 144], [119, 183, 125], [140, 188, 104],
                  [166, 190, 84], [190, 188, 72], [209, 181, 65],
                  [221, 170, 60], [228, 156, 57], [231, 140, 53],
                  [230, 121, 50], [228, 99, 45], [223, 72, 40], [218, 34, 34]]
        #part_3 = [[184, 34, 30], [149, 33, 27], [114, 30, 23], [82, 26, 19]]
        cmap_def = part_1 + part_2
        bad_def = [119, 119, 119]
        self.cmap = LinearSegmentedColormap.from_list(self.cname,
                                                      np.array(cmap_def) / 256.)
        self.cmap.set_bad(np.array(bad_def) / 256., 1.)

    def show(self):
        """
        lists names of defined colormaps
        """
        print(' '.join(repr(n) for n in self.namelist))

    def get(self, cname='rainbow_PiRd'):
        """
        returns requested colormap

        default is 'rainbow_PiRd'
        """
        self.cname = cname
        self.funcdict[cname]()
        return self.cmap

def sron_cmap(colormap):
    """
    Defines the public function which returns a SRON Matplotlib color-map
    """
    obj = SRONcmaps()
    if colormap not in obj.namelist:
        colorname = obj.namelist[5]
        print('*** Warning: requested color map not defined',
              ' known color maps are: {}.'.format(obj.namelist),
              'Using as default {}'.format(colormap))
    
    return obj.get(colormap)

def get_line_colors():
    """
    Alternative color scheme for qualitative data

    Defines 8 colors: Blue, Cyan, Green, Yellow, Red, Pink, Grey and Black
    Usage:
      - For 5 or more colors: use colors as defined in list
      - For 4 or less: use Blue (0), Red(4), Green(2) and Yellow(3)
    """
    return [ '#4477AA',   # blue
             '#66CCEE',   # cyan
             '#228833',   # green
             '#CCBB44',   # yellow
             '#EE6677',   # red
             '#AA3377',   # pink
             '#BBBBBB',   # grey
             '#000000' ]  # black