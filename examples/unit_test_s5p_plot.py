"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on S5Pplot methods: draw_signal, draw_quality,
   draw_cmp_images, draw_trend1d and draw_lines

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta
import numpy as np

from pys5p.lib.plotlib import FIGinfo
from pys5p.s5p_msm import S5Pmsm
from pys5p.s5p_plot import S5Pplot


# --------------------------------------------------
def run_draw_signal(plot):
    """
    Run unit tests on S5Pplot::draw_signal
    """
    print('Run unit tests on S5Pplot::draw_signal')
    delta = 0.01
    xx = yy = np.arange(-5.0, 5.0, delta)
    xmesh, ymesh = np.meshgrid(xx, yy)
    zz1 = np.exp(-xmesh ** 2 - ymesh ** 2)
    zz2 = np.exp(-(xmesh - 1) ** 2 - (ymesh - 1) ** 2)
    data = (zz1 - zz2) * 2

    msm = S5Pmsm(data)
    msm.set_units('Volt')
    msm.set_long_name('bogus data')

    plot.draw_signal(msm,
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=1; fig_pos=above')

    plot.draw_signal(data[500-250:500+250, :],
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=2; fig_pos=above')

    plot.set_zunit('electron.s-1')
    plot.draw_signal(data[500-125:500+125, 500-375:500+375],
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=3; fig_pos=above')

    plot.unset_zunit()
    plot.draw_signal(data[500-128:500+128, :],
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=4; fig_pos=above')

    nlines = 40 # 35
    fig_info_in = FIGinfo('right')
    for ii in range(nlines):
        fig_info_in.add('line {:02d}'.format(ii), 5 * '0123456789')
    plot.draw_signal(data, fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=1; fig_pos=right')

    plot.draw_signal(data[500-250:500+250, :], fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=2; fig_pos=right')

    plot.draw_signal(data[500-125:500+125, 500-375:500+375],
                     fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=3; fig_pos=right')

    plot.draw_signal(data[500-128:500+128, :], fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=4; fig_pos=right')


def run_draw_quality(plot):
    """
    Run unit tests on S5Pplot::draw_quality
    """
    print('Run unit tests on S5Pplot::draw_quality')
    row = np.linspace(0, 1., 1000)
    ref_data = np.repeat(row[None, :], 256, axis=0)

    data = ref_data.copy()
    data[125:175, 30:50] = 0.4
    data[125:175, 50:70] = 0.95
    data[75:125, 250:300] = 0.05
    data[125:175, 600:650] = 0.95
    data[75:125, 900:925] = 0.05
    data[75:125, 825:875] = 0.4

    plot.draw_quality(data,
                      title='Unit test of S5Pplot [draw_quality]',
                      sub_title='no reference')

    plot.draw_quality(data, ref_data=ref_data,
                      title='Unit test of S5Pplot [draw_quality]',
                      sub_title='with reference')


def run_draw_cmp_swir(plot):
    """
    Run unit tests on S5Pplot::draw_cmp_swir
    """
    print('Run unit tests on S5Pplot::draw_cmp_swir')
    delta = 0.01
    xx = yy = np.arange(-5.0, 5.0, delta)
    xmesh, ymesh = np.meshgrid(xx, yy)
    zz1 = np.exp(-xmesh ** 2 - ymesh ** 2)
    zz2 = np.exp(-(xmesh - 1) ** 2 - (ymesh - 1) ** 2)
    data = (zz1 - zz2) * 2

    data = data[500-128:500+128, :]
    ref_data = 1.001 * data

    plot.draw_cmp_swir(data, ref_data,
                       title='Unit test of S5Pplot [draw_cmp_images]',
                       sub_title='test image')
    plot.draw_cmp_swir(data, ref_data, add_residual=False,
                       title='Unit test of S5Pplot [draw_cmp_images]',
                       sub_title='test image')
    plot.draw_cmp_swir(data, ref_data, add_model=False,
                       title='Unit test of S5Pplot [draw_cmp_images]',
                       sub_title='test image')


def run_draw_trend1d(plot):
    """
    Run unit tests on S5Pplot::draw_trend1d
    """
    print('Run unit tests on S5Pplot::draw_trend1d')
    xx = np.arange(200) / 100
    hk_params = [
        ('detector_temp', 'SWIR detector temperature', 'K', np.float32),
        ('grating_temp', 'SWIR grating temperature', 'K', np.float32),
        ('obm_temp', 'SWIR OBM temperature', 'K', np.float32)]
    hk_dtype = np.dtype(
        [(parm[0], parm[3]) for parm in hk_params])
    hk_data = S5Pmsm(np.zeros(200, dtype=hk_dtype))
    hk_data.error = np.zeros((200, 2), dtype=hk_dtype)
    data = 140. + (100 - np.arange(200)) / 1000
    hk_data.value['detector_temp'][:] = data
    hk_data.error['detector_temp'][:, 0] = data - .0125
    hk_data.error['detector_temp'][:, 1] = data + .0075
    data = 202.1 + (100 - np.arange(200)) / 250
    hk_data.value['grating_temp'][:] = data
    hk_data.error['grating_temp'][:, 0] = data - .15
    hk_data.error['grating_temp'][:, 1] = data + .175
    data = 208.2 + (100 - np.arange(200)) / 750
    hk_data.value['obm_temp'][:] = data
    hk_data.error['obm_temp'][:, 0] = data - .15
    hk_data.error['obm_temp'][:, 1] = data + .175
    hk_data.set_units([parm[2] for parm in hk_params])
    hk_data.set_long_name([parm[1] for parm in hk_params])

    plot.draw_trend1d(S5Pmsm(np.sin(xx * np.pi)),
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='one dataset, no house-keeping')
    plot.draw_trend1d(S5Pmsm(np.sin(xx * np.pi)),
                      msm2=S5Pmsm(np.cos(xx * np.pi)),
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='two datasets, no house-keeping')
    hk_keys = [parm[0] for parm in hk_params]
    plot.draw_trend1d(S5Pmsm(np.sin(xx * np.pi)),
                      hk_data=hk_data, hk_keys=hk_keys[0:2],
                      msm2=S5Pmsm(np.cos(xx * np.pi)),
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='two datasets and house-keeping')
    plot.draw_trend1d(S5Pmsm(np.sin(xx * np.pi)),
                      hk_data=hk_data, hk_keys=hk_keys,
                      msm2=S5Pmsm(np.cos(xx * np.pi)),
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='two datasets and house-keeping')


def run_draw_lines(plot):
    """
    Run unit tests on S5Pplot::draw_lines
    """
    print('Run unit tests on S5Pplot::draw_lines')
    xx = np.arange(200) / 100
    plot.draw_lines(xx, np.sin(xx * np.pi), color=0,
                    label='sinus', marker='o', linestyle='-')
    plot.draw_lines(xx, np.cos(xx * np.pi), color=1,
                    label='cosinus', marker='o', linestyle='-')
    plot.draw_lines(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='Unit test of S5Pplot [draw_lines]',
                    sub_title='draw_lines [no time_axis]')

    xx = np.arange(500) / 100
    plot.draw_lines(xx, np.sin(xx * np.pi), color=0,
                    label='sinus', marker='o', linestyle='-')
    plot.draw_lines(xx, np.cos(xx * np.pi), color=1,
                    label='cosinus', marker='o', linestyle='-')
    plot.draw_lines(None, None, ylim=[-1.05, 1.05],
                    xlabel='x-axis [Pi]', ylabel='y-axis',
                    title='Unit test of S5Pplot [draw_lines]',
                    sub_title='draw_lines [no time_axis]')

    customdate = datetime(2016, 1, 1, 13, 0, 0)
    yy = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    xx = [customdate + timedelta(hours=i, minutes=4*i)
          for i in range(len(yy))]
    plot.draw_lines(xx, yy, color=0, label='mydata',
                    marker='o', linestyle='-')
    plot.draw_lines(None, None, title='Unit test of S5Pplot [draw_lines]',
                    xlabel='x-axis', ylabel='y-axis',
                    sub_title='draw_lines [time_axis]')

def run_draw_qhist(plot):
    """
    Run unit tests on S5Pplot::draw_qhist
    """
    print('Run unit tests on S5Pplot::draw_qhist')
    buff0 = np.repeat(0.9 + np.random.rand(1000) / 10, 56)
    buff1 = np.repeat(np.random.rand(1000) / 10, 10)
    buff2 = 0.1 + np.random.rand(1000) / 10
    buff3 = np.repeat(0.2 + np.random.rand(1000) / 10, 2)
    buff4 = np.repeat(0.3 + np.random.rand(1000) / 10, 3)
    buff5 = np.repeat(0.4 + np.random.rand(1000) / 10, 4)
    buff6 = np.repeat(0.5 + np.random.rand(1000) / 10, 8)
    buff7 = np.repeat(0.6 + np.random.rand(1000) / 10, 12)
    buff8 = np.repeat(0.7 + np.random.rand(1000) / 10, 20)
    buff9 = np.repeat(0.8 + np.random.rand(1000) / 10, 40)
    buffa = np.repeat(0.9 + np.random.rand(1000) / 10, 100)
    data = S5Pmsm(np.concatenate((buff0, buff1, buff2, buff3, buff4,
                                  buff5, buff6, buff7, buff8, buff9,
                                  buffa)).reshape(256, 1000))
    data_dict = {}
    data.set_long_name('pixel-quality map')
    data_dict['dpqm'] = data.copy()
    data.set_long_name('pixel-quality map (dark)', force=True)
    data_dict['dpqm_dark'] = data.copy()
    data.set_long_name('pixel-quality map (noise average)', force=True)
    data_dict['dpqm_noise'] = data.copy()
    data.set_long_name('pixel-quality map (noise variance)', force=True)
    data_dict['dpqm_noise_var'] = data
    plot.draw_qhist(data_dict,
                    title='Unit test of S5Pplot [draw_qhist]')


# --------------------------------------------------
def main():
    """
    main function
    """
    plot = S5Pplot('unit_test_s5p_plot.pdf')
    check_draw_signal = True
    check_draw_quality = True
    check_draw_cmp_images = True
    check_draw_trend1d = True
    check_draw_lines = True
    check_draw_qhist = True

    # ---------- UNIT TEST: draw_signal ----------
    if check_draw_signal:
        run_draw_signal(plot)

    # ---------- UNIT TEST: draw_quality ----------
    if check_draw_quality:
        run_draw_quality(plot)

    # ---------- UNIT TEST: draw_cmp_images ----------
    if check_draw_cmp_images:
        run_draw_cmp_swir(plot)

    # ---------- UNIT TEST: draw_lines ----------
    if check_draw_trend1d:
        run_draw_trend1d(plot)

    # ---------- UNIT TEST: draw_lines ----------
    if check_draw_lines:
        run_draw_lines(plot)

    # ---------- UNIT TEST: draw_qhist ----------
    if check_draw_qhist:
        run_draw_qhist(plot)

    # close figure
    plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
