"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on S5Pplot methods: draw_signal, draw_quality,
   draw_cmp_images, draw_trend1d and draw_lines (xarray version)

Copyright (c) 2020-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta
import numpy as np
import xarray as xr

from pys5p.lib.plotlib import FIGinfo
from pys5p.s5p_xarray import data_to_xr
from pys5p.s5p_plot import S5Pplot


def get_test_data(data_sel=None, xy_min=-5, xy_max=5, delta=0.01, error=0):
    """
    Generate synthetic data to simulate a square-detector image
    """
    if data_sel is None:
        data_sel = [(), ()]

    res = np.arange(xy_min, xy_max, delta)
    xmesh, ymesh = np.meshgrid(res[data_sel[1]], res[data_sel[0]])
    zz1 = np.exp(-xmesh ** 2 - ymesh ** 2)
    zz2 = np.exp(-(xmesh - 1) ** 2 - (ymesh - 1) ** 2)
    data = (zz1 - zz2) * 2
    data += np.random.default_rng().normal(0., error, data.shape)

    return data_to_xr(data, long_name='bogus data', units='Volt')


def run_draw_signal(plot):
    """
    Run unit tests on S5Pplot::draw_signal
    """
    msm = get_test_data(error=.1)
    # msm_ref = get_test_data(error=0.025)

    nlines = 40 # 35
    fig_info_in = FIGinfo('right')
    for ii in range(nlines):
        fig_info_in.add(f'line {ii:02d}', 5 * '0123456789')

    plot.draw_signal(msm,
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='method=data; aspect=1; fig_pos=above')
    plot.draw_signal(msm, zscale='diff',
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='method=diff; aspect=1; fig_pos=above')
    plot.draw_signal(msm, zscale='ratio',
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='method=ratio; aspect=1; fig_pos=above')
    #plot.draw_signal(msm, zscale='log',
    #                 title='Unit test of S5Pplot [draw_signal]',
    #                 sub_title='method=error; aspect=1; fig_pos=above')
    plot.draw_signal(msm, fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=1; fig_pos=right')

    msm = get_test_data(data_sel=np.s_[500-250:500+250, :], error=.1)
    plot.draw_signal(msm,
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=2; fig_pos=above')
    plot.draw_signal(msm, fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=2; fig_pos=right')

    msm = get_test_data(data_sel=np.s_[500-125:500+125, 500-375:500+375],
                        error=.1)
    plot.draw_signal(msm,
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=3; fig_pos=above')
    plot.draw_signal(msm,
                     fig_info=fig_info_in.copy(),
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=3; fig_pos=right')

    msm = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.1)
    plot.draw_signal(msm,
                     title='Unit test of S5Pplot [draw_signal]',
                     sub_title='aspect=4; fig_pos=above')
    plot.draw_signal(msm, fig_info=fig_info_in.copy(),
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

    msm = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.1)
    msm_ref = get_test_data(data_sel=np.s_[500-128:500+128, :], error=.025)

    plot.draw_cmp_swir(msm, msm_ref.values,
                       title='Unit test of S5Pplot [draw_cmp_images]',
                       sub_title='test image')
    plot.draw_cmp_swir(msm, msm_ref.values, add_residual=False,
                       title='Unit test of S5Pplot [draw_cmp_images]',
                       sub_title='test image')
    plot.draw_cmp_swir(msm, msm_ref.values, add_model=False,
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
    hk_min = np.zeros(200, dtype=hk_dtype)
    hk_avg = np.zeros(200, dtype=hk_dtype)
    hk_max = np.zeros(200, dtype=hk_dtype)
    data = 140. + (100 - np.arange(200)) / 1000
    hk_min['detector_temp'][:] = data - .0125
    hk_avg['detector_temp'][:] = data
    hk_max['detector_temp'][:] = data + .0075
    data = 202.1 + (100 - np.arange(200)) / 1000
    hk_min['grating_temp'][:] = data - .15
    hk_avg['grating_temp'][:] = data
    hk_max['grating_temp'][:] = data + .175
    data = 208.2 + (100 - np.arange(200)) / 1000
    hk_min['obm_temp'][:] = data - .15
    hk_avg['obm_temp'][:] = data
    hk_max['obm_temp'][:] = data + .175
    units = [parm[2] for parm in hk_params]
    long_name = [parm[1] for parm in hk_params]

    msm_mean = data_to_xr(hk_avg, dims=['orbit'], name='hk_mean',
                          long_name=long_name, units=units)
    msm_range = data_to_xr(np.stack([hk_min, hk_max], axis=1),
                           dims=['orbit', 'range'], name='hk_range',
                           long_name=long_name, units=units)
    hk_ds = xr.merge([msm_mean, msm_range])

    msm1 = data_to_xr(np.sin(xx * np.pi), dims=['orbit'])
    msm2 = data_to_xr(np.cos(xx * np.pi), dims=['orbit'])

    plot.draw_trend1d(msm1,
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='one dataset, no house-keeping')
    plot.draw_trend1d(msm1, msm2=msm2,
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='two datasets, no house-keeping')
    hk_keys = [parm[0] for parm in hk_params]
    plot.draw_trend1d(msm1, msm2=msm2,
                      hk_data=hk_ds, hk_keys=hk_keys[0:2],
                      title='Unit test of S5Pplot [draw_trend1d]',
                      sub_title='two datasets and house-keeping')
    plot.draw_trend1d(msm1, msm2=msm2,
                      hk_data=hk_ds, hk_keys=hk_keys,
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
    frame = np.concatenate((buff0, buff1, buff2, buff3, buff4,
                            buff5, buff6, buff7, buff8, buff9,
                            buffa)).reshape(256, 1000)
    msm = xr.merge([data_to_xr(frame, name='dpqm',
                               long_name='pixel-quality map'),
                    data_to_xr(frame, name='dpqm_dark',
                               long_name='pixel-quality map (dark)'),
                    data_to_xr(frame, name='dpqm_noise',
                               long_name='pixel-quality map (noise average)'),
                    data_to_xr(frame, name='dpqm_noise_var',
                               long_name='pixel-quality map (noise variance)')])

    plot.draw_qhist(msm, title='Unit test of S5Pplot [draw_qhist]')


# --------------------------------------------------
def main():
    """
    main function
    """
    plot = S5Pplot('unit_test_s5p_plot2.pdf')
    check_draw_signal = True
    check_draw_cmp_images = True
    check_draw_quality = True
    check_draw_qhist = True
    check_draw_trend1d = True
    check_draw_lines = True

    # ---------- UNIT TEST: draw_signal ----------
    if check_draw_signal:
        run_draw_signal(plot)

    # ---------- UNIT TEST: draw_cmp_images ----------
    if check_draw_cmp_images:
        run_draw_cmp_swir(plot)

    # ---------- UNIT TEST: draw_quality ----------
    if check_draw_quality:
        run_draw_quality(plot)

    # ---------- UNIT TEST: draw_qhist ----------
    if check_draw_qhist:
        run_draw_qhist(plot)

    # ---------- UNIT TEST: draw_trend1d ----------
    if check_draw_trend1d:
        run_draw_trend1d(plot)

    # ---------- UNIT TEST: draw_lines ----------
    if check_draw_lines:
        run_draw_lines(plot)

    # close figure
    plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
