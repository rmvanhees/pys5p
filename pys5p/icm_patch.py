'''
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Methods to simulate SWIR calibration data for patching of ICM_CA_SIR product

Simulated products are derived as follows:
 a) Background:
     - derive from offset and dark CKD
     - requires  : exposure time & co-adding factor
     - TBD: what to do with orbital variation?
 b) DLED:
     - derive from OCAL measurements (email Paul 22-Juni-2016 14:30)
     - requires  : exposure time & co-adding factor     
 c) WLS:
     - derive from OCAL measurements (email Paul 22-Juni-2016 14:30)
     - requires  : exposure time & co-adding factor
 d) SLS (ISRF):
     - derive from OCAL measurements
 e) Irradiance:   [low priority]
     - using level 2 simulations?
 f) Radiance:     [low priority]
     - using level 2 simulations?

How to use the class ICM_patch:
 1) patch particular measurements in an ICM product. Use the class ICM_io, 
select a dataset to patch, and write simulated calibration data
 2) patch as much as possible measurement datasets in an ICM product, probably
restricted to the groups "BAND%_CALIBRATION". The master script has to decide 
on the names (and or processing classes) which type of patch it has to apply
 3) patch all measurement datasets of a particular type in an ICM product. The 
advantage is that the type of patch is known.

Nearly every method return a tuple with the patch values and its errors. For 
now, the errors are only a very rough estimate.

Remarks:
 - An alternative way to pach the background measurements is to use OCAL data, 
this is more complicated and not all exposure time/coadding factor combinations 
are like to be available, however, the may lead to more realistic data and 
errors.
 - Data is not always correctly calibrated...

Copyright (c) 2016 SRON - Netherlands Institute for Space Research 
   All Rights Reserved

License:  Standard 3-clause BSD

'''
import os.path

import numpy as np
import h5py

class ICM_patch( object ):
    '''
    '''
    def background( self, exposure_time, coadding_factor ):
        ckd_dir = '/nfs/TROPOMI/ocal/ckd/ckd_release_swir'

        # read v2c CKD
        ckd_file = os.path.join(ckd_dir, 'v2c', 'ckd.v2c_factor.detector4.nc')
        with h5py.File( ckd_file, 'r' ) as fid:
            dset = fid['/BAND7/v2c_factor_swir']
            v2c_b7 = dset[...]

        v2c_swir = v2c_b7[0]['value']

        # read offset CKD
        ckd_file = os.path.join(ckd_dir, 'offset', 'ckd.offset.detector4.nc')
        with h5py.File( ckd_file, 'r' ) as fid:
            dset = fid['/BAND7/analog_offset_swir']
            offs_b7 = dset[:,:]
            dset = fid['/BAND8/analog_offset_swir']
            offs_b8 = dset[:,:]

        offset_swir = np.hstack((offs_b7, offs_b8))
    
        # read dark CKD
        ckd_file = os.path.join(ckd_dir, 'darkflux', 'ckd.dark.detector4.nc')
        with h5py.File( ckd_file, 'r' ) as fid:
            dset = fid['/BAND7/long_term_swir']
            dark_b7 = dset[:,:]
            dset = fid['/BAND8/long_term_swir']
            dark_b8 = dset[:,:]

        dark_swir = np.hstack((dark_b7, dark_b8))

        background = offset_swir['value'] * v2c_swir
        background += dark_swir['value'] * exposure_time
        background *= coadding_factor

        # fixed value at +/- 3 BU
        error = 0.0 * offset_swir['error'] + 135 / np.sqrt(coadding_factor)

        return (background, error)
        

    def dark( self, parms ):
        pass

    def dled( self, exposure_time, coadding_factor ):
        '''
        The DLED signal can be aproximated by the background signal and the
        signal-current of the DLED. Kindly provided by Paul Tol
        '''
        (signal, error) = self.background( exposure_time, 1 )

        dled_dir = '/data/richardh/Tropomi'
        dled_file = os.path.join(dled_dir, 'DledlinSw_signalcurrent_approx.h5')
        with h5py.File( dled_file, 'r' ) as fid:
            dset = fid['dled_signalcurrent_epers']
            dled_current = dset[:,:]

        signal += dled_current * exposure_time
        signal *= coadding_factor

        return (signal, error)

    def sls( self, ld_id ):
        '''
        Measurement names: 
          - SLS_MODE_0602 for ISRF LD01
            -> 2015_02_25T05_16_36_LaserDiodes_LD1_100
          - SLS_MODE_0604 for ISRF LD02
            -> 2015_02_25T16_38_20_LaserDiodes_LD2_100
          - SLS_MODE_0606 for ISRF LD03
            -> 2015_02_27T14_56_27_LaserDiodes_LD3_100
          - SLS_MODE_0608 for ISRF LD04
            -> 2015_02_27T16_58_47_LaserDiodes_LD4_100
          - SLS_MODE_0610 for ISRF LD05
            -> 2015_02_28T12_02_01_LaserDiodes_LD5_100

        Note:
          - The OCAL diode-laser measurements are processed from L0-1b, but not
          calibrated (proc_raw)!!!
          - The saved columns are defined in /PROCESSOR/goals_configuration,
          requires XML support to read this information
            * I have now used a fixed column selection for each diode-laser.
        '''
        from pynadc.stats import biweight 
        
        assert (ld_id > 0 and ld_id < 6)
        
        light_icid = 32096
        ocal_dir = '/nfs/TROPOMI/ocal/proc_raw'
        if ld_id == 1:
            band = 7
            columns = [443, 495]
            data_dir = os.path.join( ocal_dir,
                                     '2015_02_25T05_16_36_LaserDiodes_LD1_100',
                                     'proc_raw' )
            data_fl = 'trl1brb7g.lx.nc'
        elif ld_id == 2:
            band = 8
            columns = [285, 337]
            data_dir = os.path.join( ocal_dir,
                                     '2015_02_25T16_38_20_LaserDiodes_LD2_100',
                                     'proc_raw' )
            data_fl = 'trl1brb8g.lx.nc'
        elif ld_id == 3:
            band = 7
            columns = [312, 364]
            data_dir = os.path.join( ocal_dir,
                                     '2015_02_27T14_56_27_LaserDiodes_LD3_100',
                                     'proc_raw' )
            data_fl = 'trl1brb7g.lx.nc'
        elif ld_id == 4:
            band = 8
            columns = [130, 182]
            data_dir = os.path.join( ocal_dir,
                                     '2015_02_27T16_58_47_LaserDiodes_LD4_100',
                                     'proc_raw' )
            data_fl = 'trl1brb8g.lx.nc'
        elif ld_id == 5:
            band = 7
            columns = [125, 177]
            data_dir = os.path.join( ocal_dir,
                                     '2015_02_28T12_02_01_LaserDiodes_LD5_100',
                                     'proc_raw' )
            data_fl = 'trl1brb7g.lx.nc'

        # obtain start and end of measurement from engineering data
        data = {}
        with h5py.File( os.path.join(data_dir, 'engDat.nc'), 'r' ) as fid:
            gid = fid['/NOMINAL_HK/HEATERS']
            dset = gid['peltier_info']
            data['delta_time'] = dset[:, 'delta_time']
            data['icid'] = dset[:, 'icid']
            for ii in range(5):
                keyname = 'last_cmd_curr{}'.format(ii)
                buff = dset[:, keyname]
                if not np.all( buff == 0 ):
                    data['last_cmd_curr'] = buff
                    sls_id = ii+1
                    break
            assert sls_id == ld_id

            u = np.unique( data['last_cmd_curr'] )
            pcurr_min = u[1]
            i_mn = np.min( np.where( (data['icid'] == light_icid)
                                     & (data['last_cmd_curr'] > pcurr_min) ))
            i_mx = np.max( np.where( (data['icid'] == light_icid)
                                     & (data['last_cmd_curr'] > pcurr_min) ))
            delta_time_mn = data['delta_time'][i_mn]
            delta_time_mx = data['delta_time'][i_mx]

        # read measurements with diode-laser scanning
        with h5py.File( os.path.join(data_dir, data_fl), 'r' ) as fid:
            path = 'BAND{}/ICID_{}_GROUP_00001'.format(band, light_icid)
            dset = fid[path + '/GEODATA/delta_time']
            delta_time = dset[:]
            framelist = np.where( (delta_time >= delta_time_mn)
                                  & (delta_time <= delta_time_mx) )[0]
            dset = fid[path + '/OBSERVATIONS/signal']
            signal = dset[framelist[0]:framelist[-1]+1,:,columns[0]:columns[1]]

        # read background measurements
        with h5py.File( os.path.join(data_dir, data_fl), 'r' ) as fid:
            path = 'BAND{}/ICID_{}_GROUP_00000'.format(band, light_icid-1)
            dset = fid[path + '/OBSERVATIONS/signal']
            (background, background_std) = biweight( dset[1:,:,:],
                                                     axis=0, spread=True )

        # need to read background data of other band!!
        background = np.hstack( (background, background) )
        background_std = np.hstack( (background_std, background_std) )
        
        return (signal, background, background_std, columns)

    def wls( self, exposure_time, coadding_factor ):
        '''
        The WLS signal is appriximated as the DLED signal, therefore, 
        it can be aproximated by the background signal and the
        signal-current of the DLED. Kindly provided by Paul Tol
        '''
        (signal, error) = self.background( exposure_time, 1 )

        dled_dir = '/data/richardh/Tropomi'
        dled_file = os.path.join(dled_dir, 'DledlinSw_signalcurrent_approx.h5')
        with h5py.File( dled_file, 'r' ) as fid:
            dset = fid['dled_signalcurrent_epers']
            dled_current = dset[:,:]

        signal += dled_current * exposure_time
        signal *= coadding_factor

        return (signal, error)

    def irradiance( self, parms ):
        pass

    def radiance( self, parms ):
        pass

#--------------------------------------------------
def test():
    '''
    Perform some simple test to check the ICM_io and ICM_path classes
    '''
    import shutil
    
    from pynadc.tropomi import db
    from pynadc.tropomi.icm_io import ICM_io
    
    res = db.get_product_by_icid(icid=[610,1605])
    assert len(res) > 0
    
    data_dir = res[0][0]
    temp_dir = '/tmp'
    icm_file = res[0][1]
    tmp_file = icm_file.replace('_01_', '_02_')
    print( os.path.join(data_dir, icm_file) )
    
    patch = ICM_patch()
    shutil.copy( os.path.join(data_dir, icm_file),
                 os.path.join(temp_dir, tmp_file) )
    fp = ICM_io( os.path.join(temp_dir, tmp_file),
                 verbose=True, readwrite=True )
    
    fp.select( 'DARK_MODE_1605' )
    res_dark = patch.background( fp.instrument_settings['exposure_time'],
                                 fp.instrument_settings['nr_coadditions'] )
    res = { 'signal_avg' : np.split(res_dark[0], 2, axis=1),
            'signal_avg_std' : np.split(res_dark[1], 2, axis=1) }
    fp.set_data( res )
    
    fp.select( 'SLS_MODE_0610' )
    for ii in range(5):
        sls_id = ii+1
        nr_valid = np.sum(fp.housekeeping_data['sls{}_status'.format(ii+1)])
        if nr_valid > 0:
            break
        
    res_sls = patch.sls( sls_id )
    print( 'SLS values:     ', res_sls[0].shape )
    print( 'SLS background: ', res_sls[1].shape )
    print( 'SLS errors:     ', res_sls[2].shape )
    res = { 'det_lit_area_signal' : res_sls[0][:fp.delta_time.shape[0],:,:] }
    fp.set_data( res )
    
    fp.select( 'BACKGROUND_MODE_0609' )
    res = { 'signal_avg'     : np.split(res_sls[1], 2, axis=1),
            'signal_avg_std' : np.split(res_sls[2], 2, axis=1) }
    fp.set_data( res )
    del( fp )
    del( patch )

if __name__ == '__main__':
    test()
