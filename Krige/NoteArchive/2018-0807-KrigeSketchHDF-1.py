# /usr/bin/env python


print('Starting save HDF test')
# Save an HDF file
# TODO The following is currently broken
# TODO Apparently SWATH does not mean irregular.
if True:

    # Now, we use tgt_X1d and tgt_Y1d
    ny = tgt_Y1d.size
    nx = tgt_X1d.size

    # The following, no.
    x    = np.zeros((ny,nx)); x.fill(np.nan)
    y    = np.zeros((ny,nx)); y.fill(np.nan)

    # Yes, the following.
    z    = np.zeros((ny,nx)); z.fill(np.nan)
    s    = np.zeros((ny,nx)); 
    nans = np.zeros((ny,nx)); nans.fill(np.nan)

    # The following is incorrect.
    i=0
    for kr in krigeSketch_results:
        for k in range(kr.x.size):
            tgt_x_idx = np.where(tgt_X1d == kr.x[k])
            tgt_y_idx = np.where(tgt_Y1d == kr.y[k])

            if (len(tgt_x_idx) != 1) or (len(tgt_y_idx) != 1):
                print('*** tgt_?_idx error. ')
                print('*** tgt_x_idx = '+str(tgt_x_idx))
                print('*** tgt_y_idx = '+str(tgt_y_idx))
                print('*** skipping')
            else:
                x[tgt_y_idx[0],tgt_x_idx[0]] = kr.x[k]
                y[tgt_y_idx[0],tgt_x_idx[0]] = kr.y[k]
                z[tgt_y_idx[0],tgt_x_idx[0]] = kr.z[k]
                s[tgt_y_idx[0],tgt_x_idx[0]] = kr.s[k]
        
    variable_name   = krigeSketch_results[-1].zVariableName
    output_filename = "KrigeSketch_"+modis_obj.datafilename+".hdf"
    if '.hdf.hdf' in output_filename[-8:]:
        output_filename = output_filename[:-4]

    kHDF = Krige.krigeHDF(\
                          krg_name                 = variable_name+'_krg'\
                          ,krg_units               = modis_obj.units\
                          ,config                  = krigeSketch_results[-1].config\
                          ,krg_z                   = z
                          ,krg_s                   = s
                          ,krg_x                   = tgt_X1d
                          ,krg_y                   = tgt_Y1d
                          ,orig_name               = modis_obj.datafieldname\
                          ,orig_units              = modis_obj.units\
                          ,orig_z                  = nans
                          ,orig_x                  = tgt_X1d
                          ,orig_y                  = tgt_Y1d
                          ,output_filename         = output_filename\
                          ,redimension             = False\
                          ,type_hint               = 'grid'\
    )
        
    kHDF.save()
    
print('Save HDF test done.')
