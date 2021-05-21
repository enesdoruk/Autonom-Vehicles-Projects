root_data_vehicle = './data/dataset/vehicles'

root_data_non_vehicle = './data/dataset/non-vehicles'

feat_extraction_params = {'resize_h': 64,           
                          'resize_w': 64,           
                          'color_space': 'YCrCb',     
                          'orient': 9,                
                          'pix_per_cell': 8,        
                          'cell_per_block': 2,       
                          'hog_channel': "ALL",       
                          'spatial_size': (32, 32),   
                          'hist_bins': 16,         
                          'spatial_feat': True,      
                          'hist_feat': True,        
                          'hog_feat': True}          







