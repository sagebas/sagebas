# -*- coding: utf-8 -*-
"""
Makes a real-time forecast of the external magnetic field perturbation (i.e. 
 excluding the internal field contribution) at the three British Geological 
 Survey magnetometer stations: Eskdalemuir, Hartland and Lerwick.
 
 Output file column format is as follows.
  Day-Month-Year    Hour:Minute    [Model Type Flag]    Esk x    Esk y    Esk z    Had x    Had y    Had z    Ler x    Ler y    Ler z

 Where: 
 'Esk' is the observatory Eskdalemuir, 'Har' is Hartland, 'Ler' is 
  Lerwick, and 'x','y','z' are the ground geomagnetic field perturbation 
  components.
 The Model Type Flag has the following meanings.
  0: no problems: model forecast is based on the epsilon coupling function.
  1: plasma data were not available for this epoch of the real-time soalr wind 
      stream, so the forcast is based on just the magnetometer data, using the
      coupling function (IMF_magnitude^3) * (sin(clock_angle/2)^4).
  2: no solar wind data were available, so the output is nan.

 User notes.
 -- Don't compute dB/dt when there is a flag change.
 -- GGF baselines are preliminary values, and will change.
 
@author: Robert Shore: robore@bas.ac.uk
"""
output_version_identifier = 'BRTFv1p2'#Version string, for author's reference.

#%% Load packages.

#Import these packages.
import wget
import os
import sys
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib
import time
import math
import pickle

#Set geometric constants.
#Degrees to radians
rad = np.pi / 180 #scalar
#Radians to degrees
deg = 180 / np.pi #scalar

#%% Define the local time bins used by the model.

#Define a variable for the count of seconds in a day, for later use.
seconds_per_day = 24*60*60# hours * mins * secs

#Define local time bin spans by making a list of bin-edge delineations in time,
# in the form of datetimes throughout a given day.
LT_bin_edges = []#will be list of datetime objects of size [81].
date_x = datetime.datetime(2000,1,1)#arbitrary scalar datetime object for starting date, at hour 0 on some given day.
LT_bin_edges.append(date_x) 
while date_x < datetime.datetime(2000,1,2):
    #I choose to have 80 LT bins here, which equals 18 minutes long each.
    date_x += datetime.timedelta(minutes=18)
    LT_bin_edges.append(date_x)
#End iteration over LT bin definitions within a sample day.

#Define the start and end of each LT bin from the set of edges. No seconds variable,
# because the spans are at the 0th second. Convert to ordinal day fraction.
LT_bin_starts_day_fraction = np.empty([np.shape(LT_bin_edges)[0]-1,1])#np array of floats, size [80 by 1].
LT_bin_ends_day_fraction = np.empty([np.shape(LT_bin_edges)[0]-1,1])#np array of floats, size [80 by 1].
for i_LT in range(len(LT_bin_edges)-1):
    LT_bin_starts_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_edges[i_LT].time().hour, minutes=LT_bin_edges[i_LT].time().minute, seconds=0).total_seconds()/seconds_per_day
    LT_bin_ends_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_edges[i_LT+1].time().hour, minutes=LT_bin_edges[i_LT+1].time().minute, seconds=0).total_seconds()/seconds_per_day
#End loop over LT bins.

#Manually alter the last element of the LT bin end hour to be 1, rather than 0.
# This occurs because the ending date of the LT_bin_edges variable is the start 
# (i.e. hour zero) of the next day, rather than hour 24 on the same day.
LT_bin_ends_day_fraction[len(LT_bin_ends_day_fraction)-1,0] = 1.0

#Compute LT bin centroids.
LT_bin_centroids_day_fraction = np.mean(np.append(LT_bin_starts_day_fraction,LT_bin_ends_day_fraction,axis=1),axis=1)[:,np.newaxis]#np array of floats, size [LT bins (80) by 1].

#%% Load in the two regression models' coefficients.

#Take the script's filename, convert it to an absolute path, extract the 
# directory of that path, then change the current working directory to that 
# directory.
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

#Define the coefficients filename: this model predicts using epsilon as a driver.
BIRv5p5_coeffs_filename = os.path.join(os.getcwd(),'Storage_for_model_coefficients','BIRv5p5_tdmCBODSv3_reg_coefs.pkl')
#Source: C:\Users\robore\BAS_Files\Research\Code\SAGE\BGS_IMF_Regression_v5p5.py.

#Open the coefficients file.
with open(BIRv5p5_coeffs_filename,'rb') as f:
    BIRv5p5_stored_training_reg_coefs,\
    BIRv5p5_stored_training_95pcCIs_on_slopes,\
    BIRv5p5_stored_training_95pcCIs_on_intercepts,\
    BIRv5p5_stored_training_95pcCIs_on_residuals,\
    BIRv5p5_stored_training_1sigma_on_residuals,\
    BIRv5p5_stored_training_cor_coefs,\
    BIRv5p5_stored_training_PVEs = pickle.load(f)
#End indenting for this load-in.

#Define the coefficients filename: this model predicts using a 
# plasma-velocity-free solar wind coupling function as a driver.
VFERv2p0_coeffs_filename = os.path.join(os.getcwd(),'Storage_for_model_coefficients','VFERv2p0_tdmCBODSv3_reg_coefs.pkl')
#Source: C:\Users\robore\BAS_Files\Research\Code\SAGE\Velocity_Free_Epsilon_Regression_v2p0.py.

#Open the coefficients file.
with open(VFERv2p0_coeffs_filename,'rb') as f:  # Python 3: open(..., 'rb')
    VFERv2p0_stored_training_reg_coefs,\
    VFERv2p0_stored_training_95pcCIs_on_slopes,\
    VFERv2p0_stored_training_95pcCIs_on_intercepts,\
    VFERv2p0_stored_training_95pcCIs_on_residuals,\
    VFERv2p0_stored_training_1sigma_on_residuals,\
    VFERv2p0_stored_training_cor_coefs,\
    VFERv2p0_stored_training_PVEs = pickle.load(f)
#End indenting for this load-in.

#%% Download and import solar wind data.

#Define url of real-time solar wind data.
real_time_solar_wind_data_url = 'https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json'

#Define location that the real-time data will be stored locally.
real_time_solar_wind_data_filename = os.path.join(os.getcwd(),'Temp_storage_for_real_time_solar_wind_data','Geospace_propagated_solar_wind.dat')

#Download solar wind data for the last week.
wget.download(real_time_solar_wind_data_url, real_time_solar_wind_data_filename)
#Approach from https://stackabuse.com/download-files-with-python/.

#Import the solar wind data file to the workspace.
with open(real_time_solar_wind_data_filename,'rb') as json_file:
    real_time_solar_wind_data = np.array(json.load(json_file))#np array of string objects, size [((single header row) plus (minutes in week)) by 12]. Note: each row is a sub-array of size [1 by 12].
#Column format: 'time_tag', 'speed', 'density', 'temperature', 'bx', 'by', 'bz',
# 'bt', 'vx', 'vy', 'vz', 'propagated_time_tag'.

#Remove header.
real_time_solar_wind_data = np.delete(real_time_solar_wind_data,0,axis=0)#np array of string objects, size [minutes in week by 12]. Note: each row is a sub-array of size [1 by 12].

#Remove the downloaded file from local memory.
os.remove(real_time_solar_wind_data_filename)

#%% Process solar wind temporal data: add enough time-shifts to match the model's expected input data time-lag.
#Summary of required time shifts.
# -- one from 32Re to 10Re (a working substitute for the actual bow shock nose 
#     standoff).
# -- one 20 minutes long, to replicate the ionospheric reconfiguration 
#     timescale to a pertubation at the bow shock nose.

#Preallocate storage for the shifted timestamps.
shifted_times = np.empty([np.shape(real_time_solar_wind_data)[0],1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in week by 1]

#Loop over each record in the (approximately) week-long archive of real-time solar wind data.
for i_t in range(np.shape(real_time_solar_wind_data)[0]):
    #Extract string-format date of spacecraft measurement time tags, and convert 
    # directly to numpy's datetime64 format.    
    measurement_time = np.datetime64(datetime.datetime.strptime(real_time_solar_wind_data[i_t][0],'%Y-%m-%d %H:%M:%S.%f'))
    
    #Extract string-format date of shifted time tag (at 32Re), and convert 
    # directly to numpy's datetime64 format.
    shifted_time = np.datetime64(datetime.datetime.strptime(real_time_solar_wind_data[i_t][-1],'%Y-%m-%d %H:%M:%S.%f'))
    
    #Use this measurement epoch to get the epoch-specific time shift from L1 to 32Re.
    time_shift_value_L1_to_32Re = shifted_time - measurement_time#scalar, with units 'timedelta64[us]'.
    
    #Define the time shift in terms of how much time shift is applied per Earth 
    # radius, assuming a starting point of the L1 Lagrange point (about 1.5 million 
    # km) from Earth, and an end point of 32 Earth radii. Also assumes an Earth 
    # radius of 6371.2 km.
    time_shift_per_Re = time_shift_value_L1_to_32Re / ((1500000 / 6371.2) - 32)#scalar, with units 'timedelta64[us]'.
    
    #Compute the time shift to get from 32 Re to the approximate bow-shock location, 
    # assumed to be at about 10Re.
    time_shift_value_32Re_to_10Re = time_shift_per_Re * 22#scalar, with units 'timedelta64[us]'.
    
    #Add to the propagated (shifted) time values so that they continue their 
    # ballistic trajectory from the 32Re point to be at the 10Re point, 
    # representing the bow shock nose.
    shifted_time = shifted_time + time_shift_value_32Re_to_10Re
    
    #Add a further 20 minutes to the shifted time tag, in order to represent the 
    # reconfiguration timescale of the ionosphere. Following this, a time-shifted 
    # measurement with a contemporaneous time tag should be a nowcast of the 
    # ionospheric state, and all following shifted measurements will be forecasts.
    shifted_time = shifted_time + np.timedelta64(20,'m')
    
    #Store the updated propagated time tag.
    shifted_times[i_t] = shifted_time
#End loop over times (minutes in week).

#%% Account for solar wind arrival-time overlaps.
#That is, we're using a pre-propagated solar wind data set, so there exists the
# possibility that some temporal elements can 'arrive' at Earth in the reverse 
# order to that of their measurement order.  Here, we sort the propagated times
# so that this does not happen.

#Define an index which will sort the propagated solar wind dates in ascending
# order, starting from the lowest value.
indices_of_sorted_sw_data = np.argsort(shifted_times,axis=0)

#Apply the index to the solar wind data that you use in the rest of this program.
real_time_solar_wind_data = real_time_solar_wind_data[indices_of_sorted_sw_data[:,0],:]
shifted_times = shifted_times[indices_of_sorted_sw_data[:,0]]

#%% Extract the solar wind measurements of interest and convert from string to floating point values.

#Extract the solar wind data variables of interest for the entire week(ish) of 
# sorted, irregularly-propagated measurements.
sw_data_bx = real_time_solar_wind_data[:,4].astype(float)#np array of floats, size [minutes in previous hour (ish) by 0].
sw_data_by = real_time_solar_wind_data[:,5].astype(float)#np array of floats, size [minutes in previous hour (ish) by 0].
sw_data_bz = real_time_solar_wind_data[:,6].astype(float)#np array of floats, size [minutes in previous hour (ish) by 0].
sw_data_vx = real_time_solar_wind_data[:,8].astype(float)#np array of floats, size [minutes in previous hour (ish) by 0].

#%% Temporally re-grid the propagated, sorted solar wind epochs to a regular 1-min cadence, for a recent subset of times.
#Specifically, the subset spans from the now (which for the solar wind data, 
# means the closest-to-contemporaneous shifted time element), up until the most
# recent measurement (which, after propagation, gives us the forecast span).

#Create a variable for the current time, and convert to UTC if daylight saving
# is in effect.
if(time.localtime().tm_isdst > 0):
    current_time = np.datetime64(datetime.datetime.now()) - np.timedelta64(1,'h')
else:
    current_time = np.datetime64(datetime.datetime.now())
#End conditional: check if daylight savings time applies.
#Warning: may not work in the hour near clock-change, or if the CPU clock 
# differs from the UK timezone.

#Make a starting time for the temporal re-gridding based on the current time, 
# reduced to 1-min precision.
start_time = np.datetime64(current_time,'m')#scalar datetime64 object, at minute precision.
#For instance: current_time.astype(datetime.datetime).second > start_time.astype(datetime.datetime).second.

#Make an end time for the temporal re-gridding based on the last available 
# propagated measurement epoch, also reduced to 1-min precision. This applies 
# a 'floor' operation, ensureing that the range of the regular gridded series 
# is within the range of the existing data.
end_time = np.datetime64(shifted_times[-1][0],'m')

#Synthesise a 1-min cadence time series using the predefined start and end 
# times. Note that this is set to microsecond ('us') precision, because it 
# needs to match the precision of the 'shifted_times' variable for when I 
# convert both of them to ordinal floats (which will have units of 
# microseconds) for the interpolation.
shifted_times_regular_grid = np.arange(start_time,end_time+np.timedelta64(1,'m'),np.timedelta64(1,'m'), dtype='datetime64[us]')

#Interpolate the various solar wind measurement values to the regular temporal 
# grid, using the command '.astype(float)' to convert the datetime64 values to 
# ordinal microseconds.
sw_data_bx_regular_grid = np.interp(shifted_times_regular_grid.astype(float), shifted_times[:,0].astype(float), sw_data_bx)
sw_data_by_regular_grid = np.interp(shifted_times_regular_grid.astype(float), shifted_times[:,0].astype(float), sw_data_by)
sw_data_bz_regular_grid = np.interp(shifted_times_regular_grid.astype(float), shifted_times[:,0].astype(float), sw_data_bz)
sw_data_vx_regular_grid = np.interp(shifted_times_regular_grid.astype(float), shifted_times[:,0].astype(float), sw_data_vx)

#%% Compute local times at each BGS station, from the timestamps of the most-recent solar wind measurements.

#Define the three BGS station longitudes, taken from the INTERMAGNET daily file 
# metadata. In alphabetic order: Eskdalemuir, Hartland, Lerwick.
BGS_station_longitudes = [(356.8 - 360),(355.5 - 360),(358.8 - 360)]#list of size [3] -- geodetic longitude. Converted to east longitude on a +-180 scale (for later local time calculations), so it's negative because it's slightly west.

#Preallocate storage for BGS station local times.
BGS_stations_shifted_times_regular_grid_local_time_day_fraction = np.empty([np.shape(shifted_times_regular_grid)[0],3])#np array of floats, size [minutes in previous hour (ish) by 3].

#Loop over BGS stations.
for i_station in range(3):
    #Extract the longitude for this station.
    BGS_station_longitude = BGS_station_longitudes[i_station]#scalar float.
    
    #Compute the local time of this BGS station, rounded to 1-second precision, 
    # because numpy doesn't accept float inputs to timedelta64.
    BGS_single_station_shifted_times_regular_grid_with_local_time_alteration = shifted_times_regular_grid + np.timedelta64(round((BGS_station_longitude / 15) * 60 * 60),'s')#np array of datetime64[us], size [minutes in previous hour (ish) by 1] -- converts longitude to hours, then minutes, then seconds.
    
    #Loop over each time element, and convert the BGS station local time to 
    # ordinal fraction of the day.
    for i_time, time_element in enumerate(BGS_single_station_shifted_times_regular_grid_with_local_time_alteration):
        BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_time,i_station] = datetime.timedelta(\
            hours=time_element.astype(object).hour,\
            minutes=time_element.astype(object).minute,\
            seconds=time_element.astype(object).second).total_seconds()/seconds_per_day
    #End loop over each time element of the recent solar wind data subset.
#End loop over the three BGS stations.

#%% Forecast the different regression models for each epoch of the most-recent solar wind measurements.

#Preallocate storage for predictions.
model_predictions_all_stations_all_cmpnts = np.empty([np.shape(sw_data_bx_regular_grid)[0],3,3])#np array of floats, size [minutes in previous hour (ish) by BGS stations by BGS components].
model_prediction_flags = np.empty([np.shape(sw_data_bx_regular_grid)[0],1])#np array of floats, size [minutes in previous hour (ish) by 1].
#model_1sigma_error_all_stations_all_cmpnts = np.empty([np.shape(sw_data_bx_regular_grid)[0],3,3])#np array of floats, size [minutes in previous hour (ish) by BGS stations by BGS components].
model_95pcconf_error_all_stations_all_cmpnts = np.empty([np.shape(sw_data_bx_regular_grid)[0],3,3])#np array of floats, size [minutes in previous hour (ish) by BGS stations by BGS components].
#Loop over each data point in the BGS station data for all cross-validation storms.
for i_t in range(np.shape(shifted_times_regular_grid)[0]):
    #Choose which model you will use to predict this temporal element of the 
    # forecast span, dependent on what solar wind measurements are available.
    if(np.isnan(sw_data_bx_regular_grid[i_t])):
        #If the magnetic field measurements do not exist for this epoch, we
        # forecast a null value.
        model_predictions_all_stations_all_cmpnts[i_t,:,:] = np.nan
        model_prediction_flags[i_t] = 2
        
        #Skip the rest of this loop's iteration.
        continue#pertains to loop over i_t.
        
    elif(np.invert(np.isnan(sw_data_bx_regular_grid[i_t])) & np.isnan(sw_data_vx_regular_grid[i_t])):
        #If we have magnetic field values but no plasma velocity values, then 
        # we can use the velocity-free-epsilon model for the forecast.
        
        #Compute surrogate solar wind coupling function to replace epsilon, without plasma velocity data.
        #Tips from here on angles and quadrants
        # 'http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/167016'
        # 'atan' only returns angles in the range -pi/2 and +pi/2.  'atan2'
        # implicitly divides the two inputs and will produce angles between -pi and
        # +pi.  If you want to convert these to angles between 0 and 2*pi, run:
        # mod(angle,2*pi). Strictly speaking, the part inside the brackets
        # should each be multipled by rad, but the ratio is used so it would
        # have no effect.
        IMF_clock_angle = math.atan2(sw_data_by_regular_grid[i_t], sw_data_bz_regular_grid[i_t]) * deg
        
        #Calculate magnitude of full IMF vector.
        IMF_B_magnitude = math.sqrt((sw_data_bx_regular_grid[i_t] ** 2) + (sw_data_by_regular_grid[i_t] ** 2) + (sw_data_bz_regular_grid[i_t] ** 2))
        
        #Form the epsilon-surrogate plasma-free coupling function 'pfcf'.
        sw_coupling_function = (IMF_B_magnitude ** 3) * (math.sin((IMF_clock_angle / 2) * rad) ** 4)
        
        #Assign model-type identifier flag.
        model_prediction_flags[i_t] = 1
        
    elif(np.invert(np.isnan(sw_data_bx_regular_grid[i_t])) & np.invert(np.isnan(sw_data_vx_regular_grid[i_t]))):
        #If we have both magnetic field and plasma velocity values, then 
        # we can use the epsilon-driven model for the forecast.
        
        #Compute epsilon solar wind coupling function.
        #Tips from here on angles and quadrants
        # 'http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/167016'
        # 'atan' only returns angles in the range -pi/2 and +pi/2.  'atan2'
        # implicitly divides the two inputs and will produce angles between -pi and
        # +pi.  If you want to convert these to angles between 0 and 2*pi, run:
        # mod(angle,2*pi). Strictly speaking, the part inside the brackets
        # should each be multipled by rad, but the ratio is used so it would
        # have no effect.
        IMF_clock_angle = math.atan2(sw_data_by_regular_grid[i_t], sw_data_bz_regular_grid[i_t]) * deg
        
        #Calculate magnitude of full IMF vector.
        IMF_B_magnitude = math.sqrt((sw_data_bx_regular_grid[i_t] ** 2) + (sw_data_by_regular_grid[i_t] ** 2) + (sw_data_bz_regular_grid[i_t] ** 2))
        
        #Convert from nT to Tesla.
        IMF_B_magnitude_in_Tesla = IMF_B_magnitude * (1e-9)#Converted to T, from nT. #np array of floats, size [minutes in previous hour (ish) by 0].
        
        #Convert SWvx to SI units.
        vx_in_m_per_s = -sw_data_vx_regular_grid[i_t] * (1e3)#Converted to m/s from km/s. #np array of floats, size [minutes in previous hour (ish) by 0].
        
        #Assign a length-scale factor of 7Re.
        l_nought = 7 * (6371.2 * 1000) #scalar float.
        
        #Assign a factor for the amplitude of the permeability of free space, once 
        # 4 pi has been divided by mew_nought.
        four_pi_over_mew_nought = 1e7 #Note: 1/1e-7 = 1e7. #scalar float.
        
        #Form the epsilon coupling function.
        sw_coupling_function = four_pi_over_mew_nought * vx_in_m_per_s * (IMF_B_magnitude_in_Tesla ** 2) * (math.sin((IMF_clock_angle / 2) * rad) ** 4) * (l_nought ** 2)
        #Units: Watts.
        
        #Assign model-type identifier flag.
        model_prediction_flags[i_t] = 0
    #End conditional: choose forecast model based on what data are there to drive it.
    
    
    #Compute the model prediction for each component of each station.
    #Loop over BGS data components.
    for i_component in range(3):
        #Loop over BGS stations.
        for i_station in range(3):
            #Based on the decimal day-fraction of this BGS station's local time,
            # find the fiducials of the two LT bins which flank this value. 
            # Note that we are basing the model coefficient interpolation on 
            # the LT bin centroids: it dose not amtter which LT bin the station
            # is actually 'within'.
            if(BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station] < LT_bin_centroids_day_fraction[0]):
                #Here, the BGS station is between midnight and the centroid LT
                # of the first bin. That is, the BGS LT is near zero.
                
                #The LT bin which sits behind the station in LT (i.e. the 
                # 'past' LT bin) needs to have one day subtracted from its 
                # decimal day LT value, such that the effective distance between 
                # it and the LT bin which sits ahead of the station in LT (i.e. 
                # the 'future' LT bin) is accurate.
                past_LT_bin_centroid = LT_bin_centroids_day_fraction[-1] - 1
                future_LT_bin_centroid = LT_bin_centroids_day_fraction[0]
                
                #Define indices for these LT bins: these can be specified 
                # manually due to the conditions required for these indented 
                # lines of code to be activated.
                index_past_LT_bin = 79
                index_future_LT_bin = 0
            elif(BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station] > LT_bin_centroids_day_fraction[-1]):
                #Here, the BGS station is between the centroid LT of the last 
                # bin, and midnight. That is, the BGS LT is near 1.
                
                #The LT bin which sits ahead of the station in LT (i.e. the 
                # 'future' LT bin) needs to have one day added to its 
                # decimal day LT value, such that the effective distance between 
                # it and the LT bin which sits behind of the station in LT (i.e. 
                # the 'past' LT bin) is accurate.
                past_LT_bin_centroid = LT_bin_centroids_day_fraction[-1]
                future_LT_bin_centroid = LT_bin_centroids_day_fraction[0] + 1
                
                #Define indices for these LT bins: these can be specified 
                # manually due to the conditions required for these indented 
                # lines of code to be activated.
                index_past_LT_bin = 79
                index_future_LT_bin = 0
            else:
                #Here, the BGS station lies within some other pair of LT bins,
                # or directly upon one of them (we deal with that condition 
                # later).
                
                #Define indices for the flanking LT bins.
                index_past_LT_bin = np.nonzero(LT_bin_centroids_day_fraction <= BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station])[0][-1]
                index_future_LT_bin = np.nonzero(LT_bin_centroids_day_fraction >= BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station])[0][0]
                
                #Find the flanking LT bins centroid decimal day values.
                past_LT_bin_centroid = LT_bin_centroids_day_fraction[index_past_LT_bin]
                future_LT_bin_centroid = LT_bin_centroids_day_fraction[index_future_LT_bin]
            #End conditional: find the two LT bins which flank the station's precise location in LT.
            
            #Based on the solar wind data availability, the station, and the 
            # component, extract model parameter vectors, and prediction error 
            # estimates, for each of the two interpolant LT bins.
            if(np.invert(np.isnan(sw_data_bx_regular_grid[i_t])) & np.isnan(sw_data_vx_regular_grid[i_t])):
                #Extract VFERv2p0 model parameter vectors.
                past_LT_bin_model_parameter_vector =   VFERv2p0_stored_training_reg_coefs[i_station,index_past_LT_bin,i_component,:]#size [6 by 0]
                future_LT_bin_model_parameter_vector = VFERv2p0_stored_training_reg_coefs[i_station,index_future_LT_bin,i_component,:]#size [6 by 0]
                
                #Extract VFERv2p0 prediction error estimates.
                past_LT_bin_model_95pcconf_error = VFERv2p0_stored_training_95pcCIs_on_residuals[i_station,index_past_LT_bin,i_component]
                future_LT_bin_model_95pcconf_error = VFERv2p0_stored_training_95pcCIs_on_residuals[i_station,index_future_LT_bin,i_component]
            elif(np.invert(np.isnan(sw_data_bx_regular_grid[i_t])) & np.invert(np.isnan(sw_data_vx_regular_grid[i_t]))):
                #Extract BIRv5p5 model parameter vectors.
                past_LT_bin_model_parameter_vector =   BIRv5p5_stored_training_reg_coefs[i_station,index_past_LT_bin,i_component,:]#size [6 by 0]
                future_LT_bin_model_parameter_vector = BIRv5p5_stored_training_reg_coefs[i_station,index_future_LT_bin,i_component,:]#size [6 by 0]
                
                #Extract BIRv5p5 prediction error estimates.
                past_LT_bin_model_95pcconf_error = BIRv5p5_stored_training_95pcCIs_on_residuals[i_station,index_past_LT_bin,i_component]
                future_LT_bin_model_95pcconf_error = BIRv5p5_stored_training_95pcCIs_on_residuals[i_station,index_future_LT_bin,i_component]
            #End conditional: choose model coefficients based on what data are there to drive it.
            
            #Interpolate the model's prediction error estimates to the central
            # LT, and store the error on this single-minute prediction. This is
            # based on the prediction error on the training data residials.
            model_95pcconf_error_all_stations_all_cmpnts[i_t,i_station,i_component] = np.interp(\
                BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station],\
                np.array([past_LT_bin_centroid[0],future_LT_bin_centroid[0]]),\
                np.array([past_LT_bin_model_95pcconf_error,future_LT_bin_model_95pcconf_error]))
            #End indenting for this variable.
            
            #For each model coefficient, interpolate it to the central (ish) 
            # LT of the BGS station.
            interpolated_model_parameter_vector = np.empty([np.shape(BIRv5p5_stored_training_reg_coefs)[3],1])#size [6 (model parameters) by 1].
            for i_parameter in range(np.shape(BIRv5p5_stored_training_reg_coefs)[3]):
                interpolated_model_parameter_vector[i_parameter,0] = np.interp(\
                    BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station],\
                    np.array([past_LT_bin_centroid[0],future_LT_bin_centroid[0]]),\
                    np.array([past_LT_bin_model_parameter_vector[i_parameter],future_LT_bin_model_parameter_vector[i_parameter]]))
                #End indenting for this variable.
            #End loop over model parameters.
            #Note that the above interpolator code will function properly even 
            # if past bin LT = future bin LT = current BGS LT.
            
            #Define the fractional day of year of this date, specific to this station.
            BGS_date_DOY = shifted_times_regular_grid[i_t].astype(datetime.datetime).timetuple().tm_yday \
                + BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station]
            #End indenting for this variable.
            
            #Create model data kernel vector.
            data_kernel = np.array([\
                1,\
                np.sin(((BGS_date_DOY - 79) * (2 * np.pi))/365.25),\
                np.cos(((BGS_date_DOY - 79) * (2 * np.pi))/365.25),\
                sw_coupling_function,\
                np.sin(((BGS_date_DOY - 79) * (2 * np.pi))/365.25) * sw_coupling_function,\
                np.cos(((BGS_date_DOY - 79) * (2 * np.pi))/365.25) * sw_coupling_function,\
                ])
            #End indentation for this variable.
            
            #Make and store data prediction.
            model_predictions_all_stations_all_cmpnts[i_t,i_station,i_component] = np.matmul(np.transpose(data_kernel[:,np.newaxis]),interpolated_model_parameter_vector)[0][0]
            
        #End loop over BGS stations.
    #End loop over BGS components.
#End loop over each temporal element of the recent solar wind data subset.

#%% Format and save out the forecast data file.

#Define filename for real-time geomagnetic forecast output.
real_time_solar_wind_data_output_filename = os.path.join(os.getcwd(),'Temp_storage_for_output_GGF_forecast','real_time_magnetic_field_forecast_from_program_GGF_RTF_version_' + output_version_identifier + '.dat')

#Delete any existing data file of solar wind-based geomagnetic forecast outputs.
if(os.path.isfile(real_time_solar_wind_data_output_filename)):
    os.remove(real_time_solar_wind_data_output_filename)
#End:conditional: delete any already-existing output files, to avoid appending data to them.

#Open file of real time solar wind data.
file_id = open(real_time_solar_wind_data_output_filename, 'wb')

#Loop over each line of output.
for i_t in range(np.shape(model_predictions_all_stations_all_cmpnts)[0]):
    #Format the date string.
    date_string = '{i_t_day:02d}-{i_t_month:02d}-{i_t_year}  {i_t_hour:02d}:{i_t_min:02d}'\
        .format(i_t_day = shifted_times_regular_grid[i_t].astype(datetime.datetime).day,\
                i_t_month = shifted_times_regular_grid[i_t].astype(datetime.datetime).month,\
                i_t_year = shifted_times_regular_grid[i_t].astype(datetime.datetime).year,\
                i_t_hour = shifted_times_regular_grid[i_t].astype(datetime.datetime).hour,\
                i_t_min = shifted_times_regular_grid[i_t].astype(datetime.datetime).minute)
    #End indenting for this string formatting.
    
    #Format the prediction string.
    data_string = '       {data_flag:1d}       {esk_x:.2f}       {esk_y:.2f}       {esk_z:.2f}       {had_x:.2f}       {had_y:.2f}       {had_z:.2f}       {ler_x:.2f}       {ler_y:.2f}       {ler_z:.2f}'\
        .format(data_flag = model_prediction_flags[i_t,0].astype(int),\
                esk_x = model_predictions_all_stations_all_cmpnts[i_t,0,0],\
                esk_y = model_predictions_all_stations_all_cmpnts[i_t,0,1],\
                esk_z = model_predictions_all_stations_all_cmpnts[i_t,0,2],\
                had_x = model_predictions_all_stations_all_cmpnts[i_t,1,0],\
                had_y = model_predictions_all_stations_all_cmpnts[i_t,1,1],\
                had_z = model_predictions_all_stations_all_cmpnts[i_t,1,2],\
                ler_x = model_predictions_all_stations_all_cmpnts[i_t,2,0],\
                ler_y = model_predictions_all_stations_all_cmpnts[i_t,2,1],\
                ler_z = model_predictions_all_stations_all_cmpnts[i_t,2,2])
    #End indenting for this string formatting.
    
    #Collate formatted strings.
    line = date_string + data_string
    
    #Alter the string so that it has a consistent number of spaces between data elements.
    line = ' '.join(('%*s' % (10, i) for i in line.split()))
    #Solution from https://stackoverflow.com/questions/3685195/line-up-columns-of-numbers-print-output-in-table-format.
    
    #Append a carriage-return, and encode as ascii.
    ascii_output = (line + '\n').encode('ascii')
    
    #Write out this line.
    file_id.write(ascii_output)
#End loop over each time element of the forecasts.

#Close the output ascii file.
file_id.close()

#%% Plot data.

#Define BGS station names and component names.
BGS_station_names = ['Eskdalemuir','Hartland','Lerwick']
component_names = ['x','y','z']

#Initialise figure.
plt.style.use('seaborn-whitegrid')
fig, axs = plt.subplots(3,3,sharey='all',sharex='all',figsize=[12,12])

#Setting the values for all axes.
plot_limit_max = max(np.ravel(model_predictions_all_stations_all_cmpnts + model_95pcconf_error_all_stations_all_cmpnts))
plot_limit_max = plot_limit_max + (plot_limit_max / 10)
plot_limit_min = min(np.ravel(model_predictions_all_stations_all_cmpnts - model_95pcconf_error_all_stations_all_cmpnts))
plot_limit_min = plot_limit_min + (plot_limit_min / 10)
plt.setp(axs, xlim=(shifted_times_regular_grid[0]-(shifted_times_regular_grid[5]-shifted_times_regular_grid[0]),shifted_times_regular_grid[-1]), ylim=(plot_limit_min,plot_limit_max))
date_format = DateFormatter("%H:%M")

#Loop over BGS data components (order: x, y, z).
for i_component in range(3):
    #Loop over BGS stations.
    for i_station in range(3):
        #Plot data and error bars.
        axs[i_component,i_station].fill_between(shifted_times_regular_grid[:],\
            model_predictions_all_stations_all_cmpnts[:,i_station,i_component] - model_95pcconf_error_all_stations_all_cmpnts[:,i_station,i_component],\
            model_predictions_all_stations_all_cmpnts[:,i_station,i_component] + model_95pcconf_error_all_stations_all_cmpnts[:,i_station,i_component],\
            color='0.8')
        axs[i_component,i_station].plot(shifted_times_regular_grid,model_predictions_all_stations_all_cmpnts[:,i_station,i_component],color='C0')
        axs[i_component,i_station].plot(shifted_times_regular_grid,model_predictions_all_stations_all_cmpnts[:,i_station,i_component] + model_95pcconf_error_all_stations_all_cmpnts[:,i_station,i_component],'--',color='C0')
        axs[i_component,i_station].plot(shifted_times_regular_grid,model_predictions_all_stations_all_cmpnts[:,i_station,i_component] - model_95pcconf_error_all_stations_all_cmpnts[:,i_station,i_component],'--',color='C0')
        #Set x tick labels.
        axs[i_component,i_station].xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=20))
        #ticks = []
        #for i_t in range(0,len(shifted_times_regular_grid),20):
        #    ticks.append(shifted_times_regular_grid[i_t])
        #axs[i_component,i_station].set_xticks(ticks)
        axs[i_component,i_station].xaxis.set_major_formatter(date_format)
        
        #Draw on vertical line at current date.
        axs[i_component,i_station].plot([shifted_times_regular_grid[0],shifted_times_regular_grid[0]],[plot_limit_min,plot_limit_max],'k--')
        
        #Add titles and axis labels.
        if(i_component == 0):
            axs[i_component,i_station].set_title(BGS_station_names[i_station])
        if(i_station == 0):
            axs[i_component,i_station].set_ylabel(component_names[i_component] + '-component forecast (nT)')
        if(i_component == 2):
            axs[i_component,i_station].set_xlabel('UTC, starting at current time')
        #End conditional.
    #End loop over BGS stations.
#End loop over components.

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'Temp_storage_for_output_GGF_forecast', 'real_time_forecast_charts_from_program_GGF_RTF_version_' + output_version_identifier + '.eps'), format='eps')
