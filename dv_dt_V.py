import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')
sys.path.append('/Library/Python/2.7/site-packages/');
sys.path.append('/Applications/MacPorts/dv_dt_V_analysis/')
sys.path.append('/Applications/MacPorts/stimfit.app/Contents/Frameworks/stimfit/')
import numpy as np
import stf
import spells
from matplotlib import pyplot as plt


def get_dv_dt(slice_indicies=(0,0)):
    """Main function to take 1st derivative of V_trace and return an array with V_values 
    and dv_dt value for plotting
    --input tuple to use slice of trace"""
    
    #determine if using whole trace or slice
    
    if slice_indicies != 0:
    	sample_start = slice_indicies[0]
    	sample_end = slice_indicies[1]
    	
    else:
    	sample_start = 0
    	sample_end = len(stf.get_trace()) 

    #get sampling interval to create dt part of dv/dt 
    #dt is just sampling interval 	
    si = stf.get_sampling_interval()
    
    #read V values from trace, 
    V_values = stf.get_trace()[sample_start:sample_end]
    
	#compute dv and by iterating over voltage vectors
    dv = [V_values[i+1]-V_values[i] for i in range(len(V_values)-1)]
   
	#compute dv/dt
    dv_dt = [(dv[i]/si) for i in range(len(dv))]
    #V values for a dv/dt / V graph is just truncated trace with final sample point removed
    V_plot = V_values[:-1]
    #combine for a plotting function/further manipulation 
    V_dv_dt = np.vstack([V_plot, dv_dt])
    stf.new_window(dv_dt)
    
    return(V_dv_dt)
    
def get_dv_dt_as_numpy(input_array, si, slice_indicies=(0,0)):
	"""take time derivative of trace
	Inputs:
	array: 1d numpy array
	si: sampling interval for values in numpy array
	optional start/stop stime to slice array (default will calc over whole trace)"""
	#determine if using whole trace or slice
	print(slice_indicies)
	if slice_indicies[1] != 0: 
		sample_start = slice_indicies[0]
		sample_end = slice_indicies[1]
	
	else:
		sample_start = 0
		sample_end = len(stf.get_trace())
	  
	#V values from trace
	#compute dv and by iterating over voltage vectors
	#compute dv/dt	
	V_values = input_array[int(sample_start):(int(sample_end))]
	dv = [V_values[i+1]-V_values[i] for i in range(len(V_values)-1)]
	dv_dt = [(dv[i]/si) for i in range(len(dv))]
	
	return(V_values, dv_dt)
	
def plot_dv_dt(V_values, dv_dt_values):
	
	plt.plot(V_values[:-1], dv_dt_values)
	plt.show()
	
	return()
	