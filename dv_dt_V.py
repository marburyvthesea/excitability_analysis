import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')
sys.path.append('/Library/Python/2.7/site-packages/');
sys.path.append('/Applications/MacPorts/dv_dt_V_analysis/')
sys.path.append('/Applications/MacPorts/stimfit.app/Contents/Frameworks/stimfit/')
import numpy as np
import stf
import spells
from matplotlib import pyplot as plt

class excitability_trace(object):
	##class to for information for traces in excitability experiment
	def __init__(self, voltage_trace, sweep_number):
		self.voltage_trace = voltage_trace
		self.sweep_number = sweep_number 
		
	def calc_dv_dt(self):
	#calculate 1st derivative of trace
		stf.set_trace(self.sweep_number)
		dv_dt_out = get_dv_dt()
		return(dv_dt_out)

def compile_file():
	
	file_dict = {}
	for sweep in range(stf.get_size_channel()):
		stf.set_trace(sweep)
		sweep_name = str(sweep)
		file_dict[sweep_name] = excitability_trace(stf.get_trace(), sweep)
		
	return(file_dict)

def get_dv_dt_time_slice(time_start, time_end):
	"""Call get_dv_dt using input time values to take derivative of a trace slice
	"""
	#get sampling interval 
	si = stf.get_sampling_interval()
	sample_start = time_start/si
	sample_end = time_end/si 
	
	V_dv_dt_slice = get_dv_dt(sample_start, sample_end)
	
	return(V_dv_dt_slice)

def get_dv_dt(*argv):
    """Main function to take 1st derivative of V_trace and return an array with V_values 
    and dv_dt value for plotting
    --if no input then use whole trace
    --if sample values are input then use slice of trace"""
    
    #determine if using whole trace or slice
    if len(argv) > 0: 
		sample_start = argv[0]
		sample_end = argv[1]
    	
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
    
def get_dv_dt_as_numpy(input_array, si, *argv):
	"""take time derivative of trace
	Inputs:
	array: 1d numpy array
	si: sampling interval for values in numpy array
	*argv: optional start/stop stime to slice array"""
	#determine if using whole trace or slice
	if len(argv) > 0: 
		sample_start = argv[0]
		sample_end = argv[1]
	
	else:
		sample_start = 0
		sample_end = len(stf.get_trace()) 
	  
	#V values from trace
	#compute dv and by iterating over voltage vectors
	#compute dv/dt	
	V_values = input_array[sample_start:sample_end]
	dv = [V_values[i+1]-V_values[i] for i in range(len(V_values)-1)]
	dv_dt = [(dv[i]/si) for i in range(len(dv))]
	
	return(dv_dt)
	
def plot_dv_dt(V_values, dv_dt_values):
	
	plt.plot(V_values, dv_dt_values)
	plt.show()
	
	return()
	