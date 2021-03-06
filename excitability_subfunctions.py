import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')
sys.path.append('/Library/Python/2.7/site-packages/');
sys.path.append('/Applications/MacPorts/dv_dt_V_analysis/')
sys.path.append('/Applications/MacPorts/stimfit.app/Contents/Frameworks/stimfit/')
import dv_dt_V 
import numpy as np
import pandas as pd
import stf


import spells



def find_AP_peaks(start_msec, delta_msec, current_start, current_delta, threshold_value, deflection_direction, mark_option):
	
	""" count number of APs in traces with current injection/gradually increasing steps
	inputs: (time (msec) to start search, length of search region, starting current value, current delta between traces, threshold value, deflection direction ('up'/'down'), mark traces (True/False))"""	
	
	event_counts = np.zeros((stf.get_size_channel(),2)); 
	
	for trace_ in range(stf.get_size_channel()):
		event_counts[trace_][1] = spells.count_events(start_msec, delta_msec, threshold=threshold_value, up=deflection_direction, trace=trace_, mark=mark_option); 
		event_counts[trace_][0] = current_start + (current_delta*trace_) ; 
		
	loaded_file = stf.get_filename()[:-4] ; 
	np.savetxt(loaded_file + '_AP_counts.csv', event_counts, delimiter=',', newline='\n'); 
	return(event_counts)
	

def jjm_count(start, delta, threshold=0, up=True, trace=None, mark=True):
	""" Counts the number of events (e.g action potentials (AP)) in the current trace.
	Arguments:
	start -- starting time (in ms) to look for events.
	delta -- time interval (in ms) to look for events.
	threshold -- (optional) detection threshold (default = 0).
	up -- (optional) True (default) will look for upward events, False downwards.
	trace -- (optional) zero-based index of the trace in the current channel,
	if None, the current trace is selected.
	mark -- (optional) if True (default), set a mark at the point of threshold crossing
	Returns:
	An integer with the number of events.
	Examples:
	count_events(500,1000) returns the number of events found between t=500 ms and t=1500 ms
	above 0 in the current trace and shows a stf marker.
	count_events(500,1000,0,False,-10,i) returns the number of events found below -10 in the 
	trace i and shows the corresponding stf markers. """
	# sets the current trace or the one given in trace.
	if trace is None:
		sweep = stf.get_trace_index()
	else:
		if type(trace) !=int:
			print "trace argument admits only integers"
			return False
		sweep = trace
	# set the trace described in sweep
	stf.set_trace(sweep)
	# transform time into sampling points
	dt = stf.get_sampling_interval()
	pstart = int( round(start/dt) )
	pdelta = int( round(delta/dt) )
	# select the section of interest within the trace
	selection = stf.get_trace()[pstart:(pstart+pdelta)]
	# algorithm to detect events
	EventCounter,i = 0,0 # set counter and index to zero
	# list of sample points
	sample_points_absolute = []
	sample_points_relative = []
	# choose comparator according to direction:
	if up:
		comp = lambda a, b: a > b
	else:
		comp = lambda a, b: a < b
	# run the loop
	while i<len(selection):
		if comp(selection[i],threshold):
			EventCounter +=1
			if mark:
				sample_point = pstart+i; 
				sample_points_relative.append(i)
				sample_points_absolute.append(sample_point); 
				stf.set_marker(pstart+i, selection[i])
			while i<len(selection) and comp(selection[i],threshold):
				i+=1 # skip values if index in bounds AND until the value is below/above threshold again
		else:
			i+=1
	
	time_points = [sample_point*dt for sample_point in sample_points_absolute];
	return (EventCounter, sample_points_absolute)
	
def find_ADPs(AP_peak_indicies):
	ADP_values = []
	ADP_indicies = []
	##slices 
	for peak in range(len(AP_peak_indicies)-1):
		ADP_search = stf.get_trace()[AP_peak_indicies[peak]:AP_peak_indicies[peak+1]]
		min_value = np.min(ADP_search)
		min_index = AP_peak_indicies[peak] + np.argmin(ADP_search)
		stf.set_marker(min_index, min_value)
		ADP_values.append(min_value)
		ADP_indicies.append(min_index)
			
	return(ADP_values, ADP_indicies)
	
def find_thresholds(input_trace, input_trace_si, ADP_sample_points):
	#take derivative of trace, assume relevant trace is set to current trace
	#get_dv_dt_as_numpy(input_array, si, *argv)
	
	_1stderivative = dv_dt_V.get_dv_dt_as_numpy(input_trace, input_trace_si)
	#take 2nd derivative of trace
	_2ndderivative = dv_dt_V.get_dv_dt_as_numpy(_1stderivative, input_trace_si)
	_3rdderivative = dv_dt_V.get_dv_dt_as_numpy(_2ndderivative, input_trace_si)
	#will need to filter trace here
	#try with using sigma of 1
	
	_3rdderivative_filtered = _3rdderivative
	#trace_filtering.filter_1d_numpy(_3rdderivative, 1, False)
	
	#find peak time points
	#use ADP sample points, estimate an AP width of 10ms to work backword from to capture peak
	
	threshold_points = []
	threshold_indicies = []
	for x in range(len(ADP_sample_points)):
		if x == 0:
			earlier_point = int(ADP_sample_points[x]) - int(round(50/input_trace_si))
		else:
			earlier_point = int(ADP_sample_points[x-1])
		
		
		point = ADP_sample_points[x]
		deriv_peak_search = _3rdderivative_filtered[earlier_point:point]
		
		threshold_points.append(np.max(deriv_peak_search))
		threshold_index = earlier_point+np.argmax(deriv_peak_search)
		threshold_indicies.append(threshold_index)
		
		#marker function is not working here, not sure why, also maybe a good point to set a 
		#voltage bound on detecting the threshold
		stf.set_marker(threshold_index, input_trace[threshold_index])
	
	#use peak time points to pull out the voltage values from the 1st trace, these are voltage values
	return(threshold_indicies)













