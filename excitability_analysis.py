import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
sys.path.append('/Library/Python/2.7/site-packages/');
sys.path.append('/Users/johnmarshall/Documents/Analysis/PythonAnalysisScripts/stimfitscripts/')
sys.path.append('/Users/johnmarshall/miniconda3/envs/py27/lib/python2.7/site-packages')
import dv_dt_V 
import numpy as np
import pandas as pd
import stf
import trace_filtering
import xlsxwriter
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
	
def find_ADP_thresholds_for_file(current_start_file, current_delta_file, *argv):
	
	""" count number of APs in traces with current injection/gradually increasing steps
	inputs: (time (msec) to start search, length of search region, starting current value, current delta between traces, threshold value, deflection direction ('up'/'down'), mark traces (True/False))"""	
	if len(argv)>0:
		threshold_value_file = argv[0]
		deflection_direction_file = argv[1]
		mark_option_file = argv[2]
		start_msec_file = float(argv[3])
		delta_msec_file = float(argv[4])	
	else:
		threshold_value_file = 0
		deflection_direction_file = 'up'
		mark_option_file = True
		start_msec_file = float(stf.get_peak_start(True))
		delta_msec_file = float(stf.get_peak_end(True) - start_msec_file)
	
	loaded_file = stf.get_filename()[:-4] 
	event_counts = np.zeros((stf.get_size_channel(),2));
	ADPs = np.zeros((stf.get_size_channel(),2));
	thresholds = np.zeros((stf.get_size_channel(),2));  
	trace_df_dict = {}
	for trace_ in range(stf.get_size_channel()):
		AP_count_for_trace, df_for_trace = find_AP_peak_ADP_trace(trace_, threshold_value_file, deflection_direction_file, mark_option_file, start_msec_file, delta_msec_file)
		trace_df_dict['trace' + str(str(trace_).zfill(3))]= df_for_trace
	
	event_counts[trace_][1] = AP_count_for_trace
	event_counts[trace_][0] = current_start_file + (current_delta_file*trace_) ; 
		
	np.savetxt(loaded_file + '_AP_counts.csv', event_counts, delimiter=',', newline='\n'); 
	output_path = loaded_file + 'ADP_thresholds.xlsx'
	xlsx_out = pd.ExcelWriter(output_path, engine='xlsxwriter');
	for trace_name, trace_df in sorted(trace_df_dict.items()):
		trace_df.to_excel(xlsx_out, sheet_name=trace_name)
	xlsx_out.save()
	return(True)
	

def find_AP_peak_ADP_trace(*argv):
	
	""" count number of APs, find ADPs and thesholds in indicated trace with current injection/gradually increasing steps
	inputs: (time (msec) to start search, length of search region, starting current value, 
	current delta between traces, threshold value, deflection direction ('up'/'down'), mark traces (True/False))"""	
	##if times are input, use those, otherwise use peak cursor settings
	#TO DO: optional change to threshold_values and deflection_direction 
	if len(argv)>0:
		trace_selection = argv[0]
		threshold_value = float(argv[1])
		deflection_direction = argv[2]
		mark_option = argv[3]
		start_msec = float(argv[4])
		delta_msec = float(argv[5])	
	else:
		trace_selection = stf.get_trace_index()
		threshold_value = 0
		deflection_direction = 'up'
		mark_option = True
		start_msec = float(stf.get_peak_start(True))
		delta_msec = float(stf.get_peak_end(True) - start_msec)
		
	stf.set_trace(trace_selection)
	##gets AP counts and sample points in current trace 
	if deflection_direction == 'up':
		direction_input = True
	else:
		direction_input = False
		
	##count function will return number of APs in trace and sample points for subsequent functions
	trace_count, trace_sample_points_absolute = jjm_count(start_msec, delta_msec, threshold=threshold_value, up=direction_input, trace=trace_selection, mark=mark_option) 
		
	##finds afterdepolarizations--minimums between peaks
	trace_ADP_values, trace_ADP_indicies = find_ADPs(trace_sample_points_absolute)		
	trace_si = stf.get_sampling_interval()
	trace_ADP_times = [sample*trace_si for sample in trace_ADP_indicies]
	trace_AP_values, trace_AP_indicies = find_ADPs(trace_sample_points_absolute)
	trace_si = stf.get_sampling_interval()
	trace_ADP_times = [sample*trace_si for sample in trace_AP_indicies]
	trace_thresholds_indicies = find_thresholds(stf.get_trace(trace_selection), trace_si, trace_ADP_indicies)
	trace_threshold_values = [stf.get_trace(trace_selection)[index] for index in trace_thresholds_indicies]
	trace_threshold_times = [sample*trace_si for sample in trace_thresholds_indicies]
	for sample, mv in zip (trace_thresholds_indicies, trace_threshold_values):
		stf.set_marker(sample, mv)

	for x in range(len(trace_threshold_values)):
		if trace_threshold_values[x] > threshold_value or trace_threshold_values[x]<trace_ADP_values[x]:
			trace_threshold_values[x] = 'NaN'
			
	#arrays for output
	ADP_out_array = np.transpose(np.array([trace_ADP_times, trace_ADP_values]))
	threshold_out_array = np.transpose(np.array([trace_threshold_times, trace_threshold_values]))
	out_array = np.hstack([ADP_out_array, threshold_out_array])
	df_out = pd.DataFrame(out_array, columns = ['ADP time', 'ADP (mV)', 'threshold time', 'threshold (mV)'])

	return(trace_count, df_out)	
	

	
	
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

def find_APs_peaks(AP_peak_indicies):
	AP_peak_values = []
	AP_peak_indicies = []
        ##slices
    	for peak in range(len(AP_peak_indicies)-1):
            AP_peak_search = stf.get_trace()[AP_peak_indicies[peak]:AP_peak_indicies[peak+1]]
            min_value = np.min(AP_peak_search)
            min_index = AP_peak_indicies[peak] + np.argmin(AP_peak_search)
            stf.set_marker(min_index, min_value)
            AP_peak_values.append(min_value)
            AP_peak_indicies.append(min_index)


	return(AP_peak_values, AP_peak_indicies)
	
def find_thresholds(input_trace, input_trace_si, ADP_sample_points):
	#take derivative of trace, assume relevant trace is set to current trace
	#get_dv_dt_as_numpy(input_array, si, *argv)
	
	_1stderivative = dv_dt_V.get_dv_dt_as_numpy(input_trace, input_trace_si)
	#take 2nd derivative of trace
	_2ndderivative = dv_dt_V.get_dv_dt_as_numpy(_1stderivative, input_trace_si)
	_3rdderivative = dv_dt_V.get_dv_dt_as_numpy(_2ndderivative, input_trace_si)
	#will need to filter trace here
	#try with using sigma of 1
	
	_3rdderivative_filtered = trace_filtering.filter_1d_numpy(_3rdderivative, 1, False)
	
	#find peak time points
	#use ADP sample points, estimate an AP width of 10ms to work backword from to capture peak
	
	threshold_points = []
	threshold_indicies = []
	for x in range(len(ADP_sample_points)):
		if x == 0:
			earlier_point = ADP_sample_points[x] - (50/input_trace_si)
		else:
			earlier_point = ADP_sample_points[x-1]
			
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













