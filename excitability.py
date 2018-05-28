import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')
sys.path.append('/Library/Python/2.7/site-packages/')
sys.path.append('/Users/johnmarshall/miniconda3/envs/py27/lib/python2.7/site-packages')
sys.path.append('/Applications/MacPorts/dv_dt_V_analysis/')
sys.path.append('/Applications/MacPorts/stimfit.app/Contents/Frameworks/stimfit/')
import stf
import excitability_subfunctions as es
import dv_dt_V 
import numpy as np
import pandas as pd
import xlsxwriter
import stf
import spells
#import excitability_analysis


class excitability_file(object):
	##compile file with excitability traces
	def __init__(self):
		self.size_channel = stf.get_size_channel()
		self.name = stf.get_filename()[-12:]
		self.full_path = stf.get_filename()
	
	#compile APs from indiv traces by calling
	def find_AP_peaks(start_msec, delta_msec, current_start, current_delta, threshold_value, deflection_direction, mark_option):
		return()
	
	# make this do a dv_dt_by_V	
	def compile_sweeps(self):
		sweeps_dict = {}
		for sweep in range(self.size_channel):
			stf.set_trace(sweep)
			trace = excitability_trace(stf.get_trace(), sweep)
			sweeps_dict[sweep] = trace.calc_dv_x_dt()

		return(sweeps_dict)

	def output_dict(self, dictionary_with_sweeps):
		dict_df = pd.DataFrame(dictionary_with_sweeps) 
		dict_df.to_excel(str(self.name)+'.xlsx')

		return()
		
	def calc_R_input(current_start_input, current_delta_input, *argv):
		"""Calculate input resistance by measuring hyperpolarizing traces"""
		df_out = calc_R_input(current_start_input, current_delta_input, *argv)
		return(df_out)	


class excitability_trace(object):
	##class to for information for traces in excitability experiment
	def __init__(self, voltage_trace, sweep_number):
		self.voltage_trace = voltage_trace
		self.sweep_number = sweep_number 
		
	def calc_dv_x_dt(self, deriv_type=1):
		###redo this function as calc_dv_x_dt(self, deriv_type=1)
		"""calculate derivatives of trace, 1st derivative if no input, if input 2 then calcluates 2nd derivative"""
	#calculate x derivative of trace
		stf.set_trace(self.sweep_number)
		if deriv_type == 1:
			dv_dt_out = dv_dt_V.get_dv_dt_as_numpy(self.voltage_trace, stf.get_sampling_interval())
		## can actually just make this one loop
		else:
			input = self.voltage_trace
			count = 0
			while count <= deriv_type:
				dv_dt_out = dv_dt_V.get_dv_dt_as_numpy(input, stf.get_sampling_interval())
				input = dv_dt_out
				count += 1
		return(dv_dt_out)
	##add a get dv_dt_as_numpy here

	def plot_dv_dt_by_V(self):
		dv_dt_for_plot = self.calc_dv_x_dt()
		V_values =  stf.get_trace(self.sweep_number)[:-1]
		dv_dt_V.plot_dv_dt(V_values, dv_dt_for_plot)
		return(True)
	
	def plot_derivative(self, *argv):
		"""1st derivative if no input, if input 2 then calcluates 2nd derivative"""
		to_plot = self.calc_dv_x_dt(self, *argv)
		stf.new_window(to_plot)
		return(True)
	
	def get_ADP_threshold_trace(self, *argv):
		stf.set_trace(self.sweep_number)
		AP_count, ADP_threshold_df = find_ADP_thresholds_trace(*argv)
		return(AP_count, ADP_threshold_df)
			
	def save_ADP_threshold_results(self, *argv):	
		stf.set_trace(self.sweep_number)
		AP_count_for_trace, ADP_threshold_df_for_trace = find_ADP_thresholds_trace(*argv)
		output_path = stf.get_filename()[:-4] + 'trace' + str(str(trace_).zfill(3)) + '_ADP_thresholds.xlsx'
		ADP_threshold_df_for_trace.to_excel(output_path, sheet_name='trace' + str(str(trace_).zfill(3)))
		return(True)
		
	def fit_mono_exponential(self, *argv):
		stf.set_trace(self.sweep_number)
		fit_dict = stf.leastsq(1)
		##{'SSE': 0.8533321749496918, 'Tau_0': 1.1497377662822952, 'Amp_0': 904.9731610575948, 'Offset': -275.5736999511719}
		V_amplitude = fit_dict['Amp_0']
		return(V_amplitude)

##	batch load files from list and convert to excitability file objects	
	
def load_files_from_list(list_of_excitability_files):
	output_dict = {}
	for item in list_of_excitability_files:
		stf.file_open(str(item))
		output_dict[str(item)[-12:]] = excitability_file()
	return(output_dict)	

##	
def compile_file():
	"""ideally run this function and then be able to select options for indiv traces by entering names in shell"""
	file_dict = {}
	for sweep in range(stf.get_size_channel()):
		stf.set_trace(sweep)
		sweep_name = str(sweep)
		file_dict[sweep_name] = excitability_trace(stf.get_trace(), sweep)
		
	return(file_dict)	


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
		ps = stf.get_peak_start(True)
		pe = stf.get_peak_end(True)
		start_msec_file = float(ps)
		delta_msec_file = float(pe - ps)
	
	loaded_file = stf.get_filename()[:-4] 
	event_counts = np.zeros((stf.get_size_channel(),2));
	ADPs = np.zeros((stf.get_size_channel(),2));
	thresholds = np.zeros((stf.get_size_channel(),2));  
	trace_df_dict = {}
	for trace_ in range(stf.get_size_channel()):
		AP_count_for_trace, df_for_trace = find_ADP_thresholds_trace(trace_, threshold_value_file, deflection_direction_file, mark_option_file, start_msec_file, delta_msec_file)
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
	

def find_ADP_thresholds_trace(*argv):
	
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
	trace_count, trace_sample_points_absolute = es.jjm_count(start_msec, delta_msec, threshold=threshold_value, up=direction_input, trace=trace_selection, mark=mark_option) 
		
	##finds afterdepolarizations--minimums between peaks
	trace_ADP_values, trace_ADP_indicies = es.find_ADPs(trace_sample_points_absolute)		
	trace_si = stf.get_sampling_interval()
	trace_ADP_times = [sample*trace_si for sample in trace_ADP_indicies]
				
	##finds thresholds
	trace_thresholds_indicies = es.find_thresholds(stf.get_trace(trace_selection), trace_si, trace_ADP_indicies)
		
	trace_threshold_values = [stf.get_trace(trace_selection)[index] for index in trace_thresholds_indicies]
	trace_threshold_times = [sample*trace_si for sample in trace_thresholds_indicies]
		
	for sample, mv in zip (trace_thresholds_indicies, trace_threshold_values):
		stf.set_marker(sample, mv)
	##calculate instantaneous Hz based on ADP times and threshold times
	Hz_ADP = np.array([1000/float(trace_ADP_times[x+1]-trace_ADP_times[x]) for x in range(0, len(trace_ADP_times)-1)])
	Hz_threshold = np.array([1000/float(trace_threshold_times[x+1]-trace_threshold_times[x]) for x in range(0, len(trace_threshold_times)-1)])
	
	#if threshold values are clearly implausible substitutes 'NaN' for value
	for x in range(len(trace_threshold_values)):
		if trace_threshold_values[x] > threshold_value or trace_threshold_values[x]<trace_ADP_values[x]:
			trace_threshold_values[x] = 'NaN'
			
	#arrays for output
	ADP_out_array = np.transpose(np.array([trace_ADP_times, trace_ADP_values]))
	threshold_out_array = np.transpose(np.array([trace_threshold_times, trace_threshold_values]))
	out_array = np.hstack([ADP_out_array, threshold_out_array])
	df_out = pd.DataFrame(out_array, columns = ['ADP time', 'ADP (mV)', 'threshold time', 'threshold (mV)'])
	
	
	return(trace_count, df_out)	
	
def calc_R_input(current_start, current_delta, *argv):
	"""inputs: 
	current_start = value of 1st current injection
	current_delta
	list of hyperpolarizing traces, starting with 0
	e.g. -150, 50, 0, 1, 2, 3 will calculate the input resistance for the 1st four traces, assuming 1st injection was -150pA 
	and it went up by 50pA each sweep"""
	hyperpolarizing_sweeps = {}
	fit_voltage_amplitudes = []
	measured_voltage_amplitudes = []
	sweeps = [sweep for sweep in argv]
	injected_current = [(current_start+(current_delta*x)) for x in range(len(argv))]
	for sweep in argv:
		hyperpolarizing_sweeps[sweep] = excitability_trace(stf.get_trace(sweep), sweep)
		fit_voltage_amplitudes.append(hyperpolarizing_sweeps[sweep].fit_mono_exponential())
		stf.set_trace(sweep)
		measured_voltage_amplitudes.append(np.mean(stf.get_trace()[stf.get_peak_start():stf.get_peak_end()])-stf.get_base())
	
	##calculations for Rinput from fit amplitudes and measured hyperpolarizations
	# V = IR, R = V/I 
	R_input_fit = [abs(V/I) for V, I in zip(fit_voltage_amplitudes, injected_current)]
	R_input_measured = [abs(V/I) for V, I in zip(measured_voltage_amplitudes, injected_current)]
		
	to_output = np.array([sweeps, fit_voltage_amplitudes, measured_voltage_amplitudes, injected_current, R_input_fit, R_input_measured])
	
	df_to_output = pd.DataFrame(np.transpose(to_output), columns = ['sweeps', 'SS amplitude (mV) fit', 'SS amplitude (mV) measured', 'injected current', 'Rinput from fit', 'Rinput from measured values'])
	output_path = stf.get_filename()[:-4] + 'Rinput.xlsx'
	df_to_output.to_excel(output_path, sheet_name='Rinput')	
	return(True)
	

	
	
	
		
		
	
	
	
	
	