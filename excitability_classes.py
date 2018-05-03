import sys
sys.path.append('/Users/johnmarshall/Documents/Analysis/PythonAnalysisScripts/stimfitscripts/')
import dv_dt_V as dv

class exitability_trace(object):
	##class to for information for traces in excitability experiment
	def __init__(self, voltage_trace):
		self.voltage_trace = voltage_trace
		
	def calc_dv_dt(self):
	#calculate 1st derivative of trace
		dv_dt_out = dv.get_dv_dt(*argv)
		return(dv_dt_out)