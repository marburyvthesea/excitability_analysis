import sys

sys.path.append('/Library/Python/2.7/site-packages');
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')

from scipy import ndimage
import stf

def filter_current_trace(sigma):
	trace_ = stf.get_trace()
	trace_filtered = scipy.ndimage.filters.gaussian_filter(trace_, sigma)
	stf.new_window(trace_filtered) 
	
	return(trace_filtered)
	
def filter_1d_numpy(array_to_filter, sigma, *argv):
	trace_ = array_to_filter
	trace_filtered = scipy.ndimage.filters.gaussian_filter(trace_, sigma)
	if argv[0]==True:
		stf.new_window(trace_filtered)
	return(trace_filtered)
	

