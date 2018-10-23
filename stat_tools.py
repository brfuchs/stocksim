# Brody Fuchs, CSU, January 2016
# brfuchs@atmos.colostate.edu

# This is a module for dealing with some of the objective analysis statistics
# including generic distributions (homework 1), 

from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import random
from copy import deepcopy
import scipy.interpolate as interpolate
import scipy.ndimage as ND
from collections import OrderedDict


def rednoise_infinite(freqs, Te):
	rednoise = 2*Te/(1+(Te**2)*(2*np.pi*freqs**2))
	return rednoise/np.sum(rednoise)

def rednoise_finite(freqs, lag1_corr):
	real_rednoise = (1-lag1_corr**2)/(1-2*lag1_corr*np.cos(freqs*np.pi/len(freqs))+lag1_corr**2)
	return real_rednoise/np.sum(real_rednoise)


def f_value(dof):
	dofs = np.array([1,3,4,5,6,7,8,9,10,20,30,40,50,75,100])
	critical_values = np.array([4.94, 3.98, 3.51, 3.21, 2.99, 2.82, 2.69, 
    	2.59, 2.50, 2.07, 1.89, 1.80, 1.74, 1.65, 1.60])

	tck = interpolate.splrep(dofs, critical_values, s=0)
	#xnew = np.arange(0, 2*np.pi, np.pi/50)
	return interpolate.splev(dof, tck, der=0)

def f_test(val1, val2, dof = 1):
	return (val1/val2) >= f_value(dof)


def significant_peaks(spectrum, rednoise, dof = 1, freqs = None):
	# need to loop thru all the peaks and red noise time series
	ratio = spectrum/rednoise
	crit_ratio = f_value(dof)
	if freqs is not None: return freqs[np.where(ratio >= crit_ratio)]
	else: return np.where(ratio >= crit_ratio)


def folding_time(data, critical_val = 0.368): # 0.368 is 1/e or the e folding time
	# this will find the autocorrelation and the orignal number of samples
	ac = autocorrelation(data) # first get the autocorrelation
	d_ac = np.gradient(ac) # take the derivative of that autocorrelation, will be negative for a while
	first_pos = np.where(d_ac > 0)[0][0] # get the first timestep it goes negative
	valid_ac = ac[:first_pos] # subset only the first part of the autocorrelation that decreases
	f = interpolate.interp1d(valid_ac, np.arange(first_pos)) # interpolate will allow to solve for the x position
	fold_time = f(critical_val) # evaluate and return
	return fold_time + 0



def _real_and_imag(complexnumber):
	return np.real(complexnumber), np.imag(complexnumber)

def time_lag(dist1, dist2, lag = 0):

	dist1_shifted = ND.interpolation.shift(dist1, -1*lag, cval = np.NaN)
	dist1_shifted = dist1_shifted[~np.isnan(dist1_shifted)]
	# dist1 shifted takes dist1 and shifts it lag indices, replaces end values with NaN's, then gets rid of the nans
	# We need to then cut dist2 in the same amount, but how we do it depends on if the lag is positive, negative or 0
	# then do the correlations with the shifted arrays

	if lag < 0: 
		dist2_cut = dist2[-1*lag:]
	elif lag > 0:
		dist2_cut = dist2[:-1*lag]
	elif lag == 0:
		dist2_cut = dist2

	return dist1_shifted, dist2_cut


def correlation_plot(dist1, dist2, tau = 12):
	# this is a function that will plot the correlation between 2 distributions as a function of lag
	correlations = []
	for lag in np.arange(-1*tau, tau+1, 1): # loop thru each of the lags
		#print lag
		d1, d2 = time_lag(dist1, dist2, lag = lag)

		#print dist1_shifted.shape, dist2_cut.shape
		c = stats.pearsonr(d1, d2)

		correlations.append(c[0]) # c has the r value and 2 tailed p value, only want the r value

	return np.array(correlations)



def confidence_level(dist, level = 95):
	""" given some level, output the value that corresponds to that confidence. This is 2 tailed as of now.
			May want to change that in the future """
	# to do this, need the difference between the level and 100
	lowval = (100-level)/2
	highval = 100 - lowval
	lowbound = np.percentile(dist, lowval)
	highbound = np.percentile(dist, highval)
	return lowbound, highbound


def autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

# here are more general functions
def moment(data, number = 1):
	# return a moment of the distribution
	# 1 is the mean, 2 is the variance, 3 is the skewness and 4 is the kurtosis
	# no matter what, need to calculate the mean
	mean = np.average(data)
	N = data.shape[0]
	# now if moment number is 2 or more, need to calculate variance
	if number == 1:
		return mean
	else: # moment number greater than 1, need to calculate variance
		stddev = np.std(data)
		if number == 2:
			return stddev**2 # this is the variance
		elif number == 3: # skewness
			cube_diff = np.sum(np.array([(_ - mean)**3 for _ in data]))
			return cube_diff/(N*stddev**3)
		elif number == 4: # kurtosis
			fourth_diff = np.sum(np.array([(_ - mean)**4 for _ in data]))
			return (fourth_diff/(N*stddev**4))-3

def get_moments(data, moments=[1,2,3,4]):
	out = []
	for m in moments:
		out.append(moment(data, number=m))

	return out


class Distribution(object):
	# this is the generic distribution class, can take in any type of data and make a distribution out of it
	# have a feeling this will be used a lot in this class. And maybe in my research, for that matter
	def __init__(self, fname=None, data=None):
	    	self.fname = fname
		if data is not None:
			self.set_data(data)
		pass

		# COULD MAYBE DO A DECORATOR WITH THE NORMAL = TRUE OR FALSE ON A LOT OF THE FUNCTIONS

	def set_data(self, data, normal = False):
		self.rawdata = data
		if normal:
			self.standard = self.rawdata.copy()
		else:
			pass


	def read_data_file(self, fname, filetype = 'csv', **kwargs):
		""" Read in data from some type of file """
		blah = np.genfromtxt(fname, **kwargs)
		self.rawdata = blah
		self.average()
		self.sigma()
		self.standardize()

	def red_noise(self, a, nsamples = 5000):
		b = np.sqrt(1 - a**2)
		initval = np.random.normal(0, 1, 1)[0]
		data = [initval]
		for i in range(nsamples-1):
			data.append(a*data[i]+b*np.random.normal(0, 1, 1)[0])

		self.standard = np.array(data)
			

	def pdf(self, bins, dist_type = 'standard'):
		""" Take the raw data and make a pdf out of it """
		if dist_type == 'standard': # do a pdf of the standardized data
			hist = np.histogram(self.standard, bins = bins)[0]
		elif dist_type == 'raw':
			hist = np.histogram(self.rawdata, bins = bins)[0]
		return hist/np.sum(hist)


	def cdf(self):
		""" Make the cumulative distribution out of the pdf """
		pass

	def confidence_level(self, level = 95):
		""" given some level, output the value that corresponds to that confidence. This is 2 tailed as of now.
				May want to change that in the future """
		# to do this, need the difference between the level and 100
		lowval = (100-level)/2
		highval = 100 - lowval
		lowbound = np.percentile(self.standard, lowval)
		highbound = np.percentile(self.standard, highval)
		return lowbound, highbound



	def standardize(self):
		""" Take the data and standardize it with the average and standard deviation, like z instead of x """
		# first get the average and the sigma to be able to normalize
		self.standard = (self.rawdata - self.average())/self.sigma()

	def average(self):
		""" Just gives the average of the distribution """
		self.rawaverage = np.average(self.rawdata)
		return self.rawaverage
		

	def sigma(self):
		""" Gives the standard deviation of the distribution """
		
		self.rawsigma = np.std(self.rawdata)
		return self.rawsigma		

	def prob_greater(self, value, normal = True):
		if normal: # just gonna start under this assumption that it's normalized
			total_number = self.standard.shape[0]
			greater_number = np.where(self.standard >= value)[0].shape[0]
			return greater_number/total_number

	def prob_less(self, value, normal = True):
		if normal: # just gonna start under this assumption that it's normalized
			total_number = self.standard.shape[0]
			less_number = np.where(self.standard <= value)[0].shape[0]
			return less_number/total_number

	def prob_delta(self, value = 0, delta = 1, normal = True):
		# this returns the probability that the distribution is between the value +/- delta
		if normal:
			pmore = self.prob_less(value-delta)
			pless = self.prob_less(value+delta)
			return pless - pmore

	def standard_to_real(self, value):
		# assumed this is standard, dont need a check
		return value*self.sigma()+self.average()

	def real_to_standard(self, value):
		return (value - self.average())/self.sigma()

	def sample(self, N, normal = True):
		# just returns a random sample of data picked from the distribution and not replaced back in
		if normal: return np.random.choice(self.standard, N, replace = True) # replace = True is MUCH faster than replace = False
		else: return np.random.choice(self.rawdata, N, replace = True) 

	def consecutive_sample(self, N, normal = True):
		# this is dealing with timeseries and autocorrelation so need to grab N consecutive values
		totalsize = self.standard.shape[0]
		starting_spot = np.random.choice(np.arange(totalsize - N)) # cant pick the last 100 values if the starting spot is N - 20
		#print starting_spot 
		return self.standard[starting_spot:starting_spot+N]

	def acorr(self):
	    """
	    http://stackoverflow.com/q/14297012/190597
	    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
	    """
	    return autocorrelation(self.standard)

	def Nfraction(self, critical_val = 0.368): # 0.368 is 1/e or the e folding time
		# this will find the autocorrelation and the orignal number of samples
		ac = self.acorr() # first get the autocorrelation
		d_ac = np.gradient(ac) # take the derivative of that autocorrelation, will be negative for a while
		first_pos = np.where(d_ac > 0)[0][0] # get the first timestep it goes negative
		valid_ac = ac[:first_pos] # subset only the first part of the autocorrelation that decreases
		f = interpolate.interp1d(valid_ac, np.arange(first_pos)) # interpolate will allow to solve for the x position
																# where y is the critical_val, (this is done by switching x/y)
		Nfraction = f(critical_val) # evaluate and return
		return np.round(Nfraction)



	def sample_means(self, N, nsamples, normal = True):
		# this grabs a bunch of subsamples and averages them and returns the averages
		#out = []
		#for i in range(nsamples): # loop thru the number of samples needed
		#	out.append(np.average(self.sample(N, normal = normal)))
			# append the avearge of a sample of N values to the out list that will get turned into an array
		out = [np.average(self.sample(N, normal = normal)) for i in range(nsamples)]

		return np.array(out)

	def sample_pdf(self, N, nsamples, bins, normal = True):
		# this will call the sample means and just return a PDF of those values
		values = self.sample_means(N, nsamples, normal = normal)
		hist = np.histogram(values, bins)[0]
		return hist/np.sum(hist)

	def sample_prob_less(self, value, N, nsamples, normal = True):
		# given a Z value, what is the probably that the average of a sample of size N is less than 'value'?
		# first grab a bunch of means
		means = self.sample_means(N, nsamples, normal = normal)
		less = np.where(means <= value)[0].shape[0]
		return less/means.shape[0]		

	def sample_prob_greater(self, value, N, nsamples, normal = True):
		# given a Z value, what is the probably that the average of a sample of size N is less than 'value'?
		# first grab a bunch of means
		means = self.sample_means(N, nsamples, normal = normal)
		more = np.where(means >= value)[0].shape[0]
		return more/means.shape[0]

	def subset(self, condition):
		# this will make a copy of the object with subsetted data based on the condition
		# need to subset both the raw and the standardized
		new = deepcopy(self)
		new.rawdata = self.rawdata[condition]
		new.standard = self.standard[condition]

		return new

	def jacknife(self, func):
		n = len(self.standard)
		out = []
		for i in range(n): # loop thru how many samples we have
			out.append(func([self.standard[:n], self.standard[n+1:]]))


		return out


		# down here is timeseries stuff
	def power_spectrum(self, window = None, sample_rate = 1, nchunks = 1, overlap = 50, 
				dof_factor = 1.0, chunk_len = None, renormalize_chunks = False):
		# window is how to window the raw data before doing the FFT
		# sample rate is for the output frequencies
		# nchunks is the number of chunks to resample the data to boost statst
		# if nchunks is more than 1, doing chunking, but need to know how much to overlap the chunks
		# overlap is a percentage

		if chunk_len is not None: # if specifying chunk length, have to do different calculations
			N = chunk_len
			nchunks = int(len(self.rawdata)/N)

		else:
			N = int(len(self.rawdata)/nchunks)


		odd_bool = np.fmod(N,2)
		nu = np.fft.fftfreq(N, sample_rate)


		if odd_bool: freq = nu[0:int(N/2+1)]
		else: freq = np.abs(nu[0:int(N/2+1)])

		# gonna need to look thru each chunk, then take the FFT and average at the end
		# so this stuff is gonna need to be in a loop
		overlap_chunks = int(nchunks*(100/overlap) - 1)
		power = np.zeros((overlap_chunks,int((N/2)+1)))
	#	print 'output shape: {}'.format(power.shape)
		for oc in range(overlap_chunks):

			first_ind = int(oc*N/2)
			last_ind = int(oc*N/2+N)
			#print first_ind, last_ind
			indata = self.rawdata[first_ind:last_ind]
			if renormalize_chunks:
				indata -= np.average(indata)

			if window is None:
				f = np.fft.rfft(indata)/N
			
			else: # if there is a certain window specified
				windowed_data = window(N)*indata
				f = np.fft.rfft(windowed_data)/N

			im_ang = np.angle(f)
			power[oc] = 2*np.absolute(f)**2		

		avg_power = np.average(power, axis = 0)
		dof = 2*nchunks*dof_factor

		return freq, avg_power/np.sum(avg_power), im_ang, dof





	def old_school_coeffs(self):
		Ak = []
		Bk = []
		for k in range(int(t.shape[0]/2)):
			Ak.append(2*np.average((np.cos(2*np.pi*k*t/total_time)*dprime)))
			Bk.append(2*np.average((np.sin(2*np.pi*k*t/total_time)*dprime)))
		Ak = np.array(Ak)
		Bk = np.array(Bk)

		Ck = (Ak**2 + Bk**2)/2

		return Ck


	def autocorrelation_plot(self, max_lag = 10, figname = None, **kwargs):
		auto = autocorrelation(self.rawdata)
		
		fig, ax = plt.subplots(1,1)
		ax.plot(auto[:max_lag], color = 'darkorange', **kwargs)
		ax.set_xlabel('Time (lag)')
		ax.set_ylabel('Correlation')
		ax.axhline(y = 1/2.718, color = 'black', linewidth = 1, linestyle='dashed')
		ax.axhline(y = 0, color = 'black', linewidth = 1, linestyle='dashed')
		ax.grid(True)

		if figname is not None: plt.savefig(figname)
		return fig, ax





		
class Dataset(object):

	# this will first do EOF analysis, but could also do some other stuff and integrate with the Distibution class maybe??

	def __init__(self, data = None, data_type = 'attributes', method = 'svd', nx = None, ny = None, 
				weights = None, standardize = False, keep_eofs=10):
		self.data_type = data_type # can be attributes or spatiotemporal
		self.data = data
		self.method = method
		self.nx = nx # these are for spatial (or 2D) data, needed to reshape stuff down the road
		self.ny = ny
		self.weights = weights
		self.standardize = standardize
		self.keep_eofs = keep_eofs
		if self.data is not None and self.data_type == 'spatial': 
			self.rows, self.cols = self.data.shape
		elif self.data is not None and self.data_type == 'attributes':
			pass

	def set_data(self, data, labels = None):
		self.data = data
		if self.data_type == 'attributes':
			self.labels = labels
		self.rows, self.cols = self.data.shape

	def set_eof_labels(self, labels):
		self.eof_labels = labels

	def get_eof_labels(self):
		return self.eof_labels

	def set_pc_labels(self, labels):
		self.pc_labels = labels

	def get_pc_labels(self):
		return self.pc_labels


	def flip_eof_and_pc(self, rank = 1):
		# this will flip the EOF by -1 and corresponding PC timeseries if needed
		self.eigenvectors[:,rank-1] *= -1
		self.std_pc[rank-1] *= -1

	def auto_flip(self):
		for n in range(1, self.eigenvectors.shape[-1]+1):
			this_eof = self.get_eof(rank = n)
			imax = np.argmax(np.abs(this_eof))
			if np.sign(this_eof[imax]) == -1:
				self.flip_eof_and_pc(rank = n)



	def into_matrix(self, varlist = None):
		if isinstance(self.data, dict) and self.data_type == 'attributes':
			self.raw_matrix = np.zeros((len(varlist), len(self.data[self.data.keys()[0]])))
			self.anom_matrix = np.zeros_like(self.raw_matrix)
			self.standard_matrix = np.zeros_like(self.raw_matrix)

			for gv in range(len(varlist)):
				self.raw_matrix[gv,:] = self.data[varlist[gv]]
				self.anom_matrix[gv,:] = self.data[varlist[gv]]-np.average(self.data[varlist[gv]])
				self.standard_matrix[gv,:] = (self.data[varlist[gv]] - np.average(self.data[varlist[gv]]))/np.std(self.data[varlist[gv]]) 

		elif self.data_type == 'spatial':
			# need to build in a weighting option
			self.raw_matrix = deepcopy(self.data)
			#self.standard_matrix = (self.raw_matrix - np.average(self.raw_matrix))/np.std(self.raw_matrix)
			self.indata = deepcopy(self.raw_matrix.T)
			if self.weights is not None:
				# need to weight the self.indata
				self.indata = self.indata*self.weights[:,np.newaxis]
				self.weighted = True
			else:
				self.weighted = False


		else:
			print 'Data type does not match data input'
			sys.exit()


	def covariance_matrix(self):
		if self.method == 'svd':
			self.C = (1/self.standard_matrix.shape[1])*self.standard_matrix.dot(self.standard_matrix.T)
			return self.C
		elif self.method == 'covariance':
			if self.weights is not None:

				if self.rows > self.cols:
					self.C = (1/self.rows)*self.raw_matrix.T.dot(self.raw_matrix*self.weights[np.newaxis,:])
					self.sumover = 'rows'
				else:
					self.C = (1/self.cols)*(self.raw_matrix*self.weights[np.newaxis,:]).dot(self.raw_matrix.T)
					self.sumover = 'cols'

			else:

				if self.rows > self.cols:
					self.C = (1/self.rows)*self.raw_matrix.T.dot(self.raw_matrix)
					self.sumover = 'rows'
				else:
					self.C = (1/self.cols)*self.raw_matrix.dot(self.raw_matrix.T)
					self.sumover = 'cols'


			return self.C

	def calculate_eofs(self):
		if self.method == 'svd':

			if self.data_type == 'spatial':
				U, s, V = np.linalg.svds(self.indata, self.keep_eofs, full_matrices = False)
				print 'U shape: {}'.format(U.shape)
				self.nev = U.shape[-1]
				print self.ny, self.nx, self.nev
				self.eigenvectors = U.reshape(self.ny, self.nx, self.nev)
				self.variance = 100*s**2/np.sum(s**2)
				self.pc = V

			elif self.data_type =='attributes':
				#print self.standard_matrix
				# gonna try to get rid of the nans by setting them to 0 cuz that's causing the svd method to fail
				self.standard_matrix[np.isnan(self.standard_matrix)] = 0.0
				U, s, V = np.linalg.svd(self.standard_matrix, full_matrices = False)

				self.eigenvectors = U
				self.s = deepcopy(s)
				self.variance = 100*s**2/np.sum(s**2)
				self.pc = V


				# standardize the PCs
			self.std_pc = self.pc.copy()
			for ipc in range(self.std_pc.shape[0]):
			    mu = np.average(self.std_pc[ipc])
			    sig = np.std(self.std_pc[ipc])
			    self.std_pc[ipc] = (self.std_pc[ipc]-mu)/sig

			if self.data_type == 'spatial':
				# need to convert EOFs into physical units
				self.d = np.zeros_like(self.indata)
				print 'self.d.shape: {}'.format(self.d.shape)
				for j in range(self.d.shape[1]):
				    blah = (self.indata.dot(self.std_pc[j].T))/self.d.shape[1]
				    #print 'blah shape: {}'.format(blah.shape)
				    self.d[:,j] = blah

				self.d = self.d.reshape(self.ny, self.nx, self.nev)



		elif self.method == 'covariance':

			es = scipy.linalg.eig(self.C)
			# pretty sure the order depends on self.sum_over
			self.evals, self.evecs = es[0], es[1]
			self.variance = 100.0*np.real(self.evals/np.sum(self.evals))
			self.nev = self.evals.shape[0]
			self.pc = self.evecs.T
			print 'evals.shape: {}, evecs.shape: {}'.format(self.evals.shape, self.evecs.shape)

			self.evectors = self.evecs.T.dot(self.raw_matrix)
			self.eigenvectors = self.evectors.reshape(self.nev, self.ny, self.nx)

		

				# standardize the PCs
			self.std_pc = self.pc.copy()
			for ipc in range(self.std_pc.shape[0]):
			    mu = np.average(self.std_pc[ipc])
			    sig = np.std(self.std_pc[ipc])
			    self.std_pc[ipc] = (self.std_pc[ipc]-mu)/sig

			if self.data_type == 'spatial':
				# need to convert EOFs into physical units
				self.d = np.zeros_like(self.evectors)
				print 'self.d.shape: {}'.format(self.d.shape)
				for j in range(self.d.shape[0]):
				    blah = (self.std_pc[j].T.dot(self.raw_matrix))/self.d.shape[1]
				    # this looks like it should I think !!!!
				    print 'blah shape: {}'.format(blah.shape)
				    self.d[j,:] = blah

				self.d = self.d.reshape(self.nev, self.ny, self.nx)


	def get_eof(self, rank = 1, physical_units = False):
		if physical_units:
			return_val = self.d
#			return self.d[...,rank-1]
		else:
#			return self.eigenvectors[...,rank-1]
			return_val = self.eigenvectors

		if self.data_type == 'attributes':
			return return_val[...,rank-1]
		else:
			return return_val[rank-1]



	def get_pc(self, rank = 1, standard = True):
		if standard: return self.std_pc[rank-1]
		else: return self.pc[rank-1]
	


	def get_variance(self, rank = 1):
		return self.variance[rank-1]

	def get_variance_range(self, ranks = [1,2]):
		tot_variance = 0.
		for r in ranks:
			tot_variance += self.get_variance(rank = r)

		return tot_variance

	def plot_eofs(self, ranks = [1,2,3], figsize = (10,8), linewidths = 2):

		n_eofs = len(ranks)
		colors = ['black', 'purple', 'blue', 'green', 'gold', 'darkorange', 'red']

		fig, ax = plt.subplots(n_eofs, 2, figsize=figsize)
		# loop thru the eofs
		print 'N EOFs: {}'.format(n_eofs)

		for ne in range(n_eofs):
			ax[ne,0].plot(self.get_eof(rank = ranks[ne]), color = colors[ne], linewidth = linewidths)
			ax[ne,0].set_title('EOF %d'%(ranks[ne]))

			ax[ne,1].plot(self.get_pc(rank = ranks[ne]), color = colors[ne], linewidth = 1)
			ax[ne,1].set_title('PC %d: %.1f%% variance'%(ranks[ne], self.get_variance(rank = ranks[ne])))




		#ax.legend(loc = 'best', prop = {'size': 8})
		for a in ax[:,0]:
			a.set_xticks(np.arange(len(self.eof_labels)))
			a.set_xticklabels(self.eof_labels, fontsize = 12)
			a.axhline(y = 0, color = 'black', linewidth = 2)
			a.grid(True)
		  
		for a in ax[:,1]:
		    a.grid(True)
		    a.axhline(y = 0, color = 'black', linewidth = 2)
		    #a.set_xlim(time[0], time[-1])
		    #a.set_xlabel('Day of year')


		plt.tight_layout()

		return fig, ax



	def calculate_transfer_function(self):
		self.transfer_function = np.dot(self.eigenvectors[:, :self.keep_eofs], (self.s[:self.keep_eofs]* self.pc[:self.keep_eofs].T).T)
		return self.transfer_function

	def reconstruct_data(self, keys=None):
		if hasattr(self, 'transfer_function'):
			pass
		else:
			_ = self.calculate_transfer_function()
		out = OrderedDict()

		if keys is None:
			keys = self.data.keys()
# now loop thru the variables 
		for ik, _k in enumerate(keys):
    #if ik == 0:
        #print blah[ik].shape
			out[_k] = self.transfer_function[ik]*np.std(self.raw_matrix[ik])+np.average(self.raw_matrix[ik])
		return out



    	def plot_covariance_matrix(self, fig=None, ax=None):
    	#pass

    		if (fig is None) or (ax is None):
	    		fig, ax = plt.subplots(1,1, figsize=(8,6))

		C = self.covariance_matrix()
		C[np.isnan(C)] = 0
		#print C

		# Let's make a pcolormesh of this?
		#fig, ax = plt.subplots(1,1, figsize=(8,6))
		Cpc = ax.pcolormesh(np.arange(len(self.data.keys())+1), np.arange(len(self.data.keys())+1), C,
		             cmap=plt.cm.bwr, vmin=-1, vmax=1)
		Ccb = plt.colorbar(Cpc)
		Ccb.set_label('Correlation')
		ax.set_yticks(0.5+np.arange(len(self.data.keys())))
		ax.set_yticklabels(self.data.keys())

		ax.set_xticks(0.5+np.arange(len(self.data.keys())))
		ax.set_xticklabels(self.data.keys())
		plt.xticks(rotation=90)

		return fig, ax




