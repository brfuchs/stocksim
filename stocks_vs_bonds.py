from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt 

import os
import glob
import pandas as pd 
import csv
import stat_tools

DPI = 150

yield_10yr = 0.035 # % yield on a 10 year treasury
inflation = 0.02

stock_file = 'stock_returns.txt'

stock_data = pd.read_csv(stock_file, delimiter='\t', names=['year', 'returns'])

stock_data = stock_data.reindex(index=stock_data.index[::-1])


rets = stock_data['returns'].values
years = stock_data['year'].values

prices = [1]

for r in rets[:-1]:
	prices.append(prices[-1]*(1+(r/100.0)))

prices = np.array(prices)

pos_rets = len(np.where(rets >= 0)[0])
neg_rets = len(np.where(rets < 0)[0])

pct_pos = pos_rets/(pos_rets+neg_rets)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

barlist = ax.bar(years, rets, edgecolor='none', color='green', width=0.7)

neg_ind = np.where(rets < 0)[0]
for n in neg_ind:
	barlist[n].set_color('red')

ax.set_xlim(1868, 2020)
ax.grid(True)

ax.set_xlabel('Year')
ax.set_ylabel('% change')
ax.set_title('Stock market returns by year')

plt.tight_layout()

plt.savefig('ret_timeseries.png', dpi=DPI)

# now do a power spectrum

score_dev = (rets-rets.mean())
score_data = stat_tools.Distribution(data=score_dev)
score_spec = score_data.power_spectrum()

#freq, avg_power/np.sum(avg_power), im_ang, dof

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(score_spec[0], score_spec[1], color='dodgerblue', linewidth=1.5)

ax.set_xlabel('Frequency (cycles)')
ax.set_ylabel('Normalized Power')
ax.grid(True)

ax.set_title('Yearly return power spectrum')


plt.tight_layout()

plt.savefig('return_powerspec.png', dpi=DPI)

# now let's do autocorrelation

fig, ax = score_data.autocorrelation_plot(max_lag=50, linewidth=2)

ax.set_xlabel('Lag time (years)')
ax.set_ylabel('Correlation')
ax.set_title('Autocorrelation of yearly stock market returns')


plt.tight_layout()
plt.savefig('return_autocorrelation.png', dpi=DPI)






ret_bins = np.arange(-50, 50, 2.5)

ret_hist = np.histogram(rets, bins=ret_bins)[0]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.bar(ret_bins[:-1], ret_hist, edgecolor='none', color='dodgerblue')

ax.axvline(x=0.0, color='black', linewidth=3)

ax.set_xlabel('Yearly return (%)')
ax.set_ylabel('Frequency')

ax.set_title('Distribution of stock market returns since %d\nMedian: %.1f%%, Average: %.1f%%, Std dev: %.1f%%'%(\
										years[0], np.median(rets), np.average(rets), np.std(rets)))
ax.grid(True)


plt.tight_layout()
plt.savefig('ret_hist.png', dpi=DPI)

print 'pct pos: {}'.format(pct_pos*100.0)

# okay now let's do some sucessive 5, 10, 15, 30 year returns
ret_lengths = [5, 10, 15, 20, 25, 30]
ret_values_forward = []
ret_values_forward_avg = []
ret_values_backward = []
ret_values_backward_avg = []

# do the forward returns with putting it in all at once
for ir in ret_lengths:
	# need to loop thru each year
	year_returns = []
	for iy in range(len(years) - ir):
		this_val = 1
		for j in range(ir):
			this_val *= 1+(rets[iy+j]/100.0)
			#print this_val, rets[iy+j]

		year_returns.append(this_val)

	ret_values_forward.append(np.array(year_returns))


# do the forward returns with dollar cost averaging
for ir in ret_lengths:
	# need to loop thru each year
	year_returns = []
	for iy in range(len(years) - ir):
		this_val = 1/ir
		yearly_inv = 1/ir
		for j in range(ir):
			this_val = this_val*(1+rets[iy+j]/100.0) + yearly_inv
			#print this_val, rets[iy+j]

		year_returns.append(this_val)

	ret_values_forward_avg.append(np.array(year_returns))





# do the backward returns for putting it in all at once
for ir in ret_lengths:
	# need to loop thru each year
	year_returns = []
	for iy in range(ir, len(years)):
		this_val = 1
		for j in range(-1*ir, 0):
			this_val *= 1+(rets[iy+j]/100.0)
			#print this_val, rets[iy+j]

		year_returns.append(this_val)

	ret_values_backward.append(np.array(year_returns))




neg_rets = np.where(rets < -10.0)[0]
print '{:d} years with returns worse than -10 percent'.format(len(neg_rets))


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

l5 = ax.plot(years[:-5], ret_values_forward[0], color='dodgerblue', label='5 years: %.1f'%(np.median(ret_values_forward[0])), linewidth=1.5)
l10 = ax.plot(years[:-10], ret_values_forward[1], color='green', label='10 years: %.1f'%(np.median(ret_values_forward[1])), linewidth=1.5)
l15 = ax.plot(years[:-15], ret_values_forward[2], color='0.5', label='15 years: %.1f'%(np.median(ret_values_forward[2])), linewidth=1.5)
l20 = ax.plot(years[:-20], ret_values_forward[3], color='darkorange', label='20 years: %.1f'%(np.median(ret_values_forward[3])), linewidth=1.5)
l25 = ax.plot(years[:-25], ret_values_forward[4], color='purple', label='25 years: %.1f'%(np.median(ret_values_forward[4])), linewidth=1.5)
l30 = ax.plot(years[:-30], ret_values_forward[5], color='gold', label='30 years: %.1f'%(np.median(ret_values_forward[5])), linewidth=1.5)
nbar = ax.bar(years[neg_rets], rets[neg_rets]/10.0, color='red', edgecolor='none', width=0.8)

ax.set_xlabel('Starting year')
ax.set_ylabel('Returns on $1')

ax.axhline(y=0, color='black')
ax.axhline(y=1, color='0.4')
ax.set_ylim(-5, 50)
ax.set_xlim(1870, 2020)

ax.set_title('Stock market aggregate returns beginning at each year\n and yearly returns of 10% drop or worse')

ax.legend(loc='best', prop={'size': 8})
ax.grid(True)
plt.tight_layout()

plt.savefig('historical_forward_returns.png')

########################################################################################
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

l5 = ax.plot(years[:-5], ret_values_forward_avg[0], color='dodgerblue', label='5 years: %.1f'%(np.median(ret_values_forward_avg[0])), linewidth=1.5)
l10 = ax.plot(years[:-10], ret_values_forward_avg[1], color='green', label='10 years: %.1f'%(np.median(ret_values_forward_avg[1])), linewidth=1.5)
l15 = ax.plot(years[:-15], ret_values_forward_avg[2], color='0.5', label='15 years: %.1f'%(np.median(ret_values_forward_avg[2])), linewidth=1.5)
l20 = ax.plot(years[:-20], ret_values_forward_avg[3], color='darkorange', label='20 years: %.1f'%(np.median(ret_values_forward_avg[3])), linewidth=1.5)
l25 = ax.plot(years[:-25], ret_values_forward_avg[4], color='purple', label='25 years: %.1f'%(np.median(ret_values_forward_avg[4])), linewidth=1.5)
l30 = ax.plot(years[:-30], ret_values_forward_avg[5], color='gold', label='30 years: %.1f'%(np.median(ret_values_forward_avg[5])), linewidth=1.5)
nbar = ax.bar(years[neg_rets], rets[neg_rets]/10.0, color='red', edgecolor='none', width=0.8)

ax.set_xlabel('Starting year')
ax.set_ylabel('Returns on $1')

ax.axhline(y=0, color='black')
ax.axhline(y=1, color='0.4')
ax.set_ylim(-5, 50)
ax.set_xlim(1870, 2020)

ax.set_title('Stock market aggregate returns beginning at each year\n and yearly returns of 10% drop or worse')

ax.legend(loc='best', prop={'size': 8})
ax.grid(True)
plt.tight_layout()

plt.savefig('historical_forward_returns_avg.png')




# plot the backwards returns
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

l5 = ax.plot(years[5:], ret_values_backward[0], color='dodgerblue', label='5 years: %.1f'%(np.median(ret_values_backward[0])), linewidth=1.5)
l10 = ax.plot(years[10:], ret_values_backward[1], color='green', label='10 years: %.1f'%(np.median(ret_values_backward[1])), linewidth=1.5)
l15 = ax.plot(years[15:], ret_values_backward[2], color='0.5', label='15 years: %.1f'%(np.median(ret_values_backward[2])), linewidth=1.5)
l20 = ax.plot(years[20:], ret_values_backward[3], color='darkorange', label='20 years: %.1f'%(np.median(ret_values_backward[3])), linewidth=1.5)
l25 = ax.plot(years[25:], ret_values_backward[4], color='purple', label='25 years: %.1f'%(np.median(ret_values_backward[4])), linewidth=1.5)
l30 = ax.plot(years[30:], ret_values_backward[5], color='gold', label='30 years: %.1f'%(np.median(ret_values_backward[5])), linewidth=1.5)
nbar = ax.bar(years[neg_rets], rets[neg_rets]/10.0, color='red', edgecolor='none', width=0.8)

ax.set_xlabel('Ending year')
ax.set_ylabel('Returns on $1')

ax.axhline(y=0, color='black')
ax.axhline(y=1, color='0.4')
ax.set_ylim(-5, 50)
ax.set_xlim(1870, 2020)

ax.set_title('Stock market aggregate returns ending at each year\n and yearly returns of 10% drop or worse')

ax.legend(loc='best', prop={'size': 8})
ax.grid(True)
plt.tight_layout()

plt.savefig('historical_backward_returns.png')



# prices = [1]

# for r in rets[:-1]:
# 	prices.append(prices[-1]*(1+(r/100.0)))

# prices = np.array(prices)








fig, ax = plt.subplots(2, 1, figsize=(8, 10))

ax[0].plot(years, prices, color='dodgerblue', linewidth=2)
ax[0].set_ylabel('Compounded value of $1 (in %d)'%(years[0]))
ax[0].grid(True)

ax[1].plot(years, prices, color='dodgerblue', linewidth=2)
ax[1].set_ylabel('Compounded $1 (log scaled)')
ax[1].set_yscale('log')
ax[1].grid(True)

ax[0].set_title('Dollar appreciation in the stock market')

plt.tight_layout()
plt.savefig('dollar_appreciation.png', dpi=DPI)






# okay, now let's do some simulations

sims = []
nsims = 1000
nyears = 3
init_money = 100

for i in range(nsims):
	this_sim = [init_money]

	for t in range(nyears):
		this_ret = np.random.choice(rets)
		this_sim.append(this_sim[-1]*(1+(this_ret/100.0)))

	this_sim = np.array(this_sim)


	sims.append(this_sim)

sims = np.array(sims)

bonds = init_money*(1 + (yield_10yr/(2.0)) )**(np.arange(nyears+1)*2)

infl_value = init_money*(1+inflation)**nyears

stock_last_values = sims[:, -1]

stock_wins = np.where(stock_last_values >= bonds[-1])[0]
stock_win_pct = 100.0*len(stock_wins)/nsims

print '{}/{} times stocks did better'.format(len(stock_wins), nsims)

se_bins = np.arange(0, 4000, 100)

fig, ax = plt.subplots(1, 2, figsize=(11, 6))

for i in range(nsims):
	ax[0].plot(np.arange(nyears+1), sims[i], color='dodgerblue', alpha=0.5, linewidth=0.5)


ax[0].plot(np.arange(nyears+1), bonds, color='red', linewidth=3, linestyle='dashed')
ax[0].set_title('%d simulations, starting with %d dollars'%(nsims, init_money), fontsize=10)
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Value ($)')

stock_last_hist = np.histogram(stock_last_values, bins=se_bins)[0]
ax[1].bar(se_bins[:-1], stock_last_hist, width=80, color='dodgerblue', edgecolor='none')
ax[1].axvline(x=bonds[-1], color='red', linewidth=3, linestyle='dashed')

ax[1].set_title('PDF of stock market values after %d years\nStocks win %d%%, bond value: %d, avg stock value: %d, inflation: %d'%(\
								nyears, stock_win_pct, bonds[-1], np.average(stock_last_values), infl_value), fontsize=10)
ax[1].set_xlabel('Final return after %d years (dollars)'%(nyears))
ax[1].set_ylabel('Frequency')

#ax[1].text(2000, 100, 'Real bond returns: %.1f%%'%(bonds[-1]/infl_value))


plt.tight_layout()
plt.savefig('stocksims_%dyears.png'%(nyears), dpi=DPI)

























