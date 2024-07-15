import os, csv
import matplotlib
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser("graph")
parser.add_argument('-q', '--qiskit', help="Path to read Qsikit csv.", default='')
parser.add_argument('-s', '--stim', help="Path to read Stim csv.", default='')
parser.add_argument('-p', '--quasarq', help="Path to read quasarq csv.", default='')
parser.add_argument('-o', '--output', help="Path to output directory.", default='.')
args = parser.parse_args()

TOOLS = { 'stim': args.stim, 'quasarq': args.quasarq }
#TOOLS = { 'qiskit': args.qiskit, 'stim': args.stim, 'quasarq': args.quasarq }

def collect():
	RESULTS = dict()
	for tool in TOOLS.keys():
		RESULTS.update({ tool: [[]] })
	for tool, csv_path in TOOLS.items():
		if (not os.path.exists(csv_path)):
			raise Exception("%s does not exist." % csv_path)
		csv_data = list(csv.reader(open(csv_path, 'r'), delimiter=","))
		RESULTS.update({ tool: csv_data })
	return RESULTS

def plot():
	CACTUS = True
	RESULTS = collect()
	data = dict()
	for tool in TOOLS.keys():
		time = list()
		energy = list()
		for row in RESULTS[tool]:
			if row[0] == 'Circuit': continue
			if len(row) < 6:
				if row[1] != '0' or row[1] != '': 
					time.append(float(row[1]))
					energy.append(float(row[2]))
			else:
				assert(len(row) > 6)
				rowtime = 0
				for i in range(1,5):
					rowtime += float(row[i]) / 1000.0
				assert(rowtime < 100)
				time.append(rowtime)
				energy.append(float(row[6]))
		time.sort(); energy.sort()
		data.update({ tool: (time, energy)})
	data['stim'][0][:] = data['stim'][0][0:-1] # remove overshooting point
	data['stim'][1][:] = data['stim'][1][0:-1] # remove ../../../doc/draft/overshooting point
	time_x = dict()
	start = 0
	for tool in TOOLS.keys():
		time_x.update({ tool: list(range(start, len(data[tool][0]))) })
	avgSpeedup = 0
	maxSpeedup = 0
	bestQubits = 0
	for x in time_x['stim']:
		tstim=data['stim'][0][x]
		tpquarsar=data['quasarq'][0][x]
		speedup = (tstim / tpquarsar)
		if (maxSpeedup < speedup): 
			maxSpeedup = speedup
			bestQubits = x
		avgSpeedup += speedup
	avgSpeedup /= len(time_x['stim'])
	rounded_average = '%.0f' % math.ceil(avgSpeedup)
	rounded_maximum = '%.0f' % maxSpeedup
	print('Average speedup: ' + rounded_average)
	print('Maximum speedup: ' + rounded_maximum + ' at ' + str(bestQubits) + ',000 qubits')
	fig = plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=20; mW=4; lW=2.5
	x = 400
	x_part = 312
	fontSize=50
	fontWeight='bold'
	AnnofontSize=30
	circuits_label = 'Qubits in thousands'
	if CACTUS:
		xLab = circuits_label
		yLab = 'Run Time (seconds)'
		plt.plot(time_x['stim'],  data['stim'][0], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(time_x['quasarq'],  data['quasarq'][0], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	else:
		yLab = circuits_label
		xLab = 'Run Time (seconds)'
		plt.plot(data['stim'][0], time_x['stim'],  linestyle='-', color='tab:red', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(data['quasarq'][0], time_x['quasarq'],  linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	# for tool in TOOLS.keys():
	#plt.axvline(x_part, ymin=0, ymax=0.21, color='grey', linestyle='--', linewidth=2,zorder=1)
	p1 = matplotlib.patches.FancyArrowPatch((x_part + 2, 35), (330, 150), arrowstyle='<-', mutation_scale=40, linewidth=4, color='purple')
	p2 = matplotlib.patches.FancyArrowPatch((x, 25), (x, 840), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	leg = plt.legend(['Stim', 'QuaSARQ'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	# ax.add_patch(p1)
	# ax.add_patch(p2)
	# ax.annotate(str(x_part) + ',000 qubits', xy=(x_part - 20, 220), fontsize=AnnofontSize, color='black', weight="bold")
	# ax.annotate('#Partitions > 1', xy=(x_part + 5, 160), fontsize=AnnofontSize, color='purple', weight="bold")
	# ax.annotate(rounded_average + 'x Average Speedup', xy=(x + 5, x), fontsize=AnnofontSize, color='purple', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/time_cactus.pdf', dpi=1000)
	#plt.show()
	# Energy plot
	start = 0
	energy_x = dict()
	for tool in TOOLS.keys():
		energy_x.update({ tool: list(range(start, len(data[tool][1]))) })
	fig = plt.figure(figsize=(25,13))
	stim_energy = data['stim'][1][-1]
	quasarq_energy = data['quasarq'][1][-1]
	efficiency = 100.0 * (stim_energy - quasarq_energy) / stim_energy
	print('Energy efficiency: %.0f' % efficiency)
	plt.rcParams["font.family"] = "Times New Roman"
	mS=20; mW=4; lW=2.5
	CACTUS = True
	x=420
	if CACTUS:
		xLab = circuits_label
		yLab = 'Energy draw (joules)'
		#data['stim'][1].reverse()
		#data['quasarq'][1].reverse()
		plt.plot(energy_x['stim'],  data['stim'][1], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(energy_x['quasarq'],  data['quasarq'][1], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		ystim = data['stim'][1][x] - mS*50 - mW - lW
		yquasar = data['quasarq'][1][x] + mS*50 + mW + lW
		yquasar_part = data['quasarq'][1][x_part] + mS*50 + mW + lW
		#plt.axvline(x_part, ymin=0, ymax=0.24, color='grey', linestyle='--', linewidth=2,zorder=1)
		p1 = matplotlib.patches.FancyArrowPatch((x_part + 2, yquasar_part), (340, 15000), arrowstyle='<-', mutation_scale=40, linewidth=4, color='purple')
		p2 = matplotlib.patches.FancyArrowPatch((x, yquasar), (x, ystim), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	else:
		yLab = circuits_label
		xLab = 'Energy draw (joules)'
		plt.plot(data['stim'][1], energy_x['stim'],  linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(data['quasarq'][1], energy_x['quasarq'],  linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		ystim = data['stim'][1][320] - mS*50 - mW - lW
		yquasar = data['quasarq'][1][320] + mS*50 + mW + lW
		p2 = matplotlib.patches.FancyArrowPatch((ystim, x), (yquasar, x), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	leg = plt.legend(['Stim', 'QuaSARQ'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	# ax.add_patch(p1)
	# ax.add_patch(p2)
	# ax.annotate(str(x_part) + ',000 qubits', xy=(x_part - 20, 22000), fontsize=AnnofontSize, color='black', weight="bold")
	# ax.annotate('#Partitions > 1', xy=(x_part + 5, 16000), fontsize=AnnofontSize, color='purple', weight="bold")
	# reduction = '%%%.0f Average Energy\n Reduction' % efficiency
	# ax.annotate(reduction, xy=(x + 5, 40000), fontsize=AnnofontSize, color='purple', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/energy_cactus.pdf', dpi=1000)
	plt.show()

plot()
