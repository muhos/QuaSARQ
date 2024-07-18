import os, csv
import matplotlib
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser("graph")
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
	TIMELIMIT = 5000
	RESULTS = collect()
	data = dict()
	for tool in TOOLS.keys():
		results = dict()
		
		for row in RESULTS[tool]:
			if row[0] == 'Circuit': continue
			n = ''
			if 'log_q' in row[0]: 
				n = row[0].replace('log_q', '')
			elif 'q' in row[0]: 
				n = row[0].replace('q', '')
			n = n.replace('_d100', '')
			n = int(int(n) / 1000)
			time = TIMELIMIT
			energy = 0
			if len(row) == 5:
				if row[1] != '0': 
					time = float(row[1])
					energy = float(row[2])
			else:
				assert(len(row) > 5)
				rowtime = 0
				for i in range(1,4):
					rowtime += float(row[i]) / 1000.0
				assert(rowtime < 100)
				time = rowtime
				energy = float(row[4])
			results.update({ n: (time, energy) })
		data.update({ tool: sorted(results.items()) })
	avgEfficiency = 0
	avgSpeedup = 0
	maxSpeedup = 0
	bestQubits = 0
	count = 0
	stim_data = (list(), list())
	quasarq_data = (list(), list())
	for x in range(0, len(data['stim'])):
		tstim=data['stim'][x][1][0]
		tquasarq=data['quasarq'][x][1][0]
		stim_energy = data['stim'][x][1][1]
		quasarq_energy = data['quasarq'][x][1][1]
		quasarq_data[0].append(tquasarq)
		quasarq_data[1].append(quasarq_energy)
		if tstim == TIMELIMIT:
			continue
		stim_data[0].append(tstim)
		stim_data[1].append(stim_energy)
		speedup = (tstim / tquasarq)
		if stim_energy == 0:
			stim_energy = 84.5 * tstim
		efficiency = 100.0 * (stim_energy - quasarq_energy) / stim_energy
		if (maxSpeedup < speedup): 
			maxSpeedup = speedup
			bestQubits = data['quasarq'][x][0]
		avgSpeedup += speedup
		avgEfficiency += efficiency
		count += 1
	avgSpeedup /= count
	rounded_average = '%.0f' % math.ceil(avgSpeedup)
	rounded_maximum = '%.0f' % maxSpeedup
	print('Average speedup: ' + rounded_average)
	print('Maximum speedup: ' + rounded_maximum + ' at ' + str(bestQubits) + ',000 qubits')
	avgEfficiency /= count
	print('Energy efficiency: %.0f' % avgEfficiency)
	fig = plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=20; mW=4; lW=3
	x = 295
	x_part = 212
	fontSize=50
	fontWeight='bold'
	AnnofontSize=50
	circuits_label = 'Qubits in thousands'
	xLab = circuits_label
	yLab = 'Run Time (seconds)'
	stim_data[0].sort()
	quasarq_data[0].sort()
	if CACTUS:
		plt.plot(list(range(0, len(stim_data[0]))),  stim_data[0], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(list(range(0, len(quasarq_data[0]))),  quasarq_data[0], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	else:
		plt.scatter(list(range(0, len(stim_data[0]))),  stim_data[0], linestyle='-', color='tab:blue', marker='+', s=mS, linewidth=lW)
		plt.scatter(list(range(0, len(quasarq_data[0]))),  quasarq_data[0], linestyle='-', color='tab:green', marker='*', s=mS, linewidth=lW)
	# for tool in TOOLS.keys():
	#plt.axvline(x_part, ymin=0, ymax=0.21, color='grey', linestyle='--', linewidth=2,zorder=1)
	ystim = stim_data[0][x] - mS*5 - mW - lW 
	yquasar = quasarq_data[0][x] + mS*2 + mW + lW
	p2 = matplotlib.patches.FancyArrowPatch((x, yquasar), (x, ystim), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	leg = plt.legend(['CCEC (based on Stim)', 'QuaSARQ'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	ax.add_patch(p2)
	ax.annotate(rounded_average + 'x Average Speedup', xy=(x + 10, x * 3), fontsize=AnnofontSize, color='black', weight="bold")
	ax.annotate(rounded_maximum + 'x Maximum Speedup', xy=(x + 5, x * 3 - 200), fontsize=AnnofontSize, color='black', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/eqv_time_cactus.pdf', dpi=1000)
	plt.show()
	# Energy plot
	fig = plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=20; mW=4; lW=3
	CACTUS = True
	x=295
	stim_data[1].sort()
	quasarq_data[1].sort()
	if CACTUS:
		xLab = circuits_label
		yLab = 'Energy draw (joules)'
		plt.plot(list(range(0, len(stim_data[0]))),  stim_data[1], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(list(range(0, len(quasarq_data[0]))),  quasarq_data[1], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		ystim = stim_data[1][x] - mS*450 - mW - lW 
		yquasar = quasarq_data[1][x] + mS*100 + mW + lW
		#yquasar_part = data['quasarq'][x_part][1][1] + mS*50 + mW + lW
		#plt.axvline(x_part, ymin=0, ymax=0.24, color='grey', linestyle='--', linewidth=2,zorder=1)
		#p1 = matplotlib.patches.FancyArrowPatch((x_part + 2, yquasar_part), (340, 15000), arrowstyle='<-', mutation_scale=40, linewidth=4, color='purple')
		p2 = matplotlib.patches.FancyArrowPatch((x, yquasar), (x, ystim), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	else:
		yLab = circuits_label
		xLab = 'Energy draw (joules)'
		plt.plot(stim_data[1], list(range(0, len(stim_data[0]))),  linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		plt.plot(quasarq_data[1], list(range(0, len(quasarq_data[0]))),  linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
		ystim = data['stim'][x][1][1] - mS*50 - mW - lW
		yquasar = data['quasarq'][x][1][1] + mS*50 + mW + lW
		p2 = matplotlib.patches.FancyArrowPatch((ystim, x), (yquasar, x), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	leg = plt.legend(['CCEC (based on Stim)', 'QuaSARQ'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	#ax.add_patch(p1)
	ax.add_patch(p2)
	reduction = '%%%.0f Average Reduction' % avgEfficiency
	ax.annotate(reduction, xy=(x + 5, 40000), fontsize=AnnofontSize, color='black', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/eqv_energy_cactus.pdf', dpi=1000)
	plt.show()

plot()
