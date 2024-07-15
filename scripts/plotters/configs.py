import os, csv
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("graph")
parser.add_argument('-interleave5', '--interleave5', help="Path to read csv.", default='')
parser.add_argument('-interleave4', '--interleave4', help="Path to read csv.", default='')
parser.add_argument('-interleave3', '--interleave3', help="Path to read csv.", default='')
parser.add_argument('-interleave2', '--interleave2', help="Path to read csv.", default='')
parser.add_argument('-interleave', '--interleave', help="Path to read csv.", default='')
parser.add_argument('-nointerleave', '--nointerleave', help="Path to read csv.", default='')
parser.add_argument('-word8', '--word8', help="Path to read csv.", default='')
parser.add_argument('-word32', '--word32', help="Path to read csv.", default='')
parser.add_argument('-word64', '--word64', help="Path to read csv.", default='')
parser.add_argument('-output', '--output', help="Path to output directory.", default='')
args = parser.parse_args()

CONFIGS1 = { 'interleave': args.interleave, 
			'interleave2': args.interleave2, 
			'interleave3': args.interleave3, 
			'interleave4': args.interleave4, 
			'interleave5': args.interleave5,
			'nointerleave': args.nointerleave }
CONFIGS2 = {} # 'word8': args.word8, 'word32': args.word32, 'word64': args.word64 }

def collect(configs):
	table = dict()
	for config in configs.keys():
		table.update({ config: [[]] })
	for config, csv_path in configs.items():
		if (not os.path.exists(csv_path)):
			raise Exception("%s does not exist." % csv_path)
		csv_data = list(csv.reader(open(csv_path, 'r'), delimiter=","))
		table.update({ config: csv_data })
	data = dict()
	for config in configs.keys():
		time = list()
		for row in table[config]:
			if row[0] == 'Circuit': continue
			assert(len(row) > 3)
			rowtime = float(row[4])
			time.append(rowtime)
		time.sort()
		data.update({ config: time })
	return data

def plot():
	data = collect(CONFIGS1)
	time_x = dict()
	start = 100
	end = 300
	for config in CONFIGS1.keys():
		time_x.update({ config: list(range(start, end)) })
		data[config] = data[config][start-start:end-start]
	fig = plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=20; mW=4; lW=3
	x = 400
	x_part = 312
	fontSize=50
	fontWeight='bold'
	AnnofontSize=30
	time_label = 'Run time (milliseconds)'
	circuits_label = 'Qubits in thousands'
	xLab = circuits_label
	yLab = time_label
	plt.plot(time_x['interleave5'],  data['interleave5'], linestyle='--', color='tab:red', linewidth=lW)
	plt.plot(time_x['interleave4'],  data['interleave4'], linestyle='--', color='tab:blue', linewidth=lW)
	plt.plot(time_x['interleave3'],  data['interleave3'], linestyle='--', color='tab:purple', linewidth=lW)
	plt.plot(time_x['interleave2'],  data['interleave2'], linestyle='--', color='tab:brown', linewidth=lW)
	plt.plot(time_x['interleave'],  data['interleave'], linestyle='--', color='tab:orange', linewidth=lW)
	plt.plot(time_x['nointerleave'],  data['nointerleave'], linestyle='-', color='tab:green', linewidth=lW)
	leg = plt.legend(['interleave 5 cols', 'interleave 4 cols', 'interleave 3 cols', 'interleave 2 cols', 'interleave 1 col', 'no-interleave'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	# plt.axvline(x_part, ymin=0, ymax=0.21, color='grey', linestyle='--', linewidth=2,zorder=1)
	# p1 = matplotlib.patches.FancyArrowPatch((x_part + 2, 35), (330, 150), arrowstyle='<-', mutation_scale=40, linewidth=4, color='purple')
	# p2 = matplotlib.patches.FancyArrowPatch((x, 45), (x, 840), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	# ax.add_patch(p1)
	# ax.add_patch(p2)
	# ax.annotate(str(x_part) + ',000 qubits', xy=(x_part - 20, 220), fontsize=AnnofontSize, color='black', weight="bold")
	# ax.annotate('#Partitions > 1', xy=(x_part + 5, 160), fontsize=AnnofontSize, color='purple', weight="bold")
	# ax.annotate('30x Average Speedup', xy=(x + 5, x), fontsize=AnnofontSize, color='purple', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/time_interleave.pdf', dpi=1000)
	plt.show()
	# Words plot
	if len(CONFIGS2) == 0:
		return
	data = collect(CONFIGS2)
	start = 100
	end = 300
	for config in CONFIGS2.keys():
		time_x.update({ config: list(range(start, end)) })
		data[config] = data[config][start-start:end-start]
	fig = plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=15; mW=2.5; lW=2.5
	x = 400
	x_part = 312
	fontSize=50
	fontWeight='bold'
	AnnofontSize=30
	time_label = 'Run time (milliseconds)'
	circuits_label = 'Qubits in thousands'
	xLab = circuits_label
	yLab = time_label
	plt.plot(time_x['word8'],  data['word8'], linestyle='-', color='tab:red', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	plt.plot(time_x['word32'],  data['word32'], linestyle='-', color='tab:blue', marker='.', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	plt.plot(time_x['word64'],  data['word64'], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	leg = plt.legend(['8-bit word', '32-bit word', '64-bit word'], fontsize=fontSize)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	# plt.axvline(x_part, ymin=0, ymax=0.21, color='grey', linestyle='--', linewidth=2,zorder=1)
	# p1 = matplotlib.patches.FancyArrowPatch((x_part + 2, 35), (330, 150), arrowstyle='<-', mutation_scale=40, linewidth=4, color='purple')
	# p2 = matplotlib.patches.FancyArrowPatch((x, 45), (x, 840), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
	# ax.add_patch(p1)
	# ax.add_patch(p2)
	# ax.annotate(str(x_part) + ',000 qubits', xy=(x_part - 20, 220), fontsize=AnnofontSize, color='black', weight="bold")
	# ax.annotate('#Partitions > 1', xy=(x_part + 5, 160), fontsize=AnnofontSize, color='purple', weight="bold")
	# ax.annotate('30x Average Speedup', xy=(x + 5, x), fontsize=AnnofontSize, color='purple', weight="bold")
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '/time_words.pdf', dpi=1000)
	plt.show()

plot()
