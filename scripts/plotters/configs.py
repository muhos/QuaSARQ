import os, csv
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("graph")
parser.add_argument('-interleave3', '--interleave3', help="Path to read csv.", default='')
parser.add_argument('-interleave2', '--interleave2', help="Path to read csv.", default='')
parser.add_argument('-interleave', '--interleave', help="Path to read csv.", default='')
parser.add_argument('-nointerleave', '--nointerleave', help="Path to read csv.", default='')
parser.add_argument('-interleave75', '--interleave75', help="Path to read csv.", default='')
parser.add_argument('-interleave50', '--interleave50', help="Path to read csv.", default='')
parser.add_argument('-interleave25', '--interleave25', help="Path to read csv.", default='')
parser.add_argument('-nointerleave25', '--nointerleave25', help="Path to read csv.", default='')
parser.add_argument('-nointerleave50', '--nointerleave50', help="Path to read csv.", default='')
parser.add_argument('-nointerleave75', '--nointerleave75', help="Path to read csv.", default='')
parser.add_argument('-word8', '--word8', help="Path to read csv.", default='')
parser.add_argument('-word32', '--word32', help="Path to read csv.", default='')
parser.add_argument('-word64', '--word64', help="Path to read csv.", default='')
parser.add_argument('-output', '--output', help="Path to output file.", default='./output')
args = parser.parse_args()

CONFIGS1 = { 
			# 'interleave25': args.interleave25, 
			# 'interleave50': args.interleave50, 
			# 'interleave75': args.interleave75,
			'nointerleave': args.nointerleave, 
			'interleave': args.interleave, 
			#'interleave2': args.interleave2, 
			#'interleave3': args.interleave3, 
			#'interleave4': args.interleave4, 
			#'interleave5': args.interleave5,
			# 'nointerleave25': args.nointerleave25,
			# 'nointerleave50': args.nointerleave50,
			# 'nointerleave75': args.nointerleave75
			}
CONFIGS2 = {} #{'word8': args.word8, 'word32': args.word32, 'word64': args.word64 }

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
	plt.figure(figsize=(25,13))
	plt.rcParams["font.family"] = "Times New Roman"
	mS=30; mW=4; lW=6
	fontSize=70
	legFontSize=65
	fontWeight='bold'
	time_label = 'Run time (milliseconds)'
	circuits_label = 'Qubits in thousands'
	xLab = circuits_label
	yLab = time_label
	#plt.plot(time_x['interleave5'],  data['interleave5'], linestyle='--', color='tab:red', linewidth=lW)
	#plt.plot(time_x['interleave4'],  data['interleave4'], linestyle='--', color='tab:blue', linewidth=lW)
	#plt.plot(time_x['interleave3'],  data['interleave3'], linestyle='--', color='tab:purple', linewidth=lW)
	#plt.plot(time_x['interleave2'],  data['interleave2'], linestyle='--', color='tab:brown', linewidth=lW)
	plt.plot(time_x['interleave'],  data['interleave'], linestyle='--', color='tab:orange', linewidth=lW)
	plt.plot(time_x['nointerleave'],  data['nointerleave'], linestyle='-', color='tab:green', linewidth=lW)
	# plt.plot(time_x['interleave25'],  data['interleave25'], linestyle='--', color='tab:blue', linewidth=lW)
	# plt.plot(time_x['nointerleave25'],  data['nointerleave25'], linestyle='-', color='tab:blue', linewidth=lW)
	# plt.plot(time_x['interleave50'],  data['interleave50'], linestyle='--', color='tab:orange', linewidth=lW)
	# plt.plot(time_x['nointerleave50'],  data['nointerleave50'], linestyle='-', color='tab:orange', linewidth=lW)
	# plt.plot(time_x['interleave75'],  data['interleave75'], linestyle='--', color='tab:green', linewidth=lW)
	# plt.plot(time_x['nointerleave75'],  data['nointerleave75'], linestyle='-', color='tab:green', linewidth=lW)
	legentFont = matplotlib.font_manager.FontProperties(weight='bold', size=legFontSize)
	leg = plt.legend(['XZ 1-word interleave', 'separate X and Z'], prop=legentFont)
	#leg = plt.legend(['XZ 3-column interleave', 'XZ 2-column interleave', 'XZ 1-column interleave', 'separate X and Z'], prop=legentFont)
	#leg = plt.legend(['interleave (H,S = %25)', 'separate (H,S = %25)', 'interleave (H,S = %50)', 'separate (H,S = %50)', 'interleave (H,S = %75)', 'separate (H,S = %75)'], prop=legentFont)
	#leg = plt.legend(['words interleave', 'no-interleave'], prop=legentFont)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '.pdf', dpi=1000)
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
	x = 400
	x_part = 312
	fontWeight='bold'
	AnnofontSize=30
	time_label = 'Run time (milliseconds)'
	circuits_label = 'Qubits in thousands'
	xLab = circuits_label
	yLab = time_label
	plt.plot(time_x['word8'],  data['word8'], linestyle='-', color='tab:red', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	plt.plot(time_x['word32'],  data['word32'], linestyle='-', color='tab:blue', marker='.', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	plt.plot(time_x['word64'],  data['word64'], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
	leg = plt.legend(['8-bit word', '32-bit word', '64-bit word'], prop=legentFont)
	leg.get_frame().set_linewidth(4.0)
	plt.yticks(fontsize=fontSize, fontweight=fontWeight)
	plt.xticks(fontsize=fontSize, fontweight=fontWeight)
	ax = plt.gca()
	ax.xaxis.get_major_ticks()[0].set_visible(False)
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
	ax.tick_params(axis="y",direction="in", length=8, right=True)
	ax.tick_params(axis="x",direction="in", length=8, top=True)
	plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
	plt.tight_layout(pad=2)
	plt.savefig(args.output + '.pdf', dpi=1000)
	plt.show()

plot()
