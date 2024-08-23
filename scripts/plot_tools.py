import os, csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import argparse
import math

Path = os.path

main_dir = Path.dirname(Path.dirname(Path.abspath(__file__))) + '/'

parser = argparse.ArgumentParser("graph")
parser.add_argument('-s', '--stim', help="Path to read Stim csv.", default='results/stim/equivalence/equivalence.csv')
parser.add_argument('-p', '--quasarq', help="Path to read quasarq csv.", default='results/quasarq/equivalence/equivalence.csv')
parser.add_argument('-o', '--output', help="Path to output directory.", default='results')
args = parser.parse_args()

TOOLS = { 'stim': main_dir + args.stim, 'quasarq': main_dir + args.quasarq }

output_dir = main_dir + args.output

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

def show_plot(output, legend, xLab, yLab, arrow, annotStr, annotXY):
    fontSize=75; legFontSize=65; fontWeight='normal'
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.yticks(fontsize=fontSize, fontweight=fontWeight)
    plt.xticks(fontsize=fontSize, fontweight=fontWeight)
    leg = plt.legend(legend, loc='upper left', prop=font_manager.FontProperties(weight='normal', size=legFontSize))
    leg.get_frame().set_linewidth(4.0)
    ax = plt.gca()
    ax.add_patch(arrow)
    ax.annotate(annotStr, annotXY, fontsize=50, color='black', weight="normal")
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    ax.xaxis.get_offset_text().set_fontsize(fontSize)
    ax.yaxis.get_offset_text().set_fontsize(fontSize)
    ax.tick_params(axis="y",direction="in", length=8, right=True)
    ax.tick_params(axis="x",direction="in", length=8, top=True)
    plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
    plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
    plt.tight_layout(pad=2)
    plt.savefig(output + '.pdf', dpi=1000)

def plot():
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
            #n = int(int(n) / 1000)
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
    #==================================#
    #============ Time plot ===========#
    #==================================#
    fig = plt.figure(figsize=(25,13))
    plt.rcParams["font.family"] = "Times New Roman"
    mS=30; mW=4; lW=6
    x = 295
    scale = 1000
    xLab = 'Number of Qubits'
    yLab = 'Run Time (seconds)'
    legend = ['CCEC (based on Stim)', 'QuaSARQ']
    stim_data[0].sort()
    quasarq_data[0].sort()
    plt.plot(list(range(0, len(stim_data[0])*scale, scale)),  stim_data[0], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    plt.plot(list(range(0, len(quasarq_data[0])*scale, scale)),  quasarq_data[0], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    ystim = stim_data[0][x] - mS*5 - mW - lW 
    yquasar = quasarq_data[0][x] + mS*2 + mW + lW
    arrow = matplotlib.patches.FancyArrowPatch((x*scale, yquasar), (x*scale, ystim), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
    ax = plt.gca()
    ax.annotate(rounded_maximum + 'x Maximum Speedup', xy=((x + 5)*scale, (x * 3 - 200)), fontsize=50, color='black', weight="normal")
    show_plot(output_dir + '/equivalence_time', legend, xLab, yLab, arrow, rounded_average + 'x Average Speedup', ((x + 10)*scale, (x * 3)))
    #==================================#
    #=========== Energy plot ==========#
    #==================================#
    plt.figure(figsize=(25,13))
    plt.rcParams["font.family"] = "Times New Roman"
    stim_data[1].sort()
    quasarq_data[1].sort()
    yLab = 'Energy draw (joules)'
    plt.plot(list(range(0, len(stim_data[0])*scale, scale)),  stim_data[1], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    plt.plot(list(range(0, len(quasarq_data[0])*scale, scale)),  quasarq_data[1], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    ystim = stim_data[1][x] - mS*450 - mW - lW 
    yquasar = quasarq_data[1][x] + mS*100 + mW + lW
    arrow = matplotlib.patches.FancyArrowPatch((x*scale, yquasar), (x*scale, ystim), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
    reduction = '%%%.0f Average Reduction' % avgEfficiency
    show_plot(output_dir + '/equivalence_energy', legend, xLab, yLab, arrow, reduction, ((x + 5)*scale, 40000))

plot()
