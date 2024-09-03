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
parser.add_argument('-s', '--ccec', help="Path to read ccec csv.", default='results/equivalence/ccec/equivalence.csv')
parser.add_argument('-p', '--quasarq', help="Path to read quasarq csv.", default='results/equivalence/quasarq/equivalence.csv')
parser.add_argument('-o', '--output', help="Path to output directory.", default='results')
args = parser.parse_args()

TOOLS = { 'ccec': main_dir + args.ccec, 'quasarq': main_dir + args.quasarq }

output_dir = main_dir + args.output

fontSize=60; legFontSize=60; fontWeight='normal'

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
    SCALE = 1000
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
            n = int(int(n) / SCALE)
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
    ccec_data = (list(), list())
    quasarq_data = (list(), list())
    for x in range(0, len(data['ccec'])):
        tccec=data['ccec'][x][1][0]
        tquasarq=data['quasarq'][x][1][0]
        ccec_energy = data['ccec'][x][1][1]
        quasarq_energy = data['quasarq'][x][1][1]
        quasarq_data[0].append(tquasarq)
        quasarq_data[1].append(quasarq_energy)
        if tccec == TIMELIMIT:
            continue
        ccec_data[0].append(tccec)
        ccec_data[1].append(ccec_energy)
        speedup = (tccec / tquasarq)
        if ccec_energy == 0:
            ccec_energy = 70 * tccec
        efficiency = 100.0 * (ccec_energy - quasarq_energy) / ccec_energy
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
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    mS=30; mW=4; lW=6
    x = len(ccec_data[0]) - 1
    xLab = 'Number of Qubits'
    yLab = 'Run Time (seconds)'
    legend = ['CCEC (based on Stim)', 'QuaSARQ']
    ccec_data[0].sort()
    quasarq_data[0].sort()
    plt.plot(list(range(0, len(ccec_data[0])*SCALE, SCALE)),  ccec_data[0], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    plt.plot(list(range(0, len(quasarq_data[0])*SCALE, SCALE)),  quasarq_data[0], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    yccec = ccec_data[0][x] - ((mS*5 - mW - lW ) if x >= 200 else 0)
    yquasar = quasarq_data[0][x] + ((mS*2 + mW + lW) if x >= 200 else 0)
    arrow = matplotlib.patches.FancyArrowPatch((x*SCALE, yquasar), (x*SCALE, yccec), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
    ax = plt.gca()
    xy = ()
    logx = int(math.log10(x))
    xmagnitude=pow(10, logx)
    mult = 0.5
    if (xmagnitude > 1 and xmagnitude <= 10): mult = 1
    elif (xmagnitude > 10 and xmagnitude <= 100): mult = 2.5
    xpoint = ((x - xmagnitude * mult))*SCALE
    ypoint = yccec / 1.5
    if x > 500: 
        x = 295
        xy=((x + 5)*SCALE, (x * 3 - 200))
    else:
        xy=(xpoint, ypoint)
    ax.annotate(rounded_maximum + 'x Maximum Speedup', xy=xy, fontsize=50, color='black', weight="normal")
    ypoint = yccec / 2
    if x > 500:
        xy=((x + 10)*SCALE, (x * 3))
    else:
        xy=(xpoint, ypoint)
    show_plot(output_dir + '/equivalence_time', legend, xLab, yLab, arrow, rounded_average + 'x Average Speedup', xy)
    #==================================#
    #=========== Energy plot ==========#
    #==================================#
    plt.figure(figsize=(25,13))
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    ccec_data[1].sort()
    quasarq_data[1].sort()
    yLab = 'Energy draw (joules)'
    plt.plot(list(range(0, len(ccec_data[0])*SCALE, SCALE)),  ccec_data[1], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    plt.plot(list(range(0, len(quasarq_data[0])*SCALE, SCALE)),  quasarq_data[1], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    yccec = ccec_data[1][x] - mS*450 - mW - lW 
    yquasar = quasarq_data[1][x] + mS*100 + mW + lW
    arrow = matplotlib.patches.FancyArrowPatch((x*SCALE, yquasar), (x*SCALE, yccec), arrowstyle='<->', mutation_scale=40, linewidth=4, color='purple')
    reduction = '%%%.0f Average Reduction' % avgEfficiency
    xy = ()
    if x > 295: 
        x = 295
        xy=((x + 5)*SCALE, (x * 3 - 200))
    else:
        xy=((x - 0.45)*SCALE, yccec / 2)
    show_plot(output_dir + '/equivalence_energy', legend, xLab, yLab, arrow, reduction, ((x + 5)*SCALE, 40000))

plot()
