import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from parse import *

shufdir1 = sys.argv[1]
shufdir2 = sys.argv[2] 
shufdir3 = sys.argv[3]

if sys.argv[4]=='female':
    fun = best_female
elif sys.argv[4]=='male':
    fun = best_male
elif sys.argv[4]=='total':
    fun = best_total

shuf1 = parse_shuf(shufdir1, fun)
shuf2 = parse_shuf(shufdir2, fun)
shuf3 = parse_shuf(shufdir3, fun)

# Plot Stuff.
mpl.rcParams.update({'errorbar.capsize': 2})
x = list(range(len(shuf1['labels'])))

def format_fn(tick_val, tick_pos):
    if int(tick_val) in x:
        return shuf1['labels'][int(tick_val)]
    else:
        return ''

fig, ax = plt.subplots()

ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('Accuracy')
ax.set_xlabel('Female/Male Ratio')

dot_size = 3
lw = 0.5
if sys.argv[5] == 'full':
    plt.scatter(x, shuf1['vanilla'],  s=dot_size, color='blue')
    plt.scatter(x, shuf2['vanilla'],  s=dot_size, color='blue')
    plt.scatter(x, shuf3['vanilla'],  s=dot_size, color='blue')

    plt.scatter(x, shuf1['exact'],  s=dot_size, color='darkorange')
    plt.scatter(x, shuf2['exact'],  s=dot_size, color='darkorange')
    plt.scatter(x, shuf3['exact'],  s=dot_size, color='darkorange')

    plt.scatter(x, shuf1['nn'],  s=dot_size, color='green')
    plt.scatter(x, shuf2['nn'],  s=dot_size, color='green')
    plt.scatter(x, shuf3['nn'],  s=dot_size, color='green')

average_vanilla = (shuf1['vanilla']+shuf2['vanilla']+shuf3['vanilla'])/3.
errs_vanilla = [np.std(y) for y in zip(shuf1['vanilla'], shuf2['vanilla'], shuf3['vanilla'])]

average_exact = (shuf1['exact']+shuf2['exact']+shuf3['exact'])/3.
errs_exact= [np.std(y) for y in zip(shuf1['exact'], shuf2['exact'], shuf3['exact'])]

average_nn = (shuf1['nn']+shuf2['nn']+shuf3['nn'])/3.
errs_nn = [np.std(y) for y in zip(shuf1['nn'], shuf2['nn'], shuf3['nn'])]

plt.errorbar(x, average_vanilla, yerr=errs_vanilla, color='blue', linewidth=lw, label='Naive')
plt.errorbar(x, average_exact, yerr=errs_exact, color='darkorange', linewidth=lw, label='Exact')
plt.errorbar(x, average_nn, yerr=errs_nn, color='green', linewidth=lw, label='Estimated')

plt.legend()
plt.savefig("/home/vkonton/Dropbox/gender_gender_"+sys.argv[4]+"_"+sys.argv[5]+".pdf")
