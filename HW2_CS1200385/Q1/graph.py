import matplotlib.pyplot as plt

minSup = [5, 10, 25, 50, 95]

gSpan_time = []
with open('time_gspan_05.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    gSpan_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gspan_10.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    gSpan_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gspan_25.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    gSpan_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gspan_50.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    gSpan_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gspan_95.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    gSpan_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))

fsg_time = []
with open('time_pafi_05.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    fsg_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_pafi_10.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    fsg_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_pafi_25.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    fsg_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_pafi_50.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    fsg_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_pafi_95.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    fsg_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))

time_gaston_time = []
with open('time_gaston_05.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    time_gaston_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gaston_10.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    time_gaston_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gaston_25.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    time_gaston_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gaston_50.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    time_gaston_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))
with open('time_gaston_95.txt', 'r') as file:
    timeTaken = file.readlines()[-3].strip().split('\t')[1].split('m')
    time_gaston_time.append(int(timeTaken[0]) * 60 + float(timeTaken[1][:-1]))

plt.plot(minSup, gSpan_time, label='gSpan', marker='o')
plt.plot(minSup, fsg_time, label='FSG', marker='^')
plt.plot(minSup, time_gaston_time, label='Gaston', marker='s')
# plt.xscale('log')
plt.yscale('log')
plt.xlabel('Minimum Support (%)')
plt.ylabel('Running Time (s)')
plt.xticks(minSup, minSup)
# for x in minSup:
#     plt.axvline(x=x, color='blue', linestyle='dotted')
for i in range(5):
    plt.text(minSup[i], gSpan_time[i], str(gSpan_time[i]), ha='right', va='top')
    plt.text(minSup[i], fsg_time[i], str(fsg_time[i]), ha='left', va='bottom')
    plt.text(minSup[i], time_gaston_time[i], str(time_gaston_time[i]), ha='left', va='bottom')
plt.legend()
plt.savefig('running_time.png')
plt.show()