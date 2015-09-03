#!/bin/env python
import matplotlib
import re
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# get the problem arguments so that I can grab the correct files
parser = argparse.ArgumentParser(description='Make plot of eta')

parser.add_argument('--N_d', action="store", dest='N_d', default=0, type=int)
parser.add_argument('--R_d', action="store", dest='R_d', default=0, type=int)
parser.add_argument('--N_s', action="store", dest='N_s', default=0, type=int)
parser.add_argument('--R_s', action="store", dest='R_s', default=0, type=int)
parser.add_argument('--R_b', action="store", dest='R_b', default=0, type=int)

results =parser.parse_args()

N_d = results.N_d
R_d = results.R_d
N_s = results.N_s
R_s = results.R_s
R_b = results.R_b
param_string = "-"+str(N_d)+"-"+str(R_d)+"-"+str(N_s)+"-"+str(R_s)+"-"+str(R_b)

print(param_string)

# read in the data
data_actual = []
data_recon_real = []
data_recon_imag = []
x_locs = []
filename_coord = "/work/02370/kwkelly/maverick/files/results/scatter_coord"+param_string+".txt"
filename_actual = "/work/02370/kwkelly/maverick/files/results/eta_actual"+param_string+".txt"
filename_recon = "/work/02370/kwkelly/maverick/files/results/eta_recon"+param_string+".txt"
actual = open(filename_actual, "r")
recon = open(filename_recon, "r")
# read in the actual data
for line in actual:
	line = line.rstrip()
	if line:
		real, imag = re.split("(?<![e])[+]",line)
		imag = imag[:-1]
		data_actual.append(float(real))
actual.close()
# read in the reconstructed data
for line in recon:
	line = line.rstrip()
	if line:
		real, imag = re.split("(?<![e])[+]",line)
		imag = imag[:-1]
		data_recon_real.append(float(real))
		data_recon_imag.append(float(imag))
recon.close()
coord = open(filename_coord, "r")
# read in the coordinates
for i,line in enumerate(coord):
	line = line.rstrip()
	if line and (i%3 == 0):
		x_locs.append(line)
coord.close()

# and finally plot and save the plot to a file
plt.figure(1)
plt.plot(x_locs,data_actual,label="actual")
plt.plot(x_locs,data_recon_real,label="recon real part")
plt.plot(x_locs,data_recon_imag,label="recon imag part")
plt.xlabel('x')
plt.ylabel('eta')
plt.title("Eta Error, k=1000, N_d="+str(N_d)+", R_d="+str(R_d)+", N_s="+str(N_s)+", R_s="+str(R_s)+", R_b="+str(R_b))
plt.grid(True)
plt.legend(loc=2,prop={'size':10})
plt.savefig("eta_error"+param_string+".png")

