#!/usr/bin/env python3.6

import argparse
from astropy.table import Table, Column
from astropy import constants as con
from astropy import units as u
from astropy.time import Time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Code snippet to see the effcect SS bodies')
	parser.add_argument('eph', help='Ephemeris file of the moon', type=str)
	parser.add_argument('sim', help='Moons position from N-body simulation', type=str)

	args = parser.parse_args()

	moon_eph = Table.read(args.eph, format='ascii')
	moon_sim = Table.read(args.sim, format='ascii')

	ex = moon_eph['x']-moon_sim['x'] 
	ey = moon_eph['y']-moon_sim['y']
	ez = moon_eph['z']-moon_sim['z']
	e = np.sqrt(ex**2 + ey**2 + ez**2)
	ymax = e[-1]
	plt.plot(moon_sim['time'], e)
	plt.xlabel('Span of simulation in days')
	plt.ylabel('Error in kms')
	plt.title('Error (Actual-Simulated)')
	plt.text(50, 0.1*ymax, 'Max error is {e}'.format(e=np.ceil(ymax)))
	plt.show()
