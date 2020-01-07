#!/usr/bin/env python3.6

import argparse
from astropy.table import Table, Column
from astropy import constants as con
from astropy import units as u
from astropy.time import Time
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Code to generate solar system ephermeris')
	parser.add_argument('moon_eph', help='Moon Ephemeride', type=str)

	args=parser.parse_args()

	data = Table.read(args.moon_eph, format='ascii')
	x = data['x']
	y = data['y']
	z = data['z']
	d = np.sqrt(x**2+y**2+z**2)
	freq = np.fft.fftfreq(d.size, 1e8)
	fx = np.fft.fft(d)
	fv = np.fft.fftshift(freq)
	fd = np.fft.fftshift(fx.real) 
	plt.scatter(fv, fd)
	plt.show()
