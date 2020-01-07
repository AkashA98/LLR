#!/usr/bin/env python3.6

import argparse
from astropy.table import Table, Column
from astropy import constants as con
from astropy.constants import G
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, get_body, get_body_barycentric_posvel, solar_system_ephemeris
import numpy as np
from calendar import monthrange
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math, time
from matplotlib.gridspec import GridSpec


class vector:

    def __init__(self, x, y, z):

        self.x = x
        self.y = y
        self.z = z

    def  __add__(self, other):

        return vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):

        return vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):

        return vector(self.x*other, self.y*other, self.z*other)

    def __div__(self, other):

        return vector(self.x/other, self.y/other, self.z/other)

    def __eq__(self, other):

        if isinstance(other, vector):
            return self.x == other.x and self.y == other.y and self.z == other.z

        return False

    def __ne__(self, other):

        return not self.__eq__(other)

    def __str__(self):

        return '({x}, {y}, {z})'.format(x=self.x, y=self.y, z=self.z)

    def __abs__(self):

        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def norm(self):

        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def precision(self):

        x = np.round(self.x,10)
        y = np.round(self.y,10)
        z = np.round(self.z,10)

        return vector(x,y,z)


origin = vector(0.0, 0.0, 0.0)

class nbody:

    def __init__(self, table,span, ts):
        
        global au

        data = table
        self.g = G.to(((u.km)**3)/(u.kg*((u.day)**2))).value
        self.bodies = data['body']
        self.span = float(span)
        self.t = float(ts*u.hour.to(u.day))
        self.timestep = int(self.span/self.t)

        
        self.masses = [0.0 for i in range(len(self.bodies))]
        self.pos = [origin for i in range(len(self.bodies))]
        self.vel = [origin for i in range(len(self.bodies))]
        #self.pos_err = [origin for i in range(len(self.bodies))]
        self.barycenter = origin
        self.tot_mass = 0.0
        
        self.tab = [Table(names=['body', 'cycle', 'time', 'x', 'y', 'z', 'vx',\
            'vy', 'vz'], dtype=['S16', 'i8', 'f16', 'f16', 'f16', 'f16', 'f16',\
            'f16', 'f16']) for i in range(len(self.bodies))]

        for i in range(len(self.bodies)):


            self.masses[i] = data['mass'][i]
            self.pos[i] = vector(data['x'][i], data['y'][i], data['z'][i])
            #self.pos_err[i] = vector(data['ex'][i], data['ey'][i], data['ez'][i])
            self.vel[i] = vector(data['vx'][i], data['vy'][i], data['vz'][i])
            self.barycenter = self.barycenter + (self.pos[i])*(self.masses[i])
            self.tot_mass = self.tot_mass + self.masses[i]

        self.barycenter = self.barycenter*(1/self.tot_mass)

        for i in range(len(self.bodies)):

            self.tab[i].add_row([self.bodies[i], 0, 0, self.pos[i].x,\
            self.pos[i].y, self.pos[i].z, self.vel[i].x, self.vel[i].y,\
            self.vel[i].z])

    def update(self, body, err):
        for i in range(len(self.bodies)):
            if self.bodies[i]==body:
                self.pos[i]= self.pos[i] + self.pos_err[i]*err
                self.tab[i].remove_row(0)
                self.tab[i].add_row([self.bodies[i], 0, 0, self.pos[i].x,\
                self.pos[i].y, self.pos[i].z, self.vel[i].x, self.vel[i].y,\
                self.vel[i].z])

        return None

    
    def __compute_acc(self):
        self.acc = [origin for i in range(len(self.bodies))]
        for i in range(len(self.bodies)):
            for j in range(len(self.bodies)):
                if i!=j:
                    r = self.pos[j] - self.pos[i]
                    mag = self.g*self.masses[j]/(math.pow(r.norm(),3.0))
                    self.acc[i] = self.acc[i] + r*(mag)
        return None


    def __compute_vel(self):

        for i in range(len(self.bodies)):
            self.vel[i] = self.vel[i] + self.acc[i]*(self.t)
        return None


    def __compute_pos(self):

        for i in range(len(self.bodies)):
            self.pos[i] = self.pos[i] + self.vel[i]*(self.t)
        return None


    def __resolve_collisions(self):

        for i in range(len(self.bodies)):
            for j in range(len(self.bodies)):
                if self.pos[i]==self.pos[j]:
                    (self.vel[i], self.vel[j]) = (self.vel[j], self.vel[i])

        return None


    def simulate(self):

        self.__compute_acc()
        self.__compute_vel()
        self.__compute_pos()

        return None


    def results(self, cyc):

        for i in range(len(self.bodies)):
            self.tab[i].add_row([self.bodies[i], cyc+1, (cyc+1)*self.t,\
            self.pos[i].x, self.pos[i].y, self.pos[i].z, self.vel[i].x,\
            self.vel[i].y, self.vel[i].z])
        
        return None


def ear_moon(nb, body):

    tab= Table(names=['time', 'x', 'y', 'z', 'd', 'vx', 'vy', 'vz', 'v'],\
        dtype=['f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16'])

    if body=='earth-moon':
        for i in nb.bodies:
            if i=='earth':
                ind = list(nb.bodies).index(i)
        ear = nb.tab[ind]
        moon = nb.tab[ind+1]

        for i in range(len(moon)):

            x = moon['x'][i]-ear['x'][i]    
            y = moon['y'][i]-ear['y'][i]    
            z = moon['z'][i]-ear['z'][i]
        
            d = np.sqrt(x**2 + y**2 + z**2)

            vx = moon['vx'][i]-ear['vx'][i]    
            vy = moon['vy'][i]-ear['vy'][i]    
            vz = moon['vz'][i]-ear['vz'][i]    

            v = np.sqrt(vx**2 + vy**2 + vz**2)

            tab.add_row([moon['time'][i], x, y, z, d, vx, vy, vz, v])

    else:
        for i in nb.bodies:
            if i==body:
                ind = list(nb.bodies).index(i)
        b = nb.tab[ind]

        for i in range(len(b)):

            x = b['x'][i]
            y = b['y'][i]    
            z = b['z'][i]
        
            d = np.sqrt(x**2 + y**2 + z**2)

            vx = b['vx'][i]    
            vy = b['vy'][i]    
            vz = b['vz'][i]    

            v = np.sqrt(vx**2 + vy**2 + vz**2)

            tab.add_row([b['time'][i], x, y, z, d, vx, vy, vz, v])
    
    return tab

def err_check(args, nbody, table, err):

    plotfile = PdfPages('/home/akash/results/em_error.pdf')

    fig = plt.figure()
    gs = GridSpec(2,2)

    nb=nbody(table, args.span, args.step)
    #nb2=nbody(table, args.span, args.step)
    #nb2.update('sun', err)

    for k in range(nb.timestep):
        nb.simulate()
        nb.results(k)
        #nb2.simulate()
        #nb2.results(k)

    #Earth error    
    plt.clf()
    e_t = Table.read('/home/akash/results/actual/earth_Ephemeride.txt', format='ascii')[0:nb.timestep+1]
    e_t1 = ear_moon(nb, 'earth')
    #e_t2 = ear_moon(nb2, 'earth')
    ax1 = plt.subplot(gs[0])
    plt.plot(e_t1['time'], (e_t['d']-e_t1['d'])*1e6, label='Earth')
    plt.xlabel('Time in days')
    plt.ylabel('Error in mm')
    plt.legend()

    #Moon error
    #plt.clf()
    m_t = Table.read('/home/akash/results/actual/moon_Ephemeride.txt', format='ascii')[0:nb.timestep+1]
    m_t1 = ear_moon(nb, 'moon')
    #m_t2 = ear_moon(nb2, 'moon')
    ax2 = plt.subplot(gs[1])
    plt.plot(m_t1['time'], (m_t['d']-m_t1['d'])*1e6, label='Moon')
    plt.xlabel('Time in days')
    plt.ylabel('Error in mm')
    plt.legend()

    ax3 = plt.subplot(gs[2])
    plt.plot(e_t1['time'], (e_t['d']-e_t1['d'])*1e6, 'r', label='Earth' )
    plt.plot(e_t1['time'], (m_t['d']-m_t1['d'])*1e6, 'y', label='Moon')
    #plt.plot(e_t1['time'], (e_t2['d']-e_t1['d'])*1e6, 'red', label='Earth' )
    #plt.plot(e_t1['time'], (m_t2['d']-m_t1['d'])*1e6, 'yellow', label='Moon')
    plt.xlabel('Time in days')
    plt.ylabel('Error in mm')
    plt.legend()
    #plotfile.savefig()
    
    #Earth-moon error
    #plt.clf()
    em_t = m_t['d'] - e_t['d']
    em_t1 = ear_moon(nb, 'earth-moon')
    #em_t2 = ear_moon(nb2, 'earth-moon')
    ax4 = plt.subplot(gs[3])
    plt.plot(em_t1['time'], (em_t-em_t1['d'])*1e6, 'r', label='E-M distance')
    #plt.plot(em_t1['time'], (em_t2['d']-em_t1['d'])*1e6, 'red', label='E-M distance')
    plt.xlabel('Time in days')
    plt.ylabel('Error in mm')
    plt.legend()
    plt.tight_layout()
    

    plotfile.savefig()
    plotfile.close()

def pos_fit(args, nbody, table, body, ini_err, err_range, flag=True):
    
    res_err = []
    for er in err_range:
        nb=nbody(table,args.span, args.step)
        nb.update(body, ini_err+er)
        
        for k in range(nb.timestep):
            nb.simulate()
            nb.results(k)

        for i in range(len(nb.bodies)):
            if nb.bodies[i]=='moon':
                ex = nb.tab[i]['x'] - nb.obs['x'][0:nb.timestep+1]
                ey = nb.tab[i]['y'] - nb.obs['y'][0:nb.timestep+1]
                ez = nb.tab[i]['z'] - nb.obs['z'][0:nb.timestep+1]

                t = nb.tab[i]['time']
                e = np.sqrt(ex**2 + ey**2 + ez**2)*1e9
        res_err.append([er, np.round(np.max(e),0)])
    res_err = np.array(res_err)    
    if flag==True:
        plt.plot(res_err[:,0], res_err[:,1])
        plt.show()
    return res_err


def sim(args,table, nbody):
    nb = nbody(table, args.span, args.step)
    for k in range(nb.timestep):
        nb.simulate()
        nb.results(k)        
    for i in range(len(nb.bodies)):
        nb.tab[i].write("/home/akash/results/simulations/{n}.txt".format(n=nb.bodies[i]), format='ascii', delimiter='\t', overwrite=True)


def optima(args, nbody, table, body, ini_err, err_range, min_mask=False):
    
    new_err_range = err_range
    while min_mask==False:
        err_fit = pos_fit(args, nbody, table, body, ini_err, new_err_range)
        local_min = err_fit[np.argmin(err_fit[:,1]),0]
        shift = local_min
        
        if local_min==err_fit[0][0] or local_min==err_fit[-1][0]:
            min_mask = False
            print("\nOptima not found in the range [{a}, {b}]\n".format(a=err_fit[0][0], b=err_fit[-1][0]))    
            
        else: 
            min_mask = True
            print("\nOptima found !! Located at {s} from the original mean\n".format(s=shift))        
        
        new_err_range= err_range + shift
    return shift, np.min(err_fit[:,1]), err_fit

        
def check(err_fit):
    err = err_fit[:,1]
    ind = np.where(err==0.)[0]
    print(ind)
    print(err_fit[ind[0]:ind[-1]+1,0])    
    return err_fit[ind[0]:ind[-1]+1,0]    

def fit(body, err, ini, table):

    for i in range(len(body)):
        print("Finding optimum position for the body {b}".format(b=body[i]))
    
        err_range = np.linspace(-abs(ini[i]), abs(ini[i]), 25)
        print("\nInitial error range for {bod} is [{a}, {b}]\n".format(bod=body[i], a=err_range[0], b=err_range[-1]))    

        shift, err_max, err_fit = optima(args, nbody, table, body[i], err[i], err_range)
        print(shift)    
        print(err_max)
        
        while err_max!=0:
            err_range = np.linspace(shift-abs(shift)/10., shift+abs(shift)/10., 25)
            shift, err_max, err_fit = optima(args, nbody, req, body[i], err[i], err_range)
            print(shift)    
            print(err_max)
    '''    
        re_err_range = np.arange(shift-shift/10, shift+shift/10, shift/200)
        err_fit = pos_fit(args, nbody, table, body, err[i], re_err_range)
        box = check(err_fit)
        
        print("\nInterval in which the error is 0 is {s} km from the mean value of the initial error and of width {w} m".format(s=(np.mean(box)),w=abs(box[-1]- box[0])*1000))    
    '''    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='N-body simulation code')

    parser.add_argument('ephemeris', help='Input ephemeris file', type=str)
    parser.add_argument('--initial', help='Input initial error conditions file', type=str)
    parser.add_argument('--span', help='Timespan for the simulation (in days), default=6', type=float, default=6)
    parser.add_argument('--step', help='Timestep between consecutive nodes of simulation (in sec), default=1 hr', type=float, default=1)
    parser.add_argument('-p', '--planet', help='Planets that are to be considered for simulation, mer-1, ven-2, ear-3, moon-4, mars-5 ....', type=int, action='append')
    
    args = parser.parse_args()

    start_time = time.time()
    global au
    au = u.au.to(u.m)
    
    ss_bodies={'sun':0, 'mercury':1, 'venus':2, 'earth':3, 'moon':4, 'mars':5, 'jupiter':6, 'saturn':7, 'uranus':8, 'neptune':9}
    
    
    if args.initial is None:
        if args.planet is None:
            req=np.arange(11)
            #req=[0,3,4]
        else:
            req=args.planet
    else:
        params = Table.read(args.initial, format='ascii')
        body = np.array(params['Body'])    
        err = np.array(params['Error'])
        ini = np.array(params['Initial'])
        req=[3,4]
        for b in body:
            req.append(ss_bodies[b])

        req.sort()

    args.planet=req
    

    eph = Table.read(args.ephemeris, format='ascii')
    table = eph[req]
    
    err_check(args, nbody, table, 1000)    
    #sim(args, table, nbody)
