#!/usr/bin/env python3.5

import argparse, os
from astropy.table import Table, Column
from astropy import constants as con
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, get_body, get_body_barycentric_posvel, solar_system_ephemeris
import numpy as np
from calendar import monthrange
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy.polynomial.chebyshev as che
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec



def eph(time, ephmeride):
    t = Table(names=['body', 'mass', 'x', 'y', 'z', 'd', 'ex', 'ey', 'ez', 'vx', 'vy', 'vz', 'v'], dtype=['S8', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16']) 

    mass = {'sun':0.295912208285591100e-03, 'mercury':0.491248045036476000e-10, 'venus':0.724345233264412000e-09,\
        'earth':0.888769244512563400e-09, 'moon':0.109318945074237400e-10, 'mars':0.954954869555077000e-10 ,\
        'jupiter':0.282534584083387000e-06 , 'saturn':0.845970607324503000e-07 , 'uranus':0.129202482578296000e-07,\
        'neptune':0.152435734788511000e-07, 'pluto':0.217844105197418000e-11}
    #These are the values of GM in the units of au**3/day**2

    #err = {'sun':100.0, 'mercury':50.0, 'venus':20.0, 'earth':0, 'moon':0, 'mars':20, 'jupiter':10000, 'saturn':3000, 'uranus':1000000, 'neptune':5000000}
    
    body = np.array(list(mass.keys()))

    for obj in body: 
        pos, vel = get_body_barycentric_posvel(obj, time, ephemeris=ephmeride) 
        t.add_row([obj, mass[obj]*((u.au.to(u.m)**3/(u.day.to(u.second)**2))/con.G.value),\
            pos.x.value, pos.y.value, pos.z.value, pos.norm().value,\
            abs(pos.x.value)/pos.norm().value, abs(pos.y.value)/pos.norm().value, abs(pos.z.value)/pos.norm().value,\
            vel.x.value, vel.y.value, vel.z.value, vel.norm().value])

    return t


def get_times(a,b):
    times=[]
    a = Time(a, format='jd'); b=Time(b, format='jd')
    jd_times = np.arange(a.jd, b.jd, 12*u.hour.to(u.day))
    for i in jd_times:
        #You can leave it at jd is well. This is just to visualize date better
        times.append(Time(Time(i, format='jd').isot))
    return times


def get_body(times, body, savefile=False):
    tab = Table()
    t= Table(names=['utc', 'jd', 'x', 'y', 'z', 'd', 'vx', 'vy', 'vz', 'v'], dtype=['S32', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16', 'f16'])
    for time in times:
        pos, vel = get_body_barycentric_posvel(body, time, ephemeris='de430')
        t.add_row([time.value, time.jd, pos.x.value, pos.y.value, pos.z.value, pos.norm().value, vel.x.value/(u.day.to(u.second)), vel.y.value/(u.day.to(u.second)), vel.z.value/(u.day.to(u.second)), vel.norm().value/(u.day.to(u.second))]) 
    if savefile==True:
        t.write('/home/akash/results/{b}_Ephemeride.txt'.format(b=body), format='ascii', delimiter='\t', overwrite=True)
    return t


def get_moon(times):
    
    t = Table()
    
    earth = get_body(times, 'earth')
    moon = get_body(times, 'moon')

    em_x = moon['x']-earth['x']
    em_y = moon['y']-earth['y']
    em_z = moon['z']-earth['z']
    em_d = np.sqrt(em_x**2 + em_y**2 + em_z**2)
    em_vx = moon['vx']-earth['vx']
    em_vy = moon['vy']-earth['vy']
    em_vz = moon['vz']-earth['vz']
    em_v = np.sqrt(em_vx**2 + em_vy**2 + em_vz**2)
    t.add_column(moon['utc'])
    t.add_column(moon['jd'])
    t.add_column(em_x)
    t.add_column(em_y)
    t.add_column(em_z)
    t.add_column(em_d, name='d')
    t.add_column(em_vx)
    t.add_column(em_vy)
    t.add_column(em_vz)
    t.add_column(em_v, name='v')
    t.add_column(np.arcsin(em_z/em_d), name='dec')
    t.add_column(np.arctan2(em_y, em_x), name='RA')

    return t


def orbit(tab): 
    apog=[] 
    peri=[] 
    ecc=[]; e=[]
    i=0 
    mu = con.G.value*(con.M_earth.value + 7.1e22)
    while i<len(tab): 
        chunk = tab[i:i+30*4] 
        h=[]
        ap = np.max(abs(chunk['d']))
        t_ap = chunk['jd'][np.argmax(chunk['d'])] 
        pe = np.min(abs(chunk['d']))
        t_pe =  chunk['jd'][np.argmin(chunk['d'])]
        for t in [np.argmin(chunk['d']), np.argmax(chunk['d'])]:
                E = 0.5*1e6*chunk['v'][t]**2 - mu/(chunk['d'][t]*1000)
                H = np.cross([chunk['x'][t], chunk['y'][t], chunk['z'][t]], [chunk['vx'][t], chunk['vy'][t], chunk['vz'][t]])
                h = np.linalg.norm(H)*1e6
                e = np.sqrt(1 + (2*E*h**2)/(mu**2))
                ecc.append([Time(chunk['jd'][t], format='jd').decimalyear, e]) 
        apog.append([Time(t_ap, format='jd').decimalyear, ap]) 
        peri.append([Time(t_pe, format='jd').decimalyear, pe]) 
        i= i+30*4
    return np.array(apog), np.array(peri), np.array(ecc)


def cheb_fit(tab, ind, mask=True):

    plot_dir = '/home/akash98/work/IUCAA/simulations/results'
    try:
        os.mkdir(plot_dir+'/cheb_fit_ind_{id}'.format(id=ind))
        
    except FileExistsError:
        pass
    
    plot_dir = plot_dir+'/cheb_fit_ind_{id}'.format(id=ind)

    xdata = tab['jd']
    xdata_norm = -1 + 2*(xdata - xdata[0])/(xdata[-1]-xdata[0])

    req = {'x':'EM x distance', 'y':'EM y distance', 'z':'EM z distance', 'd':'EM distance'}

    for j,i in enumerate(req):

        if i=='d':  #Remove it if you want all the co-ordinates
            fit_ind = 50 #int(0.97*len(tab[i]))
            ydata = tab[i]
            poly = che.chebfit(np.float64(xdata_norm[:fit_ind]), np.float64(ydata[:fit_ind]), ind, full=True)

            if mask==True:
                fig, ax = plt.subplots()
                color='tab:red'
                ax.plot(xdata_norm, ydata, 'r.', label='Observed '+req[i])
                ax.plot(xdata_norm, che.chebval(xdata_norm, poly[0]), 'g-', label='Chebyshev fit for '+req[i])

                err = che.chebval(xdata_norm, poly[0]) - ydata
                textstr = '\n'.join((r'Extrp. error ($mm$)', r'$min$={min:2.3e}'.format(min=np.min(abs(err[fit_ind:,]))*1e6),\
                r'$max$={max:2.3e}'.format(max = np.max(abs(err[fit_ind:,]))*1e6),\
                r'$avg$={rms:2.3e}'.format(rms= np.mean(abs(err[fit_ind:,]))*1e6)))
                
                ax.set_xlabel('Normalized Julian day')
                ax.set_ylabel('EM distance in km', color=color)
                ax.tick_params(axis='y', labelcolor=color)
                ax.set_title('Chebyshev fit for {s}'.format(s=req[i]))
                ax.axvline(xdata_norm[fit_ind], linestyle='--', color='black')
                plt.tight_layout()
                ax.text(-1, np.min(np.array([np.min(ydata), np.min(che.chebval(xdata_norm, poly[0]))])),textstr,\
                bbox=dict(boxstyle='round', facecolor='white', alpha=1))
                
                ax2 = ax.twinx()
                color='tab:blue'
                ax2.plot(xdata_norm, err, label='Error in fit')
                ax2.set_ylabel('Error in kms', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                plt.tight_layout()
                
                lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                #fig.legend(lines, labels, loc='lower center')
                #plt.show()
                plt.savefig(plot_dir+'/cheb_fit_err_extrp_{s}.png'.format(s=i))

            else:
                plt.clf()
                plt.plot(xdata_norm, ydata, '.', label='Actaul '+req[i])
                plt.plot(xdata_norm, che.chebval(xdata_norm, poly[0]), label='Estimated '+req[i])
                plt.xlabel('Normalized Julian day')
                plt.ylabel('EM distance in km')
                plt.title('Chebyshev fit for {s}'.format(s=req[i]))
                plt.tight_layout()
                plt.legend()
                plt.text(-0.25,0.5*(np.min(ydata)+np.max(ydata)),textstr, bbox=dict(boxstyle='round', facecolor='white', alpha=1))
                plt.savefig(plot_dir+'/cheb_fit_{s}.png'.format(s=i))

                plt.clf()
                plt.plot(xdata_norm, err, label=i)
                plt.xlabel('Normalized Julian day')
                plt.ylabel('Error in kms')
                plt.title('Chebyshev error in {s}'.format(s=req[i]))
                plt.tight_layout()
                plt.text(-0.25,0.5*(np.min(err)+np.max(err)),textstr, bbox=dict(boxstyle='round', facecolor='white', alpha=1))
                plt.legend()
                plt.savefig(plot_dir+'/cheb_err_{s}.png'.format(s=i))


def func(x, *a): 
    exp=0
    f = a[0]*np.sin(a[1]*x + a[2]) + a[3]*(np.cos(a[4]*x + a[5])) 
    for i in range(len(a)-5): 
        exp = exp+ a[i+5]*x**i 
    
    return f*exp


def err_stats(err):
    err_min = np.min(abs(err))
    err_max = np.max(abs(err))
    err_avg = np.mean(abs(err))
    return err_min, err_max, err_avg


def opt_fit(xdata, ydata,a):

    def opt_fun(xdata, *a):
        exp=0 
        f = a[0]*np.sin(a[1]*np.arcsin(xdata) + a[2]) +  np.exp(a[3]*xdata)*np.sin(a[4]*np.arcsin(xdata) + a[5])
        for i in range(len(a)-6):  
            exp = exp+ a[i+6]*xdata**i  
        return f+exp

    popt, pconv = curve_fit(opt_fun, xdata, ydata, p0=a)
    post_fit = opt_fun(xdata, *popt)
    err = ydata- post_fit
    plt.plot(xdata, ydata, 'b.')
    plt.plot(xdata, post_fit, '-')
    plt.show()
    plt.clf()
    plt.plot(xdata, err)
    plt.show()
    print(err_stats(err))


def plot(apo, peri, ecc):
    ''' 
    plotfile = PdfPages("/home/akash/results/lunar_var.pdf")
    plt.plot(apo[:,0], apo[:,1]/1000, label='Apogee')
    plt.xlabel("Year")
    plt.ylabel("Variations in lunar orbit (x 1000km)")
    plt.legend()
    plt.tight_layout()
    plotfile.savefig()
   
    plt.clf()
    plt.plot(peri[:,0], peri[:,1]/1000, label='Perigee')
    plt.xlabel("Year")
    plt.ylabel("Variations in lunar orbit (x 1000km)")
    plt.legend()
    plt.tight_layout()
    plotfile.savefig()
    plt.clf()
    '''
    plt.plot(ecc[:,0], ecc[:,1], label='Eccentricity')
    plt.xlabel("Year")
    plt.ylabel("Variations in lunar orbit")
    plt.ylim([0.00, 0.10])
    for i in range(len(apo[:,0])):
        plt.scatter(apo[i][0], ecc[2*i+1][1], c='red')
        plt.scatter(peri[i][0], ecc[2*i][1], c='yellow')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ecc_var.pdf")

    plt.clf()
    plt.plot(apo[:,0], apo[:,1]/1000, label='Apogee')
    plt.plot(peri[:,0], peri[:,1]/1000, label='Perigee')
    plt.xlabel("Year")
    plt.ylabel("Variations in lunar orbit (x 1000km)")
    plt.legend()
    plt.ylim([352, 410])
    plt.tight_layout()
    plt.savefig("axis_var.pdf")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Code to generate solar system ephermeris')
    parser.add_argument('--time', help='UTC of interest to generate ephermeris, must be of isot format (2018-01-01T00:00:00)', type=str, default='2008-01-01T00:00:00')
    parser.add_argument('--eph', help='Planetary ephermeris to be used, eg: de430, de432s', type=str, default='de430')
    parser.add_argument('--year', help='Year of observation', type=str, default='2019')

    args = parser.parse_args()

    global m_moon
    m_moon = 7.346030775889471e+22

    if args.time=='now':
        time = Time.now()
        time.format='isot'
    else:
        time=Time(args.time, scale='utc', format='isot')
    
    # To generate ephemeris
    '''
    ephmeride = args.eph
    print(time)
    tab = eph(time, ephmeride)
    print(tab)
    tab.write('/home/akash/results/SS_Ephereride.txt', format='ascii', delimiter='\t', overwrite=True)
    '''

    # To get lunar variation
    '''
    print("\nGenerating timestamps\n")
    obs_times = get_times(time.jd, time.jd+5*u.year.to(u.day))
    print("\nGetting lunar positions")
    moon = get_moon(obs_times)
    apo, peri, ecc = orbit(moon)
    plot(apo, peri, ecc) '''

    #Chebyshev fit and err

    obs_times = get_times(time.jd, time.jd+ 30)
    emoon = get_moon(obs_times)
    cheb_fit(emoon, 60)