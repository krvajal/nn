import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
from math import pi, sqrt


def compute_variability(trials):
    """
        Computes the variabilty of the sample data
        Parameters:
            trials -> list
    """
    # filter each array
    t = np.linspace(0, 1, 10000)  # map the time
    intervals = []
    for i in range(0, 128):
        spike_times = t[trials[i]]  # mask the times that contain spikes
        trial_intervals = np.diff(spike_times)  # compute the diff between spike times
        intervals.append(trial_intervals)

    total_intervals = np.concatenate(tuple(intervals))  # merge all in a big dataset

    n, bins, patches = plt.hist(total_intervals, 50, normed=True)
    plt.plot(bins, mlab.normpdf(bins, np.mean(total_intervals), np.std(total_intervals)))
    plt.xlabel("I")
    plt.ylabel("Probabilidad")
    plt.title("file='spikes.dat', num realizaciones = 128")
    Cv = np.std(total_intervals) / np.mean(total_intervals)  # coeficiente de variabilidad
    plt.text(0.03, 80, r"$C_v = %lf$" % Cv)
    plt.text(0.03, 70, r"$\mu = %lf$" % np.mean(total_intervals))
    plt.text(0.03, 60, r"$\sigma^2 = %lf$" % np.var(total_intervals))
    plt.show()


def compute_fano(trials):
    """ Compute the Fano factor of the trials
        The Fano factor is defined as $F = \frac{\sigma_n^2}{<N>}$
        If the Fano factor is bigger than 1 it means that
    """
    num_spikes = np.sum(trials, axis=1)
    mean_num_spikes = np.mean(num_spikes)
    var_num_spikes = np.var(num_spikes)
    print(mean_num_spikes)
    print(var_num_spikes)
    F = var_num_spikes / mean_num_spikes  # Fano factor

    n, bins, patches = plt.hist(num_spikes, 20, normed=1)
    plt.plot(bins, mlab.normpdf(bins, mean_num_spikes, sqrt(var_num_spikes)))
    plt.xlabel("Numero de spikes")
    plt.ylabel("Probabilidad")
    plt.title("file='spikes.dat', num realizaciones = 128")
    plt.text(90, 0.025, r"$\mu = %lf$" % mean_num_spikes)
    plt.text(90, 0.023, r"$\sigma^2 = %lf$" % var_num_spikes)
    plt.text(90, 0.020, r"$F=%lf$" % F)
    plt.show()

def  compute_psth(signal,trials, dt):
    """ Compute the Peri-Stimulus-Time Histogram"""
    num_rep =  trials.shape[0]
    time_step = 0.0001 # [s] 1ms time between samples
    total_time =  1.0 # [s] duration of the experiment
    num_bins = int(total_time// dt) + 1
    bin_size =  10000//num_bins
    hist =  np.zeros(num_bins) #create empty buckets
    cumulated =  np.sum(trials,axis=0)/num_rep
    time_points = np.linspace(0,1,num_bins)
    for i  in range(1,num_bins):
        start = (i-1) * bin_size
        end = (i) * bin_size
        hist[i-1] = np.sum(cumulated[start:end])

    # make the plots
    # upper plot
    plt.subplot(2,1,1)
    plt.plot(signal[:,0]/1000, signal[:,1])
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Intensidad [db]");
    plt.title(r"Senal de estimulo")
    # bottom plot
    plt.subplot(2,1,2)
    plt.plot(time_points, hist,'r')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Average num spikes");
    plt.show()




def compute_filter(signal, trials):
    """ Compute the filter based on the signal and the responses"""
    time_window  = 50 # [ms] time to look back
    dt = 0.1 # [ms] time between signals
    response_dt = 0.1
    l = int(time_window/dt)
    previous_stimulus = np.zeros(l)
    count = 0

    for trial in trials:
        print "Trial", trial,"of" , trials.size
        indexes = np.argwhere(trial) # find the spikes indexes
        for idx in indexes:
            """ Compute the indexes of the start and end
                of the stimulus window that is 50 ms before the 
                 spike was generated
            """
            idx_start = int(idx - l)
            idx_end =  int(idx)
            if idx_start > 0 : # take only spikes that started after 50 ms
                previous_stimulus = previous_stimulus + signal[idx_start:idx_end,1]
                count = count + 1 # increment the count
    print("Spikes averaged",count)
    previous_stimulus = previous_stimulus / count

    plt.plot(np.linspace(0,50,l), previous_stimulus[::-1])
    plt.xlabel(r" Tiempo [ms]")
    plt.ylabel(r"Intensidad [dB]")
    plt.title(r"Filtro")
    plt.text(20,3,r"$\Delta t = %lf ms$"%time_window)
    plt.show()



if __name__ == "__main__":


    trials = np.loadtxt("../data/spikes.dat", dtype=bool)
    signal  = np.loadtxt("../data/stimulus.dat")

    # plot the stimulos

    # print(trials.shape)

    #compute_variability(trials)
    #compute_fano(trials)
    dt =  0.01 # [s] sampling interval
    #compute_psth(signal,trials, dt)
    compute_filter(signal, trials)
