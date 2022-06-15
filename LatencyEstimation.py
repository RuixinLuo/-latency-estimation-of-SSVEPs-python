
# load modules
import numpy as np
import scipy.io as io
import os
from scipy.fftpack import fft
from scipy.signal  import detrend
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def freqDomian(signal, fs, nfft):
    """ Calculate amplitude and phase using fft

    Parameters
    ----------
    signal: ndarray
        eeg data, shape (nTimes,).
    Fs: float
        Sampling Rate
    nfft: int
        the number of fft

    Returns
    -------
    phiAns: ndarray
        phases of signal , shape (nfft,).
    freqAns: ndarray
        amplitude of signal , shape (nfft,).
    f: ndarray
        frequency corresponding to the fft results , shape (nfft,).
    """

    mf = fft(detrend(signal), nfft)
    phiAns = np.angle(mf , deg=True) - 180  # why -180
    freqAns = abs(mf) * 2 / len(signal)
    f = np.arange(0, len(mf), 1) * fs / len(mf)
    f = np.around(f, decimals=2 )
    return phiAns, freqAns, f

def func(x, a, b):
    return a * x + b

def count_Latency(Data ,Fre_list ,Fs, draw_fig=True,sub_id=0):
    """ SSVEP latency estimation based on paper [1]_.
    Adapted from https://github.com/iqgnat/SSVEP_phase_latency
    from MATLAB to python

    Parameters
    ----------
    Data: ndarray
        eeg data, shape (nTimes, nEvents, nTrials).
    Fre_list: ndarray
        stimulus frequency , shape (nEvents,).
    Fs: float
        Sampling Rate
    draw_fig: bool
        whether to draw

    Returns
    -------
    latency: float
        estimated latency of SSVEP


    References
    ----------
    .. [1] Jie P , Gao X , Fang D , et al. Enhancing the classification accuracy of steady-state visual evoked potential-based
          brain-computer interfaces using phase constrained canonical correlation analysis[J].Journal of Neural Engineering, 2011, 8(3):036027.
    """

    [nTimes, nEvents, nTrials] = Data.shape

    # calculate phase and SNR
    phiS = np.zeros((nEvents, nTrials))
    SNR = np.zeros((nEvents, nTrials))
    for i in range(nEvents):
        for j in range(nTrials):
            dat = Data[:,i,j]
            # calculate phase
            phiAns, freqAns, f = freqDomian(signal=dat, fs=fs, nfft=Fs*10)
            f_loc,f_loc_low,f_loc_high = np.where(f==np.around(Fre_list[i],decimals=2))[0],np.where(f==np.around((Fre_list[i]-1),decimals=2))[0],np.where(f==np.around((Fre_list[i]+1),decimals=2))[0]
            phiS[i,j] = phiAns[f_loc]
            # calculate phase
            snr = freqAns[f_loc] / np.mean(freqAns[int(f_loc_low):int(f_loc_high)+1])
            SNR[i,j] = snr

    stdPhase = phiS.copy()
    minPhase = phiS.copy()
    # 6 points in each freq, jump from the highest one, 5 jumps at most
    stepStd = np.zeros((nEvents, nTrials))
    for i in range(nEvents):
        stepStd[i,0] = np.std(phiS[i,:],ddof=1)
        for jump in range(5):
            phiTop = np.argmax(phiS[i,:])
            stdPhase[i, phiTop] = phiS[i, phiTop] - 360  # jump from the highest one
            stepStd[i, jump + 1] = np.std(stdPhase[i,:],ddof=1)  # the std change with jumping

    #   jump back from the lowest point to the best situation
    for i in range(nEvents):
        minStd = np.argmin(stepStd[i, :])
        if minStd != 0:
            for rejump in range(minStd):
                phiTop = np.argmax(stdPhase[i, :])
                minPhase[i, phiTop] = phiS[i, phiTop] - 360  # jump from the highest one

    #  find n
    nstep = np.zeros((2,nEvents))
    for i in range(nEvents):
        nstep[0, i] = np.ceil(0.08*Fre_list[i]-1)
        nstep[1, i] = np.floor(0.22 * Fre_list[i])

    # find linear regression (least square) that fits best
    W = np.reshape(SNR, -1)
    fre = np.repeat(Fre_list,nTrials,-1)
    R_all = []
    slope_all = []
    intercept_all = []
    nf_all = []
    for n0 in range(np.int(nstep[0, 0]),np.int(nstep[1, 0]+1), 1):
        for n1 in range(np.int(nstep[0, 1]), np.int(nstep[1, 1] + 1), 1):
            for n2 in range(np.int(nstep[0, 2]), np.int(nstep[1, 2] + 1), 1):
                for n3 in range(np.int(nstep[0, 3]), np.int(nstep[1, 3] + 1), 1):
                    for n4 in range(np.int(nstep[0, 4]), np.int(nstep[1, 4] + 1), 1):
                        for n5 in range(np.int(nstep[0, 5]), np.int(nstep[1, 5] + 1), 1):
                            for n6 in range(np.int(nstep[0, 6]), np.int(nstep[1, 6] + 1), 1):
                                for n7 in range(np.int(nstep[0, 7]), np.int(nstep[1, 7] + 1), 1):
                                    for n8 in range(np.int(nstep[0, 8]), np.int(nstep[1, 8] + 1), 1):
                                        for n9 in range(np.int(nstep[0, 9]), np.int(nstep[1, 9] + 1), 1):
                                            # Get N
                                            N = np.zeros((nEvents,nTrials))
                                            N[0,:] = n0 * np.ones((nTrials))
                                            N[1, :] = n1 * np.ones((nTrials))
                                            N[2, :] = n2 * np.ones((nTrials))
                                            N[3, :] = n3 * np.ones((nTrials))
                                            N[4, :] = n4 * np.ones((nTrials))
                                            N[5, :] = n5 * np.ones((nTrials))
                                            N[6, :] = n6 * np.ones((nTrials))
                                            N[7, :] = n7 * np.ones((nTrials))
                                            N[8, :] = n8 * np.ones((nTrials))
                                            N[9, :] = n9 * np.ones((nTrials))
                                            b = minPhase - 360 * N
                                            B = np.reshape(b, -1)
                                            # linear regression
                                            WLS = LinearRegression()
                                            X = fre.reshape(-1, 1)
                                            y = B.reshape(-1, 1)

                                            WLS.fit(X=X, y=y, sample_weight=W)
                                            slope = WLS.coef_
                                            intercept = WLS.intercept_
                                            R2 = np.linalg.norm(func(X, slope, intercept) - y,ord=2)
                                            latency_tmp = (slope / -360) * 1000
                                            # slope due to latency 80ms-220ms
                                            if slope<-28.8 and slope>-79.2 and latency_tmp>79 and latency_tmp <219:
                                                R_all.append(R2)
                                                slope_all.append(slope)
                                                intercept_all.append(intercept)
                                                nf_all.append(N)

    inx = R_all.index(min(R_all))
    max_R = R_all[inx]
    max_slope = slope_all[inx]
    max_intercept = intercept_all[inx]
    max_nf = np.array(nf_all[inx])
    max_phiE = phiS - max_nf * 360

    if draw_fig:
        plt.scatter(fre,max_phiE.reshape(-1))
        plt.plot(fre, np.squeeze(func(X, max_slope, max_intercept)), 'b-')
        plt.title('Sub'+str(sub_id))
        plt.show()
        print('sub_no=',sub_id,'Latency=',-max_slope/360*1000,'ms',' MSE=', max_R,'nf=',max_nf[:,0])

    return -max_slope/360*1000

if __name__ == '__main__':
    #  setting
    filepath = r'\Bench'
    taskTime = 5
    fs = 250


    Fre_list =np.around(
    [8, 12, 11.2, 15.2, 10.4, 14.4, 9.6, 13.6, 8.8, 12.8],decimals=2 ) # only using frequency when phase_init=0
    Phase0_loc = [0, 4, 11, 15, 18, 22, 25, 29, 32, 36]
    channel = np.array([62]) - 1  # channel Oz
    #   main
    subject_list =  ['S'+'{:02d}'.format(idx_subject+1) for idx_subject in range(35)]


    L = []
    sub_id = 0
    for id in subject_list:
        # load data
        path = os.path.join(filepath, str(id) + '.mat')
        data = io.loadmat(path)
        dataAll = data['data']
        data1 = np.squeeze(dataAll[channel, int(fs * 0.5):int(fs * (taskTime + 0.5)), ...]) # choose signal from Oz
        data_pre = data1[:, Phase0_loc, :]  # choose signal when phase_init=0
        del dataAll, data, data1
        # count Latency as a
        a = count_Latency(Data=data_pre, Fre_list=Fre_list, Fs=fs, draw_fig=True, sub_id=sub_id+1)
        L.append(a)
        sub_id = sub_id+1
    L =np.squeeze(np.array(L))
    print('mean_latency=',np.mean(L))
    # if everything is ok, you will get the mean_latency ≈ 136ms
    # the result is same as the source code from https://github.com/iqgnat/SSVEP_phase_latency


