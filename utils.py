import pyabf
import pandas as pd
import numpy as np


def find_ap(
    abf,
    sampling_rate = 10,
    detection_threshold = 0,
    rate_threshold = 0.5
    ):
    time = abf.sweepX
    trace = abf.sweepY

    # Sample signals to reduce noise
    ind_sample = np.where(np.arange(len(trace))%sampling_rate == 0)[0]
    trace_ = pd.Series(trace[ind_sample])
    time_ = pd.Series(time[ind_sample])

    # Compute time derivative
    rate = np.gradient(trace_)

    # Find indices of the action potential segments:
    # 1) time derivative higher than the rate threshold
    # 2) amplitude higher than the detection threshold
    ap_segments = np.where((rate >= rate_threshold) & 
                           (trace_ >= detection_threshold))[0]

    # Find action potential time and amplitude
    ap_amplitude = []
    ap_time = []
    if len(ap_segments) == 1:
        ind = ap_segments[0]
        max_ind = np.argmax(trace_[ind:ind+2])
        amplitude = max(trace_[ind:ind+2])
        ap_amplitude.append(amplitude)
        ap_time.append(time_[max_ind])
    if len(ap_segments) > 1:
        for i in range(1, len(ap_segments)):
            if ap_segments[i] != ap_segments[i-1]+1:
                ind = ap_segments[i-1]
            elif i == len(ap_segments)-1:
                ind = ap_segments[-1]
            else:
                continue
            # Check if the peak happens after the actional potential segments
            max_ind = np.argmax(trace_[ind:ind+2])
            amplitude = max(trace_[ind:ind+2])
            ap_amplitude.append(amplitude)
            ap_time.append(time_[max_ind])
    return ap_amplitude, ap_time


def measure_current(abf):
    time = pd.Series(abf.sweepX)
    current = pd.Series(abf.sweepY)

    # Find the start and end of the current step period
    # defined as the period of time when the current is higher
    # then the median value
    current_step = np.where(current >= (max(current)+min(current))*0.5)[0]
    
    if len(current_step) == 0:
        start = None
        end = None
        current_amplitude = 0
    else:
        start = time[current_step[0]]
        end = time[current_step[-1]]

        # Find baseline indices
        baseline = np.logical_and(time >= start-0.04,
                                  time <= start-0.02)
        # Find current step sample indices
        cs_sample = np.logical_and(time >= start+0.05,
                                   time <= end-0.05)
        # Compute amplitude
        current_amplitude = round(np.median(current[cs_sample])-np.median(current[baseline]))//10*10
        
    return current_amplitude, start, end


def read_abf(
    file_name,
    sampling_rate = 10,
    detection_threshold = 0,
    rate_threshold = 0.5
    ):
    abf = pyabf.ABF(file_name)
    ap_amplitude_list = []
    ap_time_list = []
    current_amplitude_list = []
    total_ap_count_list = []
    in_cs_ap_count_list = []

    for i in abf.sweepList:
        # Find action potential stat
        abf.setSweep(i, channel=0)
        ap_amplitude, ap_time = find_ap(abf=abf,
                                        sampling_rate=sampling_rate,
                                        detection_threshold=detection_threshold,
                                        rate_threshold=rate_threshold)
        ap_amplitude_list.append(ap_amplitude)
        ap_time_list.append(ap_time)
        total_ap_count_list.append(len(ap_amplitude))

        # Find current stat
        abf.setSweep(i, channel=1)
        current_amplitude, start, end = measure_current(abf)
        current_amplitude_list.append(current_amplitude)

        # Count number of action potentials within the current step period
        if (len(ap_time) == 0) | (current_amplitude == 0):
            in_cs_ap_count_list.append(0)
        else:
            ap_time_ = pd.Series(ap_time)
            in_cs_ap_count = len(ap_time_[(ap_time_ >= start) & (ap_time_ <= end)])
            in_cs_ap_count_list.append(in_cs_ap_count)
    
    df = pd.DataFrame({"sweep": abf.sweepList,
                      "current_amplitude": current_amplitude_list,
                      "in_cs_ap_count": in_cs_ap_count_list,
                      "total_ap_count": total_ap_count_list,
                      "ap_amplitude": ap_amplitude_list,
                      "ap_time": ap_time_list})
    return df


def find_derivatives(
    abf,
    ap_time,
    cs_start,
    cs_end,
    sampling_rate = 10
):
    time = abf.sweepX
    trace = abf.sweepY
    
    # Find time of occurence within the current step period
    ap_time_ = pd.Series(ap_time)[(ap_time >= cs_start) & (ap_time <= cs_end)]

    # Sample signals to reduce noise
    ind_sample = np.where(np.arange(len(trace))%sampling_rate == 0)[0]
    trace_ = pd.Series(trace[ind_sample])
    time_ = pd.Series(time[ind_sample])

    # Compute time derivative
    # Need to adjust the scale based on the sampling rate
    rate = pd.Series(np.gradient(trace_)*10, index=time_)

    # Compute second-order derivative
    shape = pd.Series(np.gradient(rate), index=time_)
    
    return rate, shape

