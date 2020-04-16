"""
Power analyzer (single phase)
Created on Wed Apr  1 18:12:57 2020
@author: Richard Parsons

Voltage and current of an AC source measured using a Hantek 6022BE oscilloscope.
Channel 1: 10x compensated probe
Channel 2: Hantek current clamp (1mV/10mA [0.1 to 15A] or 1mV/100mA [0.1 to 65A])

To do:
    - add support for wave form estimation from PWM
    - Use numpy arrays in format_data function
    - remove sine wave fitting since its not really needed

"""
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

#### Parameters ###
sample_rate = 1E6 # An int, the sample rate in Hz. See 6022BE notes for available rates
num_samples = 4E4 # 256, 512, 2048 etc.
output_file = False # Outputs csv file into sigrok-cli dir instead of returning data
sigrok_dir = 'C:\Program Files (x86)\sigrok\sigrok-cli' # sigrok install dir
dump_num = 600 # An int. Number of samples to exclude to remove artifact.
v_scaler = 10 # Probe attenuation (i.e. 1x or 10x)
i_scaler = 10 / 4 #An int, [probe scal/num. turns] 10 = 1mV/10mA or 100 = 1mV/100mA
v_correction = 0.9238 # An int, in volts, for correction of non-zero isolated measurement
i_correction = 0.151 # An int, in amps, for correction of non-zero isolated measurement

def acquire_data(sample_rate, num_samples, output_file = False,
                sigrok_dir = 'C:\Program Files (x86)\sigrok\sigrok-cli'):
    """Returns a set number of voltage readings for Ch1 anc Ch2 at the 
    specified sample rate as a string. String includes Ch1 and ch2 label and units
    for ever reported value. All ch1 values returned first, then ch2 values.
    
    :param sample_rate: An int, the oscilloscope sample rate in Hz.
    :param num_samples: An int, the number of samples to record.
    :param output_file: A boolean, if true saves a csv file in sigrok-cli dir.
    :param sigrok_dir: A string, the directory of sigrok-cli.
    """
    # complie a string from imput parameters for use in sigrok-cli console
    sample_rate = int(sample_rate)
    # add 600 extra sampels to be dumped later to remove measurement artifact
    if int(num_samples) > 512:
        num_samples = int(num_samples) + dump_num
    else:
        num_samples = int(num_samples)
    if output_file == True:
        sigrok_cmd = str('sigrok-cli --driver hantek-6xxx --config samplerate=' 
                         + str(sample_rate) + ' --output-file test.csv --output-format csv --samples=' + num_samples)
        
    else:
        sigrok_cmd = str('sigrok-cli --driver hantek-6xxx --config samplerate=' 
                         + str(sample_rate) + ' --samples=' + str(num_samples))

    print('Sigrok command:', sigrok_cmd)
    # Send sigrok-cli command through console and store the returned oscilloscope data
    os.chdir(sigrok_dir) # set working dir to sigrok_dir
    output = subprocess.run(sigrok_cmd, shell=True, check=True, capture_output=True, text=True)
    raw_data = output.stdout
    return raw_data

def format_data(raw_data):
    """Formats oscilloscope voltage data and return a list of lists with 
    the format [time, ch1, ch2].
    
    :param raw_data: A string, raw data from acquire_data().
    """
    time_interval = 1000/ sample_rate # time interval in milliseconds
    time_list = []
    data_list = []
    # Split raw_data into a list and strip out channel names
    data = raw_data.split('\n')
    del data[-1] # last entry is a new line, so it is removed
    for i in range(len(data)):
        data[i] = data[i].strip("CH V") # strip channel and voltage label
        data[i] = data[i][2:] # remove the channel number
        data[i] = data[i].strip() # strip any white space
        data[i] = float(data[i])
        time_list.append(i * time_interval)
    # Combine into a single list, dump first 600 samples
    num_samples = int(len(data) * 0.5)
    if num_samples > 512:
        data_list.append(time_list[:num_samples-dump_num])
        data_list.append(data[dump_num:num_samples])
        data_list.append(data[dump_num+num_samples:])
    else:
        data_list.append(time_list[:int(len(data)*0.5)])
        data_list.append(data[:int(len(data)*0.5)])
        data_list.append(data[int(len(data)*0.5):])
    return data_list

def sine_func(time, amplitude, freq, phase):
    """Using with scipy optimize to fit sine wave
    
    :param time: A float, time data.
    :param amplitude: A float, the peak deviation of the function from zero.
    :param angular_freq: the rate of change of the function argument in units of radians per second
    :param phase: A float, specifies (in radians) where in its cycle the oscillation is at t = 0
    """
    return amplitude * np.sin(2* np.pi * freq * time + (phase/2* np.pi * freq))


def plot(time, time_unit, ch1, ch1_unit, ch2, ch2_unit, power, power_unit):
    """Plot the voltage, current and power output.
    
    :param time: A list, time data.
    :param time_unit: A string, time label unit. eg. s or ms
    :param ch1: A list, ch1 voltage data
    :param ch1_unit: A string, ch1 voltage label unit. e.g. V or mV
    :param ch2: A list, ch2 current data
    :param ch2_unit: A string, ch2 current label unit. e.g. A or mA
    :param ch1: A list, power data
    :param ch1_unit: A string, power label unit. e.g. W or mW
    """
    # plot the data
    plt.figure(figsize=(6, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, ch1, '', label='Measured data')
    plt.title('Single phase measurement')
    plt.ylabel('Voltage (' + ch1_unit + ')')
    plt.axhline(0, color='black', lw=2)
    plt.plot(time, sine_func(time, amplitude_v, freq_v, phase_v),
         label='Fitted function')
    plt.legend(loc='right')
    
    plt.subplot(3, 1, 2)
    plt.plot(time, ch2, '', label='Measured data')
    plt.ylabel('Current (' + ch2_unit + ')')
    plt.axhline(0, color='black', lw=2)
    plt.plot(time, sine_func(time, amplitude_i, freq_i, phase_i),
         label='Fitted function')
    plt.legend(loc='right')
    
    
    plt.subplot(3, 1, 3)
    plt.plot(time, power, '')
    plt.xlabel('Time (' + time_unit + ')')
    plt.ylabel('Real Power (' + power_unit + ')')
    plt.fill_between(time, 0, power)
    
    # Adjust the subplot layout
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
    plt.show()

def statistics():
    """Prints detected frequency, V RMS, I RMS, Vpp, Ipp, power
    """ 
    print('Frequency =', freq_v * 1000)
    print('V RMS  =', v_rms, 'V')
    print('V average =', v_avg, 'V')
    print('Vpeak to peak =', v_pp, 'V')
    print('I RMS  =', i_rms, 'A')
    print('I average =', i_avg, 'A')
    print('Ipeak to peak =', i_pp, 'A')
    print('Apparent Power =', apparent_power, 'W')
    print('True Power =', np.mean(true_power), 'W')
    print('Power factor =', powerfactor)

### Script ###
# Read data from the oscilloscope
raw_data = acquire_data(sample_rate, num_samples, output_file, sigrok_dir)
#  Remove the ch1 names and units and place the data into a list
formated_data = format_data(raw_data)

# Seperate data into time, ch1 and ch2 and convert into a numpy arrary
time = np.array(formated_data[0])
ch1 = np.array(formated_data[1])
ch2 = np.array(formated_data[2])

# Scale and correct the data
ch1_corrected  = ch1 * v_scaler - v_correction
ch2_corrected  = ch2 * i_scaler - i_correction

# fit sine wave to V data
params_v, params_covariance = optimize.curve_fit(sine_func, time, ch1_corrected,
                                               bounds=(0, [50, 0.1, 100]))
amplitude_v = params_v[0]
freq_v = params_v[1]
phase_v = params_v[2]

# fit sine wave to I data
params_i, params_covariance = optimize.curve_fit(sine_func, time, ch2_corrected,
                                               bounds=(0, [50, 0.1, 100]))
amplitude_i = params_i[0]
freq_i = params_i[1]
phase_i = params_i[2]

# calculate values of interest
v_sq = ch1_corrected ** 2
i_sq = ch2_corrected ** 2

v_rms = np.sqrt(v_sq.sum()/ch1_corrected.size)
i_rms = np.sqrt(i_sq.sum()/ch1_corrected.size)  

v_pp = ch1_corrected.max() - ch1_corrected.min()
i_pp = ch2_corrected.max() - ch2_corrected.min()

v_avg = np.mean(ch1_corrected)
i_avg = np.mean(ch2_corrected) 

power = np.abs(ch1_corrected) * np.abs(ch2_corrected)
apparent_power = np.mean(v_rms * i_rms)
true_power = np.mean(np.abs(ch1_corrected) * np.abs(ch2_corrected)) #summation
powerfactor = true_power / apparent_power

# plot the data
plot(time, 'ms', ch1_corrected, 'V', ch2_corrected, 'A', power, 'W')
    
# report statistics
statistics()
