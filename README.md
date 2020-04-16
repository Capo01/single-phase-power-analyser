# single-phase-power-analyser
A low cost single phase power analyser capable of working with high frequency arbitrary waveforms. Makes use of a USB oscilloscope and sigrok-cli for acquiring data and the numerical python library for analysis. 

## Features
* As it makes use of an oscilloscope, it can measure arbitrary waveforms (current and voltage) at high frequency.
* As the wave forms are acquired and stored in a python array, they can be charted and analysed using the powerful numpy, scipy, matplotlib or similar python libraries.
* With minor changes, this script could be extended for use with PWM voltage (such as from motor controllers) more oscilloscope channels for 3 phase power analysis. Note that multiple oscilloscopes cannot be used simultaneously unless there is a way to synchronise them due to sigrok-cli only working with one device at a time.

## Design 
A Hantek 6022BE two channel oscilloscope is used to monitor the voltage (CH1) and current (CH2) of a alternative source. The current can be measured through a shunt resistor, or as I did, with a Hantek current clamp. The oscilloscope acquires a user specified set number of samples at a desired sample rate. This data is then pulled off the oscilloscope over USB using sigrok-cli. As sigrok-cli operates by command prompt only, the python 'os' library is used. The data is then analysed to calculate the apparent power, power factor, peak to peak voltage/current etc. Scipy is also used to fit a sine wave to the voltage and current wave forms for frequency estimation, but is not required for anything else.

Example of the output data acquired from a 50 Hz transformer power some incandescent light globes.

![alt text](https://github.com/Capo01/single-phase-power-analyser/blob/master/example.png "Example of output data")
