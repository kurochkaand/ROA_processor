I am working on the program for processing Raman and Raman optical activity (ROA) spectra.
I want you to plan how you write a python program that would use numpy, pandas, scipy, matplotlib or other libs.
Don't write any code yet. Just plan the functionality, arcitecture, and data processing pipeline, ideas how to create a good program for spectral analisis.
Plan a CLI application that should work on Windows, Visual Studio Code.
Here are the functions I want to have: spicke removal, denoising, baseline subtraction, baseline correction and normalisation; perform spectral unmixing; and visualise the results.
When I am measuring Raman and ROA, the instrument is producing the files with names like this:

prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_A-000_out.txt

Where:

"prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7" is a sample name, I write it manually;

"N20" is a numbers of scan cycles after which a new file is saved;

"500mW" incident laser power, power at sample is a more valuable parameter;

"146791s" acquisition time of camera (means 1.46791 seconds), this value is proportional to signal intensity;

"031" arbitrary number of experiment;

"2026-06-21" date;

"A" camera (there is also a camera "B", but I don't need it now);

"000" index of the block (it also means a file which was saved every 20 scan cycles), every next block has index incremented by one;

"out.txt" is a constant ending of the file.

Inside of the file 1st row is a header:

"""

#1/cm sum, dif: SCP DCPI DCPII SCPc # Gain 9.4 e/ADc # Power at sample 321 mW # Cycles 20 # Total times [s] 470.69 470.69 470.69 470.69

"""

where:

"1/cm" is a constant, which means units of measurement for the first column, it's reciprocal centimeters, this column goes from largest to lowest and the numbers don't change during the measurements and even between the measurements;

"sum" means that 2nd, 4th, 6th, 8th columns are sums of scattered light intensities for right and left circularly polarized light (IR + IL);

"dif" means that 3rd, 5th, 7th, 9th columns are differences of scattered light intensities for right and left circularly polarized light (IR - IL);

"SCP DCPI DCPII SCPc" are 4 different modalities of the spectral acquisition, for now lets focus on SCP (it means incident light is unpolarized and scattered light is split into right and left circularly polarized components;

"Gain 9.4 e/ADc" this value doesn't change during the measurement;

"# Power at sample 321 mW" it's constant during the measurement, it's an important parameter because it's proportional to signal intensity, the value of laser power that I enter in file name is less important;

"Cycles 20" this value means the number of scan cycles summed up to yield this block, this value gets larger with every next block, in this example it increments by 20.

"Total times [s] 470.69 470.69 470.69 470.69" is a total duration of measurement in seconds, the 4 numbers just repeat, the numbers get bigger with each block, this number is also proportional to signal intensity.

Here is how rows look like from 2nd to the last one:
"""
2609.36 3.60509e+08 1215 0 0 0 0 0 0
...
-48.0254 3.70329e+08 -156602 0 0 0 0 0 0
"""
So 1st column is the wavenumber scale, it's fixed for all the measurements and all blocks for camera A (we dont take into account camera B for now).
The 2nd column is Raman (IR+IL), it's ussually not noisy and it requires baseline subtraction, baseline is recorded in a separate experiment.
The 3rd column is ROA (IR-IL), it's very noicy and I would like to denoise it, baseline doesnt have to be subtracted for ROA, but sometimes it's curved, and can be staitened.
ROA is much more prone to cosmic spyckes appearance which have to be removed.
I will give a path to the info file:
X:\Cations_and_PGA\ROA-data\260621_CdCl2_precipitated\prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_info.txt
The content of the info file are not important but the begining of name of the file ("prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21") is same as the begining of all \*out.txt files.
In this examples there are 42 files (blocks).
Each block is a comulative summ of all previous plocks, maybe as a first step we should mathematically subtract (i+1) - i, to get isolated blocks.
then I think it will be easier to remove spickes.
then I thinke we can try to denoise (maybe presence of multiple blocks can be used for it).
Propose the ideas and ask me questions.
