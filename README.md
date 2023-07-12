# Self-Organizing-Map
Particle Matching, Tracing and Velocimetry with Unsupervised Machine Learning

## Description
This self-organizing map takes in particle position coordinates from images or image sequences and matches particles from two consecutive images. The matches can be appended to trace particles over image sequences and display their traces. Angular velocities of particles on their traces can be calculated and a particle flow field can be generated.

## Installation
The self-organizing map can be imported in Python with import SOMacc. Some other packages are needed for execution as seen in the examples notebook. The python package "Numba" is used for acceleration. If "Numba" is not accessable and faster processing is not needed, the package "SOM" can be imported rather then "SOMacc". 

## Usage
A minimal example is shown in the according notebook. A Unet is used for gathering particle positions from the example images. Other methods such as TrackPy could also be used. The SOM just reads in 2-column Numpy Arrays with unsorted particle position coordinates.
