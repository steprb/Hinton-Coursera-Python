### Python version of programming assignments for **"Neural Networks for Machine Learning"** **Coursera** course taught by **Geoffrey Hinton**.

This is basically a line-by-line conversion from Octave/Matlab to Python3 of four programming assignments from 2013 Coursera course "Neural Networks for Machine Learning" taught by Geoffrey Hinton.

All python functions for one assignment are gathered in one file: assignment1.py, assignment2.py, etc. Data files are identical to original Octave assignments and are in *.mat format.

Python packages in use:
* numpy for vector and matrix operations;
* scipy.io for loading *.mat files to Python;
* matplotlib for drawing plots.

Solutions are not included. But all scripts were tested and let you successfully pass assignments as of Feb 24, 2017 (provided you add the code required in each assignment).

PS The course is really good. 

Minor notes:
* Comments are from original Octave scripts. Changed some of them to match variable names. Added some comments of my own, they usually start with "PS: ".

* When plots appear on screen, just close them to continue script execution.

* Assignment 3, question 3: 
a3(1e7, 7, 10, 0, 0, false, 4) doesn't pass gradient check. I haven't figured out why. Doesn't matter for answering the question.

* Assignment 4, last question: haven't tried that.

* Versions: Python 3.4.1, numpy 1.11.0, matplotlib 1.5.1, scipy 0.17.1







