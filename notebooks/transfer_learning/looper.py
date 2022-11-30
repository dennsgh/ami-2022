'''
https://stackoverflow.com/questions/44112399/automatically-restart-a-python-program-if-its-killed
This script allows you to enforce running the filename script despite memory leaks.
'''

import subprocess

filename = 'ilsvrc_optuna_trial.py'
while True:
    p = subprocess.Popen('python3 ' + filename, shell=True).wait()

    if p != 0:
        continue
    else:
        break
