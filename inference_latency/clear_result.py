import os

for first in os.listdir():
    if os.path.isdir(first):
        for second in os.listdir(os.path.join('./',first)):
            if os.path.exists(os.path.join('./',first,second,'result.txt')):
                f = open(os.path.join('./',first,second,'result.txt'),'w')
                f.close()