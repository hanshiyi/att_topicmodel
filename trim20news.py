import os

outputfile = '20newspuresentence'
outputfile = open(outputfile,'w')
uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowercase = 'abcdefghijklmnopqrstuvwxyz'
datapath = '/home/shiyi/rlfortopicmodel/data/20news-18828'
for (dirname, subdir, subfile) in os.walk(datapath):
    if len(subfile)>0:
        for filename in subfile:
            # print dirname+'/'+filename
            file = open(dirname+'/'+filename)
            content = file.readlines()[2:]
            for line in content:
                if len(line.strip()) == 0:
                    continue
                newstr = ''
                for i in line.lower():
                    if i in lowercase:
                        newstr += i
                    else:
                        newstr += ' '
                newstr += '\n'
                outputfile.write(newstr)

