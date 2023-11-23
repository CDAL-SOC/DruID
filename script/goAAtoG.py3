import sys
#from datetime import datetime
#now=datetime.now()
#date_time = now.strftime("_%Y%m%d_%H.%M.%S.txt")
 
cmdargs = str(sys.argv)
print ("Args list: %s " % cmdargs)
# Pharsing args one by one 
print ("Script name: %s" % str(sys.argv[0]))
print ("First argument: %s" % str(sys.argv[1]))

#date_time = now.strftime("_%Y%m%d_%H.%M.%S.txt")
#date_time = sys.argv[1]+date_time

f=open(sys.argv[1],'r')
#fout=open(date_time,'w')
fout=open(sys.argv[3],'w')

for x in f:
    y = x.split(':')
    yy = y[1].split('.')
    idk = yy[1].index('>')
    print(y[0])
    print(yy)
    str="{} {} {} {} {} {}\n".format(y[0],yy[1][0:idk-1],yy[1][0:idk-1],yy[1][idk-1],yy[1][idk+1],sys.argv[2])
    print(str)
    fout.write(str)

f.close()
fout.close()