#! /usr/bin/env python
"""
Usage: lut2sphinxtbl.py lutfile sphinxfile atlasname
"""
import os
import re
import sys
import numpy as np

if len(sys.argv) < 2:
    print __doc__
    sys.exit(0)


lutfile = sys.argv[1]
spxfile = sys.argv[2]
namelist = []
for i, arg in enumerate(sys.argv):
    if i > 2:
        namelist.append(arg)
atlasname = " ".join(namelist)    

lutarr = np.genfromtxt(lutfile, str)
lutarr = lutarr[:,:2]
maxid = 0
maxname = 0
for row in lutarr:
    if len(row[0]) > maxid:
        maxid = len(row[0])
    if len(row[1]) > maxname:
        maxname = len(row[1])
leftbar = max(maxid, 3)
rightbar = max(maxname, 20) 

fid = open(spxfile, "w")

fid.write(".. _%s:\n\n" % os.path.splitext(os.path.split(spxfile)[1])[0])
fid.write("%s\n" % atlasname)
for i in range(len(atlasname)):
    fid.write("-")
fid.write("\n\n")
leftline = ""
for i in range(leftbar):
    leftline = "".join([leftline, "="])
rightline = ""
for i in range(rightbar):
    rightline = "".join([rightline, "="])
fid.write("%s   %s\nID     Region\n%s   %s\n" % (leftline, rightline, leftline, rightline))
for row in lutarr:
    name = row[1]
    if not re.match("[rR](h|ight|\-).*", name) and not re.match("[Uu]nknown", name):
        id = row[0][-3:]
        if len(id) > 3:
            id = int(id[-3:])
        else:
            id = int(id)
        m = re.match("(([lL])(h|eft|\-)(\-*))(.*)", name)
        if m:
            name = name[len(m.group(1)):].capitalize()
        space = ""
        for i in range(7-len(str(id))):
            space = "".join([space, " "])
        fid.write("%d%s%s\n" % (id, space, name))

fid.write("%s   %s\n\n" % (leftline, rightline))

