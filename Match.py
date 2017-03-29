#!/usr/bin/python

#     This file is part of PyMatchInterface.
# 
#     PyMatchInterface is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     Foobar is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os, re
from matplotlib import pyplot as plt

class Serie(object):
    '''
    Represents one 1D series, can be 2 or 3 columns,
    exports self.x, self.y and self.s. Is capable of:

        Normalize :: 
            -0.5 -> 0.5 or to scale any other Series given
    '''
    def __init__(self, filename, label = None):
        self.original_filename = filename
        self.filename = filename
        self.start = None
        self.end   = None
        self.x     = None
        self.y     = None
        self.s     = None
        self.label = 'Noname' if label is None else label
        
        data = np.loadtxt(filename, unpack=True)
        if data.shape[0] == 2:
            self.x, self.y = data
            self.s = None
        else:
            self.s,self.x,self.y = data
    
    def limits(self, start = None, end = None):
        self.start = start
        self.end   = end
        return self
    
    def write(self, filename = None):
        filename = filename if filename is not None else self.filename
        
        if self.s is None:
            data = np.array(zip(self.x, self.y))
        else:
            data = np.array(zip(self.s, self.x, self.y))
        
        np.savetxt(filename, data)
        self.filename = filename
        
        return True
    
    def normalize(self, to = None):
        s = 1.0 if to == None else np.max(to.y) - np.min(to.y)
        l = -0.5 if to == None else np.min(to.y)

        self.y -= np.min(self.y)
        self.y *= s/np.max(self.y)
        self.y += l

        return self

    def plot(self):
        m1 = self.x.min() if self.start is None else self.start
        m2 = self.x.max() if self.end is None else self.end
        plt.axvspan(m1, m2, 0.05, 0.15, alpha=0.75, color="0.6", label='Used Segment')
        plt.plot(self.x, self.y, label='Series %s [%s]' % (self.label if self.label else '-',self.filename))
        plt.legend()

class MatchConfFile(object):
    ##
    # Location of the MATCH executable code
    ##
    MATCHCMD = "match" 

    @staticmethod
    def __pkv(fio, ek = None):
        pos = fio.tell()
        result = re.match("([^ ]+)[ ]+([^ ]+)", fio.readline().strip())

        if result is None:
            fio.seek(pos)
            KeyError("No decode !")

        k, v = result.groups()
        
        if ek and k != ek:
            fio.seek(pos)
            raise KeyError("Bad Expected Key")
        
        return v

    def __init__(self, filename):
        self.filename = filename
        self.__read()

    def __write(self, fio, variable):
        v = getattr(self, variable)
        if v == None: return
        if variable == "speeds": v = ",".join(v)
        print >>fio,"%-13s" % variable, "%s" % v

    def __read(self):
        fio = open(self.filename, "r")

        self.series1 = self.__pkv(fio, "series1")
        self.begin1  = float(self.__pkv(fio, "begin1"))
        self.end1    = float(self.__pkv(fio, "end1"))
        self.numintervals1 = int(self.__pkv(fio, "numintervals1"))

        fio.readline()
        
        self.series2 = self.__pkv(fio, "series2")
        self.begin2  = float(self.__pkv(fio, "begin2"))
        self.end2    = float(self.__pkv(fio, "end2"))
        self.numintervals2 = int(self.__pkv(fio, "numintervals2"))

        fio.readline()

        self.nomatch      = float(self.__pkv(fio, "nomatch"))
        self.speedpenalty = float(self.__pkv(fio, "speedpenalty"))
        self.targetspeed  = self.__pkv(fio, "targetspeed")
        self.speedchange  = float(self.__pkv(fio, "speedchange"))
        self.tiepenalty   = float(self.__pkv(fio, "tiepenalty"))
        self.gappenalty   = float(self.__pkv(fio, "gappenalty"))
        
        fio.readline()

        speeds = self.__pkv(fio, "speeds")
        if speeds:
            self.speeds = speeds.split(",")

        fio.readline()

        try:
            self.tiefile = self.__pkv(fio, "tiefile")
        except KeyError:
            self.tiefile = None

        try:
            self.series1gaps = self.__pkv(fio, "series1gaps")
        except KeyError:
            self.series1gaps = None

        try:
            self.series2gaps = self.__pkv(fio, "series2gaps")
        except KeyError:
            self.series2gaps = None

        self.matchfile   = self.__pkv(fio, "matchfile")
        self.logfile     = self.__pkv(fio, "logfile")

        fio.close()

    def write(self, filename):
        fio = open(filename, "w")
        
        self.__write(fio, "series1")
        self.__write(fio, "begin1")
        self.__write(fio, "end1")
        self.__write(fio, "numintervals1")
        
        fio.write("\n")
        
        self.__write(fio, "series2")
        self.__write(fio, "begin2")
        self.__write(fio, "end2")
        self.__write(fio, "numintervals2")

        fio.write("\n")

        self.__write(fio, "nomatch")
        self.__write(fio, "speedpenalty")
        self.__write(fio, "targetspeed")
        self.__write(fio, "speedchange")
        self.__write(fio, "tiepenalty")
        self.__write(fio, "gappenalty")

        fio.write("\n")

        self.__write(fio, "speeds")

        fio.write("\n")
        
        self.__write(fio, "tiefile")
        self.__write(fio, "series1gaps")
        self.__write(fio, "series2gaps")

        self.__write(fio, "matchfile")
        self.__write(fio, "logfile")

        fio.close()

        self.filename = filename

    def clean(self):
        if os.path.isfile(self.logfile):   os.unlink(self.logfile)
        if os.path.isfile(self.matchfile): os.unlink(self.matchfile)

        f = "%s.new" % self.series1
        if os.path.isfile(f): os.unlink(f)
        
        f = "%s.new" % self.series2
        if os.path.isfile(f): os.unlink(f)

    def getSeries(self, which):
        if which == 1:
            s = Serie(self.series1, '#1')
            s.limits(self.begin1, self.end1)
            return s
        elif which == 2:
            s = Serie(self.series2, '#2')
            s.limits(self.begin2, self.end2)
            return s
        else:
            raise Exception("No such serie")
    
    def run(self):
        if not os.path.isfile(self.filename):
            raise Exception("Filename '%s' Not Found." % self.filename)
        
        if not os.path.isfile(MatchConfFile.MATCHCMD):
            raise Exception("Program 'match' was not found in this system !")
        
        results = os.popen('./match -v %s 2>&1' % self.filename)
        
        items = {
            "Total" : -1,
            "point" : -1,
            "nomatch" : -1,
            "speed" : -1,
            "speedchange" : -1,
            "tie" : -1,
            "gap" : -1,
            "tie nomatch": -1
        }
        
        go = False
        for line in results:
            line = line.strip()
            if line == "---Penalties---":
                go = True
                continue
            
            if not go: continue
            
            try:
                k,v = line.split(":")
                k = k.strip()
                v = v.strip()
                if k not in items.keys(): continue
            except:
                continue

            items[k] = float(v)

        return items

    def optimize(self, parameter, values):
        if parameter not in self.__dict__:
            print "%s not in self" % parameter
            return

        rms = []
        x   = []
        s   = []

        for iv, v in zip(range(len(values)), values):
            print ".",
            setattr(self,parameter,v)

            self.write("lala.conf")
            results = self.run()
            _, x1, y1 = np.loadtxt(self.series1, unpack=True)
            _, x2, y2 = np.loadtxt(self.series2 + ".new" , unpack=True)
            xnew = np.linspace(min(x1), max(x2), 2000)
            y1n = np.interp(xnew, x1, y1)
            y2n = np.interp(xnew, x2, y2)
            r = np.sqrt( np.mean( (y1n-y2n)**2 ))
            self.clean()

            try:
                v = float(v)
            except:
                v = iv

            x.append(v)
            s.append(results['Total'])
            rms.append(r)

        _, ax1 = plt.subplots()
        
        ax1.plot(x, rms, "r-o", label='RMS')
        ax1.set_ylabel('RMS')
        formats, labels = ax1.get_legend_handles_labels()
        
        ax2 = ax1.twinx()
        ax2.plot(x, s, "k+-", label='Total Score')
        ax2.set_ylabel('Total Score')
        f, l = ax2.get_legend_handles_labels()
        
        formats.extend(f)
        labels.extend(l)

        plt.xlabel("%s" % (parameter))
        plt.legend(formats, labels)
        
    def report(self):
        print " ** Series **"
        print "%13s" % "Series:", self.series1, "begin:", self.begin1, "end:", self.end1, "num. Intervals:", self.numintervals1
        print "%13s" % "Series:", self.series2, "begin:", self.begin2, "end:", self.end2, "num. Intervals:", self.numintervals2
        print ""
        print "%13s" % "speeds:", ",".join(self.speeds)
        print ""
        print " ** Penalties **"
        print "%13s" % "nomatch:", self.nomatch
        print "%13s" % "speedpenalty:", self.speedpenalty
        print "%13s" % "targetspeed:", self.targetspeed
        print "%13s" % "speedchange:", self.speedchange
        print "%13s" % "tiepenalty:", self.tiepenalty
        print "%13s" % "gappenalty:", self.gappenalty
        print ""
        print " ** Constrains & Files **"
        print "%13s" % "series1gaps:", self.series2gaps
        print "%13s" % "series2gaps:", self.series1gaps
        print "%13s" % "tiefile:", self.tiefile
        print "%13s" % "matchfile:", self.matchfile
        print "%13s" % "logfile:", self.logfile
