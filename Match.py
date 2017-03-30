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
        self.xm    = None
        self.ym    = None
        self.sm    = None
        self.ismatched = False
        self.label = 'Noname' if label is None else label
        
        self.x, self.y, self.s = self.__read(self.filename)
        
        try:
            filename = self.filename + ".new"
            self.xm, self.ym, self.sm = self.__read(filename)
            self.ismatched = True
        except:
            pass
    
    @staticmethod
    def __read(filename):
        if not os.path.isfile(filename):
            raise Exception("File does not exists !")
        
        data = np.loadtxt(filename, unpack=True)
        if data.shape[0] == 2:
            x, y = data
            s = None
        else:
            s,x,y = data
        return x,y,s
    
    def limits(self, start = None, end = None):
        '''
        Sets limits of use for the data serie by match
        '''
        self.start = start
        self.end   = end
        return self
    
    def write(self, filename = None):
        '''
        Save this data serie
        '''
        filename = filename if filename is not None else self.filename
        
        if self.s is None:
            data = np.array(zip(self.x, self.y))
        else:
            data = np.array(zip(self.s, self.x, self.y))
        
        np.savetxt(filename, data)
        self.filename = filename
        
        return True
    
    def normalize(self, to = None):
        '''
        Normalize this serie between -0.5 and 0.5, or 
        if to is an Serie instance match this one to the other.
        '''
        if to is not None and not isinstance(to, Serie):
            raise Exception("Bad destination Serie instance on to variable !")
        
        s = 1.0 if to == None else np.max(to.y) - np.min(to.y)
        l = -0.5 if to == None else np.min(to.y)

        self.y -= np.min(self.y)
        self.y *= s/np.max(self.y)
        self.y += l

        return self
    
    def plotcomp(self, other):
        '''
        Make a comparison plot of matched data series.
        Other is another Serie instance that was matched to self
        '''
        if not isinstance(other, Serie):
            raise Exception("Need the other serie that was matched to compare !")
        
        if not self.ismatched or not other.ismatched:
            raise Exception("Both series needs to be Matched !")
        
        plt.subplot(2,1,1)
        plt.plot(self.x, self.y, label=self.filename)
        plt.plot(other.xm, other.ym, label=other.filename)
        plt.axvspan(self.start, self.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % self.filename)
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(self.xm, self.ym, label=self.filename)
        plt.plot(other.x, other.y, label=other.filename)
        plt.axvspan(other.start, other.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % other.filename)
        plt.legend()
    
    def plot(self):
        '''
        Plot this data serie and its parameters
        '''
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
        self.series1 = None
        self.begin1  = None
        self.end1    = None
        self.numintervals1 = None
        self.series2 = None
        self.begin2  = None
        self.end2    = None
        self.numintervals2 = None
        self.nomatch      = None
        self.speedpenalty = None
        self.targetspeed  = None
        self.speedchange  = None
        self.tiepenalty   = None
        self.gappenalty   = None
        self.speeds = None
        self.tiefile = None
        self.series1gaps = None
        self.series2gaps = None
        self.matchfile   = None
        self.logfile     = None
        self.filename = filename
        
        if os.path.isfile(filename):
            self.__read()
        else:
            self.matchfile = self.filename.replace(".conf","") + ".match"
            self.logfile = self.filename.replace(".conf","") + ".log"

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

        try:
            self.nomatch      = float(self.__pkv(fio, "nomatch"))
        except:
            print "Failed to read Nomatch"

        try:
            self.speedpenalty = float(self.__pkv(fio, "speedpenalty"))
        except:
            print "Failed to read SpeedPenalty"
        
        try:
            self.targetspeed  = self.__pkv(fio, "targetspeed")
        except:
            print "Failed to read TargetSpeed"

        
        try:
            self.speedchange  = float(self.__pkv(fio, "speedchange"))
        except:
            print "Failed to read Speedchange"

        
        try:
            self.tiepenalty   = float(self.__pkv(fio, "tiepenalty"))
        except:
            print "Failed to read Tiepenalty"
        
        try:
            self.gappenalty   = float(self.__pkv(fio, "gappenalty"))
        except:
            pass

        fio.readline()

        try:
            speeds = self.__pkv(fio, "speeds")
            if speeds:
                self.speeds = speeds.split(",")
        except:
            speeds = []

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

    def write(self, filename  = None):
        if filename is None:
            filename = self.filename
        
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

    def setSeries(self, which, filename, begin, end, nintervals):
        if not os.path.isfile(filename):
            raise Exception("Not a file ! ")

        if which == 1:
            self.series1 = filename
            self.begin1 = float(begin) 
            self.end1   = float(end)
            self.numintervals1 = int(nintervals)
        elif which == 2:
            self.series2 = filename
            self.begin2 = float(begin) 
            self.end2   = float(end)
            self.numintervals2 = int(nintervals)
        else:
            raise Exception("time series not know !")
        return self

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
        print "%13s" % "speeds:", ",".join(self.speeds) if self.speeds else "-"
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