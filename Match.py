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
import os
from matplotlib import pyplot as plt
from mhlib import isnumeric

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
        self.begin = None
        self.end   = None
        self.x     = None
        self.y     = None
        self.s     = None
        self.xm    = None
        self.ym    = None
        self.sm    = None
        self.ismatched = False
        self.label = 'Noname' if label is None else label
        self._ties  = {}
        
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
    
    def setLimits(self, begin = None, end = None):
        '''
        Sets data series usefull limits
        '''
        self.begin = begin
        self.end   = end
        return self
    
    def _window(self, var):
        s = self.begin
        e = self.end

        if s == None and e == None:
            return var
        
        if s == None and e != None:
            return var[ self.x < e ]
        
        if s != None and e == None:
            return var[ self.x > s ]

        return var[ (self.x > s) & (self.x < e) ]
    
    @property
    def ties(self):
        return self._ties.copy()
    
    @property
    def x_window(self):
        return self._window(self.x)
    
    @property
    def y_window(self):
        return self._window(self.y)
    
    def setTie(self, label, value):
        if value < self.x.min() or value > self.x.max():
            raise Exception("Bad Value, outside of series definition range.")

        self._ties[label] = value
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
    
    def normalizeStd(self):
        m = self.y.mean()
        s = self.y.std()
        self.y = (self.y - m) / s
    
        return self
    
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
        plt.axvspan(self.begin, self.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % self.filename)
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(self.xm, self.ym, label=self.filename)
        plt.plot(other.x, other.y, label=other.filename)
        plt.axvspan(other.begin, other.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % other.filename)
        plt.legend()
    
    def plot(self):
        '''
        Plot this data serie and its parameters
        '''
        m1 = self.x.min() if self.begin is None else self.begin
        m2 = self.x.max() if self.end is None else self.end
        plt.axvspan(m1, m2, 0.05, 0.15, alpha=0.75, color="0.6", label='Used Segment')
        plt.plot(self.x, self.y, label='Series %s [%s]' % (self.label if self.label else '-',self.filename))
        
        for k,v in self.ties.iteritems():
            plt.axvline(v, 0.05, 0.2, color ='k')
        
        plt.legend()
    
    def report(self):
        print "Serie: Label: %s Filename: %s Original Filename: %s" %(self.label, self.filename, self.original_filename)
        print "\nStatistics:        %8s / %-8s"  % ("Full", "Window")
        print " Number of Points: %8d / %-8d" % (len(self.x), len(self.x_window))
        print "            x-Min: %8f / %-8f" % (self.x.min(), self.x_window.max())
        print "            y-Min: %8f / %-8f" % (self.y.min(), self.y_window.min())
        print "            y-Max: %8f / %-8f" % (self.y.max(), self.y_window.max())
        print "           y-Mean: %8f / %-8f" % (self.y.mean(), self.y_window.mean())
        print "          y-StdEv: %8f / %-8f" % (self.y.std(), self.y_window.std())
        print "\nAssociated Information:"
        print "   Begin is: %s and End is: %s" % (self.begin, self.end)
        print "   Total of %d tie points." % (len(self.ties))
        i = 0
        for k,x in self.ties.iteritems():
            print "      Tie #%d, Label: %s Position: %s" % (i, k, x)
            i += 1

class MatchConfFile(object):
    ##
    # Location of the MATCH executable code
    ##
    _MATCHCMD = "./match" 
    
    def __init__(self, filename):
        self._series1 = None
        self._begin1  = None
        self._end1    = None
        self._numintervals1 = None
        self._series2 = None
        self._begin2  = None
        self._end2    = None
        self._numintervals2 = None
        self._nomatch      = None
        self._speedpenalty = None
        self._speedchange  = None
        self._tiepenalty   = None
        self._gappenalty   = None
        self._tiefile = None
        self._series1gaps = None
        self._series2gaps = None
        self._matchfile   = None
        self._logfile     = None
        self._filename = filename
        self._issaved = False
        
        self._tiepoints = {}
        
        # These are property
        self._speeds =  []
        self._targetspeed  = None
        
        if os.path.isfile(filename):
            self.__read()
            self._issaved = True
        else:
            self.matchfile = self.filename.replace(".conf","") + ".match"
            self.logfile = self.filename.replace(".conf","") + ".log"
    
    ''' Getters
    '''
    def _addTie(self, label, v1, v2):
        self._tiepoints[label] = (float(v1), float(v2))
    
    def _tie(self, label, which = None):
        t = self._tiepoints[label]
        return t if which == None else t[which - 1]
    
    def _ties(self):
        return self._tiepoints.keys()
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def series1(self):
        return self._series1
    
    @property
    def begin1(self):
        return self._begin1
    
    @property
    def end1(self):
        return self._end1
    
    @property
    def numintervals1(self):
        return self._numintervals1
    
    @property
    def series2(self):
        return self._series2
    
    @property
    def begin2(self):
        return self._begin2
    
    @property
    def end2(self):
        return self._end2
    
    @property
    def numintervals2(self):
        return self._numintervals2
    
    @property
    def nomatch(self):
        return self._nomatch
    
    @property
    def speedpenalty(self):
        return self._speedpenalty
    
    @property
    def targetspeed(self):
        return self._targetspeed
    
    @property
    def speedchange(self):
        return self._speedchange
    
    @property
    def tiepenalty(self):
        return self._tiepenalty
    
    @property
    def gappenalty(self):
        return self._gappenalty
    
    @property
    def speeds(self):
        return self._speeds
    
    @property
    def tiefile(self):
        return self._tiefile
    
    @property
    def series1gaps(self):
        return self._series1gaps
    
    @property
    def series2gaps(self):
        return self._series2gaps
    
    @property
    def matchfile(self):
        return self._matchfile
    
    @property
    def logfile(self):
        return self._logfile

    ''' Setters
    '''
    def __fitvalue(self, which, value):
        value = float(value)
        sa = self.getSeries(which)
        if value < sa.x.min() or value > sa.x.max():
            print "Adjusting end by time series range !"
            value = max(sa.x.min(), min(sa.x.max(), value))
        return value
    
    def __maxsubdiv(self, which, value):
        value = int(value)
        sa = self.getSeries(which)
        if len(sa.x) / value <= 0:
            print "Adjusting maximum sub-division so that there is at least 2 samples per segment"
            value = len(sa.x) // 2
        return value
    
    @filename.setter
    def filename(self, value):
        self._filename = str(value)
        self._issaved = False
    
    @series1.setter
    def series1(self, value):
        if value == None:
                self._series1 = None
                return
        if not os.path.isfile(value):
            raise Exception("File '%s' not found." % value)
        self._series1 = str(value)
        self._issaved = False

    @begin1.setter
    def begin1(self, value):
        self._begin1 = self.__fitvalue(1, value) if value != None else None
        self._issaved = False

    @end1.setter
    def end1(self, value):
        self._end1 = self.__fitvalue(1, value) if value != None else None
        self._issaved = False
    
    @numintervals1.setter
    def numintervals1(self, value):
        self._numintervals1 = self.__maxsubdiv(1, value) if value != None else None
        self._issaved = False
    
    @series2.setter
    def series2(self, value):
        if value == None:
                self._series2 = None
                return
        if not os.path.isfile(value):
            raise Exception("File '%s' not found." % value)
        self._series2 = str(value)
        self._issaved = False
    
    @begin2.setter
    def begin2(self, value):
        self._begin2 = self.__fitvalue(2, value) if value != None else None
        self._issaved = False
    
    @end2.setter
    def end2(self, value):
        self._end2 = self.__fitvalue(2, value) if value != None else None
        self._issaved = False
    
    @numintervals2.setter
    def numintervals2(self, value):
        self._numintervals2 = self.__maxsubdiv(2, value) if value != None else None
        self._issaved = False
    
    @nomatch.setter
    def nomatch(self, value):
        value = float(value) if value != None else None
        if value is not None and value == 0.0:
            raise Exception("Value cannot be 0.0")
        self._nomatch = value
        self._issaved = False
    
    @speedpenalty.setter
    def speedpenalty(self, value):
        self._speedpenalty = float(value) if value != None else None
        self._issaved = False
    
    @targetspeed.setter
    def targetspeed(self, value):
        if value is None:
            self._targetspeed = value
            return
        
        value = str(value)
        
        if value.find(":") == -1:
            raise Exception("Invalid value of target speed '%s'" % value)
        
        if value not in self._speeds:
            raise Exception("Target speed is not in speeds !!")
        
        self._targetspeed = value
        self._issaved = False
    
    @speedchange.setter
    def speedchange(self, value):
        self._speedchange = float(value) if value != None else None
        self._issaved = False
    
    @tiepenalty.setter
    def tiepenalty(self, value):
        self._tiepenalty = float(value) if value != None else None
        self._issaved = False
    
    @gappenalty.setter
    def gappenalty(self, value):
        self._gappenalty = float(value) if value != None else None
        self._issaved = False
    
    @speeds.setter
    def speeds(self, value):
        if value is None:
            self._speeds = []
            return

        if isinstance(value, str):
            comas = value.count(",")
            ddots = value.count(":")
            if comas != ddots - 1:
                raise Exception("Invalid speeds values given, need list or string with ','")
            value = value.split(",")
        
        if not isinstance(value, list):
            raise Exception("Invalid speeds values given, need list or string with ','")
        
        for v in value:
            if v.find(":") == -1:
                raise Exception("Invalid value '%s' in speeds!" % v)
        
        self._speeds = value
        self._issaved = False
    
    @tiefile.setter
    def tiefile(self, value):
        self._tiefile = str(value) if value != None else None
        self._issaved = False
    
    @series1gaps.setter
    def series1gaps(self, value):
        self._series1gaps = str(value) if value != None else None
        self._issaved = False
    
    @series2gaps.setter
    def series2gaps(self, value):
        self._series2gaps = str(value) if value != None else None
        self._issaved = False
    
    @matchfile.setter
    def matchfile(self, value):
        self._matchfile = str(value) if value != None else None
        self._issaved = False
    
    @logfile.setter
    def logfile(self, value):
        self._logfile = str(value) if value != None else None
        self._issaved = False
    
    ''' io method
    '''
    
    @staticmethod
    def __pkv(line):
        items = line.strip().split()
        
        try:
            k = items[0]
        except IndexError:
            k = None
        
        try:
            v = items[1]
        except IndexError:
            v = None
        
        return k, v
    
    def _loadtie(self):
        if self.tiefile == "" or self.tiefile == None:
            return
        
        if not os.path.isfile(self.tiefile):
            print "Tie file not found."
            return
        
        label = ['A']
        with open(self.tiefile) as fio:
            for line in fio:
                v1, v2 = line.strip().split()
                v1 = float(v1)
                v2 = float(v2)
                self._addTie("".join(label), v1, v2)
                if label[-1] == "Z":
                    label[-1] = 'A'
                    label.append("A")
                else:
                    label = chr(ord(label[-1]) + 1)

    def __read(self):
        validkeys = [
            "series1", "begin1", "end1", "numintervals1",
            "series2", "begin2", "end2", "numintervals2",
            "nomatch", "speedpenalty", "targetspeed",
            "speedchange", "tiepenalty", "gappenalty",
            "speeds", "tiefile", "series1gaps", "series2gaps",
            "matchfile", "logfile" ]
        
        targetspeed = None
        
        with open(self.filename) as fio:
            for line in fio:
                k, v = self.__pkv(line)
                
                if k not in validkeys:
                    if k != None and k != "":
                        print "Invalid key in config file ignored: '%s'  !" % k
                    continue
                
                if k != "targetspeed":
                    setattr(self, k, v)
                else:
                    targetspeed = v

        try:
            self.targetspeed = targetspeed
        except Exception,e:
            print "Error -- ",e.message
        
        self._loadtie()
    
    def __write(self, fio, variable):
        v = getattr(self, variable)
        if v == None: return
        if variable == "speeds": v = ",".join(v)
        print >>fio,"%-13s" % variable, "%s" % v
    
    def write(self, filename  = None, regenerate_matchlogfiles = False):
        if filename != None:
            self.filename = filename

        ## Check consistency of data before write !
        if self.series1 == None or not os.path.isfile(self.series1):
            raise Exception("Series1: '%s', is not defined or file does not exists." % self.series1)
        
        if self.series2 == None or not os.path.isfile(self.series2):
            raise Exception("Series2: '%s', is not defined or file does not exists." % self.series2)
        
        s1 = Serie(self.series1)
        if self.begin1 == None:
            self.begin1 = s1.x.min()
            print "Setting series1 begin to %s" % self.begin1

        if self.end1 == None:
            self.end1 = s1.x.max()
            print "Setting series1 end to %s" % self.end1

        if self.numintervals1 == None:
            self.numintervals1 = len(s1.x) // 7
            print "Setting series1 numintervals to %s" % self.numintervals1

        s2 = Serie(self.series2)
        if self.begin2 == None:
            self.begin2 = s2.x.min()
            print "Setting series2 begin to %s" % self.begin2

        if self.end2 == None:
            self.end2 = s2.x.max()
            print "Setting series2 end to %s" % self.end2

        if self.numintervals2 == None:
            self.numintervals2 = len(s2.x) // 7
            print "Setting series2 numintervals to %s" % self.numintervals2

        if regenerate_matchlogfiles or self.matchfile == None or self.matchfile == "":
            self.matchfile = self.filename.replace(".conf","") + ".match"

        if regenerate_matchlogfiles or self.logfile == None or self.logfile == "":
            self.logfile = self.filename.replace(".conf","") + ".log"

        fio = open(self.filename, "w")
        
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

        if self.speeds != None and len(self.speeds) > 0: self.__write(fio, "speeds")

        fio.write("\n")
        
        self.__write(fio, "tiefile")
        self.__write(fio, "series1gaps")
        self.__write(fio, "series2gaps")

        self.__write(fio, "matchfile")
        self.__write(fio, "logfile")

        fio.close()
        self._issaved = True
    
    ''' Utils
    '''
    def computeSpeeds(self, first, last, exchanged = True):
        if not isinstance(first, list):
            first = range(1,first+1)
        if not isinstance(last, list):
            last = range(1,last+1)
        if len(first) == 0 or len(last) == 0:
            raise Exception("Not enough values for pair")
        
        items = []
        for f in first:
            for l in last:
                item = "%d:%d" % (f,l)
                if item not in items: items.append(item)
                if exchanged:
                    item = "%d:%d" % (l,f)
                    if item not in items: items.append(item)
        self.speeds = items

    def setSeries(self, which, filename, begin, end, nintervals):
        if not os.path.isfile(filename):
            raise Exception("Not a file ! ")
        
        if which == 1:
            self.series1 = filename
            serie = Serie(filename)
            self.begin1 = float(begin) if begin != None else serie.x.min() 
            self.end1   = float(end) if end != None else serie.x.max()
            self.numintervals1 = int(nintervals) if nintervals != None else len(serie.x) // 7
        elif which == 2:
            self.series2 = filename
            serie = Serie(filename)
            self.begin2 = float(begin) if begin != None else serie.x.min()
            self.end2   = float(end) if end != None else serie.x.max()
            self.numintervals2 = int(nintervals) if nintervals != None else len(serie.x) // 7
        else:
            raise Exception("time series not know !")
        return self
    
    def guessParameters(self):
        '''
        This method try to guess the values for each penalties, algorithm 
        is taken directly from Matlab code from Lorraine's !! 
            -- Need still to check numbers !
        Don't blame me on that !
        '''
        sa = self.getSeries(1)
        sb = self.getSeries(2)

        da = sa.y_window
        db = sb.y_window

        mean1 = np.mean(da)
        mean2 = np.mean(db)
        std1  = np.std(da)
        std2  = np.std(db)
        m = 2*np.abs(mean1-mean2)
        s = max(std1, std2)**2
        d = self.end2 - self.begin2
        
        self.nomatch      = round(150.0*(s+m*m))/10.0
        self.speedpenalty = round(350.0*(0.2*s+m*m))/100.0
        self.speedchange  = round(350.0*(0.15*s+m*m))/100.0
        self.tiepenalty   = round(5000.0*s/d+m*m)
        self.gappenalty   = round(1000.0*s/d+0.8*m*m)
    
    def clean(self):
        if os.path.isfile(self.logfile):   os.unlink(self.logfile)
        if os.path.isfile(self.matchfile): os.unlink(self.matchfile)

        f = "%s.new" % self.series1
        if os.path.isfile(f): os.unlink(f)
        
        f = "%s.new" % self.series2
        if os.path.isfile(f): os.unlink(f)
    
    def getSeries(self, which):
        if which == 1:
            if self.series1 == None: raise Exception("Serie is Unset.")
            s = Serie(self.series1, '#1')
            s.setLimits(self.begin1, self.end1)
            map(lambda k: s.setTie(k, self._tie(k, 1)), self._ties())
            return s
        elif which == 2:
            if self.series2 == None: raise Exception("Serie is Unset.")
            s = Serie(self.series2, '#2')
            s.setLimits(self.begin2, self.end2)
            map(lambda k: s.setTie(k, self._tie(k, 2)), self._ties())
            return s
        else:
            raise Exception("No such serie")
    
    def run(self, autosave = False):
        if self._issaved == False and autosave == False:
            raise Exception("Please save Conf file first !")
        
        if not os.path.isfile(self.filename):
            raise Exception("Filename '%s' Not Found." % self.filename)
        
        if not os.path.isfile(MatchConfFile._MATCHCMD):
            raise Exception("Program 'match' was not found in this system !")
        
        if autosave:
            print "Auto-saving file '%s'" % self.filename
            self.write()
        
        results = os.popen('%s -v %s 2>&1' % (MatchConfFile._MATCHCMD, self.filename))
        
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
    
    def optimize(self, parameter, values, update = True, plot = True):
        if not hasattr(self, parameter):
            raise Exception("%s not in self" % parameter)
        
        if parameter not in ["nomatch", "speedpenalty",
                             "targetspeed", "speedchange",
                             "tiepenalty", "gappenalty",
                             "begin1", "end1",
                             "begin2", "end2",
                             "numintervals1", "numintervals2"
                             ]:
            raise Exception("%s is not optimizable" % parameter)
        
        rms = []
        x   = []
        s   = []
        
        print "Working on optimizing %s from %s - %s" % (parameter, values[0], values[-1]) 
        
        bvalue = None
        mrms   = None
        original_filename = self.filename
        for iv, v in zip(range(len(values)), values):
            if plot: print ".",
            setattr(self,parameter,v)

            self.write("lala.conf")
            results = self.run()
            
            sa = self.getSeries(1)
            sb = self.getSeries(2)
            x1 = sa.x
            y1 = sa.y
            x2 = sb.xm
            y2 = sb.ym
            xnew = np.linspace(min(x1), max(x2), 2000)
            y1n = np.interp(xnew, x1, y1)
            y2n = np.interp(xnew, x2, y2)
            r = np.sqrt( np.mean( (y1n-y2n)**2 ))
            self.clean()
            os.unlink("lala.conf")

            if mrms == None or r < mrms:
                bvalue = v
                mrms = r

            try:
                v = float(v)
            except:
                v = iv

            x.append(v)
            s.append(results['Total'])
            rms.append(r)

        self.filename = original_filename
        if update:
            print ""
            print "Final parameter %s = %s"  % (parameter, bvalue)
            setattr(self, parameter, bvalue)
            self.write()
            self.run()
        
        if plot:
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
        print ""
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
        
        print ""
        print " ** Ties **"
        print " %d ties are defined" % len(self._tiepoints)

        for k in sorted(self._ties()):
            v1, v2 = self._tie(k)
            print "  Label '%-s' %5s = %-5s" % (k,v1,v2)

def create_tie(label, series_list, ages_list):
    label = str(label)
    
    series_list = [ series_list ] if not isinstance(series_list, list) else series_list
    ages_list = [ ages_list ] if not isinstance(ages_list, list) else ages_list
    
    print ages_list
    
    for s in series_list:
        if not isinstance(s, Serie):
            raise Exception("Object in series_list is not of type Serie")

    for a in ages_list:
        if not isnumeric(str(a)):
            raise Exception("Object in agest_list is not a Number")

    for s,a in zip(series_list, ages_list):
        s.setTie(label, a)
    
    return

def show_tie(label,  series_list):
    series_list = [series_list] if not isinstance(series_list, list) else series_list

    for s in series_list:
        if not isinstance(s, Serie):
            raise Exception("Object in series_list is not of type Serie")

    print "%15s  " % " ",
    for s in series_list:
        print "%-10s" % s.label,
    print ""

    print "%15s: " % label,
    for s in series_list:
        try:
            print "%-10s" % s.ties[label],
        except:
            print "%-10s" % "-",
    print ""
    return
