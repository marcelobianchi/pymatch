#! /usr/bin/python

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

class Tie(object):
    def __init__(self, filename):
        if filename == "":
            raise Exception("Need a valid filename or None")
        
        self.__label = None
        self._filename = filename
        self.__tiepoints =  { }
        
        if self.filename != None and os.path.isfile(self.filename):
            with open(self.filename) as fio:
                for line in fio:
                    v1, v2 = line.strip().split()
                    v1 = float(v1)
                    v2 = float(v2)
                    self.set_tie(v1, v2)
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def tie_labels(self):
        return self.__tiepoints.keys()
    
    @property
    def nextlabel(self):
        if self.__label == None:
            self.__label = "A"
        else:
            indices = map(lambda x: ord(x), self.__label)
            updated = False
            for i in range(1,len(indices)+1):
                if indices[-i] == 90:
                    indices[-i] = 65
                    continue
                indices[-i] += 1
                updated = True
                break
            if not updated: indices.insert(0, 65)
            self.__label = "".join(map(lambda x: chr(x), indices))
        return self.__label
    
    ''' Utils
    '''
    def set_tie(self, value_one, value_two, segment_one = 0, segment_two = 0, label = None):
        label = label if label != None else self.nextlabel
        self.__tiepoints[label] = (float(value_one), float(value_two),
                                   int(segment_one), int(segment_two))
    
    def tie(self, label):
        return self.__tiepoints[label]
    
    def saveas(self, filename):
        self._filename = filename
        return self.save()
    
    def save(self):
        fio = open(self.filename, "w")
        for label in self.tie_labels:
            v1, v2, _, _ = self.tie(label)
            print >>fio, v1, v2
        fio.close()
        
        return True    
    
    def report(self):
        n = len(self.tie_labels)
        print 'Total of %d labels are defined.' % n
        for label in sorted(self.tie_labels):
            v1, v2, s1, s2 = self.tie(label)
            print "  %s => %d,%f = %d,%f" % (label, s1,v1,s2,v2)

class Serie(object):
    '''
    Represents one 1D series, can be 2 or 3 columns,
    exports self.x, self.y and self.s. Is capable of:

        Normalize :: 
            -0.5 -> 0.5 or to scale any other Series given
    '''
    def __init__(self, filename, label = None):
        self.__filename = filename
        self.begin = None
        self.end   = None
        self._numintervals = None
        self.x     = None
        self.y     = None
        self.s     = None
        self.xm    = None
        self.ym    = None
        self.sm    = None
        self.ismatched = False
        self.label = 'Noname' if label is None else label
        self._ties  = { }
        
        self.x, self.y, self.s = Serie.read(self.filename)
        
        try:
            filename = self.filename + ".new"
            self.xm, self.ym, self.sm = Serie.read(filename)
            self.ismatched = True
        except:
            pass
    
    @staticmethod
    def read(filename):
        if not os.path.isfile(filename):
            raise Exception("File does not exists !")
        
        data = np.loadtxt(filename, unpack=True)
        if data.shape[0] == 2:
            x, y = data
            s = None
        else:
            s,x,y = data
        return x,y,s
    
    def setLimits(self, begin = None, end = None, cut = False):
        '''
        Sets data series usefull limits
        '''
        if begin and end and begin >= end: 
            raise Exception("Begin should be smaller than end !")
        
        self.begin = begin
        self.end   = end
        
        if cut:
            self.y = self.__window(self.y)
            self.x = self.__window(self.x)
            t = {}
            for t in self._ties.keys():
                val = self._ties[t]
                if begin and val < begin: del self._ties[t]
                if end and val > end: del self._ties[t]
        
        return self
    
    def __window(self, var):
        s = self.begin
        e = self.end

        if s == None and e == None:
            return var
        
        if s == None and e != None:
            return var[ self.x < e ]
        
        if s != None and e == None:
            return var[ self.x >= s ]

        return var[ (self.x >= s) & (self.x <= e) ]
    
    @property
    def numintervals(self):
        return self._numintervals
    
    @numintervals.setter
    def numintervals(self, value):
        if value is None:
            self._numintervals = 1
            return
        
        if (len(self.x) / value) < 2:
            raise Exception("Value of nintervals too high !")
        
        self._numintervals = value
    
    @property
    def filename(self):
        return self.__filename
    
    @property
    def ties(self):
        return self._ties.copy()
    
    def tie(self, value):
        return self._ties[value]
    
    @property
    def x_window(self):
        return self.__window(self.x)
    
    @property
    def y_window(self):
        return self.__window(self.y)
    
    def setTie(self, label, value):
        if value < self.x.min() or value > self.x.max():
            raise Exception("Bad Tie Value = %f, outside of series x-definition range %f to %f." % (value,self.x.min(), self.x.max()))

        self._ties[label] = value
        return self
    
    def save(self):
        '''
        Save this data serie
        '''
        
        if self.s is None:
            data = np.array(zip(self.x, self.y))
        else:
            data = np.array(zip(self.s, self.x, self.y))
        
        np.savetxt(self.filename, data)
        
        return True
    
    def saveas(self, filename):
        if os.path.isfile(filename):
            print "Will overwrite filename '%s'." % filename
        self.__filename = filename
        return self.save()
    
    def normalizeStd(self, respect = True):
        if respect:
            m = self.y_window.mean()
            s = self.y_window.std()
        else:
            m = self.y.mean()
            s = self.y.std()
        self.y = (self.y - m) / s
    
        return self
    
    def normalize(self, respect = True, to = None):
        '''
        Normalize this serie between -0.5 and 0.5, or 
        if to is an Serie instance match this one to the other.
        '''
        if to is not None and not isinstance(to, Serie):
            raise Exception("Bad destination Serie instance on to variable !")
        
        selfy  = self.y if respect == False else self.y_window
        
        s = 1.0
        l = -0.5
        
        if to is not None:
            othery = to.y if respect == False else to.y_window
            s = np.max(othery) - np.min(othery)
            l = np.min(othery)
        
        offset = selfy.min()
        norm   = np.max(selfy - offset)
        
        self.y -= np.min(selfy)
        self.y *= (s / norm)
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
        for t in self.ties:
            plt.axvline(self.tie(t), 0.1, 0.15, color='k')
        plt.plot(other.xm, other.ym, label=other.filename)
        plt.axvspan(self.begin, self.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % self.filename)
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(self.xm, self.ym, label=self.filename)
        for t in other.ties:
            plt.axvline(other.tie(t),0.1, 0.15, color='k')
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
        
        for v in self.ties.values():
            plt.axvline(v, 0.05, 0.2, color ='k')
        
        plt.legend()
    
    def report(self, simple = False):
        if simple:
            print "%13s" % "Series:", self.filename
            print "%13s" % "","begin:", self.begin, "end:", self.end, "num. Intervals:", self.numintervals," len:",len(self.x)
        else:
            print "Serie: Label: %s Filename: %s" %(self.label, self.filename)
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
            print ""

class MatchLog(object):
    def __init__(self, matchfile, logfile, sa = None, sb = None):
        self.logfile   = logfile
        self.matchfile = matchfile
        
        self.x1 = None
        self.x2 = None
        
        self.sa = sa
        self.sb = sb
        
        self.scores = {
            "Total" : -1,
            "point" : -1,
            "nomatch" : -1,
            "speed" : -1,
            "speedchange" : -1,
            "tie" : -1,
            "gap" : -1,
            "tie nomatch": -1
        }
        
        self.params = {
            "begin1" : None, "end1" : None, "numintervals1" : None,
            "begin2" : None, "end2" : None, "numintervals2" : None,
            "nomatch" : None, "speedpenalty" : None, "targetspeed" : None,
            "speedchange" : None, "tiepenalty" : None, "gappenalty" : None
        }

        self._parse_log()
        self._parse_match()
        
        self._rms = None
        self._cor = None
    
    def _parse_match(self):
        if not os.path.isfile(self.matchfile):
            raise Exception("No Match file found.")
        _, x1, _, x2 = np.loadtxt(self.matchfile, unpack = True)
        self.x1 = x1
        self.x2 = x2
    
    def _parse_log(self):
        if not os.path.isfile(self.logfile):
            raise Exception("No Match file found.")
        
        with open(self.logfile) as results:
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
                    if k not in self.scores.keys(): continue
                except:
                    continue
                self.scores[k] = float(v)
    
    @property
    def correlation(self):
        if self.sa == None or self.sb == None:
            return False
        
        if self._cor == None:
            sa = self.sa
            sb = self.sb
            
            xnew = np.linspace(min(sa.x), max(sb.xm), 2000)
            y1n = np.interp(xnew, sa.x, sa.y)
            y2n = np.interp(xnew, sb.xm, sb.ym)
            self._cor = np.correlate(y1n, y2n)[0]
            
        return self._cor
    
    @property
    def good(self):
        return self.scores["Total"] != -1
    
    @property
    def rms(self):
        if self.sa == None or self.sb == None:
            return False
        
        if self._rms == None:
            sa = self.sa
            sb = self.sb
            
            xnew = np.linspace(min(sa.x), max(sb.xm), 2000)
            y1n = np.interp(xnew, sa.x, sa.y)
            y2n = np.interp(xnew, sb.xm, sb.ym)
            self._rms = np.sqrt( np.mean( (y1n-y2n)**2 ))
        
        return self._rms
    
    def plot(self):
        labels = ["Total", "point", "nomatch", "speed", "speedchange", "tie", "gap", "tie nomatch"]
        
        # Bar Plot
        values = map(lambda x: self.scores[x], labels)
        positions = range(len(labels))
        plt.figure(figsize=(15,3))
        plt.bar(positions, values, align='center')
        plt.xticks(positions, labels)
        plt.ylabel("Score")
        
        # Sedimentation Ratio
        _, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(self.x1, self.x2)
        
        if self.sb != None:
            ax2 = ax1.twinx()
            ax2.plot(self.sb.xm, self.sb.ym)
            ax2.set_ylim(min(self.sb.ym),max(self.sb.ym)*15)

class MatchConfFile(object):
    ##
    # Location of the MATCH executable code
    ##
    _MATCHCMD = "./match" 
    
    def __init__(self, filename, autonormalize = False):
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
        self._filename = None
        self._issaved = False
        
        self._autonormalize = autonormalize
        
        # These are property
        self._speeds =  []
        self._targetspeed  = None
        
        if os.path.isfile(filename):
            self.__read(filename)
            self._issaved = True
        else:
            self.__set_filename(filename)
    
    def __params(self):
        return {
            "begin1" : self.begin1, "end1" : self.end1, "numintervals1" : self.numintervals1,
            "begin2" : self.begin2, "end2" : self.end2, "numintervals2" : self.numintervals2,
            "nomatch" : self.nomatch, "speedpenalty" : self.speedpenalty, "targetspeed" : self.targetspeed,
            "speedchange" : self.speedchange, "tiepenalty" : self.tiepenalty, "gappenalty" : self.gappenalty
        }
    
    ''' Getters
    '''
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
            value = max(sa.x.min(), min(sa.x.max(), value))
        return value
    
    def __maxsubdiv(self, which, value):
        value = int(value)
        sa = self.getSeries(which)
        if len(sa.x) / value <= 0:
            print "Adjusting maximum sub-division so that there is at least 2 samples per segment"
            value = len(sa.x) // 2
        return value
    
    def __set_filename(self, value):
        self._filename = str(value)
        
        base = self.filename
        if ".conf" in self.filename:
            base = base[:self.filename.find(".conf")]
        
        self._matchfile = base + ".match"
        self._logfile = base + ".log"
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
        if value == None:
            self._tiefile = None
        else:
            if not os.path.isfile(value):
                raise Exception("File '%s' not found." % value)
            
            ##
            # Test tie
            ##
            tt = Tie(value)
            sa = self.getSeries(1)
            map(lambda label: sa.setTie(label, tt.tie(label)[0]), tt.tie_labels)
            
            sb = self.getSeries(2)
            map(lambda label: sb.setTie(label, tt.tie(label)[1]), tt.tie_labels)
            
            ##
            # Ok
            ##
            self._tiefile = value
        
        self._issaved = False
    
    @series1gaps.setter
    def series1gaps(self, value):
        self._series1gaps = str(value) if value != None else None
        self._issaved = False
    
    @series2gaps.setter
    def series2gaps(self, value):
        self._series2gaps = str(value) if value != None else None
        self._issaved = False
    
    ''' Io method
    '''
    def __read(self, filename):
        def pkv(line):
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
        validkeys = [
            "series1", "begin1", "end1", "numintervals1",
            "series2", "begin2", "end2", "numintervals2",
            "nomatch", "speedpenalty", "targetspeed",
            "speedchange", "tiepenalty", "gappenalty",
            "speeds", "tiefile", "series1gaps", "series2gaps",
            "matchfile", "logfile" ]
        
        targetspeed = None
        
        self._filename = filename
        
        with open(self.filename) as fio:
            for line in fio:
                k, v = pkv(line)
                
                if k not in validkeys:
                    if k != None and k != "":
                        print "Invalid key in config file ignored: '%s'  !" % k
                    continue
                
                if k == "targetspeed":
                    targetspeed = v
                elif k == "matchfile":
                    self._matchfile = v
                elif k == "logfile":
                    self._logfile = v
                else:
                    setattr(self, k, v)
        
        if self.matchfile in ["", None] or self.logfile in ["", None]:
            print "No match & log file indicated, generating default values."
            self.__set_filename(filename)
        
        try:
            self.targetspeed = targetspeed
        except Exception,e:
            print "Error -- ",e.message
    
    def __write(self, fio, variable):
        v = getattr(self, variable)
        if v == None: return
        if variable == "speeds": v = ",".join(v)
        print >>fio,"%-13s" % variable, "%s" % v
    
    def saveas(self, filename):
        '''
        Save the conf file as filename
        '''
        self.__set_filename(filename)
        return self.save()
    
    def save(self):
        '''
        Save the conf file
        '''
        # Check consistency of data before write !
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
        
        return self._issaved
    
    ''' Utils
    '''
    def generateSpeeds(self, first, last, exchanged = True):
        if not isinstance(first, list):
            first = range(1,first+1)
        
        if not isinstance(last, list):
            last = range(1,last+1)
        
        if len(first) == 0 or len(last) == 0:
            raise Exception("Not enough values for pair")
        
        items = []
        
        first.reverse()
        last.reverse()
        
        if exchanged:
            for f in first:
                for l in last:
                    item = "%d:%d" % (l,f)
                    if item not in items: items.append(item)
        
        first.reverse()
        last.reverse()
        
        for f in first:
            for l in last:
                item = "%d:%d" % (f,l)
                if item not in items: items.append(item)
        
        self.speeds = items
    
    def setSeries(self, which, filename, begin = None, end = None, nintervals = None):
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
        '''
        sa = self.getSeries(1)
        sb = self.getSeries(2)
        
        if self._autonormalize:
            print "Normalizing series (1) from %.1f to %.1f and series (2) from %.1f to %.1f" % (self.begin1, self.end1, self.begin2, self.end2)
            sb.normalizeStd(True).save()
            sb.normalizeStd(True).save()
        
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
        
        return self
    
    def clean(self):
        if os.path.isfile(self.logfile):   os.unlink(self.logfile)
        if os.path.isfile(self.matchfile): os.unlink(self.matchfile)

        f = "%s.new" % self.series1
        if os.path.isfile(f): os.unlink(f)
        
        f = "%s.new" % self.series2
        if os.path.isfile(f): os.unlink(f)
    
    def getSeries(self, which):
        t = Tie(self.tiefile) if self.tiefile != None else None
        if which == 1:
            if self.series1 == None: raise Exception("Serie is Unset.")
            s = Serie(self.series1, '#1')
            s.setLimits(self.begin1, self.end1, cut = False)
            s.numintervals = self.numintervals1
            if t:
                map(lambda label: s.setTie(label, t.tie(label)[0]), t.tie_labels)
            return s
        elif which == 2:
            if self.series2 == None: raise Exception("Serie is Unset.")
            s = Serie(self.series2, '#2')
            s.setLimits(self.begin2, self.end2, cut = False)
            s.numintervals = self.numintervals2
            if t:
                map(lambda label: s.setTie(label, t.tie(label)[1]), t.tie_labels)
            return s
        else:
            raise Exception("No such serie")
    
    def run(self, autosave = False, plotresults = True):
        if self._issaved == False and autosave == False:
            raise Exception("Please save CONF file first !")
        
        if autosave:
            print "Auto-saving file '%s'" % self.filename
            self.save()
        
        if not os.path.isfile(self.filename):
            raise Exception("Filename '%s' Not Found." % self.filename)
        
        if not os.path.isfile(MatchConfFile._MATCHCMD):
            raise Exception("Program 'match' was not found in this system !")
        
        if self._autonormalize:
            print "Normalizing series (1) from %.1f to %.1f and series (2) from %.1f to %.1f" % (self.begin1, self.end1, self.begin2, self.end2)
            self.getSeries(1).normalizeStd(True).save()
            self.getSeries(2).normalizeStd(True).save()
        
        os.popen('%s -v %s 2>&1' % (MatchConfFile._MATCHCMD, self.filename))
        
        ml = MatchLog(self.matchfile,
                      self.logfile,
                      self.getSeries(1),
                      self.getSeries(2)
                      )
        
        ml.params.update(self.__params())
        
        if plotresults: ml.plot()
        
        return ml 
    
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
        
        x   = []
        s   = []
        
        print "Working on optimizing %s from %s - %s" % (parameter, values[0], values[-1]) 
        best_value = None
        max_cor    = None
        original_filename = self.filename
        pcount     = 0
        rounds = 0
        for iv, v in zip(range(len(values)), values):
            if plot:
                if pcount in  [5,10,15, 20, 25, 30, 35, 40]: print " ",
                if pcount == 45:
                    rounds += 1
                    print "%03s %%" % ((100*rounds*45)/len(values))
                    pcount = 0
                print ".",
                if iv == len(values)-1: print "100 %"
            pcount += 1
            
            setattr(self, parameter, v)
            
            self.saveas("lala.conf")
            results = self.run(plotresults = False)
            self.clean()
            os.unlink("lala.conf")
            
            if max_cor == None or results.correlation > max_cor:
                best_value = v
                max_cor = results.correlation
            
            x.append(float(v) if not isinstance(v, str) else iv)
            s.append(results)
        
        self.__set_filename(original_filename)
        
        print "Final parameter %s = %s"  % (parameter, best_value)
        
        if update:
            setattr(self, parameter, best_value)
            self.run(autosave = True, plotresults=plot)
        
        if plot:
            _, ax1 = plt.subplots(figsize=(10,7))
            values = map(lambda item: item.correlation, s)
            ax1.plot(x, values, "r-o", label='Correlation')
            formats, labels = ax1.get_legend_handles_labels()
            
            ax2 = ax1.twinx()
            for k in ["Total", "point", "nomatch", "speed", "speedchange", "tie", "gap", "tie nomatch"]:
                values = map(lambda item: item.scores[k], s)
                ax2.plot(x, values, label = k)
            
            f, l = ax2.get_legend_handles_labels()
            
            formats.extend(f)
            labels.extend(l)
            
            ax1.set_ylabel('Correlation')
            ax2.set_ylabel('Total Score')
            ax1.set_xlabel("%s" % (parameter))
            plt.legend(formats,
                       labels,
                       loc='upper left',
                       ncol=1,
                       shadow=True,
                       bbox_to_anchor=(1.1, 1.0)
                       )
        
        return s
    
    def report(self):
        sa = self.getSeries(1)
        sb = self.getSeries(2)
        print ""
        print " ** Series **"
        sa.report(simple = True)
        print ""
        sb.report(simple = True)
        print ""
        
        print " ** Penalties **"
        print "%13s" % "nomatch:", self.nomatch
        print "%13s" % "speedpenalty:", self.speedpenalty
        print "%13s" % "targetspeed:", self.targetspeed
        print "%13s" % "speedchange:", self.speedchange
        print "%13s" % "tiepenalty:", self.tiepenalty
        print "%13s" % "gappenalty:", self.gappenalty
        print ""
        
        print "%13s" % "speeds:", ",".join(self.speeds) if self.speeds else "-"
        print ""
        
        print " ** Constrains & Files **"
        print "%13s" % "series1gaps:", self.series2gaps
        print "%13s" % "series2gaps:", self.series1gaps
        print "%13s" % "tiefile:", self.tiefile
        print "%13s" % "matchfile:", self.matchfile
        print "%13s" % "logfile:", self.logfile
        
        if self.tiefile != None and self.tiefile != "":
            tt = Tie(self.tiefile)
            print ""
            print " ** Tie is Defined **"
            tt.report()
            print ""

class Optimizer(object):
    def __init__(self, mcf):
        self._mcf = mcf
        self.mls = None
        self.tests = {}
        
        self.xs = None
        self.ys = None
    
    def targetspeed(self, values):
        self.tests["targetspeed"] = values
    
    def nomatch(self, values):
        self.tests["nomatch"] = values
    
    def speedchangepenalty(self, values):
        self.tests["speedchangepenalty"] = values
    
    def speedpenalty(self, values):
        self.tests["speedpenalty"] = values
    
    def gappenalty(self, values):
        self.tests["gappenalty"] = values
    
    def tiepenalty(self, values):
        self.tests["tiepenalty"] = values
    
    def nintervals(self, vals1_int1, vals_int2):
        self.tests["nintervals"] = (vals1_int1, vals_int2)
    
    def __run_nintervals(self, params):
        b1, e1, v1, b2, e2, v2 = params

        stop = False
        for i in range(4):
            if stop: break
        
            before = self._mcf.numintervals1
            ll = self._mcf.optimize("numintervals1", range(b1, e1, v1), True, False)
            self.mls.extend(ll)
            after = self._mcf.numintervals1
        
            stop = (before == after)
            print "Numintervals1",i,before, after, stop
            print ""
        
            before = self._mcf.numintervals2
            ll = self._mcf.optimize("numintervals2", range(b2, e2, v2), True, False)
            self.mls.extend(ll)
            after = self._mcf.numintervals2
        
            stop = stop and (before == after)
            print "Numintervals2",i,before, after, stop
            print ""
    
    def __run(self, param, values):
        self.mls.extend(self.mcf.optimize(param, values, True, False))
    
    def run(self, plot = True):
        # Zero stats
        self.mls = list()
        
        ## Run tests
        for k in ["nintervals", "targetspeed", "nomatch", "speedchangepenalty", "speedpenalty", "gappenalty", "tiepenalty"]:
            if k not in self.tests:
                print "Skipping optimize: %s" % k
                continue
            
            if k == "nintervals":
                self.__run_nintervals(self.tests[k])
                continue
            
            self.__run(k, self.tests[k])

        ## Estimate
        xmin = min(self.mls[0].x1)
        xmax = max(self.mls[0].x1)
        for x in self.mls:
            xmin = min(xmin, min(x.x1))
            xmax = max(xmax, max(x.x1))

        self.xs = np.linspace(xmin, xmax, 100.)
        self.ys  = []
        print "Average",len(self.mls),"objects"
        for x in self.mls:
            ynew = np.interp(self.xs, x.x1, x.x2)
            self.ys.append(ynew)
        self.ys = np.array(self.ys)
        
        ## Plot
        if plot:
            plt.figure(figsize=(20,10))
            _  = map(lambda x: plt.plot(x.x1, x.x2), self.mls)
            _ = plt.plot(self.xs, np.median(self.ys,axis=0), linewidth=6, color='w')
            _ = plt.errorbar(self.xs, np.median(self.ys,axis=0), yerr = self.ys.std(axis=0), capsize=3, color='k')
            ## _  = plt.legend(loc='upper left', ncol=2, shadow=False, bbox_to_anchor=(1.01, 1.0))
            
        return

    def estimate(self, cm = None, t = None):
        xcm = self.xs
        xt  = np.median(self.ys, axis=0)
        et  = self.ys.std(axis=0)
        
        if cm is None and t is not None:
            cm = np.interp(t, xt, xcm)
            st = np.interp(t, xt, et)
        elif cm is not None and t is None:
            t = np.interp(cm, xcm, xt)
            st = np.interp(cm, xcm, et)
        else:
            raise Exception("need cm or t")
        
        return t, cm, st
