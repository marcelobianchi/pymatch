#! /usr/bin/python
#
################################################################################
#     This file is part of PyMatchInterface.                                   #
#                                                                              #
#     PyMatchInterface is free software: you can redistribute it and/or modify #
#     it under the terms of the GNU General Public License as published by     #
#     the Free Software Foundation, either version 3 of the License, or        #
#     (at your option) any later version.                                      #
#                                                                              #
#     PyMatchInterface is distributed in the hope that it will be useful,      #
#     but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the             #
#     GNU General Public License for more details.                             #
#                                                                              #
#     You should have received a copy of the GNU General Public License        #
#     along with PyMatchInterface.  If not, see <http://www.gnu.org/licenses/>.#
################################################################################

from __future__ import print_function, division

import os, sys, subprocess, psutil
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

class Helpers(object):
    def __init__(self):
        raise Exception("This is just a helper class")

    @staticmethod
    def applyTie(tie_obj, serie_obj, column = 1):
        if column not in [1,2]:
            print("Column should be 1 or 2")
            return
        
        if not isinstance(serie_obj, Serie):
            print("Expect a serie object on serie_obj !")
            return
        
        if not isinstance(tie_obj, Tie):
            print("Expect a tie object on tie_obj !")
            return
        
        i = 0
        for t in tie_obj.tie_labels:
            serie_obj.setTie(t, tie_obj.tie(t)[column-1])
            i+=1
        
        print("Applied %d ties to %s" % (i, serie_obj.label))
        return i

class Tie(object):
    ''' Represents a tie file as loaded by match. Can be created from a file,
    or from none. A tie file is a file with two columns with values related
    to positions in different series.
    '''
    def __init__(self, filename):
        if filename == "":
            raise Exception("Need a valid filename or None")
        
        self.__label = None
        self._filename = filename
        self.__tiepoints =  { }
        
        if self.filename != None and os.path.isfile(self.filename):
            with open(self.filename) as fio:
                for line in fio:
                    line = line.strip()
                    if len(line) == 0: continue
                    try:
                        s1, v1, s2, v2 = line.strip().split()
                    except ValueError:
                        v1, v2 = line.strip().split()
                        s1 = None
                        s2 = None
                    self.set_tie(v1, v2, s1, s2)
    
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
            indices = list(map(lambda x: ord(x), self.__label))
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
    
    ''' Utilities methods
    '''
    def set_tie(self, value_one, value_two, segment_one = None, segment_two = None, label = None):
        label = label if label is not None else self.nextlabel
        if segment_one is not None: segment_one = int(segment_one)
        if segment_two is not None: segment_two = int(segment_two)
        
        self.__tiepoints[label] = (float(value_one), float(value_two),
                                   segment_one, segment_two)
    
    def tie(self, label):
        return self.__tiepoints[label]
    
    def saveas(self, filename):
        self._filename = filename
        return self.save()
    
    def save(self):
        if self.filename is None:
            raise Exception("Bad filename for saving tie - give me a name first!")
        fio = open(self.filename, "w")
        for label in self.tie_labels:
            v1, v2, s1, s2 = self.tie(label)
            if s1 is None and s2 is None:
                print(v1, v2, file = fio)
            else:
                print(s1, v1, s2, v2, file = fio)
        fio.close()
        
        return True    
    
    def report(self):
        n = len(self.tie_labels)
        print('Total of %d labels are defined.' % n)
        for label in sorted(self.tie_labels):
            v1, v2, s1, s2 = self.tie(label)
            print("  %s => %s,%f = %s,%f" % (label, s1 if s1 is not None else "-",v1, s2 if s2 is not None else "-",v2))

class Serie(object):
    '''
    Represents one 1D series, can be 2 or 3 columns.
    If a file name ".new" exists, read it and assumes
    it is the matched version of the Serie.
    
    The class should be created from a filename, the filename is a two column
    file with data points, normally depth / value or time / value.
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
    
    def setData(self, xdata, ydata, segmentcode = None):
        if len(xdata) != len(ydata): raise Exception("Cannot load data with different size")
        if segmentcode is not None and len(segmentcode) != len(xdata): raise Exception("Cannot load data amd segment size with different size")
        
        self.x = np.array(xdata, dtype = np.float)
        self.y = np.array(ydata, dtype = np.float)
        self.s = np.array(segmentcode, dtype = np.int) if segmentcode is not None else segmentcode
        
        return self

    def setGap(self, gapstart, gapend, force = False):
        if gapstart >= gapend:
            raise Exception("gapstart should be smaller than gapend")
        
        if not force and (len(self.x[ (self.x >= gapstart) & (self.x <= gapend) ]) > 0):
            raise Exception("There is data in gap region [x>=gapstart && x<=gapend] !")
        
        if self.s is None: self.s = np.zeros(self.x.shape)
        
        self.s += 1
        
        self.s[ self.x <= gapstart ] -= 1
        self.s[ self.x >= gapend ] += 1
    
        return self


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
        
        if s is not None:
            s = np.array(s, dtype = np.int)
        
        return x,y,s
    
    def setLimits(self, begin = None, end = None, cut = False):
        '''
        Sets data series useful limits
        '''
        if begin and end and begin >= end: 
            raise Exception("Begin should be smaller than end !")
        
        self.begin = begin
        self.end   = end
        
        if cut:
            self.s = self.__window(self.s) if self.s is not None else self.s
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
        
        #if (len(self.x) / value) < 2:
        #    raise Exception("Value of nintervals too high !")
        
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
            data = np.array(list(zip(self.x, self.y)))
        else:
            data = np.array(list(zip(self.s, self.x, self.y)))
        
        np.savetxt(self.filename, data)
        
        return True
    
    def saveas(self, filename):
        if os.path.isfile(filename):
            print("Will overwrite filename '%s'." % filename)
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
        
        ax = plt.subplot(2,1,1)
        for sid in self._gapids():
            x, y = self._segment(sid)
            ax.plot(x, y, "-o", label="%s sID: %d" % (self.filename, sid))
        for t in self.ties:
            ax.axvline(self.tie(t), 0.1, 0.15, color='k')
        ax.plot(other.xm, other.ym, label="Matched %s" % other.filename)
        ax.axvspan(self.begin, self.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % self.filename)
        ax.legend(bbox_to_anchor = (1.0, 1.45), ncol = 3)
        
        plt.subplot(2,1,2)
        plt.plot(self.xm, self.ym, label="Matched: %s" % self.filename)
        for t in other.ties:
            plt.axvline(other.tie(t),0.1, 0.15, color='k')
        plt.plot(other.x, other.y, "-o", label=other.filename)
        plt.axvspan(other.begin, other.end, 0.1, 0.2, alpha=0.3, color='80', label='Used part from %s' % other.filename)
        plt.legend(bbox_to_anchor = (1.0, -0.15), ncol = 3)
    
    def _gapids(self):
        return np.array(self.s[sorted(np.unique(self.s, True)[1])], dtype = np.int) if self.s is not None else [ None ]
    
    def _segment(self, sid):
        if sid is None or ((sid == 0) and (self.s is None)):
            return self.x, self.y
        return self.x[(self.s == sid)], self.y[(self.s == sid)]
    
    def plot(self, symbols = False):
        '''
        Plot this data serie and its parameters
        '''
        m1 = self.x.min() if self.begin is None else self.begin
        m2 = self.x.max() if self.end is None else self.end
        
        # Data Span
        plt.axvspan(m1, m2, 0.05, 0.10, alpha=0.75, color="0.6", label='Used Segment')
        
        # Segments
        c = 'm'
        for sid in self._gapids():
            x, y = self._segment(sid)
            plt.axvspan(x.min(), x.max(), 0.11, 0.15, alpha = 0.75, color = c)
            c = 'm' if c == 'y' else 'y'
            plt.plot(x, y, label = 'Series %s [%s] sID %s' % (self.label if self.label else '-', self.filename, sid))
            if symbols: plt.plot(x, y, 'o')
        
        for v in self.ties.values():
            plt.axvline(v, 0.05, 0.2, color ='k')
        
        plt.legend()
    
    def report(self, simple = False):
        if simple:
            print("%13s" % "Series:", self.filename)
            print("%13s" % "","begin:", self.begin, "end:", self.end, "num. Intervals:", self.numintervals," len:",len(self.x))
        else:
            print("Serie: Label: %s Filename: %s" %(self.label, self.filename))
            print("\nStatistics:        %8s / %-8s"  % ("Full", "Window"))
            print(" Number of Points: %8d / %-8d" % (len(self.x), len(self.x_window)))
            print("            x-Min: %8f / %-8f" % (self.x.min(), self.x_window.max()))
            print("            y-Min: %8f / %-8f" % (self.y.min(), self.y_window.min()))
            print("            y-Max: %8f / %-8f" % (self.y.max(), self.y_window.max()))
            print("           y-Mean: %8f / %-8f" % (self.y.mean(), self.y_window.mean()))
            print("          y-StdEv: %8f / %-8f" % (self.y.std(), self.y_window.std()))
            print("x-spac. (min/max): %8.4f / %-8.4f" % (np.min(self.x[1:] - self.x[:-1]), np.max(self.x[1:] - self.x[:-1])))
            print("\nAssociated Information:")
            print("   Begin is: %s and End is: %s" % (self.begin, self.end))
            print("   Total # of series/gaps: %s" % str(len(set(self.s))) if self.s is not None else "-")
            print("   Total of %d tie points." % (len(self.ties)))
            i = 0
            for k,x in self.ties.items():
                print("      Tie #%d, Label: %s Position: %s" % (i, k, x))
                i += 1
            print("")
        
        return self

class MatchLog(object):
    '''
    This object represents a log file from a match run.
    It loads it up, all the scores and penalties. Also computes the correlation
    between the time series. It also holds all params used to run match.
    '''
    
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
        # BUggy - we are ignoring the cores again!
        _, x1, _, x2 = np.loadtxt(self.matchfile, unpack = True)
        self.x1 = x1
        self.x2 = x2
    
    def _parse_log(self):
        if not os.path.isfile(self.logfile):
            raise Exception("No Match file found.")
        
        success = False
        
        with open(self.logfile) as results:
            go = False
            for line in results:
                line = line.strip()
                
                if line == "Execution successful.": success = True
                
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
        
        if success is False: raise Exception("Match Failed to Run - No Success @ end")
        if not go: raise Exception("Match Failed to Run - No Penalties")
        
    
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
            
            a = np.correlate(y1n,y1n)[0]
            b = np.correlate(y2n,y2n)[0]
            self._cor = np.correlate(y1n, y2n)[0]/np.sqrt(a*b)
            
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
        labels = ["Total", "point", "nomatch", "speed", "speedchange", "tie", "gap", "tie nomatch", "rms", "corr."]
        
        plt.figure(figsize=(15,9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

        # Bar Plot
        ##
        values = list(map(lambda x: self.scores[x], labels[:-2]))
        positions = list(range(len(labels)))
        values.append(self.rms)
        values.append(self.correlation)
        
        ax0 = plt.subplot(gs[0])
        for pos,_ in enumerate(labels):
            ax0.text(0.1, pos, str("%.3f" %  values[pos]), fontsize=20, horizontalalignment='left', verticalalignment='center')
        ax0.barh(positions, values, align='center')
        plt.yticks(positions, labels, rotation=0)
        plt.xlabel("Score")
        
        # age-model plot
        ax1 = plt.subplot(gs[1])
        ax1.plot(self.x2, self.x1,"-", c="black", lw=2, label="Fitted age-model")
        plt.xlabel("#2 x-value")
        plt.ylabel("#1 x-value")
        plt.legend(loc=1)
        
        
        if self.sb != None:
            ax2 = ax1.twinx()
            
            ax2.plot(self.sb.x, self.sb.y,'--.', c="orange", label="%s" % self.sb.label)
            ax2.plot(self.sa.xm, self.sa.ym,'-', c="red", label=self.sa.label)
            vv1 = min(self.sb.ym.min(), self.sa.y.min())
            vv2 = max(self.sb.ym.max(), self.sa.y.max())
            ax2.set_ylim([vv1, vv2*12.])
            for tl in self.sb.ties:
                ax2.axvline(self.sb.tie(tl), 0.0, 0.03, lw=2, c='k')
            plt.legend(loc=4)
            plt.ylabel("Matched #2")
            
            ax3 = ax1.twiny()
            ax3.plot(self.sb.ym,self.sb.xm,"-", c="orange")
            ax3.plot(self.sa.y,self.sa.x,"--", c="red")
            vv1 = min(self.sb.y.min(), self.sa.ym.min())
            vv2 = max(self.sb.y.max(), self.sa.ym.max())
            ax3.set_xlim([vv1, vv2*12.])
            for tl in self.sa.ties:
                ax3.axhline(self.sa.tie(tl), 0.0, 0.03, lw=2, c='k')
            plt.xlabel("Matched #1")
    
    def save_gmt(self, filename = None):
        ff = open(filename, "w")
        kks = [
            "Total", "point",
            "nomatch", "speed", "speedchange",
            "tie", "gap", "tie nomatch"
        ]
        
        pps = [
            "begin1", "end1", "numintervals1", "begin2", "end2", "numintervals2",
            "nomatch", "speedpenalty", "speedchange", "tiepenalty", "gappenalty"
        ]
        
        for ik, k in enumerate(kks):
            ff.write("#Score>%d %f 12 0 0 TM %s\n" % (ik, self.scores[k], k))
        
        for ik, k in enumerate(pps):
            ff.write("#Penalty>%d %f 12 0 0 TM %s\n" % (ik, self.params[k], k))
        ff.write("#Penalty>%d %s 12 0 0 TM %s\n" % (ik+1,self.params["targetspeed"], "targetspeed"))
        
        ff.write("> %f %f %f\n" % (self.rms, self.correlation, self.scores['Total']))
        
        for xx, yy in zip(self.x1, self.x2):
            ff.write("%f %f\n" % (xx, yy))
        ff.close()

class MatchConfFile(object):
    '''
    This class represents a MatchConfFile, i.e. the configuration file for match
    if the given filename exists parameters are obtained from file, otherwise a 
    new file is created but not saved. Use the save() call to save it.
    
    Use the interface supplied to set each of the match penalties. Use the run()
    method to run match. Before that, give the series filename to the series1
    and series2 variables. This will automatically attach the series to the
    conf-file. Alternatively use the setSeries() method.

    If you want to optimize a parameter, use the optimize() method supplying the
    parameter name and array of values to be tested. The optimization method
    should be one of ["RMS", "CORRELATION", "TOTAL_SCORE"] and supplied while
    instantiating the class.

    run() and runcopy() returns a MatchLog object, optimize() returns a list of
    MatchLog.
    '''
    ##
    # Location of the MATCH executable code
    ##
    
    _MATCHCMD = "./match" if not psutil.WINDOWS else "match.exe"
    
    def __init__(self, filename, autonormalize = False, optmization_method = "Total_Score"):
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
        if optmization_method.upper() not in [ "RMS", "CORRELATION", "TOTAL_SCORE" ]:
            raise Exception('Wrong optimization method, use one of: "RMS", "CORRELATION", "TOTAL_SCORE"')
        
        self._optimize_method = optmization_method.upper()
        
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
            print("Adjusting maximum sub-division so that there is at least 2 samples per segment")
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
                self.begin1   = None
                self.end1     = None
                self.numintervals1 = None
                return
        if not os.path.isfile(value):
            raise Exception("File '%s' not found." % value)
        self.setSeries(1, value, None, None, None)
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
        #self._numintervals1 = self.__maxsubdiv(1, value) if value != None else None
        self._numintervals1 = value
        self._issaved = False
    
    @series2.setter
    def series2(self, value):
        if value == None:
                self._series2 = None
                self.begin2   = None
                self.end2     = None
                self.numintervals2 = None
                return
        if not os.path.isfile(value):
            raise Exception("File '%s' not found." % value)
        self.setSeries(2, value, None, None, None)
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
        #self._numintervals2 = self.__maxsubdiv(2, value) if value != None else None
        self._numintervals2 = value
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
        if value is None:
            self._tiefile = None
        else:
            if not os.path.isfile(value):
                raise Exception("File '%s' not found." % value)
            
            ##
            # Test tie
            ##
            tt = Tie(value)
            sa = self.getSeries(1)
            sb = self.getSeries(2)
            for label in tt.tie_labels:
                ### BUGGY - We should check that the core s1,s2 is valid for serie
                v1, v2, s1, s2 = tt.tie(label)
                sa.setTie(label, v1)
                sb.setTie(label, v2)
            
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
                        print("Invalid key in config file ignored: '%s'  !" % k)
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
            print("No match & log file indicated, generating default values.")
            self.__set_filename(filename)
        
        try:
            self.targetspeed = targetspeed
        except Exception as e:
            print("Error -- ",e.message)
    
    def __write(self, fio, variable):
        v = getattr(self, variable)
        if v == None: return
        if variable == "speeds": v = ",".join(v)
        print("%-13s" % variable, "%s" % v, file = fio)
    
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
            print("Setting series1 begin to %s" % self.begin1)
        
        if self.end1 == None:
            self.end1 = s1.x.max()
            print("Setting series1 end to %s" % self.end1)
        
        if self.numintervals1 == None:
            self.numintervals1 = len(s1.x) // 7
            print("Setting series1 numintervals to %s" % self.numintervals1)
        
        s2 = Serie(self.series2)
        if self.begin2 == None:
            self.begin2 = s2.x.min()
            print("Setting series2 begin to %s" % self.begin2)
        
        if self.end2 == None:
            self.end2 = s2.x.max()
            print("Setting series2 end to %s" % self.end2)
        
        if self.numintervals2 == None:
            self.numintervals2 = len(s2.x) // 7
            print("Setting series2 numintervals to %s" % self.numintervals2)
        
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
            first = list(range(1,first+1))
        
        if not isinstance(last, list):
            last = list(range(1,last+1))
        
        if len(first) == 0 or len(last) == 0:
            raise Exception("Not enough values for pair")
        
        items = []
        
        first.reverse()
        last.reverse()
        
        if exchanged:
            for f in first:
                for l in last:
                    if f == 0 or l == 0: continue
                    #if l == f and l != 1: continue
                    item = "%d:%d" % (l,f)
                    if item not in items: items.append(item)
        
        first.reverse()
        last.reverse()
        
        for f in first:
            for l in last:
                if f == 0 or l == 0: continue
                #if l == f and l != 1: continue
                item = "%d:%d" % (f,l)
                if item not in items: items.append(item)
        
        self.speeds = items
    
    def setSeries(self, which, filename, begin = None, end = None, nintervals = None):
        if not os.path.isfile(filename):
            raise Exception("Not a file ! ")
        
        if which == 1:
            serie = Serie(filename)
            self._series1 = filename
            self.begin1 = float(begin) if begin != None else serie.x.min() 
            self.end1   = float(end) if end != None else serie.x.max()
            self.numintervals1 = int(nintervals) if nintervals != None else len(serie.x) // 7
        elif which == 2:
            serie = Serie(filename)
            self._series2 = filename
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
            print("Normalizing series (1) from %.1f to %.1f and series (2) from %.1f to %.1f" % (self.begin1, self.end1, self.begin2, self.end2))
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
            s = Serie(self.series1, "#1: " + self.series1)
            s.setLimits(self.begin1, self.end1, cut = False)
            s.numintervals = self.numintervals1
            if t:
                list(map(lambda label: s.setTie(label, t.tie(label)[0]), t.tie_labels))
            return s
        elif which == 2:
            if self.series2 == None: raise Exception("Serie is Unset.")
            s = Serie(self.series2, "#2: " + self.series2)
            s.setLimits(self.begin2, self.end2, cut = False)
            s.numintervals = self.numintervals2
            if t:
                list(map(lambda label: s.setTie(label, t.tie(label)[1]), t.tie_labels))
            return s
        else:
            raise Exception("No such serie")
    
    def run(self, autosave = False, plotresults = True):
        if self._issaved == False and autosave == False:
            raise Exception("Please save CONF file first !")
        
        if autosave:
            print("Auto-saving file '%s'" % self.filename)
            self.save()
        
        if not os.path.isfile(self.filename):
            raise Exception("Filename '%s' Not Found." % self.filename)
        
        if not os.path.isfile(MatchConfFile._MATCHCMD):
            raise Exception("Program 'match' was not found in this system !")
        
        if self._autonormalize:
            print("Normalizing series (1) from %.1f to %.1f and series (2) from %.1f to %.1f" % (self.begin1, self.end1, self.begin2, self.end2))
            self.getSeries(1).normalizeStd(True).save()
            self.getSeries(2).normalizeStd(True).save()
        
        err = subprocess.call([MatchConfFile._MATCHCMD, self.filename])
        if err != 0:
            print("Failed to RUN match -- look at the log file -- %s" % self.logfile)
            return None
        
        try:
            ml = MatchLog(self.matchfile,
                          self.logfile,
                          self.getSeries(1),
                          self.getSeries(2)
                          )
        except Exception as e:
            print(str(e))
            return None
        
        ml.params.update(self.__params())
        
        if plotresults: ml.plot()
        
        return ml 
    
    def runcopy(self):
        original_filename = self.filename
        self.saveas("lala.conf")
        results = self.run(plotresults = False)
        self.clean()
        os.unlink("lala.conf")
        self.__set_filename(original_filename)
        self.save()
        return results
    
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
        
        if plot: print("Working on optimizing %s from %s - %s" % (parameter, values[0], values[-1]))
        best_value = None
        max_metric    = None
        original_filename = self.filename
        pcount     = 0
        rounds = 0
        for iv, v in zip(range(len(values)), values):
            if plot:
                if pcount in  [5,10,15, 20, 25, 30, 35, 40]: print(" ", end = "")
                if pcount == 45:
                    rounds += 1
                    if plot: print("%03s %%" % ((100*rounds*45)/len(values)))
                    pcount = 0
                print(".", end = "")
                if iv == len(values)-1: print("100 %")
            pcount += 1
            
            setattr(self, parameter, v)
            
            self.saveas("lala.conf")
            results = self.run(plotresults = False)
            self.clean()
            os.unlink("lala.conf")
            
            if self._optimize_method == "RMS":
                CMAX = results.rms
            elif self._optimize_method == "CORRELATION":
                CMAX = (1.0 - results.correlation)
            elif self._optimize_method == "TOTAL_SCORE":
                CMAX = results.scores["Total"]
            else:
                raise Exception("Bad Optimization Method -- use one of: 'RMS', 'CORRELATION', 'TOTAL_SCORE'")
            
            if max_metric == None or CMAX < max_metric:
                best_value = v
                max_metric = CMAX
            x.append(float(v) if not isinstance(v, str) else iv)
            s.append(results)
        
        self.__set_filename(original_filename)
        
        if plot: print("Final parameter %s = %s"  % (parameter, best_value))
        
        if update:
            setattr(self, parameter, best_value)
            self.run(autosave = True, plotresults=plot)
        
        if plot:
            _, ax1 = plt.subplots(figsize=(10,7))
            values = list(map(lambda item: item.correlation, s))
            ax1.plot(x, values, "r-o", label='Correlation')
            formats, labels = ax1.get_legend_handles_labels()
            
            ax2 = ax1.twinx()
            for k in ["Total", "point", "nomatch", "speed", "speedchange", "tie", "gap", "tie nomatch"]:
                values = list(map(lambda item: item.scores[k], s))
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
        print("")
        print(" ** Series **")
        sa.report(simple = True)
        print("")
        sb.report(simple = True)
        print("")
        
        print(" ** Penalties **")
        print("%13s" % "nomatch:", self.nomatch)
        print("%13s" % "speedpenalty:", self.speedpenalty)
        print("%13s" % "targetspeed:", self.targetspeed)
        print("%13s" % "speedchange:", self.speedchange)
        print("%13s" % "tiepenalty:", self.tiepenalty)
        print("%13s" % "gappenalty:", self.gappenalty)
        print("")
        
        print("%13s" % "speeds:", ",".join(self.speeds) if self.speeds else "-")
        print("")
        
        print(" ** Constrains & Files **")
        print("%13s" % "series1gaps:", self.series2gaps)
        print("%13s" % "series2gaps:", self.series1gaps)
        print("%13s" % "tiefile:", self.tiefile)
        print("%13s" % "matchfile:", self.matchfile)
        print("%13s" % "logfile:", self.logfile)
        
        if self.tiefile != None and self.tiefile != "":
            tt = Tie(self.tiefile)
            print("")
            print(" ** Tie is Defined **")
            tt.report()
            print("")

class Optimizer(object):
    '''
    This class is able to run perform a full optimization of the penalties.
    While it performs the optimization it also stores all the logfiles as 
    matchlog objects that are used in the end to obtain an age model weigthed
    by the Correlation.
    
    Tests can be run in a given order, or in an automated order depende on what
    method is used: run() or run_ordered().
    
    An exhaustive test can be made using runallcomb().
    ''' 
    def __init__(self, mcf):
        self._mcf = mcf
        self.mls = None
        
        # This is for run, one per 
        # param using Lisiecks order
        self.tests = {}
        
        # Those are for ordered running
        self.orderactivated = [ ]
        self.orderedparams = [ ]
        
        self.xs = None
        self.ys = None
    
    ''' Optimization parameters
    '''
    def targetspeed(self, values):
        self.orderactivated.append("targetspeed")
        self.orderedparams.append(values)
        self.tests["targetspeed"] = values
    
    def nomatch(self, values):
        self.orderactivated.append("nomatch")
        self.orderedparams.append(values)
        self.tests["nomatch"] = values
    
    def speedchange(self, values):
        self.orderactivated.append("speedchange")
        self.orderedparams.append(values)
        self.tests["speedchange"] = values
    
    def speedpenalty(self, values):
        self.orderactivated.append("speedpenalty")
        self.orderedparams.append(values)
        self.tests["speedpenalty"] = values
    
    def gappenalty(self, values):
        self.orderactivated.append("gappenalty")
        self.orderedparams.append(values)
        self.tests["gappenalty"] = values
    
    def tiepenalty(self, values):
        self.orderactivated.append("tiepenalty")
        self.orderedparams.append(values)
        self.tests["tiepenalty"] = values
    
    def nintervals(self, vals1_int1, vals_int2):
        self.orderactivated.append("nintervals")
        self.orderedparams.append((vals1_int1, vals_int2))
        self.tests["nintervals"] = (vals1_int1, vals_int2)
    
    def __run_nintervals(self, params):
        v1, v2 = params
        
        stop = False
        for _ in range(6):
            if stop: break
        
            before = self._mcf.numintervals1
            ll = self._mcf.optimize("numintervals1", v1, True, False)
            self.mls.extend(ll)
            after = self._mcf.numintervals1
        
            stop = (before == after)
        
            before = self._mcf.numintervals2
            ll = self._mcf.optimize("numintervals2", v2, True, False)
            self.mls.extend(ll)
            after = self._mcf.numintervals2
        
            stop = stop and (before == after)

        print("Nintervals1 from ",v1[0]," to ",v1[-1]," is: ",getattr(self._mcf, "numintervals1"))
        print("Nintervals2 from ",v2[0]," to ",v2[-1]," is: ",getattr(self._mcf, "numintervals2"))
    
    def __run(self, param, values):
        oldv = getattr(self._mcf, param)
        try:
            self.mls.extend(self._mcf.optimize(param, values, True, False))
        except:
            setattr(self._mcf, param, oldv)
            self._mcf.save()
            print("Failed to optimize %s with %s" % (param, values))
        
        print(param," from ",values[0]," to ",values[-1]," is: ",getattr(self._mcf, param))
    
    ''' Utility methods
    '''
    def plot_results(self):
        mls = []
        for i,ml in enumerate(self.mls):
            c = 1.0 - ml.correlation
            if c > 1: continue
            mls.append((i,c))
        mls = sorted(mls, key=lambda tup: tup[1], reverse=True)
        
        # Start Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[1, 3]}, sharex=True, figsize=(6,7.5))
        
        sa = self._mcf.getSeries(1)
        sb = self._mcf.getSeries(2)
        
        xta, xa, vxa = self.estimate(cm=sa.x)
        sa.x = xta
        
        ax1.plot(sb.x, sb.y,'-', c='red')
        for s in sa._gapids():
            x, y = sa._segment(s)
            ax1.plot(x, y, "-k")
        ax1.yaxis.set_label_text("$\delta O^{18}$")
        
        xs, mys, eys, rxs, rmys, reys = self.median_estimate()
        ax2.plot(rxs, rmys, lw=2, c='red', zorder=11)
        ax2.errorbar(rxs[::3], rmys[::3], reys[::3], c='red', elinewidth=2, lw=0, zorder=10)
        
        cmap = plt.cm.magma
        norm = mpl.colors.Normalize(0.0, 0.05)
        c = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        for iml in mls:
            ml = self.mls[iml[0]]
            ax2.plot(ml.x2, ml.x1, c = c.to_rgba(iml[1]), zorder = 1)
        ax2.yaxis.set_label_text("Depth (km)")
        ax2.xaxis.set_label_text("Time (ka)")
        
        ttt = Tie(self._mcf.tiefile)
        for label in ttt.tie_labels:
            tie = ttt.tie(label)
            ax2.axvline(tie[1],0.02,0.05, ls='--')
            ax2.axhline(tie[0],0.02, 0.05, ls='--')
        
        ax3 = ax2.figure.add_axes([1.0, 0.1, 0.02, 0.3])
        mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, label='1-$\phi$')
        plt.tight_layout()
        
        return
    
    def median_estimate(self, power = 4.0, limit = 1.0):
        xs  = None
        mys = None
        eys = None
        
        rxs = None
        rmys = None
        reys = None
        
        if self.mls:
            xmin = min(self.mls[0].x1)
            xmax = max(self.mls[0].x1)
            xlen = len(self.mls[0].x1)
            rxmin = min(self.mls[0].x2)
            rxmax = max(self.mls[0].x2)
            rxlen = len(self.mls[0].x2)
            for x in self.mls:
                xlen = max(xlen, len(x.x1))
                xmin = min(xmin, min(x.x1))
                xmax = max(xmax, max(x.x1))

                rxlen = max(rxlen, len(x.x2))
                rxmin = min(rxmin, min(x.x2))
                rxmax = max(rxmax, max(x.x2))
            
            # We Use 2xLength of curves given by Lisiecks
            xs = np.linspace(xmin, xmax, 2*xlen)
            ys  = []
            wys = []
            
            rxs = np.linspace(rxmin, rxmax, 2*rxlen)
            rys  = []
            
            # print("Average",len(self.mls),"objects")
            for x in self.mls:
                if limit is not None and (1-x.correlation) > limit: continue
                ys.append(np.interp(xs, x.x1, x.x2, left = np.nan, right = np.nan))
                rys.append(np.interp(rxs, x.x2, x.x1, left = np.nan, right = np.nan))
                wys.append(1.0/(1.0-x.correlation))
            
            if len(ys) == 0:
                raise Exception("No good fit exists --- run again !")
            
            ##
            # Go to masked array
            #
            ys = np.array(ys)
            ys = np.ma.array(ys, mask=np.isnan(ys))
            
            rys = np.array(rys)
            rys = np.ma.array(rys, mask=np.isnan(rys))
            
            
            ##
            # Process weights
            #
            wys = np.array(wys)**power
            
            ##
            # Compute average
            #
            mys = np.ma.average(ys, axis=0, weights = wys)
            variance = np.ma.average((ys-mys)**2, weights=wys, axis=0)
            eys = np.sqrt(variance)
        
            rmys = np.ma.average(rys, axis=0, weights = wys)
            variance = np.ma.average((rys-rmys)**2, weights=wys, axis=0)
            reys = np.sqrt(variance)
        
        return (xs, mys, eys, rxs, rmys, reys)
    
    def run_ordered(self, plot = True):
        if self.mls is None: self.mls = list()
        
        for k,v in zip(self.orderactivated, self.orderedparams):
            if k == "nintervals":
                self.__run_nintervals(v)
                continue
            self.__run(k, v)
        
        if not self.mls:
            self.orderactivated = []
            self.orderedparams  = []
            print(" -- No tests was performed. -- ")
            return
        
        if plot:
            self.plot_results()
        
        self.orderactivated = []
        self.orderedparams  = []
        
        return
    
    def run(self, plot = True):
        self.mls = list()
        
        for k in ["nintervals", "targetspeed", "nomatch", "speedchange", "speedpenalty", "gappenalty", "tiepenalty"]:
            if k not in self.tests:
                print("Skipping optimize: %s" % k)
                continue
            
            if k == "nintervals":
                self.__run_nintervals(self.tests[k])
                continue
            
            self.__run(k, self.tests[k])
        
        if not self.mls:
            self.tests = {}
            print(" -- No tests was performed. -- ")
            return
        
        if plot:
            self.plot_results()
        
        self.tests = {}
        return
    
    def runallcomb(self):
        self.mls = list()
        
        def _G(k):
            try:
                return self.tests[k]
            except KeyError:
                return [ getattr(self._mcf, k) ]
            return []
        
        for targetspeed in _G("targetspeed"):
            self._mcf.targetspeed = targetspeed
            for nomatch in _G("nomatch"):
                self._mcf.nomatch = nomatch
                for speedchange in _G("speedchange"):
                    self._mcf.speedchange = speedchange
                    for speedpenalty in _G("speedpenalty"):
                        self._mcf.speedpenalty = speedpenalty
                        for tiepenalty in _G("tiepenalty"):
                            self._mcf.tiepenalty = tiepenalty
                            self._mcf.save()
                            if "nintervals" in self.tests:
                                self.__run_nintervals(self.tests['nintervals'])
                            else:
                                self.mls.append(self._mcf.runcopy())
        
        return 
    
    def estimate(self, cm = None, t = None):
        xcm, xt, et, _, _, _ = self.median_estimate()
        
        if cm is None and t is not None:
            cm = np.interp(t, xt, xcm, left = np.nan, right = np.nan)
            st = np.interp(t, xt, et, left = np.nan, right = np.nan)
        elif cm is not None and t is None:
            t = np.interp(cm, xcm, xt, left = np.nan, right = np.nan)
            st = np.interp(cm, xcm, et, left = np.nan, right = np.nan)
        else:
            raise Exception("need cm or t")
        
        return t, cm, st
    
    def export_csv(self, filename_or_file = sys.stdout, sep=","):
        need_close = False
        
        if isinstance(filename_or_file, str):
            need_close = True
            filename_or_file = open(filename_or_file, "w")
        
        xs, ys, eys, _, _, _ = self.median_estimate()
        for x, y, e in zip(xs,ys,eys):
            print("%f%s%f%s%f" % (x,sep,y,sep,e), file = filename_or_file)
        
        if need_close:
            filename_or_file.close()
        
        return
    
    def curves_fitted(self):
        sa = self._mcf.getSeries(1)
        sb = self._mcf.getSeries(2)
        
        xa = sa.x
        va = sa.y
        
        xb = sb.x
        vb = sb.y
        
        xat, _, sta = self.estimate(cm=xa)
        _, xbx, stb = self.estimate(t=xb)

        #
        # Clean UP np.nan from output
        #
        A = np.array([xa, xat, sta, va])
        A = A.transpose()
        A = A[~np.isnan(A).any(axis=1)]
        xa, xat, sta, va = A.transpose()
        
        B = np.array([xbx, xb, stb, vb])
        B = B.transpose()
        B = B[~np.isnan(B).any(axis=1)]
        xbx, xb, stb, vb = B.transpose()
        
        return xa, xat, sta, va, xbx, xb, stb, vb
    
    def curves(self):
        plt.figure(figsize=(20,10))
        
        xa, xat, _, va, xbx, xb, _, vb = self.curves_fitted()
        
        plt.subplot(1,2,1)
        plt.plot(xa, va, "r", xbx, vb, "k")
        
        plt.subplot(1,2,2)
        plt.plot(xat, va, "r", xb, vb, "k")
        
        return
    
    def save_gmt(self, folder = "GMT"):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        i = 0
        for ml in self.mls:
            ml.save_gmt(os.path.join(folder, "op_%08d.dat" % i))
            i += 1

        l = self._mcf.run(False, False)
        l.save_gmt(os.path.join(folder, "best.dat"))
        
        xcm, xt, et, _, _, _ = self.median_estimate()
        ff = open(os.path.join(folder, "median.dat"), "w")
        ff.write("> Median Age Model\n")
        for x,t,e in zip(xcm,xt,et):
            ff.write("%f %f %f\n" % (x,t,e))
        ff.close()

