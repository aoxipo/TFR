# pysigproc.py -- P. Demorest, 2016/04
#
# Simple functions for generating sigproc filterbank
# files from python.  Not all possible features are implemented.
# now works with python3 also!

import mmap
import struct
import sys
from collections import OrderedDict
from astropy.io import fits 
import platform
import numpy
import astropy.io.fits as pyfits
from decimal import Decimal

SYS_OPERATE = platform.system()

class SigprocFile(object):

    ## List of types
    _type = OrderedDict()
    _type['rawdatafile'] = 'string'
    _type['source_name'] = 'string'
    _type['machine_id'] = 'int'
    _type['barycentric'] = 'int'
    _type['pulsarcentric'] = 'int'
    _type['telescope_id'] = 'int'
    _type['src_raj'] = 'double'
    _type['src_dej'] = 'double'
    _type['az_start'] = 'double'
    _type['za_start'] = 'double'
    _type['data_type'] = 'int'
    _type['fch1'] = 'double'
    _type['foff'] = 'double'
    _type['nchans'] = 'int'
    _type['nbeams'] = 'int'
    _type['ibeam'] = 'int'
    _type['nbits'] = 'int'
    _type['tstart'] = 'double'
    _type['tsamp'] = 'double'
    _type['nifs'] = 'int'

    def __init__(self, fp=None, copy_hdr=None, data_source = None):
        self.data_source = data_source
        if data_source == None:
            # init all items to None
            for k in list(self._type.keys()):
                setattr(self, k, None)
      
            if copy_hdr is not None:
                for k in list(self._type.keys()):
                    setattr(self,k,getattr(copy_hdr,k))
            
            if fp is not None:
                try:
                    self.fp = open(fp,'rb')
                except TypeError:
                    self.fp = fp
                
                if(SYS_OPERATE == "Windows"):
                    self._mmdata = mmap.mmap(self.fp.fileno(), 0, None, access = mmap.ACCESS_READ)
                else:
                    self._mmdata = mmap.mmap(self.fp.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
                    
            self.read_header(self.fp)
        elif data_source == "国台天文台":
            self.fp_name = fp
            self.hdulist = pyfits.open(fp)
            
            self.read_header()
            
            self._mmdata, self.total_time = self.zipdata(self.hdulist[1].data['data'])
            
            
        
        
            
            
        # print(self.rawdatafile)
    ## See sigproc send_stuff.c
    def zipdata(self, data1):
        if len(data1.shape)>2:
           a,b,c,d,e = data1.shape
           data = data1[:,:,0,:,:].squeeze().reshape((-1,d))
           l, m = data.shape
           total_time = self.tsamp*l
           data = data.reshape(l//64, 64, d).sum(axis=1)
        else:
           data=data1.reshape(-1,int(self.obsnchan))
           l,m=data.shape
           data = data.reshape(l//64, 64, int(self.obsnchan)).sum(axis=1)
        total_time = self.tsamp*l
        return data, total_time
    
    @staticmethod
    def send_string(val,f=sys.stdout):
        val=val.encode()
        f.write(struct.pack('i',len(val)))
        f.write(val)

    def send_num(self,name,val,f=sys.stdout):
        self.send_string(name,f)
        f.write(struct.pack(self._type[name][0],val))

    def send(self,name,f=sys.stdout):
        if not hasattr(self,name): return
        if getattr(self,name) is None: return
        if self._type[name]=='string':
            self.send_string(name,f)
            self.send_string(getattr(self,name),f)
        else:
            self.send_num(name,getattr(self,name),f)

    ## See sigproc filterbank_header.c

    def filterbank_header(self,fout=sys.stdout):
        self.send_string("HEADER_START",f=fout)
        for k in list(self._type.keys()):
            self.send(k,fout)
        self.send_string("HEADER_END",f=fout)

    ## See sigproc read_header.c

    @staticmethod
    def get_string(fp):
        """Read the next sigproc-format string in the file."""
        nchar = struct.unpack('i',fp.read(4))[0]
        if nchar > 80 or nchar < 1:
            return (None, 0)
        out = fp.read(nchar)
        return (out, nchar+4)

    def read_header(self,fp=None):
        """Read the header from the specified file pointer."""
        if self.data_source == "国台天文台":
            self.data_type = 1
            
            self.nchans = self.hdulist[0].header['OBSNCHAN'] # 周期
            
            self.nifs = 1
            self.rawdatafile = self.fp_name
            self.src_raj = None
            self.az_start = 0
            self.za_start = 0
            self.nifs = 1
            self.telescope_id = None
            self.nbits = 8
            self.fch1 = self.hdulist['SUBINT'].data[0]['DAT_FREQ'][0]
            self.foff = -1
            self.src_dej = None
            self.machine_id = None
            
            self.tsamp = self.hdulist[1].header['TBIN']
            self.obsnchan = self.hdulist[0].header['OBSNCHAN']
            
            self.secperday = 3600*24
            self.subintoffset = self.hdulist[1].header['NSUBOFFS']
            self.samppersubint  = int(self.hdulist[1].header['NSBLK'])
            self.tsamp = self.hdulist[1].header['TBIN']
            self.tstart = "%.13f" % (Decimal(self.hdulist[0].header['STT_IMJD']) + Decimal(self.hdulist[0].header['STT_SMJD'] + self.tsamp * self.samppersubint * self.subintoffset )/ self.secperday )
            
        else:
            if fp is not None: self.fp = fp
            self.hdrbytes = 0
            (s,n) = self.get_string(self.fp)
            if s != b'HEADER_START':
                self.hdrbytes = 0
                return None
            self.hdrbytes += n
            while True:
                (s,n) = self.get_string(self.fp)
                s=s.decode()
                self.hdrbytes += n
                if s == 'HEADER_END': return
                if self._type[s] == 'string':
                    (v,n) = self.get_string(self.fp)
                    self.hdrbytes += n
                    setattr(self,s,v)
                    print("s,v",s,v)
                else:
                    datatype = self._type[s][0]
                    datasize = struct.calcsize(datatype)
                    val = struct.unpack(datatype,self.fp.read(datasize))[0]
                    setattr(self,s,val)
                    self.hdrbytes += datasize
                    print("s,val",s,val)
           
        

    @property
    def dtype(self):
        if self.nbits==8:
            return numpy.uint8
        elif self.nbits==16:
            return numpy.uint16
        elif self.nbits==32:
            return numpy.float32
        else:
            raise RuntimeError('nbits=%d not supported' % self.nbits)

    @property
    def bytes_per_spectrum(self):
        return self.nbits * self.nchans * self.nifs / 8

    @property
    def nspectra(self):
        return (self._mmdata.size() - self.hdrbytes) / self.bytes_per_spectrum

    @property
    def tend(self):
        return self.tstart + self.nspectra*self.tsamp/86400.0

    def get_data(self, nstart, nsamp, offset=0):
        """Return nsamp time slices starting at nstart."""
        if(self.data_source == "国台天文台"):
            return self._mmdata
        else:
            bstart = int(nstart) * self.bytes_per_spectrum
            nbytes = int(nsamp) * self.bytes_per_spectrum
            b0 = self.hdrbytes + bstart + (offset*self.bytes_per_spectrum)
            b1 = b0 + nbytes
            # print(self.dtype, self.nifs, self.nchans)
            return numpy.frombuffer(self._mmdata[int(b0):int(b1)],
                    dtype=self.dtype).reshape((-1,self.nifs,self.nchans))

    def unpack(self,nstart,nsamp):
        """Unpack nsamp time slices starting at nstart to 32-bit floats."""
        if self.nbits >= 8:
            return self.get_data(nstart,nsamp).astype(numpy.float32)
        bstart = int(nstart) * self.bytes_per_spectrum
        nbytes = int(nsamp) * self.bytes_per_spectrum
        b0 = self.hdrbytes + bstart
        b1 = b0 + nbytes
        # reshape with the frequency axis reduced by packing factor
        fac = 8 / self.nbits
        d = numpy.frombuffer(self._mmdata[b0:b1],
                dtype=numpy.uint8).reshape(
                        (nsamp,self.nifs,self.nchans/fac))
        unpacked = numpy.empty((nsamp,self.nifs,self.nchans),
                dtype=numpy.float32)
        for i in range(fac):
            mask = 2**(self.nbits*i)*(2**self.nbits-1)
            unpacked[...,i::fac] = (d & mask) / 2**(i*self.nbits)
        return unpacked

    @property
    def chan_freqs(self):
        if(self.data_source == "国台天文台"):
            return self.fch1 + numpy.arange(256)*self.foff
        return self.fch1 + numpy.arange(self.nchans)*self.foff

    @property
    def bandpass(self):
        if(self.data_source == "国台天文台"):
            return self._mmdata.mean(0)
        else:
            return self.get_data(nstart=0,nsamp=int(self.nspectra))[:,0,:].mean(0)
