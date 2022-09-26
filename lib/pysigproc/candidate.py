#!/usr/bin/env python3
import logging

import h5py
import numpy as np
from scipy.optimize import golden
from skimage.transform import resize
import cv2
import copy
from .pysigproc import SigprocFile

logger = logging.getLogger(__name__)


def _decimate(data, decimate_factor, axis, pad=False, **kwargs):
    """

    :param data: data array to decimate
    :param decimate_factor: number of samples to combine
    :param axis: axis along which decimation is to be done
    :param pad: Whether to apply padding if the data axis length is not a multiple of decimation factor
    :param args: arguments of padding
    :return:
    """
    if data.shape[axis] % decimate_factor and pad is True:
        logging.info(f'padding along axis {axis}')
        pad_width = closest_number(data.shape[axis], decimate_factor)
        data = pad_along_axis(data, data.shape[axis] + pad_width, axis=axis, **kwargs)
    elif data.shape[axis] % decimate_factor and pad is False:
        raise AttributeError('Axis length should be a multiple of decimate_factor. Use pad=True to force decimation')

    if axis:
        return data.reshape(int(data.shape[0]), int(data.shape[1] // decimate_factor), int(decimate_factor)).mean(2)
    else:
        return data.reshape(int(data.shape[0] // decimate_factor), int(decimate_factor), int(data.shape[1])).mean(1)


def _resize(data, size, axis, **kwargs):
    """

    :param data: data array to resize
    :param size: required size of the axis
    :param axis: axis long which resizing is to be done
    :param args: arguments for skimage.transform resize function
    :return:
    """
    if axis:
        return resize(data, (data.shape[0], size), **kwargs)
    else:
        return resize(data, (size, data.shape[1]), **kwargs)


def crop(data, start_sample, length, axis):
    """

    :param data: Data array to crop
    :param start_sample: Sample to start the output cropped array
    :param length: Final Length along the axis of the output
    :param axis: Axis to crop
    :return:
    """
    if data.shape[axis] > start_sample + length:
        if axis:
            return data[:, start_sample:start_sample + length]
        else:
            return data[start_sample:start_sample + length, :]
    elif data.shape[axis] == length:
        return data
    else:
        raise OverflowError('Specified length exceeds the size of data')


def pad_along_axis(array: np.ndarray, target_length, loc='end', axis=0, **kwargs):
    """

    :param array: Input array to pad
    :param target_length: Required length of the axis
    :param loc: Location to pad: start: pad in beginning, end: pad in end, else: pad equally on both sides
    :param axis: Axis to pad along
    :return:
    """
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array
        # return a

    npad = [(0, 0) for x in range(axis_nb)]

    if loc == 'start':
        npad[axis] = (int(pad_size), 0)
    elif loc == 'end':
        npad[axis] = (0, int(pad_size))
    else:
        npad[axis] = (int(pad_size // 2), int(pad_size // 2))

    return np.pad(array, pad_width=npad, **kwargs)


def closest_number(big_num, small_num):
    """
    Finds the difference between the closest multiple of a smaller number with respect to a bigger number
    :param big_num: The bigger number to find the closest of
    :param small_num: Number whose multiple is to be found and subtracted
    :return:
    """
    if big_num % small_num == 0:
        return 0
    else:
        q = big_num // small_num
        return (q + 1) * small_num - big_num


class Candidate(SigprocFile):
    def __init__(self, fp=None, copy_hdr=None, dm=None, tcand=0, width=0, label=-1, snr=0, min_samp=256, device=0,
                 kill_mask=None, data_source = None):
        """

        :param fp: Filepath of the filterbank
        :param copy_hdr: Custom header to the filterbank file
        :param dm: DM of the candidate
        :param tcand: Time of the candidate in filterbank file (seconds)
        :param width: Width of the candidate (number of samples)
        :param label: Label of the candidate (1: for FRB, 0: for RFI)
        :param snr: SNR of the candidate
        :param min_samp: Minimum number of time samples to read
        :param device: If using GPUs, device is the GPU id
        :param kill_mask: Boolean mask of channels to kill
        """
        SigprocFile.__init__(self, fp, copy_hdr, data_source)
        self.dm = dm
        self.tcand = tcand
        self.width = width
        self.label = label
        self.snr = snr
        
        self.id = 'cand_tstart_{:.12}_tcand_{:.7}_dm_{:.5}_snr_{:.5}'.format(self.tstart, self.tcand, self.dm, self.snr)
        #print(self.id)
        self.data = None
        self.dedispersed = None
        self.dmt = None
        self.device = device
        self.min_samp = min_samp
        self.dm_opt = -1
        self.snr_opt = -1
        self.kill_mask = kill_mask

    def save_h5(self, out_dir=None, fnout=None):
        """
        Generates an h5 file of the candidate object
        :param out_dir: Output directory to save the h5 file
        :param fnout: Output name of the candidate file
        :return:
        """
        cand_id = self.id
        if fnout is None:
            fnout = cand_id + '.h5'
        if out_dir is not None:
            fnout = out_dir + fnout
        with h5py.File(fnout, 'w') as f:
            f.attrs['cand_id'] = cand_id
            f.attrs['tcand'] = self.tcand
            f.attrs['dm'] = self.dm
            f.attrs['dm_opt'] = self.dm_opt
            f.attrs['snr'] = self.snr
            f.attrs['snr_opt'] = self.snr_opt
            f.attrs['width'] = self.width
            f.attrs['label'] = self.label

            # Copy over header information as attributes
            for key in list(self._type.keys()):
                if getattr(self, key) is not None:
                    f.attrs[key] = getattr(self, key)
                else:
                    f.attrs[key] = b'None'

            freq_time_dset = f.create_dataset('data_freq_time', data=self.dedispersed, dtype=self.dedispersed.dtype,
                                              compression="lzf")
            freq_time_dset.dims[0].label = b"time"
            freq_time_dset.dims[1].label = b"frequency"

            if self.dmt is not None:
                dm_time_dset = f.create_dataset('data_dm_time', data=self.dmt, dtype=self.dmt.dtype, compression="lzf")
                dm_time_dset.dims[0].label = b"dm"
                dm_time_dset.dims[1].label = b"time"
        return fnout

    def dispersion_delay(self, dms=None):
        """
        Calculates the dispersion delay at a specified DM
        :param dms: DM value to get dispersion delay
        :return:
        """
        if dms is None:
            dms = self.dm
        if dms is None:
            return None
        else:
            return 4148808.0 * dms * (1 / np.min(self.chan_freqs) ** 2 - 1 / np.max(self.chan_freqs) ** 2) / 1000

    def get_chunk(self, tstart=None, tstop=None):
        """
        Read the data around the candidate from the filterbank
        :param tstart: Start time in the fiterbank, to read from
        :param tstop: End time in the filterbank, to read till
        :return:
        """
        if self.data_source == "国台天文台":
            self.data_origin = copy.deepcopy(self._mmdata)
            self.median = np.median(self.data_origin)
            self.std= np.std(self.data_origin)
#            self.max = np.max(self.data_origin)
#            self.min = np.min(self.data_origin)
            
            self.data_origin = (self.data_origin - self.median)/self.std
#            self.data_origin = (self.data_origin - self.min)/(self.max - self.min)                  
#            self.data = cv2.resize(self.data_origin, (256,256), cv2.INTER_NEAREST)
            self.data = resize(self.data_origin, (256, 256))
            return self
        
        if tstart is None:
            tstart = self.tcand - self.dispersion_delay() - self.width * self.tsamp
            #if tstart < 0:
            #    tstart = 0
        if tstop is None:
            tstop = self.tcand + self.dispersion_delay() + self.width * self.tsamp
            #if tstop > self.tend:
            #    tstop = self.tend
        nstart = int(tstart / self.tsamp)
        nsamp = int((tstop - tstart) / self.tsamp)
        if self.width < 2:
            nchunk = self.min_samp
        else:
            nchunk = self.width * self.min_samp // 2
        if nsamp < nchunk:
            # if number of time samples less than nchunk, make it min_samp.
            nstart -= (nchunk - nsamp) // 2
            nsamp = nchunk
        if nstart < 0:
            self.data = self.get_data(nstart=0, nsamp=nsamp + nstart)[:, 0, :]
            logging.debug('median padding data as nstart < 0')
            self.data = pad_along_axis(self.data, nsamp, loc='start', axis=0, mode='median')
        elif nstart + nsamp > self.nspectra:
            self.data = self.get_data(nstart=nstart, nsamp=self.nspectra - nstart)[:, 0, :]
            logging.debug('median padding data as nstop > nspectra')
            self.data = pad_along_axis(self.data, nsamp, loc='end', axis=0, mode='median')
        else:
            self.data = self.get_data(nstart=nstart, nsamp=nsamp)[:, 0, :]

        if self.kill_mask is not None:
            assert len(self.kill_mask) == self.data.shape[1]
            data_copy = self.data.copy()
            data_copy[:, self.kill_mask] = 0
            self.data = data_copy
            del data_copy
        return self

    def dedisperse(self, dms=None, target='CPU'):
        """
        Dedisperse Frequency time data at a specified DM
        :param dms: DM to dedisperse at
        :return:
        """
        if dms is None:
            dms = self.dm
        #print("here :")
        if self.data is not None:
            if target == 'CPU':
                nt, nf = self.data.shape
                #assert nf == len(self.chan_freqs)
                delay_time = 4148808.0 * dms * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2) / 1000
                #print("delay_time:", delay_time)
                delay_bins = np.round(delay_time / self.tsamp).astype('int64')
                #print("delay_bins:", delay_bins)
                self.dedispersed = np.zeros(self.data.shape, dtype=np.float32)
                for ii in range(nf):
                    self.dedispersed[:, ii] = np.concatenate(
                        [self.data[-delay_bins[ii]:, ii], self.data[:-delay_bins[ii], ii]])
            elif target == 'GPU':
                from gpu_utils import gpu_dedisperse
                gpu_dedisperse(self, device=self.device)
        else:
            self.dedispersed = None
        return self

    def dedispersets(self, dms=None):
        """
        Dedisperse Frequency time data at a specified DM and return a time series
        :param dms: DM to dedisperse at
        :return: time series
        """
        if dms is None:
            dms = self.dm
        if self.data is not None:
            nt, nf = self.data.shape
            #assert nf == len(self.chan_freqs)
            delay_time = 4148808.0 * dms * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2) / 1000
            
            #print("delay_time:", delay_time, " delay_bins:", self.tsamp)
            
            delay_bins = np.round(delay_time / self.tsamp).astype('int64')
            
            ts = np.zeros(nt, dtype=np.float32)
            for ii in range(nf):
                ts += np.concatenate([self.data[-delay_bins[ii]:, ii], self.data[:-delay_bins[ii], ii]])
            return ts

    def dmtime(self, dmsteps=256, target='CPU'):
        """
        Generates DM-time array of the candidate by dedispersing at adjacent DM values
        dmsteps: Number of DMs to dedisperse at
        :return:
        """
        if target == 'CPU':
            range_dm = self.dm
            dm_list = self.dm + np.linspace(-range_dm, range_dm, dmsteps)
            self.dmt = np.zeros((dmsteps, self.data.shape[0]), dtype=np.float32)
#            print(self.dmt)
            for ii, dm in enumerate(dm_list):
                temp = self.dedispersets(dms=dm)
                self.dmt[ii, :] = temp
                #print( temp.shape) 
            #print(self.dmt)
        elif target == 'GPU':
            from gpu_utils import gpu_dmt
            gpu_dmt(self, device=self.device)
        return self

    def get_snr(self, time_series=None):
        """
        Calculates the SNR of the candidate
        :param time_series: time series array to calculate the SNR of
        :return:
        """
        if time_series is None and self.dedispersed is None:
            return None
        if time_series is None:
            x = self.dedispersed.mean(1)
        else:
            x = time_series
        argmax = np.argmax(x)
        mask = np.ones(len(x), dtype=np.bool)
        mask[argmax - self.width // 2:argmax + self.width // 2] = 0
        x -= x[mask].mean()
        std = np.std(x[mask])
        return x.max() / std

    def optimize_dm(self):
        """
        Calculate more precise value of the DM by interpolating between DM values to maximise the SNR
        This function has not been fully tested.
        :return: optimnised DM, optimised SNR
        """
        if self.data is None:
            return None

        def dm2snr(dm):
            time_series = self.dedispersets(dm)
            return -self.get_snr(time_series)

        try:
            out = golden(dm2snr, full_output=True, brack=(-self.dm / 2, self.dm, 2 * self.dm), tol=1e-3)
        except (ValueError, TypeError):
            out = golden(dm2snr, full_output=True, tol=1e-3)
        self.dm_opt = out[0]
        self.snr_opt = -out[1]
        return out[0], -out[1]

    def decimate(self, key, decimate_factor, axis, pad=False, **kwargs):
        """
        TODO: Update candidate parameters as per decimation factor
        :param key: Keywords to chose which data to decimate
        :param decimate_factor: Number of samples to average
        :param axis: Axis to decimate along
        :param pad: Optional argument if padding is to be done
        :args: arguments for numpy pad
        :return:
        """
        if key == 'dmt':
            logging.debug(
                f'Decimating dmt along axis {axis}, with factor {decimate_factor},  pre-decimation shape: {self.dmt.shape}')
            self.dmt = _decimate(self.dmt, decimate_factor, axis, pad, **kwargs)
            logging.debug(f'Decimated dmt along axis {axis}, post-decimation shape: {self.dmt.shape}')
        elif key == 'ft':
            logging.debug(
                f'Decimating ft along axis {axis}, with factor {decimate_factor}, pre-decimation shape: {self.dedispersed.shape}')
            self.dedispersed = _decimate(self.dedispersed, decimate_factor, axis, pad, **kwargs)
            logging.debug(f'Decimated ft along axis {axis}, post-decimation shape: {self.dedispersed.shape}')
        else:
            raise AttributeError('Key can either be "dmt": DM-Time or "ft": Frequency-Time')
        return self

    def resize(self, key, size, axis, **kwargs):
        """
        TODO: Update candidate parameters as per final size
        :param key: Keywords to chose which data to decimate
        :param size: Final size of the data array required
        :param axis: Axis to resize alone
        :param args: Arguments for skimage.transform resize function
        :return:
        """
        if key == 'dmt':
            self.dmt = _resize(self.dmt, size, axis, **kwargs)
        elif key == 'ft':
            self.dedispersed = _resize(self.dedispersed, size, axis, **kwargs)
        else:
            raise AttributeError('Key can either be "dmt": DM-Time or "ft": Frequency-Time')
        return self
