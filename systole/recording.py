# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import socket
import time
from struct import unpack
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from systole.detection import ppg_peaks
from systole.plots import plot_events, plot_raw


class Oximeter:
    """Recording PPG signal with Nonin pulse oximeter.

    Attributes
    ----------
    instant_rr :
        Time serie of instantaneous heartrate.
    n_channels : int | None
        Number of additional channels.
    recording :
        Time serie of PPG signal.
    sfreq :
        Sampling frequnecy. Default value is 75 Hz.
    threshold :
        The threshold used to detect beat peaks. Will use the average +
        standars deviation.
    times :
        Time vector (in seconds).
    diff :
        Records the differential of the PPG signal. Used to detect heartbeat
        peaks.
    peaks :
        List of 0 and 1. 1 index detected peaks.
    channels :
        Additional channels to record. Will continuously record *n_channels*
        additional channels in parallel of `recording` with default `0` as
        defalut value.
    serial :
        PySerial object indexing the USB port to read.
    rr :
        RR intervals time course. The time course will be generated if
        :py:func:`self.find_peaks` is used.

    Examples
    --------
    First, you will need to define a :py:func:`serial` instance, indexing the
    USB port where the Nonin Pulse Oximeter is plugged.

    >>> import serial
    >>> ser = serial.Serial('COM4')

    This instance is then used to create an :py:func:`Oximeter` instance that
    will be used for the recording.

    >>> from systole.recording import Oximeter
    >>> oximeter = Oximeter(serial=ser, sfreq=75)

    Use the :py:func:`setup` method to initialize the recording. This will find
    the start byte to ensure that all the forthcoming data is in Synch. You
    should not wait more than ~10s between the setup and the recording,
    otherwise the buffer will start to overwrite the data.

    >>> oximeter.setup()

    Two methods are availlable to record PPG signal:

    1. The :py:func:`read` function.

    Will continuously record for certain amount of time (specified by the
    *duration* parameter, in seconds). This is the easiest and most robust
    method, but it is not possible to run instructions in the meantime.

    >>> oximeter.read(duration=10)

    2. The :py:func:`readInWaiting` function.

    Will read all the availlable bytes (up to 10 seconds of recording). When
    inserted into a while loop, it allows to record PPG signal together with
    other scripts.

    >>> import time
    >>> tstart = time.time()
    >>> while time.time() - tstart < 10:
    >>>     oximeter.readInWaiting()
    >>>     # Insert code here

    The recorded signal can latter be inspected using the :py:func:`plot()`
    method.

    >>> oximeter.plot()

    .. warning:: Data read from the serial port are appended to list and
      processed for pulse detection and instantaneous heart rate estimation.
      The time required to append new data to the recording will increase as
      its size increase. You should beware that this processing time does not
      exceed the sampling frequency (i.e. 75Hz or 0.013 seconds per sample for
      Nonin pulse oximeters) to allow continuous recording and fast processing
      of in waiting samples. We recommend storing regularly 5 minutes recording
      as .npy file using the :py:func:save() function.
    """

    def __init__(
        self,
        serial,
        sfreq: int = 75,
        add_channels: Optional[int] = None,
        data_format: str = "2",
    ):
        """
        Parameters
        ----------
        serial :
            The `serial` instance interfacing with the USB port.
        sfreq :
            The sampling frequency of the recording. Defautl is 75 Hz.
        add_channels :
            If int, will create as many additionnal channels. If None, no
            additional channels created.
        data_format :
            Data format returned by the USB dongle ("2" or "7"). See
            https://www.nonin.com/wp-content/uploads/6000-7000-CP-7602-000-11_ENG.pdf
            for details. The pulse waveform value is automatically normalized and
            range between 0 and 255 both for data format "2" and "7".
        """
        self.reset(serial, sfreq, add_channels, data_format)

    def reset(
        self,
        serial,
        sfreq: int = 75,
        add_channels: Optional[int] = None,
        data_format: str = "2",
    ):
        """Initialize/restart the recording instance.

        Parameters
        ----------
        serial :
            The `serial` instance interfacing with the USB port.
        sfreq :
            The sampling frequency of the recording. Defautl is 75 Hz.
        add_channels :
            If int, will create as many additionnal channels. If None, no
            additional channels created.
        data_format :
            Data format returned by the USB dongle ("2" or "7"). See
            https://www.nonin.com/wp-content/uploads/6000-7000-CP-7602-000-11_ENG.pdf
            for details. The pulse waveform value is automatically normalized and
            range between 0 and 255 both for data format "2" and "7".
        """
        self.serial = serial
        self.lag = 0
        self.sfreq = sfreq
        self.dist = int(self.sfreq * 0.2)

        # Initialize recording with empty lists
        self.instant_rr: List[float] = []
        self.recording: List[float] = []
        self.times: List[float] = []
        self.n_channels: Optional[int] = add_channels
        self.threshold: List[float] = []
        self.diff: List[float] = []
        self.peaks: List[int] = []
        if add_channels is not None:
            self.channels: Optional[Dict[str, List]] = {}
            for i in range(add_channels):
                self.channels[f"Channel_{i}"] = []
        else:
            self.channels = None

        # Set the get value function depending on the data format
        self.data_format = data_format
        if data_format == "2":
            self.get_value = self.data_format2
        elif data_format == "7":
            self.get_value = self.data_format7
        else:
            raise ValueError('Data format should be "2" or "7"')

        return self

    def add_paquet(self, value: int, window: float = 1.0):
        """Read a portion of data.

        Parameters
        ----------
        value :
            The data to record. Should be an integer between 0 and 255.
        window :
            Length of the window used to compute threshold (seconds). Default
            is `1.0`.

        Notes
        -----
        Will automatically calculate the differential, threshold and increment
        additional channles with 0 if provided.
        """

        # Store new data
        self.recording.append(value)
        self.peaks.append(0)

        # Add 0 to the additional channels
        if self.channels is not None:
            for ch in self.channels:
                self.channels[ch].append(0)

        # Update times vector
        if not self.times:
            self.times = [0]
        else:
            self.times.append(len(self.times) / self.sfreq)

        # Update threshold
        window = int(window * self.sfreq)
        new_threshold = float(
            np.mean(self.recording[-window:]) + np.std(self.recording[-window:])
        )
        self.threshold.append(new_threshold)

        # Store new differential if not exist
        if not self.diff:
            self.diff = [0]
        else:
            self.diff.append(self.recording[-1] - self.recording[-2])

            # Is it a threshold crossing value?
            if value > self.threshold[-1]:
                # Is the new differential zero or crossing zero?
                if (self.diff[-1] <= 0) & (self.diff[-2] > 0):
                    # Is it far enough from the previous peak (0.2 s)?
                    if not any(self.peaks[-15:]):
                        self.peaks[-1] = 1

        # Update instantaneous heart rate
        if sum(self.peaks) >= 2:
            self.instant_rr.append(
                (np.diff(np.where(self.peaks)[0])[-1] / self.sfreq) * 1000
            )
        else:
            self.instant_rr.append(float("nan"))

        return self

    def check(self, paquet: list):
        """Check if the provided paquet is correct

        Parameters
        ----------
        paquet :
            The paquet to inspect.
        """
        check = False
        if len(paquet) >= 5:
            if (paquet[0] == 1) | (paquet[0] >= 128):
                if (paquet[1] >= 0) & (paquet[1] <= 255):
                    if (paquet[2] >= 0) & (paquet[2] <= 255):
                        if paquet[3] <= 127:
                            if paquet[4] == sum(paquet[:4]) % 256:
                                check = True

        return check

    def data_format2(self, paquet):
        """Extract pulse waveform value for data format 2.

        Parameters
        ----------
        paquet :
            A list containg 5 items.
        """
        return paquet[2]

    def data_format7(self, paquet):
        """Extract pulse waveform value for data format 7.

        Parameters
        ----------
        paquet :
            A list containg 5 items.
        """
        return ((paquet[1] * 256 + paquet[2]) / 65535) * 255

    def find_peaks(self, **kwargs):
        """Find peaks in recorded signal.

        Returns
        -------
        Oximeter instance. The peaks occurences are stored in the `peaks` attribute.

        Other Parameters
        ----------------
        **kwargs ::py:func:`systole.detection.ppg_peaks` properties.
        """
        # Peak detection
        resampled_signal, peaks = ppg_peaks(
            self.recording, sfreq=75, new_sfreq=75, **kwargs
        )

        # R-R intervals (in miliseconds)
        self.rr = (np.diff(np.where(peaks)[0]) / self.sfreq) * 1000

        # Beats per minutes
        self.bpm = 60000 / self.rr

        return self

    def plot_events(self, n_channel: str = "Channel_0", **kwargs):
        """Visualize the distribution of events stored in additional channels.

        Parameters
        ----------
        n_channel :
            The name of the channel encoding the events of interest. Defaults to
            `""Channel_0""`.
        kwargs:
            Other keyword arguments are passed down to
            py:`func:systole.plots.plot_evoked()`.

        Returns
        -------
        fig, ax :
            The figure and axe instances.
        """
        if self.channels is not None:
            triggers = self.channels[n_channel]

        return plot_events(triggers=triggers, sfreq=75, **kwargs)

    def plot_raw(self, **kwargs):
        """Plot the raw PPG signal.

        Parameters
        ----------
        **kwargs :
            Additional arguments will be passed to `:py:func:systole.plots.plot_raw`.

        Returns
        -------
        plot :
            The matplotlib axes, or the boken figure containing the plot.
        """

        return plot_raw(signal=self.recording, sfreq=75, **kwargs)

    def read(self, duration: float):
        """Read PPG signal for some amount of time.

        Parameters
        ----------
        duration :
            Length of the desired recording time.
        """
        tstart = time.time()
        while time.time() - tstart < duration:
            if self.serial.inWaiting() >= 5:
                # Store Oxi level
                paquet = list(self.serial.read(5))
                if self.check(paquet):
                    self.add_paquet(value=self.get_value(paquet))
                else:
                    self.setup()
        return self

    def readInWaiting(self, stop: bool = False):
        """Read in wainting oxi data.

        Parameters
        ----------
        stop :
            Stop the recording when an error is detected. Default is *False*.
        """
        # Read oxi
        while self.serial.inWaiting() >= 5:
            # Store Oxi level
            paquet = list(self.serial.read(5))
            if self.check(paquet):
                self.add_paquet(value=self.get_value(paquet))
            else:
                if stop is True:
                    raise ValueError("Synch error")
                else:
                    print("Synch error")
                    while True:
                        self.serial.reset_input_buffer()
                        paquet = list(self.serial.read(5))
                        if self.check(paquet=paquet):
                            break

    def save(self, fname: str):
        """Save the recording instance.

        Parameters
        ----------
        fname :
            The file name. The file extension can be `.npy` for
            :class:`numpy.array` or `.txt` for :class:`pandas.DataFrame`. If
            no extension is provided, will use the `.npy` extension by default.

        Notes
        -----
        If the signal is saved as a :class:`pandas.DataFrame`, the resulting data
        frame will contain the following columns:
            * `signal`
            * `peaks`
            * `instant_rr`
            * `time`
        If stim channels are provided, additional columns are appended as Channel_`i`,
        for `i` additional channels.

        If the signal is saved as a :class:`numpy.array`, the first dimension
        will encode the channels in that order, and the second dimension the
        samples.
        """
        # Sanity checks
        if len(self.peaks) != len(self.recording):
            self.peak = [0 * len(self.recording)]

        if len(self.instant_rr) != len(self.recording):
            self.instant_rr = [0 * len(self.recording)]

        if len(self.times) != len(self.recording):
            self.times = [0 * len(self.recording)]

        # Data that should be saved
        saveList = [
            np.asarray(self.recording),
            np.asarray(self.peaks),
            np.asarray(self.instant_rr),
            np.asarray(self.times),
        ]

        # Add stim channels if provided
        if self.channels is not None:
            for i in range(len(self.channels)):
                if len(self.channels[f"Channel_{i}"]) != len(self.recording):
                    self.channels["Channel_" + str(i)] = [0 * len(self.recording)]
                saveList.append(np.asarray(self.channels[f"Channel_{i}"]))

        # Check data format and save
        if fname.endswith(".txt"):
            colnames = ["signal", "peaks", "instant_rr", "time"]
            if self.n_channels:
                for i in range(self.n_channels):
                    colnames.extend(["Channel_" + str(i)])
            pd.DataFrame(np.array(saveList).T, columns=colnames).to_csv(
                fname, index=False
            )
        else:
            recording = np.array(saveList)
            np.save(fname, recording)

    def setup(
        self, read_duration: float = 1.0, clear_peaks: bool = True, nAttempts: int = 100
    ):
        """Find start byte and read a portion of signal.

        Parameters
        ----------
        read_duration :
            Length of signal to record after setup. Default is set to 1 second.
        clear_peaks :
            If *True*, will remove detected peaks.
        nAttempts :
            Number of attempts to read pulse oximeter signal from the USB. If no
            readable signal has been receive after `nAttemps`, a RuntimeError is raised.

        Notes
        -----
        .. warning:: setup() clear the input buffer and will remove previously
        recorded data from the Oximeter instance. Peaks detected during this
        procedure are automatically removed.
        """
        # Reset recording instance
        self.reset(
            serial=self.serial,
            add_channels=self.n_channels,
            data_format=self.data_format,
        )
        completed, i = False, 0
        while True:
            i += 1
            self.serial.reset_input_buffer()
            paquet = list(self.serial.read(5))
            if self.check(paquet=paquet):
                completed = True
                break
            if i > nAttempts:
                break
        if completed is False:
            raise RuntimeError("Unable to read signal from the USB port.")
        self.read(duration=read_duration)

        # Remove peaks
        if clear_peaks is True:
            self.peaks = [0] * len(self.peaks)

        return self

    def waitBeat(self):
        """Read Oximeter until a heartbeat is detected."""
        while True:
            if self.serial.inWaiting() >= 5:
                # Store Oxi level
                paquet = list(self.serial.read(5))
                if self.check(paquet):
                    self.add_paquet(value=self.get_value(paquet))
                    if any(self.peaks[-2:]):  # Peak found
                        break
                else:
                    print("Synch error")
        return self


class BrainVisionExG:
    """Recording ECG signal through TCPIP.

    Parameters
    ----------
    ip :
        The IP address of the recording computer.
    sfreq :
        The sampling frequency.
    port :
        The port to listen. Default is 51244 (32 bits). Change port to 51234 to
        connect to 16Bit RDA-port

    Examples
    --------
    This instance is then used to create an :py:func:`BrainVisionExG` instance
    that will be used for the recording.

    >>> from ecg.recording import BrainVisionExG
    >>> exg = BrainVisionExG(ip='xxx.xxx.xx', sfreq=1000).read(30)

    Use the :py:func:`read` method to record some signal and save it in the
    `exg` dictionary.

    .. warning:: The signals received fom the host are appened to a list. This
       process can require more time at each iteration as the signal length
       increase in memory. You should alway make sure that this will not
       interfer with other task and regularly save intermediate recording to
       save resources.

    Notes
    -----
    This class is adapted from the RDA client for python made available by
    Brain Products on the following link: https://www.brainproducts.com/downloads.php?kid=2
    """

    def __init__(self, ip, sfreq, port=51244):
        self.ip = ip
        self.port = port
        self.sfreq = sfreq
        self.dist = int(self.sfreq * 0.2)
        self.recording = []

        # Create a tcpip socket
        self.con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # block counter to check overflows of tcpip buffer
        self.lastBlock = -1

        # Connect to recorder host
        self.con.connect((self.ip, self.port))

        # Marker dict for storing marker information
        self.marker = {
            "position": 0,
            "points": 0,
            "channel": -1,
            "type": "",
            "description": "",
        }

    def RecvData(self, requestedSize):
        """Helper function for receiving whole message"""
        returnStream = b""
        while len(returnStream) < requestedSize:
            databytes = self.con.recv(requestedSize - len(returnStream))
            if not databytes:
                raise RuntimeError
            returnStream += databytes

        return returnStream

    def SplitString(self, raw):
        """Helper function for splitting a raw array of zero terminated strings (C) into
        an array of python strings"""

        raw = raw.decode()
        stringlist = []
        s = ""
        for i in range(len(raw)):
            if raw[i] != "\x00":
                s = s + raw[i]
            else:
                stringlist.append(s)
                s = ""
        return stringlist

    def GetProperties(self, rawdata):
        """Helper function for extracting ExG properties from a raw data array read from
        tcpip socket"""

        # Extract numerical data
        (channelCount, samplingInterval) = unpack("<Ld", rawdata[:12])

        # Extract resolutions
        resolutions = []
        for c in range(channelCount):
            index = 12 + c * 8
            restuple = unpack("<d", rawdata[index : index + 8])
            resolutions.append(restuple[0])

        # Extract channel names
        channelNames = self.SplitString(rawdata[12 + 8 * channelCount :])

        return (channelCount, samplingInterval, resolutions, channelNames)

    def GetData(self, rawdata, channelCount):
        """Helper function for extracting eeg and marker data from a raw data array read
        from tcpip socket"""

        # Extract numerical data
        (block, points, markerCount) = unpack("<LLL", rawdata[:12])

        # Extract eeg data as array of floats
        data = []
        for i in range(points * channelCount):
            index = 12 + 4 * i
            value = unpack("<f", rawdata[index : index + 4])
            data.append(value[0])

        # Extract markers
        markers = []
        index = 12 + 4 * points * channelCount
        for m in range(markerCount):
            markersize = unpack("<L", rawdata[index : index + 4])

            ma = self.marker.copy()
            (ma["position"], ma["points"], ma["channel"]) = unpack(
                "<LLl", rawdata[index + 4 : index + 16]
            )
            typedesc = self.SplitString(rawdata[index + 16 : index + markersize[0]])
            ma["type"] = typedesc[0]
            ma["description"] = typedesc[1]

            markers.append(ma)
            index = index + markersize[0]

        return (block, points, markerCount, data, markers)

    def read(self, duration):
        """Read incoming signals.

        Parameters
        ----------
        duration :
            The length of the recording.

        Returns
        -------
        recording :
            Dictionary with channel name as key.

        Notes
        -----
        Duration will be converted to expected signal length (duration * sfreq) to
        ensure consistent recording.
        """
        while True:
            # Get message header as raw array of chars
            rawhdr = self.RecvData(24)

            # Split array into usefull information id1 to id4 are constants
            (id1, id2, id3, id4, msgsize, msgtype) = unpack("<llllLL", rawhdr)

            # Get data part of message, which is of variable size
            rawdata = self.RecvData(msgsize - 24)

            # Perform action dependend on the message type
            if msgtype == 1:
                # Start message, extract eeg properties and display them
                (
                    channelCount,
                    samplingInterval,
                    resolutions,
                    channelNames,
                ) = self.GetProperties(rawdata)
                # reset block counter
                self.lastBlock = -1

                print(
                    "Reading TCP/IP connection ("
                    + str(channelCount)
                    + " channels found). "
                    + str(resolutions)
                    + " Hz. "
                    + str(samplingInterval)
                    + " samples. "
                    + str(channelNames)
                )

            elif msgtype == 4:
                # Data message, extract data and markers
                (block, points, markerCount, data, markers) = self.GetData(
                    rawdata, channelCount
                )

                # Check for overflow
                if self.lastBlock != -1 and block > self.lastBlock + 1:
                    print(
                        "*** Overflow with "
                        + str(block - self.lastBlock)
                        + " datablocks ***"
                    )
                self.lastBlock = block

                # Print markers, if there are some in actual block
                if markerCount > 0:
                    for m in range(markerCount):
                        print(
                            "Marker "
                            + markers[m]["description"]
                            + " of type "
                            + markers[m]["type"]
                        )

                # Put data at the end of actual buffer
                self.recording.extend(data)
                if ((len(self.recording) / self.sfreq) / channelCount) >= duration:
                    break
            elif msgtype == 3:
                # Stop message, terminate program
                print("Stop")

        recording = {}
        for ch_name, ch_nb in zip(channelNames, range(channelCount)):
            recording[ch_name] = np.array(self.recording[ch_nb::channelCount])

        return recording

    def close(self):
        """Close TCPIP connections"""
        self.con.close()





class Nonin3231USB:
    """Recording Nonin 3231 USB HR signal through USB connection

    Parameters
    ----------
    
    serial : pySerial object
        The `serial` instance interfacing with the USB port.
        

    Examples
    --------
    First, you will need to define a :py:func:`serial` instance, indexing the
    USB port where the Nonin 3231 Pulse Oximeter is plugged.

    >>> import serial
    >>> ser = serial.Serial('COM4')

    This instance is then used to create an :py:func:`Nonin3231USB` instance
    that will be used for the recording from a Nonin 3231 USB device.

    >>> from ecg.recording import Nonin3231USB
    >>> exg = Nonin3231USB(serial).read(30)

    Use the :py:func:`read` method to record some signal and save it in the
    `exg` dictionary.

    .. warning:: The signals received fom the host are appened to a list. This
       process can require more time at each iteration as the signal length
       increase in memory. You should alway make sure that this will not
       interfer with other task and regularly save intermediate recording to
       save resources.

    Notes
    -----
    """

    def __init__(
        self,
        serial,
        add_channels: Optional[int] = 1,
    ):
        self.reset(serial, add_channels)

    def reset(
        self,
        serial,
        add_channels: Optional[int] = 1,
    ):
        """Initialize/restart the recording instance.

        Parameters
        ----------
        serial : pySerial object
            The `serial` instance interfacing with the USB port.

        Returns
        -------
        Nonin3231USB instance.
        """
        self.serial = serial

        # Initialize recording with empty lists
        self.recording: List[float] = []
        self.bpm: List[float] = []
        self.SpO2: List[float] = []
        self.times: List[float] = []
        self.n_channels: Optional[int] = add_channels
        if add_channels is not None:
            self.channels: Optional[Dict[str, List]] = {}
            for i in range(add_channels):
                self.channels[f"Channel_{i}"] = []
        else:
            self.channels = None
        
        # print('channels: '+str(self.channels))

        return self
    
    def setup(self):
        self.reset(
            serial=self.serial,
        )

        return self
    
    def read(self, duration: float):
        """Read PPG signal for some amount of time.

        Parameters
        ----------
        duration : int or float
            Length of the desired recording time.
        """
        # init start ime
        tstart = time.time()
        
        # read for length of duration 
        while time.time() - tstart < duration:
            # assert one full line of data is there
            if self.serial.inWaiting() >= 12:
                # Store line of data 
                data = list(self.serial.readline())
                # assert full data line
                if len(data) < 12: 
                    continue
                # get bpm reading 
                self.bpm.append(data[8])
                # give bpm to recording as well
                self.recording.append(data[8])
                # get SpO2 reading (we don't use it)
                self.SpO2.append(data[6])
                # get 'time' reading (seconds) 
                self.times.append(data[5])
                # Add 0 to the additional channels
                if self.channels is not None:
                    for ch in self.channels:
                        self.channels[ch].append(0)
        
        return self
    
    
    def readInWaiting(self, bool = False):
        """Read in waiting Nonin data.

        Parameters
        ----------
        stop : bool
            Stop the recording when an error is detected. Default is *False*.
        """

        #  nonin device reads out 12 bits per message
        while self.serial.inWaiting() >= 12:

            # Store full message line
            data = list(self.serial.readline())
            # assert that there is a full line of data
            if len(data) < 12: 
                continue
            # get bpm reading 
            self.bpm.append(data[8])
            # give bpm to recording as well
            self.recording.append(data[8])
            # get SpO2 reading (we don't use it)
            self.SpO2.append(data[6])
            # get 'time' reading (seconds) 
            self.times.append(data[5])
            # Add 0 to the additional channels for triggers
            if self.channels is not None:
                for ch in self.channels:
                    self.channels[ch].append(0)
        

    
    def save(self, fname: str):
        """
        Not saving the data as of now, only results
        """
        None

    
    def close(self):
        """Close serial connections"""
        self.serial.close()
