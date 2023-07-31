#!/usr/bin/python
##############################################################
# Trilogy - Color / grayscale image maker (from FITS input)
# Dan Coe
# http://www.stsci.edu/~dcoe/trilogy/
##############################################################

# Trilogy produces nice color / grayscale images based on your input FITS files
# Just tell the input file (e.g., trilogy.in)
#   which images you'd like applied to which channel (RGB)

# When assigning multiple filters to a channel (R,G,B)
#   Trilogy currently adds the data from those filters

# If you do not specify channels, a grayscale image will be made

# To determine scaling,
#   Samples a (samplesize,samplesize) section of the image core (center)

#################################
# Usage:

# alias trilogy python ~/trilogy/trilogy.py  (or wherever you put it)

# trilogy
# - Assumes an input file trilogy.in

# trilogy acsir.in
# - sets the input file; output name is acsir.png by default

# trilogy acsir.in -indir ../images -outname a383 -showwith PIL -sampledx 300
# - set the input file and override some options from the command line

#################################
# Example input file (everything between the quotes)
# and default values (beginning with outdir):
"""
B
a383_uvis_f225w.fits
a383_uvis_f275w.fits
a383_uvis_f336w.fits
a383_uvis_f390w.fits

G
a383_acs_f435w.fits
a383_acs_f475w.fits
a383_acs_f606w.fits
a383_acs_f625w.fits
a383_acs_f775w.fits
a383_acs_f814w.fits
a383_acs_f850lp.fits

R
a383_ir_f105w.fits
a383_ir_f110w.fits
a383_ir_f125w.fits
a383_ir_f140w.fits
a383_ir_f160w.fits

indir  /Users/dcoe/CLASH/a383/
outname  a383
outdir   .
samplesize 1000
sampledx  0
sampledy  0
stampsize  1000
showstamps  0
satpercent  0.001
noiselum    0.10
colorsatfac  1
deletetests  0
testfirst    1
show  1
showstamps  0
showwith  open
deletetests  0
scaling  None
legend  1
thumbnail  None
maxstampsize  6000
"""

# Can also set noiselum individually for each filter with:
# noiselums 0.10,0.15,0.20

# Some explanations:
"""
    'noiselum':0.10,  # Make higher to dig into noise more (between 0 - 1)
    'satpercent':0.001,  # *Percentage* of pixels which will be saturated
    # (satpercent = 0.001 means 1 / 100,000 pixels will be saturated)
    'samplesize':1000,  # to determine levels
    'sampledx':0,  # offset
    'sampledy':0,  # offset
    'stampsize': 1000,  # for making final color image one section at a time (just a memory issue)
    'maxstampsize':6000,   # My memory can't handle an array larger than 6000x6000
    'showwith':'open',  # Command to display images; set to 0 to display with PIL (as lossy jpeg)
    'scaling':None,  # Use an input scaling levels file
    'legend':1,  # Adds legend to top-left corner indicating which filters were used
"""

#################################

# TO DO:
# Images larger than 6000
# More robust input? if just input 3 fits files, have them be RGB?
# Change temperature to make image redder/bluer, if desired
# I should allow for input of a weight image and make use of it, but I don't currently
# better way to combine images within channel?
# allow one to add a constant to the image before scaling

#################################
# Requirements (python libraries):
# PIL - Python Image Library
# pyfits - FITS handler
# numpy - handles data arrays
# scipy - "golden" root finder (maybe there's another way that doesn't require scipy?)

#################################
# Log scaling constrained at 3 data points: "tri-log-y"

# Inputs:
# % of pixels that saturate
# output brightness of noise
# color saturation boost

# y = np.log10( k * (x - xo) + 1 ) / r

# input data (x) -> output scaling (y) from 0-1 (in gray / color image)
# x0 yields 0
# x1 yields y1
# x2 yields 1

# Current settings:
# x0: 0 (0 in the input yields black in the output)
# x1: mean + std (1-sigma above the noise)
# x2: set so only some small fraction of pixels saturate (with output = 1)

# DERIVATION

# np.log10( k * (x - xo) + 1 ) / r
# x0, x1, x2  YIELD  0, 0.5, 1,  RESPECTIVELY

# (0): np.log10( k (x0 - xo) + 1 ) / r = 0
# (1): np.log10( k (x1 - xo) + 1 ) / r = 0.5
# (2): np.log10( k (x2 - xo) + 1 ) / r = 1

# (0)        gives xo = x0
# (2) - (0)  gives r = log( k (x2-x0) + 1 )
# (2) = 2(1) gives k = (x2 - 2*x1 + x0) / (x2 - x0)**2

# This is not easily generalized to output values other than (0, 0.5, 1)
# The first two require y0 = 0
# The last step is possible because y2 / y1 = 1 / 0.5 = 2

# Of course one could always solve numerically...

#################################
# More resources:

# Robert Lupton
# http://www.astro.princeton.edu/~rhl/PrettyPictures/

# Robert Hurt
# http://hea-www.harvard.edu/~ascpub/viztalks/talks/Chandra%20Dynamic%20Range.ppt

#################################

import sys
from os.path import join
import astropy.io.fits as pyfits
import string
import numpy as np
from PIL import Image, ImageDraw
import os, sys
from scipy.optimize import golden
from os.path import exists, join
from glob import glob


defaultvalues = {
    'indir':'',
    #'outname':'trilogy',
    'outname':None,
    'outdir':'',
    'noiselum':0.15,  # output luminosity of "noise" (between 0 - 1)
    'noiselums':{},   # set independently for each channel
    'satpercent':0.001,  # *Percentage* of pixels which will be saturated
    # (satpercent = 0.001 means 1 / 100,000 pixels will be saturated)
    'colorsatfac':1,  # > 1 to boost color saturation
    'thumbnail':None,
    'samplesize':1000,  # to determine levels
    'sampledx':0,  # offset
    'sampledy':0,  # offset
    'stampsize': 1000,  # for making final color image (just a memory issue)
    'testfirst':1,
    'show':1,
    'showstamps':0,
    'showwith':'open',  # Command to display images; set to 0 to display with PIL (as lossy jpeg)
    'deletetests':0,
    'scaling':None,
    'maxstampsize':6000,   # My memory can't handle an array larger than 6000x6000
    'legend':1,  # Adds legend to top-left corner indicating which filters were used
    'invert':0,  # Invert luminosity (black on white)
    'combine':'average',  # average or sum.  sum was the previous default (not explicitly defined)
    'noise':None,     # determined automatically if None: image data value of the "noise"
    'saturate':None,  # determined automatically if None: image data value allowed to saturate
    'bscale':1,  # Multiply all images by this value
    'bzero':0,   # Add this value to all images
    'correctbias':0,   # Measure data noise mean (otherwise assume = 0)
    'noisesig':1,    # Data noise level output to noiselum: measured sigma above the measured mean
    'noisesig0':2,   # Data noise level: measured sigma above the measured mean
    }

imfilt = ''  # Initialize

#################################
# A few general tools


class StatRobust: #meanstd_robust
    # Generates robust statistics using a sigma np.clipping
    # algorithm. It is controlled by the parameters n_sigma
    # and n, the number of iterations
    # - from Narciso Benitez
    def __init__(self, x, n_sigma=3, n=5, reject_fraction=None):
        self.x = x
        self.n_sigma = n_sigma
        self.n = n
        self.reject_fraction = reject_fraction

    def run(self):
        good = np.ones(len(self.x))
        nx = np.sum(good)
        if self.reject_fraction is None:
            for i in range(self.n):
                if i > 0:
                    xs = np.compress(good, self.x)
                else:
                    xs = self.x
                aver = np.median(xs)
                std1 = np.std(xs)
                good = good * np.less_equal(np.abs(self.x - aver), self.n_sigma * std1)
                nnx = np.sum(good)
                if nnx == nx:
                    break
                else:
                    nx = nnx
        else:
            np_float = float(len(self.x))
            nmin = int((0.5 * self.reject_fraction) * np_float)
            nmax = int((1. - 0.5 * self.reject_fraction) * np_float)
            orden = np.argsort(self.x)
            np.arange(len(self.x)), np.sort(self.x)
            good = np.greater(orden, nmin) * np.less(orden, nmax)

        self.remaining = np.compress(good, self.x)
        self.max_value = np.max(self.remaining)
        self.min_value = np.min(self.remaining)
        self.mean = np.mean(self.remaining)
        self.rms = np.std(self.remaining)
        # self.rms0 = rms(self.remaining)  # --DC
        self.median = np.median(self.remaining)
        self.outliers = np.compress(np.logical_not(good), self.x)
        self.n_remaining = len(self.remaining)
        self.n_outliers = len(self.outliers)
        self.fraction = 1. - (float(self.n_remaining) / float(len(self.x)))



def rms(x):
    return np.sqrt(np.mean(x**2))

class MeanStdRobust:
    # Generates robust statistics using a sigma np.clipping
    # algorithm. It is controlled by the parameters n_sigma
    # and n, the number of iterations
    # ADAPTED from Txitxo's stat_robust
    # Now much quicker for large arrays
    def __init__(self, x, n_sigma=3, n=5, sortedalready=False):
        self.x = x
        self.n_sigma = n_sigma
        self.n = n
        self.sortedalready = sortedalready

    def run(self):
        ihi = nx = len(self.x)
        ilo = 0
        # self.x[np.isnan(self.x)] = 0  # set all nan values to zero
        if not self.sortedalready:
            print('sorting...')
            self.xsort = np.sort(self.x)
        else:
            self.xsort = self.x
        for i in range(self.n):
            # print i
            xs = self.xsort[ilo:ihi]
            # print 'median'
            # aver = median(xs)
            # print xs
            # print xs[-1]
            # print xs[-2]
            # print len(xs)
            imed = (ilo + ihi) // 2
            # print imed
            aver = xs[imed]
            # print 'std'
            std1 = np.std(xs)
            std1 = rms(xs - aver)
            # print 'lohi'
            lo = aver - self.n_sigma * std1
            hi = aver + self.n_sigma * std1
            # print 'searching...'
            ilo = np.searchsorted(self.xsort, lo)
            ihi = np.searchsorted(self.xsort, hi, side='right')
            nnx = ihi - ilo
            # print ilo, ihi, nnx, nx, lo, hi
            if nnx == nx:
                break
            else:
                nx = nnx

        self.remaining = xrem = xs[ilo:ihi]
        self.mean = np.mean(xrem)
        self.std = rms(xrem - self.mean)


def strend(string, phrase):
    return string[-len(phrase):] == phrase

def decapfile(name, ext=''):
    """REMOVE EXTENSION FROM FILENAME IF PRESENT
    IF ext LEFT BLANK, THEN ANY EXTENSION WILL BE REMOVED"""
    if ext:
        if ext[0] != '.':
            ext = '.' + ext
        n = len(ext)
        if name[-n:] == ext:
            name = name[:-n]
    else:
        if strend(name, '.gz'):
            name = name[:-3]
        i = name.rfind('.')
        if i > -1:
            name = name[:i]
    return name

def stringsplitatof(string, separator=''):
    """Splits a string into floats"""
    if separator:
        words = string.split(separator)
    else:
        words = string.split()
    vals = []
    for word in words:
        vals.append(float(word))
    return vals

def str2num(string, rf=0):
    """CONVERTS A STRING TO A NUMBER (INT OR FLOAT) IF POSSIBLE
    ALSO RETURNS FORMAT IF rf=1"""
    try:
        num = int(string)
        format = 'd'
    except:
        try:
            num = float(string)
            format = 'f'
        except:
            if not string.strip():
                num = None
                format = ''
            else:
                words = string.split()
                if len(words) > 1:
                    num = list(map(str2num, tuple(words)))
                    format = 'l'
                else:
                    num = string
                    format = 's'
    if rf:
        return num, format
    else:
        return num

def clip2(m, m_min=None, m_max=None):
    if m_min is None:
        m_min = min(m)
    if m_max is None:
        m_max = max(m)
    return np.clip(m, m_min, m_max)

def striskey(string):
    """IS string AN OPTION LIKE -C or -ker
    (IT'S NOT IF IT'S -2 or -.9)"""
    is_key = 0
    if string:
        if string[0] == '-':
            is_key = 1
            if len(string) > 1:
                is_key = string[1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
    return is_key

def params_cl(convert_to_numbers=True):
    """RETURNS PARAMETERS FROM COMMAND LINE ('cl') AS DICTIONARY:
    KEYS ARE OPTIONS BEGINNING WITH '-'
    VALUES ARE WHATEVER FOLLOWS KEYS: EITHER NOTHING (''), A VALUE, OR A LIST OF VALUES
    ALL VALUES ARE CONVERTED TO INT / FLOAT WHEN APPROPRIATE"""
    list = sys.argv[:]
    i = 0
    dictionary = {}
    old_key = ""
    key = ""
    list.append('')  # EXTRA ELEMENT SO WE COME BACK AND ASSIGN THE LAST VALUE
    while i < len(list):
        if striskey(list[i]) or not list[i]:  # (or LAST VALUE)
            if key:  # ASSIGN VALUES TO OLD KEY
                if value:
                    if len(value) == 1:  # LIST OF 1 ELEMENT
                        value = value[0]  # JUST ELEMENT
                dictionary[key] = value
            if list[i]:
                key = list[i][1:]  # REMOVE LEADING '-'
                value = None
                dictionary[key] = value  # IN CASE THERE IS NO VALUE!
        else:  # VALUE (OR HAVEN'T GOTTEN TO KEYS)
            if key:  # (HAVE GOTTEN TO KEYS)
                if value:
                    if convert_to_numbers:
                        value.append(str2num(list[i]))
                    else:
                        value = value + ' ' + list[i]
                else:
                    if convert_to_numbers:
                        value = [str2num(list[i])]
                    else:
                        value = list[i]
        i += 1

    return dictionary

#################################
# TRILOGY-specific tools

def determinescaling(data, unsatpercent, noisesig=1, correctbias=True, noisesig0=2):
    """Determines data values (x0,x1,x2) which will be scaled to (0,noiselum,1)"""
    datasorted = np.sort(data.flat)
    datasorted[np.isnan(datasorted)] = 0  # set all nan values to zero
    if datasorted[0] == datasorted[-1]:
        levels = 0, 1, 100  # whatever
    else:
        s = MeanStdRobust(datasorted, sortedalready=True)
        s.run()
        m = s.mean
        r = s.std

        if correctbias:
            x0 = m - noisesig0 * r
        else:
            x0 = 0
        x1 = m + noisesig * r
        x2 = setlevels(datasorted, np.array([unsatpercent]), sortedalready=True)[0]
        levels = x0, x1, x2
    return levels

def setlevels(data, pp, stripneg=False, sortedalready=False):
    if sortedalready:
        vs = data
    else:
        print('sorting...')
        vs = np.sort(data.flat)
    if stripneg:
        i = np.searchsorted(vs, 0)
        vs = vs[i+1:]
    else:
        vs = np.clip(vs, 0, None)
    ii = np.array(pp) * len(vs)
    ii = ii.astype(int)
    ii = np.clip(ii, 0, len(vs)-1)
    levels = vs.take(ii)
    return levels

def imscale1(data, levels):
    x0, x1, x2 = levels
    k = (x2 - 2 * x1 + x0) / float(x1 - x0) ** 2
    r1 = np.log10(k * (x2 - x0) + 1)
    v = np.ravel(data)
    v = np.clip(v, 0, None)
    d = k * (v - x0) + 1
    d = np.clip(d, 1e-30, None)
    z = np.log10(d) / r1
    z = np.clip(z, 0, 1)
    z.shape = data.shape
    z = z * 255
    z = z.astype(int)
    return z




def da(k):
    a1 = k * (x1 - x0) + 1
    a2 = k * (x2 - x0) + 1
    a1n = a1**n
    a1n = np.abs(a1n)  # Don't want the solutions where a1 & a2 are both negative!
    da1 = a1n - a2
    k = np.abs(k)
    if k == 0:
        return da(1e-10)
    else:
        da1 = da1 / k  # To avoid solution k = 0!
    return np.abs(da1)

def imscale2(data, levels, y1):
    # x0, x1, x2  YIELD  0, y1, 1,  RESPECTIVELY
    global n, x0, x1, x2  # So that golden can use them
    x0, x1, x2 = levels
    if y1 == 0.5:
        k = (x2 - 2 * x1 + x0) / float(x1 - x0) ** 2
    else:
        n = 1 / y1
        k = np.abs(golden(da))
    r1 = np.log10(k * (x2 - x0) + 1)
    v = np.ravel(data)
    v = clip2(v, 0, None)
    d = k * (v - x0) + 1
    d = clip2(d, 1e-30, None)
    z = np.log10(d) / r1
    z = np.clip(z, 0, 1)
    z.shape = data.shape
    z = z * 255
    z = z.astype(np.uint8)
    return z

def satK2m(K):
    m00 = rw * (1-K) + K
    m01 = gw * (1-K)
    m02 = bw * (1-K)

    m10 = rw * (1-K)
    m11 = gw * (1-K) + K
    m12 = bw * (1-K)

    m20 = rw * (1-K)
    m21 = gw * (1-K)
    m22 = bw * (1-K) + K

    m = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
    return m

# Luminance vector
rw, gw, bw = 0.212671, 0.715160, 0.072169  # D65: red boosted, blue muted a bit, I like it

# Function to adjust the color saturation of an image
def adjsat(RGB, K):
    """Adjust the color saturation of an image.  K > 1 boosts it."""
    m = satK2m(K)
    three, nx, ny = RGB.shape
    RGB.shape = three, nx*ny
    RGB = np.dot(m, RGB)
    RGB.shape = three, nx, ny
    return RGB

# Function to convert RGB data to an Image object
def RGB2im(RGB):
    data = RGB
    data = np.transpose(data, (1, 2, 0))
    data = np.clip(data, 0, 255)
    data = data.astype(np.uint8)
    three = data.shape[-1]
    if three == 3:
        im = Image.fromarray(data)
    elif three == 1:
        im = Image.fromarray(data[:, :, 0], 'L')
    else:
        print('Data shape not understood: expect last number to be 3 for RGB, 1 for L', data.shape)
        raise Exception
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im

# Function to scale RGB data to an Image object with optional saturation adjustment and luminance inversion
def RGBscale2im(RGB, levdict, noiselums, colorsatfac, mode='RGB', invlum=0):
    three, nx, ny = RGB.shape
    if nx * ny > 2000 * 2000:
        print('Warning: You should probably feed smaller stamps into RGBscale2im.')
        print("This may take a while...")

    scaled = np.zeros(RGB.shape, float)
    for i in range(three):
        channel = mode[i]
        levels = levdict[channel]
        noiselum = noiselums[channel]
        scaled[i] = imscale2(RGB[i], levels, noiselum)

    if (colorsatfac != 1) and (mode == 'RGB'):
        scaled = adjsat(scaled, colorsatfac)

    if invlum:
        scaled = 255 - scaled

    im = RGB2im(scaled)
    return im

# Function to create a grayscale Image object from scaled data
def grayimage(scaled):
    ny, nx = scaled.shape
    im = Image.new('L', (nx, ny))
    im.putdata(scaled.np.ravel())
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im

# Function to create a grayscale Image object from a stamp using specific levels and noiselum
def grayscaledimage(stamp, levels, noiselum):
    global scaled
    scaled = imscale2(stamp, levels, noiselum)
    im = grayimage(scaled)
    return im

# Function to save level information to a text file
def savelevels(levdict, outfile='levels.txt', outdir=''):
    outfile = join(outdir, outfile)
    with open(outfile, 'w') as fout:
        for filt in levdict.keys():
            levels = x0, x1, x2 = levdict[filt]
            line = '%s  %g  %g  %g' % (filt, x0, x1, x2)
            fout.write(line + '\n')

# Function to load a file and return its content as a list of strings
def loadfile(filename, dir="", silent=0, keepnewlines=0):
    infile = join(dir, filename)
    if not silent:
        print("Loading ", infile, "...\n")
    with open(infile, 'r') as fin:
        sin = fin.readlines()
    if not keepnewlines:
        for i in range(len(sin)):
            sin[i] = sin[i][:-1]
    return sin
# Function to load a dictionary from a file
def loaddict(filename, dir="", silent=0):
    lines = loadfile(filename, dir, silent)
    dict = {}
    for line in lines:
        if line[0] != '#':
            words = line.split()
            key = str2num(words[0])
            val = ''  # if nothing there
            valstr = ' '.join(words[1:])
            valtuple = False
            valarray = True
            if valstr[0] in '[(' and valstr[-1] in '])':  # LIST / TUPLE!
                valtuple = valstr[0] == '('
                valstr = valstr[1:-1].replace(',', '')
                words[1:] = valstr.split()
            if len(words) == 2:
                val = str2num(words[1])
            elif len(words) > 2:
                val = []
                for word in words[1:]:
                    val.append(str2num(word))
                if valtuple:
                    val = tuple(val)
                if valarray:
                    val = np.array(val)

            dict[key] = val
    return dict

#################################
# Apply offsets
# Implement below??

offsets = {}
def offsetarray(data, offset):
    new = np.zeros(data.shape)
    dx, dy = offset
    if dy >= 0:
        if dx >= 0:
            new[dy:, dx:] = data[:-dy, :-dx]
        else:
            new[dy:, :-dx] = data[:-dy, dx:]
    else:
        if dx >= 0:
            new[:-dy, dx:] = data[dy:, :-dx]
        else:
            new[:-dy, :-dx] = data[dy:, dx:]
    return new

for channel in offsets.keys():
    dataRGB[channel] = offsetarray(dataRGB[channel], offsets[channel])

# Function to extract the filter from a FITS header
def extractfilter(header):
    """Extracts filter from a FITS header"""
    filt = header.get('FILTER', '')
    if filt == '':
        filt1 = header.get('FILTER1', '')
        if filt1 != '':
            if type(filt1) == str:
                if filt1[:5] == 'CLEAR':
                    filt2 = header.get('FILTER2', '')
                    filt = filt2
                else:
                    filt = filt1
    return filt

# Function to load FITS image data
def loadfitsimagedata(image, indir='', silent=1, bscale=1, bzero=0):
    global imfilt, imfilts
    if image[-1] == ']':
        iext = int(image[-2])
        image = image[:-3]  # Remove [0]
    else:
        iext = 0

    image0 = decapfile(image)

    image = join(indir, image)
    hdulist = pyfits.open(image, memmap=1)
    hdu = hdulist[iext]
    data = hdu.data
    if image0 in imfilts.keys():
        imfilt = imfilts[image0]
    else:
        imfilt = extractfilter(hdu.header)
        imfilts[image0] = imfilt
    if not silent:
        print(image + '[%d]' % iext, data.shape, imfilt)

    data = datascale(data, bscale, bzero)

    return data


imfilts = {}

# Function to process the image name, remove extensions, and store filters
def processimagename(image):
    global imfilts
    if image[-1] == ')':
        i = image.find('(')
        filt = image[i + 1:-1]
        image = image[:i]
        imfilts[image] = filt
    if image[-1] == ']':
        ext = image[-3:]
        image = image[:-3]
    else:
        ext = ''
    if not image.endswith('.fits') and not image.endswith('.fits.gz'):
        image += '.fits'
    image = image + ext
    return image

# Function to apply scaling to the data array
def datascale(data, bscale, bzero):
    if (bscale != 1) or (bzero != 0):
        return bscale * data + bzero
    else:
        return data


class Trilogy:
    def __init__(self, infile=None, images=None, imagesorder='BGR', **inparams):
        self.nx = None  # image size
        self.ny = None  # image size
        self.xlo = 0
        self.ylo = 0
        self.xhi = None
        self.yhi = None
        self.imagesRGB = {'R': [], 'G': [], 'B': [], 'L': []}  # File names
        self.inkeys = []
        self.mode = 'L'  # reset below if color
        self.weightext = None  # No weighting unless weight images are declared
        # Can use either:
        # weightext drz wht
        # weightext drz -> wht

        print('From input file', infile, ':')
        self.infile = infile
        if infile:
            self.loadinputs()

        self.images = images
        if images:
            self.setimages()

        self.inparams = inparams
        if inparams:
            self.setinparams()

        # if infile or inparams:
        self.setdefaults()

        # self.setoutfile()

    def setinparams(self):
        print('From input parameters:')
        bekeys = 'invert '.split()
        inkeys = list(self.inparams.keys())
        for key in inkeys:
            if key in bekeys:
                continue
            val = self.inparams[key]
            cmd = 'self.%s = val' % key
            print(key, '=', val)
            exec(cmd)
            self.inkeys.append(key)
        for key in bekeys:
            val = key in inkeys
            cmd = 'self.%s = val' % key
            print(key, '=', val)
            exec(cmd)
            self.inkeys.append(key)  # added later


    def setdefaults(self):
        print('Default:')
        for key in defaultvalues.keys():
            if key not in self.inkeys:
                val = defaultvalues[key]
                cmd = 'self.%s = val' % key
                exec(cmd)
                print(key, '=', val, '(default)')

    def setimages(self, images=None):
        images = images or self.images
        if images is not None:
            if type(images) == str:  # Single image
                images = processimagename(images)
                self.imagesRGB['L'] = [images]
                self.mode = 'L'
            elif type(images[0]) == str:  # List of images
                images = list(map(processimagename, images))  # Need to convert map to list
                self.imagesRGB['L'] = images
                self.mode = 'L'
            else:  # List of 3 lists of images, one for each channel
                self.mode = 'RGB'
                for i in range(3):
                    channel = imagesorder[i]
                    channelimages = list(map(processimagename, images[i]))  # Need to convert map to list
                    self.imagesRGB[channel] = channelimages

    def setnoiselums(self):
        for channel in self.mode:
            self.noiselums[channel] = self.noiselum

    def loadinputs(self):
        """Load R,G,B filenames and options"""
        #self.images = []  # List of images
        #self.channels = []

        with open(self.infile) as f:
            prevline = ''
            #channel = 'L'  # if no channel declared, then it's grayscale!
            channel = 'L'  # if no channel declared, then it's grayscale!
            self.noiselums = {}
            for line in f:
                if line.startswith('#'):
                    continue

                word = line.strip()
                if len(word):
                    words = word.split()
                    if len(words) == 1:  # Channel or image name
                        if (word in 'RGB') and (prevline == ''):
                            channel = word
                            self.mode = 'RGB'
                        else:
                            image = word
                            image = processimagename(image)
                            self.imagesRGB[channel].append(image)
                            print(channel, image)
                            #if channel not in self.channels:
                            #    self.channels.append(channel)
                            #if image not in self.images:
                            #    self.images.append(image)
                    else:  # parameter and value(s)
                        key = words[0]
                        val = str2num(' '.join(words[1:]))
                        if key == 'weightimages':
                            print(words)
                            if len(words[1:]) == 2:  # drz wht
                                keyagain, self.imext, self.weightext = words
                            else:  # drz -> wht
                                self.imext = words[1]
                                self.weightext = words[3]
                        elif key == 'noiselums':
                            if ',' in val:
                                val = val.split(',')
                                val = list(map(float, val))
                            for i, channel in enumerate(self.mode[::-1]):
                                self.noiselums[channel] = val[i]
                        else:
                            cmd = 'self.%s = val' % key
                            #print(cmd)
                            exec(cmd)
                        print(key, '=', val)
                        self.inkeys.append(key)
                prevline = word

        if not self.noiselums:
            if 'noiselum' in self.inkeys:
                for channel in self.mode:
                    self.noiselums[channel] = self.noiselum
                    self.inkeys.append('noiselums')

    def setoutfile(self, outname=None):
        self.outname = outname or self.outname or self.infile
        if self.outname is None:
            self.outname = self.images
            if self.outname:
                self.outname = os.path.basename(self.outname)
                self.outname = decapfile(self.outname)
                #if self.outname[-5:] == '.fits':
                #    self.outname = self.outname[:-5]

        #self.outname = join(self.outdir, self.outname)
        if len(self.outname) > 4 and (self.outname[-4] == '.'):
            # Has extension
            self.outfile = self.outname  # Use whatever extension they picked
            self.outname = self.outname[:-4]  # Remove extension
        else:  # Just root
            self.outfile = self.outname + '.png'


    def outfilterfile(self):
        outfile = os.path.join(self.outdir, self.outname + '_filters.txt')
        return outfile

    def loadimagesize(self):
        global filters
        print()
        print("Loading image data. If multiple filters per channel, adding data.")
        filters = {'B': [], 'G': [], 'R': [], 'L': []}
        fout = open(self.outfilterfile(), 'w')

        for channel in self.mode[::-1]:
            if channel in 'RGB':
                ichannel = 'RGB'.index(channel)
            else:  # L
                ichannel = 0
            outline = channel + ' = '

            for iimage, image in enumerate(self.imagesRGB[channel]):
                print(channel, end=' ')
                sgn = 1
                if image[0] == '-':
                    sgn = -1
                    image = image[1:]
                if iimage:
                    sgnsym = '.+-'[sgn]
                    outline += ' %s ' % sgnsym
                elif sgn == -1:
                    outline += '- '

                data = loadfitsimagedata(image, self.indir, silent=0)
                filt = imfilt
                if filt == '':
                    filt = imfilts.get(image, '')
                filters[channel].append(filt)
                outline += filt
                ny, nx = data.shape

                if self.ny is None:
                    self.ny = ny
                    self.nx = nx
                    self.yc = ny // 2
                    self.xc = nx // 2
                else:
                    if (self.ny != ny) or (self.nx != nx):
                        print("Input FAIL. Your images are not all the same size as (%d,%d)." % (self.ny, self.nx))
                        for channel in self.mode[::-1]:  # 'BGR'
                            for image in self.imagesRGB[channel]:
                                data = loadfitsimagedata(image, self.indir, silent=0)
                        raise Exception("Images have different sizes.")

            fout.write(outline+'\n')
            print(outline)
            print()

        fout.close()
        print()

        if 0:
            fout = open(self.outfilterfile(), 'w')
            for channel in 'BGR':
                filtstr = ' + '.join(filters[channel])
                fout.write('%s = %s\n' % (channel, filtstr))

            fout.close()

        # quit()

        # Allow the user to just create part of the image
        # xlo  1000
        # xhi  5000
        # xhi -1000
        # if hi is negative, then trims from that edge (as 1000:-1000)
        if self.xhi is None:
            self.xhi = self.nx
        elif self.xhi < 0:
            self.xhi = self.nx + self.xhi

        if self.yhi is None:
            self.yhi = self.ny
        elif self.yhi < 0:
            self.yhi = self.ny + self.yhi


    def loadstamps(self, limits, silent=1):
        ylo, yhi, xlo, xhi = limits

        ylo = max(ylo, 0)
        yhi = min(yhi, self.ny)
        xlo = max(xlo, 0)
        xhi = min(xhi, self.nx)

        ny = yhi - ylo
        nx = xhi - xlo

        three = len(self.mode)
        stampRGB = np.zeros((three, ny, nx), dtype=float)
        weighting = self.weightext is not None
        if weighting:
            weightstampRGB = np.zeros((three, ny, nx), dtype=float)

        for ichannel, channel in enumerate(self.mode):
            for image in self.imagesRGB[channel]:
                if not silent:
                    print(channel, end=' ')
                sgn = 1
                if image[0] == '-':
                    sgn = -1
                    image = image[1:]
                data = loadfitsimagedata(image, self.indir, silent=silent)
                stamp = data[ylo:yhi, xlo:xhi]
                stamp = datascale(stamp, self.bscale, self.bzero)

                # weight image?
                if weighting:
                    weightimage = image.replace(self.imext, self.weightext)
                    weightfile = os.path.join(self.indir, weightimage)
                    if os.path.exists(weightfile):
                        weight = loadfitsimagedata(weightimage, self.indir, silent=silent)
                        weightstamp = weight[ylo:yhi, xlo:xhi]
                        weightstamp = greater(weightstamp, 0)  # FLAG IMAGE!!  EITHER 1 or 0
                        weightstampRGB[ichannel] = weightstampRGB[ichannel] + sgn * weightstamp
                        stamp = stamp * weightstamp
                    else:
                        print(weightfile, 'DOES NOT EXIST')

                stampRGB[ichannel] = stampRGB[ichannel] + sgn * stamp

        if weighting:
            for ichannel, channel in enumerate(self.mode):
                stampRGB[ichannel] = np.where(weightstampRGB[ichannel],
                                             stampRGB[ichannel] / weightstampRGB[ichannel], 0)
        elif self.combine == 'average':
            for ichannel, channel in enumerate(self.mode):
                stampRGB[ichannel] = stampRGB[ichannel] / len(self.imagesRGB[channel])

        return stampRGB


    def determinescalings(self):
        """Determine data scalings
        will sample a (samplesize x samplesize) region of the (centered) core
        make color image of the core as a test if desired"""

        self.testimages = []
        redo = True
        while redo:  # Until the user is happy with the test image of the core
            dx = dy = self.samplesize
            print()
            unsatpercent = 1 - 0.01 * self.satpercent
            self.levdict = {}

            if dx * dy == 0:
                print('By setting samplesize = 0, you have asked to sample the entire image to determine the scalings.')
                print('(Note this will be np.clipped to a maximum of %dx%d.)' % (self.maxstampsize, self.maxstampsize))
                dx = dy = self.maxstampsize  # Maximum size possible

            ylo = np.clip(self.yc - dy // 2 + self.sampledy, 0, self.ny)
            yhi = np.clip(self.yc + dy // 2 + self.sampledy, 0, self.ny)
            xlo = np.clip(self.xc - dx // 2 + self.sampledx, 0, self.nx)
            xhi = np.clip(self.xc + dx // 2 + self.sampledx, 0, self.nx)
            dy = yhi - ylo
            dx = xhi - xlo
            print("Determining image scaling based on %dx%d core sample" % (dx, dy), end=' ')
            if self.sampledx or self.sampledy:
                print('offset by (%d,%d)' % (self.sampledx, self.sampledy), end=' ')

            print('...')

            limits = ylo, yhi, xlo, xhi
            stampRGB = self.loadstamps(limits)
            for ichannel, channel in enumerate(self.mode):
                self.levdict[channel] = determinescaling(stampRGB[ichannel], unsatpercent, noisesig=self.noisesig, correctbias=self.correctbias, noisesig0=self.noisesig0)
                print(channel, end=' ')
                print(' %f  %f  %f' % self.levdict[channel])

            savelevels(self.levdict, outdir=self.outdir)

            redo = False
            if self.testfirst:
                im = RGBscale2im(stampRGB, self.levdict, self.noiselums, self.colorsatfac, self.mode, self.invert)

                outfile = '%s_test_%g_%g_%g.png' % (self.outname, self.satpercent, self.noiselum, self.colorsatfac)
                outfile = os.path.join(self.outdir, outfile)
                self.testimages.append(outfile)

                print("Creating test image", outfile)
                im.save(outfile)
                print(f"Image={Image}")
                if self.show:
                    self.showimage(outfile, Image)

                print('Like what you see?')
                print('If so, press <Enter> a few times')
                print('Otherwise, enter new values:')

                inp = input('  noise yields brightness: %g? ' % self.noiselum)
                if inp.strip() != '':
                    self.noiselum = float(inp)
                    for channel in self.mode:
                        self.noiselums[channel] = self.noiselum
                    redo = True

                inp = input('  %% of pixels that saturate: %g? ' % self.satpercent)
                if inp.strip() != '':
                    self.satpercent = float(inp)
                    redo = True

                if self.mode == 'RGB':
                    inp = input('  color saturation factor: %g? ' % self.colorsatfac)
                    if inp.strip() != '':
                        self.colorsatfac = float(inp)
                        redo = True

                inp = input('  Sample size: %d? ' % self.samplesize)
                if inp.strip() != '':
                    self.samplesize = int(inp)
                    redo = True

                inp = input('  Sample offset x: %d? ' % self.sampledx)
                if inp.strip() != '':
                    self.sampledx = int(inp)
                    redo = True

                inp = input('  Sample offset y: %d? ' % self.sampledy)
                if inp.strip() != '':
                    self.sampledy = int(inp)
                    redo = True




    def determinescalings2(self):
        """Determine data scalings
        will sample a (samplesize x samplesize) region of the (centered) core
        make a color image of the core as a test if desired"""

        self.testimages = []
        redo = True
        while redo:  # Until the user is happy with the test image of the core
            dx = dy = self.samplesize
            # print()
            unsatpercent = 1 - 0.01 * self.satpercent
            self.levdict = {}

            if dx * dy == 0:
                print('By setting samplesize = 0, you have asked to sample the entire image to determine the scalings.')
                print('(Note this will be np.clipped to a maximum of %dx%d.)' % (self.maxstampsize, self.maxstampsize))
                dx = dy = self.maxstampsize  # Maximum size possible

            ylo = np.clip(self.yc - dy // 2 + self.sampledy, 0, self.ny)
            yhi = np.clip(self.yc + dy // 2 + self.sampledy, 0, self.ny)
            xlo = np.clip(self.xc - dx // 2 + self.sampledx, 0, self.nx)
            xhi = np.clip(self.xc + dx // 2 + self.sampledx, 0, self.nx)
            dy = yhi - ylo
            dx = xhi - xlo
            print("Determining image scaling based on %dx%d core sample" % (dx, dy), end=' ')
            if self.sampledx or self.sampledy:
                print('offset by (%d,%d)' % (self.sampledx, self.sampledy), end=' ')

            print('...')

            limits = ylo, yhi, xlo, xhi
            stampRGB = self.loadstamps(limits)
            for channel in self.mode:
                self.levdict[channel] = 0, self.noise, self.saturate  # x0, x1, x2

            savelevels(self.levdict, outdir=self.outdir)

            redo = False
            if self.testfirst:
                im = RGBscale2im(stampRGB, self.levdict, self.noiselums, self.colorsatfac, self.mode, self.invert)

                # outfile = '%s_test_%g_%g_%g.png' % (self.outname, self.satpercent, self.noiselum, self.colorsatfac)
                outfile = '%s_test_%g_%g_%g.png' % (self.outname, self.noiselum, self.noise, self.saturate)
                outfile = os.path.join(self.outdir, outfile)
                self.testimages.append(outfile)

                print("Creating test image", outfile)
                im.save(outfile)

                if self.show:
                    self.showimage(outfile, Image)

                print('Like what you see?')
                print('If so, press <Enter> a few times')
                print('Otherwise, enter new values:')

                inp = input('  noise yields brightness: %g? ' % self.noiselum)
                if inp.strip() != '':
                    self.noiselum = float(inp)
                    for channel in self.mode:
                        self.noiselums[channel] = self.noiselum
                    redo = True

                inp = input('  noise image input data value: %g? ' % self.noise)
                if inp.strip() != '':
                    self.noise = float(inp)
                    redo = True

                inp = input('  saturation level image input data value: %g? ' % self.saturate)
                if inp.strip() != '':
                    self.saturate = float(inp)
                    redo = True

                if self.mode == 'RGB':
                    inp = input('  color saturation factor: %g? ' % self.colorsatfac)
                    if inp.strip() != '':
                        self.colorsatfac = float(inp)
                        redo = True

                inp = input('  Sample size: %d? ' % self.samplesize)
                if inp.strip() != '':
                    self.samplesize = int(inp)
                    redo = True

                inp = input('  Sample offset x: %d? ' % self.sampledx)
                if inp.strip() != '':
                    self.sampledx = int(inp)
                    redo = True

                inp = input('  Sample offset y: %d? ' % self.sampledy)
                if inp.strip() != '':
                    self.sampledy = int(inp)
                    redo = True


    def makecolorimage(self):
        """Make color image (in sections)"""
        if (self.stampsize == self.samplesize == 0) and self.testfirst:
            # Already did the full image!
            print('Full size image already made.')
            imfile = self.testimages[-1]
            outfile = os.path.join(self.outdir, self.outfile)
            if self.deletetests:
                print('Renaming to', outfile)
                os.rename(imfile, outfile)
            else:
                print('Copying to', outfile)
                shutil.copy(imfile, outfile)
            imfull = Image.open(outfile)
            return imfull

        # Clean up: Delete test images
        if self.deletetests:
            for testimage in self.testimages:
                if os.path.exists(testimage):
                    os.remove(testimage)

        dx = dy = self.stampsize
        if dx * dy == 0:
            dx = dy = self.maxstampsize

        imfull = Image.new(self.mode, (self.nx, self.ny))

        print()
        if self.mode == 'RGB':
            print('Making the full color image, one stamp (section) at a time...')
        elif self.mode == 'L':
            print('Making the full grayscale image, one stamp (section) at a time...')

        for yo in range(self.ylo, self.yhi, dy):
            dy1 = min([dy, self.yhi - yo])
            for xo in range(self.xlo, self.xhi, dx):
                dx1 = min([dx, self.xhi - xo])
                print('%5d, %5d  /  (%d x %d)' % (xo, yo, self.nx, self.ny))
                limits = yo, yo + dy, xo, xo + dx
                stamps = self.loadstamps(limits)
                im = RGBscale2im(stamps, self.levdict, self.noiselums, self.colorsatfac, self.mode, self.invert)
                if self.show and self.showstamps:
                    im.show()

                imfull.paste(im, (xo, self.ny - yo - dy1, xo + dx1, self.ny - yo))

        outfile = os.path.join(self.outdir, self.outfile)
        if self.legend:
            self.addlegend(im=imfull)
        else:
            print('Saving', outfile, '...')
            imfull.save(outfile)

        if self.show:
            self.showimage(outfile, Image)

        return imfull


    def makethumbnail1(self, outroot, width, fmt='jpg'):
        nx = width
        ny = int(1000 * (self.ny / float(nx)))
        im = Image.open(outroot + '.png')
        im2 = im.resize((nx, ny))
        im2.save(f"{self.outname}_{width}.{fmt}")
        return im2

    def makethumbnail(self):
        if self.thumbnail not in [None, 'None']:
            outname = self.thumbnail
            if outname[-4] == '.':
                outname = outname[:-4]
                fmt = outname[-3:]
            width = int(outname)
            self.makethumbnail1(self.outname, width, fmt)

    def showsample(self, outfile):
        dx = dy = self.samplesize
        if dx * dy == 0:
            print('By setting samplesize = 0, you have asked to sample the entire image to determine the scalings.')
            print('(Note this will be np.clipped to a maximum of %dx%d.)' % (self.maxstampsize, self.maxstampsize))
            dx = dy = self.maxstampsize  # Maximum size possible

        ylo = np.clip(self.yc - dy // 2 + self.sampledy, 0, self.ny)
        yhi = np.clip(self.yc + dy // 2 + self.sampledy, 0, self.ny)
        xlo = np.clip(self.xc - dx // 2 + self.sampledx, 0, self.nx)
        xhi = np.clip(self.xc + dx // 2 + self.sampledx, 0, self.nx)
        dy = yhi - ylo
        dx = xhi - xlo
        print("Showing %dx%d core sample" % (dx, dy), end=' ')
        if self.sampledx or self.sampledy:
            print('offset by (%d,%d)' % (self.sampledx, self.sampledy), end=' ')

        print('...')

        limits = ylo, yhi, xlo, xhi
        stampRGB = self.loadstamps(limits)
        im = RGBscale2im(stampRGB, self.levdict, self.noiselums, self.colorsatfac, self.mode, self.invert)

        outfile = os.path.join(self.outdir, outfile)
        # self.testimages.append(outfile)

        print("Creating test image", outfile)
        im.save(outfile)

        if self.show:
            self.showimage(outfile, Image)

        print('Like what you see?')
        inp = input()
        # pause()

    def addlegend(self, outfile=None, im=None):
        if im is None:
            outfile1 = os.path.join(self.outdir, self.outfile)
            print('Adding legend to', outfile1, '...')
            im = Image.open(outfile1)
        else:
            print('Adding legend...')

        nx, ny = im.size
        draw = ImageDraw.Draw(im)

        x = 20
        y0 = 20
        dy = 15

        txt = loadfile(self.outfilterfile(), silent=1)

        if self.mode == 'L':
            white = 255
            line = txt[0][4:]  # get rid of leading "L = "
            draw.text((x, y0), line, fill=white)
        else:
            blue = tuple(255 * np.array([0, 0.5, 1]))
            green = tuple(255 * np.array([0, 1, 0]))
            red = tuple(255 * np.array([1, 0, 0]))

            colors = blue, green, red
            colors = np.array(colors).astype(int)

            for i, line in enumerate(txt):
                y = y0 + dy * i
                ichannel = 'BGR'.index(line[0])
                color = tuple(colors[ichannel])
                draw.text((x, y), line, fill=color)

        if outfile is None:
            outfile = os.path.join(self.outdir, self.outfile)

        print('Saving', outfile, '...')
        im.save(outfile)



    def showimage(self, outfile, Image):
        cmd = self.showwith
        if (not cmd) or (cmd.upper() == 'PIL'):
            Image.open(outfile).show()
        else:
            try:
                os.system(cmd + ' ' + outfile)
            except:  # In case "open" doesn't work on their system (not a Mac)
                # Although it may not work but not raise an error either!
                # Should do better error handling here
                Image.open(outfile).show()

    def addtofilterlog(self):
        fout = open('trilogyfilterlog.txt', 'a')
        fout.write(self.outfile + '\n')
        txt = loadfile(self.outfilterfile())
        for line in txt:
            fout.write(line + '\n')

        fout.write('\n')
        fout.close()

    def run(self):
        # self.setinputs()
        # self.loadimages()
        self.setimages()  # not needed from command line
        self.setoutfile()  # adds .png if necessary to outname
        self.loadimagesize()
        self.addtofilterlog()
        if 'justaddlegend' in self.inkeys:
            self.addlegend()
            quit()

        if not self.noiselums:
            self.setnoiselums()
        if self.scaling is None:
            if self.noise and self.saturate:
                self.determinescalings2()
                ## self.levdict = {}
                ## for channel in self.mode:
                ##     self.levdict[channel] = 0, self.noise, self.saturate  # x0, x1, x2
            else:
                self.determinescalings()
        else:
            print('Loading scaling saved in', self.scaling)
            self.levdict = loaddict(self.scaling)
            scaleroot = os.path.basename(self.scaling)[:-4]
            if self.testfirst:
                self.showsample(f"{self.outname}_{scaleroot}.png")
            print("Scalings:")
            for channel in self.mode:
                print(channel, self.levdict[channel])
            self.makecolorimage()

            # Legend added in makecolorimage
            # if self.legend:
            #     self.addlegend()

        self.makethumbnail()



def pause(text=''):
    inp = input(text)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        infile = 'trilogy.in'
        images = None
        Trilogy(infile, images=images, **params_cl()).run()
    else:  # > 1
        input1 = sys.argv[1]
        if ('*' in input1) or ('?' in input1):
            indir = params_cl().get('indir', '')
            input1 = join(indir, input1)
            images = glob(input1)
            for image in images:
                Trilogy(images=image, **params_cl()).run()
        else:
            images = None
            if input1.endswith('.fits') or input1.endswith('.fits.gz'):
                images = input1
                infile = None
            else:
                infile = input1

            Trilogy(infile, images=images, **params_cl()).run()


    #print 'infile', infile
    #print 'images', images
    #pause()
