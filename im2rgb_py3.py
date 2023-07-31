# im2rgbfits CL0024.png -over -header det.fits
# WILL HONOR WCS FROM headerfile

# im2rgbfits.py
# ~/ACS/CL0024/color/production/color.py
# ALSO SEE pyfits.pdf (Pyfits manual)


from PIL import Image
import astropy.io.fits as pyfits
import sys
import os
from os.path import exists, join
import numpy as np

def str2num(string, rf=0):
    """CONVERTS A STRING TO A NUMBER (INT OR FLOAT) IF POSSIBLE
    ALSO RETURNS FORMAT IF rf=1"""
    try:
        num = int(string)
        format = 'd'
    except ValueError:
        try:
            num = float(string)
            format = 'f'
        except ValueError:
            if not string.strip():
                num = None
                format = ''
            else:
                words = string.split()
                if len(words) > 1:
                    num = list(map(str2num, words))
                    format = 'l'
                else:
                    num = string
                    format = 's'
    if rf:
        return (num, format)
    else:
        return num

def params_cl(converttonumbers=True):
    """RETURNS PARAMETERS FROM COMMAND LINE ('cl') AS DICTIONARY:
    KEYS ARE OPTIONS BEGINNING WITH '-'
    VALUES ARE WHATEVER FOLLOWS KEYS: EITHER NOTHING (''), A VALUE, OR A LIST OF VALUES
    ALL VALUES ARE CONVERTED TO INT / FLOAT WHEN APPROPRIATE"""
    arguments = sys.argv[:]
    i = 0
    dictionary = {}
    oldkey = ""
    key = ""
    arguments.append('')  # EXTRA ELEMENT SO WE COME BACK AND ASSIGN THE LAST VALUE
    while i < len(arguments):
        if striskey(arguments[i]) or not arguments[i]:  # (or LAST VALUE)
            if key:  # ASSIGN VALUES TO OLD KEY
                if value:
                    if len(value) == 1:  # LIST OF 1 ELEMENT
                        value = value[0]  # JUST ELEMENT
                dictionary[key] = value
            if arguments[i]:
                key = arguments[i][1:]  # REMOVE LEADING '-'
                value = None
                dictionary[key] = value  # IN CASE THERE IS NO VALUE!
        else:  # VALUE (OR HAVEN'T GOTTEN TO KEYS)
            if key:  # (HAVE GOTTEN TO KEYS)
                if value:
                    if converttonumbers:
                        value.append(str2num(arguments[i]))
                    else:
                        value = value + ' ' + arguments[i]
                else:
                    if converttonumbers:
                        value = [str2num(arguments[i])]
                    else:
                        value = arguments[i]
        i += 1

    return dictionary

def striskey(string):
    """IS str AN OPTION LIKE -C or -ker
    (IT'S NOT IF IT'S -2 or -.9)"""
    is_key = 0
    if string:
        if string[0] == '-':
            is_key = 1
            if len(string) > 1:
                is_key = string[1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
    return is_key

def strend(string, phr):
    return string[-len(phr):] == phr

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

def loadrgb(infile):
    im = Image.open(infile)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    rgb = np.asarray(im)  # numpy
    print(rgb.shape)
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = rgb[:3]  # in case there's an alpha channel on the end
    # rgb.flags.writeable = True  # DEFAULT IS CAN'T EDIT IT!
    return rgb

def im2rgbfits(infile, rgbfile='', overwrite=False, headerfile=None, flip=False):
    if rgbfile == '':
        rgbfile = decapfile(infile) + '_RGB.fits'

    if exists(rgbfile):
        if overwrite:
            os.remove(rgbfile)
        else:
            print(rgbfile, 'EXISTS')
            sys.exit(1)

    data = loadrgb(infile)

    header = headerfile and pyfits.getheader(headerfile)
    hdu = pyfits.PrimaryHDU(None, header)
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(rgbfile)

    try:  # If there's a 'SCI' extension, then that's where the WCS is
        header = pyfits.getheader(headerfile, 'SCI')
    except:
        pass

    if header is not None:
        if 'EXTNAME' in header:
            del header['EXTNAME']

    for i in range(3):
        print('RGB'[i])
        data1 = data[i]
        if flip:
            data1 = np.flipud(data1)
        pyfits.append(rgbfile, data1, header)

    print(rgbfile, 'NOW READY FOR "Open RGB Fits Image" in ds9')


if __name__ == '__main__':
    infile = sys.argv[1]

    outfile = ''
    if len(sys.argv) > 2:
        file2 = sys.argv[2]
        if file2[0] != '-':
            outfile = file2

    params = params_cl()
    overwrite = 'over' in params
    headerfile = params.get('header', None)

    im2rgbfits(infile, outfile, overwrite=overwrite, headerfile=headerfile)
