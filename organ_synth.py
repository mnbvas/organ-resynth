#! usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, generators 
# checking for dependecies #########################################################################
depends = [
    ('numpy', 'NumPy not found. If you do not have it, check http://www.scipy.org/scipylib/download.html')
     ]

import imp
for module, text in depends:
    try:
        __import__(module)
    except ImportError:
        print(text)
        sys.exit(9)

# actual code #####################################################################################
import sys, os

filename = 'sound.wav'
warning = True

for arg in sys.argv[1:]:
    if arg == '--help':
        print('''
            Usage:
            python organ_synth.py [--file some_wav_file.wav] [--out output_file.wav] [--no-warn] [--help]
            ''')
    
    elif arg == '--file':
        i = sys.argv.index(arg)
        if i + 1 <= len(sys.argv):
            filename = sys.argv[i + 1]

    elif arg == '--out':
        i = sys.argv.index(arg)
        if i + 1 <= len(sys.argv):
            out_name = sys.argv[i + 1]

    elif arg == '--no-warn':
        warning = False

out_name = filename[:-4] + ' resynth.wav'

try:
    wav = open(filename, 'rb')
except IOError:
    print('Could not open', filename)
    sys.exit(8)

try:
    out = open(out_name, 'wb')
except IOError:
    print('Could not open', out.name)
    if not out.closed:
        out.close()
    try:
        os.remove(out_name)
    except OSError:
        pass
    sys.exit(7)

import struct

def read_int32():
    r, = struct.unpack('i', wav.read(4))
    return r

def read_int16():
    r, = struct.unpack('h', wav.read(2))
    return r

def read_byte():
    r, = struct.unpack('b', wav.read(1))
    return r

# reading metadata ################################################################################
chunk_id = wav.read(4)

if chunk_id != 'RIFF':
    print('File', filename, 'is not a .wav file.')
    sys.exit(1)

file_size = read_int32()
riff_type = wav.read(4)

if riff_type != 'WAVE':
    print('File', filename, 'is not a .wav file.')
    sys.exit(2)

format_id = read_int32() # likely 'fmt '
format_size = read_int32()
format_code = read_int16()
channels = read_int16()
sample_rate = read_int32()
format_avg_bps = read_int32()
format_block_align = read_int16()
bit_depth = read_int16()

if bit_depth not in [8, 16, 32]:
    print('File', filename, 'contains really interesting data. Too bad I\'m not ready to parse it.')
    sys.exit(3)

if format_size == 18:
    extra_size = read_int16()
    wav.read(extra_size)

data_id = wav.read(4) # should be 'data'
data_size = read_int32()

while data_id != 'data': # any chunk besides data is meta, not interested
    temp_data = wav.read(data_size)
    if len(temp_data) < data_size: # got EOF and still no data
        print('File', filename, 'is probably corrupt.')
        sys.exit(4)
    data_id = wav.read(4)
    data_size = read_int32()

# reading wave data ############################################################################

import numpy as np
import math

raw = np.fromfile(file = wav,  count = data_size, dtype = (
    np.int8 if bit_depth == 8 else (
        np.int16 if bit_depth == 16 else (
            np.int32 ))))
wav.close()

raw_wave = raw[0::channels] # I'll discard all but first channels, should suffice

# reading relevant frequencies #################################################################

# meta:
# since higher octaves are just (oct0 * 2^n), they will be in the same fft tables
# therefore, it is enough to run 12 ffts per sample duration (~8 times faster)

SAMPLE_DURATION = 0.010  # 10 ms

def freq(note):
    return 27.5 * (2 ** ((note - 9) / 12)) # 27.5 for A0, -9 for C0

def round_up_power_2(n):
    return int(2 ** math.ceil(math.log(n, 2)))

import multiprocessing
thread_count = multiprocessing.cpu_count()

import threading

note_chunks = [ [] for i in range(thread_count) ]

def analyze_chunk(chunk_no):
    chunk_length = int(float(len(raw_wave)) / sample_rate / SAMPLE_DURATION / thread_count)
    chunk_offset = chunk_no * chunk_length
    HALF_TONE_RATIO = 2 ** (1/12)
    for t in xrange(chunk_offset, chunk_offset + chunk_length):
        start_offset = int(t * sample_rate * SAMPLE_DURATION)
        notes = []
        for i in xrange(12): # base halftones
            sample_duration = int(sample_rate / freq(i))
            sample = raw_wave[start_offset : start_offset + sample_duration]
            if sample_duration != len(sample): # at the end, and don't have enough samples
                continue
            xvals = np.linspace(0, sample_duration - 1, num = sample_duration, endpoint = False)
            fft = np.absolute(np.fft.fft(sample))
            avg_mag = np.mean(fft[:len(fft) / 2.0])
            for j in xrange(8): # octaves 0-7
                tone = 2 ** j
                if fft[tone] > avg_mag and fft[tone * 2] * 3 > fft[tone] and fft[tone - 1] < fft[tone] > fft[tone + 1]:
                    interp_tone = ((fft[tone - 1] * (tone - 1) +
                                    fft[tone]     * tone +
                                    fft[tone + 1] * (tone + 1)) /
                                           
                                   (fft[tone - 1] + fft[tone] + fft[tone + 1]))
                    interp_tone *= HALF_TONE_RATIO ** i # current halftone ratio
                    note = math.log(interp_tone, HALF_TONE_RATIO) # interpolated note played
                    inote = int(round(note, 0))
                    if (abs(note - round(note, 0)) < 1 and
                         inote % 12 == i):  # belongs in this halftone (e. g. Cn or Bn)
                            notes.append( (inote, fft[tone] ** 1.5) )

        note_chunks[chunk_no].append(notes)

# duration warning #########################################################
duration_minutes = round(float(data_size) / bit_depth * 8 / 60 / sample_rate, 2)
duration_processing = round(duration_minutes * (0.8 + 1.8 / thread_count), 2)
if warning:
    print('Selected .wav is {:.2} minutes long.'.format(duration_minutes),
          'It will take about {:.2} minutes to process it (usually overestimates).'.format(duration_processing), 
          'Press ^C (Ctrl+C) any time to cancel, Enter to continue...')
    raw_input()
else:
    print('ETA %d minutes.' % duration_processing)
############################################################################

print('1/5: Analyzing data.')

all_notes = []

def analyze():
    global all_notes, only_notes, thread_count
    threads = [threading.Thread(group=None, target=analyze_chunk, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    progress = 0.0
    for t in threads:
        t.join()
        progress += 1
        print('\t%d%%' % (round(progress / thread_count, 2) * 100))
    for chunk in note_chunks:
        all_notes += chunk

analyze()

print('2/5: Reordering data.')
# reordering notes into note/time/magnitude from time/note/magnitude #######################
# and normalizing magnitudes

all_notes += [ [], [] ] # add two empty samples to end
note_histogram = [ np.zeros(len(all_notes)) for i in xrange(8 * 12) ] # a histogram for each note, filled with zeros

max_magnitude = -1;
for tick in all_notes:
    if len(tick) > 0:
        max_magnitude = max(max_magnitude, np.max(tick, axis=0)[1])

for i, notes in enumerate(all_notes):
    if 0 < i < len(all_notes) - 2:
        for n, m in notes:
            if n not in [nn for nn, mm in all_notes[i - 1]] and n not in [nn for nn, mm in all_notes[i + 1]]:
                notes.remove( (n, m) )

for i, notes in enumerate(all_notes):
    for n, m in notes:
        note_histogram[n][i] = m / max_magnitude

print('3/5: Smoothing data.')
# smoothing ####################################################################################

SYNTH_SAMPLE_DURATION = SAMPLE_DURATION #0.003 # 3ms
TICK_DURATION = int(SYNTH_SAMPLE_DURATION * sample_rate)
note_smooth_histogram = note_histogram

# wave generators ##############################################################################
                
def sine_wave(phase):
    return np.sin(phase * 2 * math.pi)

MAX_INT = 2 ** 15 - 1

# approximates as addition : Σ (sine(f * 2^i) / 2^i), i=0..4
def organ_wave(offset, period, length, start_scale, end_scale):
    multiplier = 0.25 * MAX_INT
    values = np.zeros((length,), dtype = np.int16)
    for i in range(5): # harmonic count
        phase = offset / period
        values += sine_wave(phase) * multiplier
        multiplier /= 2
        period /= 2

    scaling = np.linspace(start_scale, end_scale, length)
    values *= scaling

    return values

def organ(note, index):
    count = int(SYNTH_SAMPLE_DURATION * sample_rate)
    period = sample_rate / freq(note)
    offset = index * TICK_DURATION
    ret = np.fromfunction(lambda o:
                          organ_wave(offset + o, period, count,
                                     note_smooth_histogram[note][index],
                                     note_smooth_histogram[note][index + 1]),
                          (count,))                   
    return ret

print('4/5: Synthesizing.')
# synthesizing ################################################################################

out_wave = np.zeros(len(note_smooth_histogram[0]) * TICK_DURATION)            

def synth_note(halftone):
    for octave in xrange(8):
        for i in xrange(len(note_smooth_histogram[0]) - 1):
            offset = i * TICK_DURATION
            if len(out_wave[offset:offset + TICK_DURATION]) == TICK_DURATION: # maybe out of samples, will lose at most ~3ms
                out_wave[offset:offset + TICK_DURATION] += organ(halftone + octave * 12, i)

for i in xrange(12): # halftones
    synth_note(i)
    print('\t%d%%' % (round(float(i + 1) / 12, 2) * 100))

print('5/5: Writing to new file')
# write metadata to new file ##################################################################

def write_int32(i):
    out.write(struct.pack('i', i))
    
def write_int16(i):
    out.write(struct.pack('h', i))

def write_int8(i):
    out.write(struct.pack('b', i))

out.write(bytearray('RIFF'))
write_int32(0) # will return later
out.write(bytearray('WAVEfmt '))
write_int32(16) # format size
write_int16(1) # format code, not sure about this
write_int16(1) # channels
write_int32(sample_rate)
write_int32(sample_rate * bit_depth / 8) # avg bps
write_int16(2) # block align
write_int16(16) # bit_depth
out.write(bytearray('data'))
write_int32(len(out_wave) * 2) # datasize

# writing waves out ###########################################################################

out_wave *= MAX_INT * 0.75 / np.max(out_wave)
out_wave = out_wave.astype('int16')

out_wave.tofile(out)
file_size = out.tell() - 4
out.seek(4)
write_int32(file_size)
out.close()

print('Synthesized to', out.name)
