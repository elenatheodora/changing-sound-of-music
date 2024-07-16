## Elena Georgieva
## April 2024
## PYIN + simple range function + entropy calculations for ISMIR paper. 20 Splts numbered 0 - 19 

## IMPORTS ------------
import librosa
import numpy as np
import os
from os import path
import scipy
from scipy import signal
import csv
import pandas as pd
import pysftp
import warnings
from scipy.stats import entropy


## HIDING WARNINGS --------
warnings.filterwarnings('ignore') 

## PATHS ALL on HPC ------------
msd_path = "/scratch/work/marl/datasets/mir_datasets/MSD_30sec/songs_msd/"
processed_path = '/scratch/etg259/ismirsf_work/pyin_ismirsf/' # where we write csvs to
vocals_output_path = '/scratch/etg259/ismirsf_work/demucs_ismirsf/split_0/htdemucs/' # change here*
pyin_path = '/scratch/etg259/ismirsf_work/pyin_ismirsf/split_0/' # all subdirectories 


## ENTROPY FUNCTION ----------------------
def compute_entropy(f0, x):
    # Remove frames with 0, NaN, or inf values from f0_array
    f0_array = np.array(f0, dtype=np.float64)
    valid_indices = ~(np.isnan(f0_array) | np.isinf(f0_array) | (f0_array == 0))
    f0_nonzero = f0_array[valid_indices]

    # Convert cents to Hz
    f0_hz = 16.35 * 2**(f0_nonzero/1200) # C0 = 16.35

    ## PITCH CLASS ENTROPY --------------------------------
    # Convert Hz to pitch classes, dropping octave info
    pitch_classes = [note[:-1] for note in librosa.hz_to_note(f0_hz)]

    # Count the frequency of elements
    unique_pitch_classes, unique_counts = np.unique(pitch_classes, return_counts=True)
    pitch_class_counts = dict(zip(unique_pitch_classes, unique_counts))
    total_pitch_class_counts = np.sum(unique_counts)
    
    # Normalize pitch class counts so it adds up to 1
    normalized_pitch_class_count = {note1: count1/total_pitch_class_counts for note1, count1 in pitch_class_counts.items()}
    
    # Calculate pitch entropy
    pitch_class_entropy = scipy.stats.entropy(list(normalized_pitch_class_count.values()), base=2)

    ## PITCH ENTROPY --------------------------------
    # Convert f0_hz to notes-- retaining octave info
    notes = [librosa.hz_to_note(f) for f in f0_hz]

    # Count the frequency of elements 
    unique_elements, counts = np.unique(notes, return_counts=True)
    note_counts = dict(zip(unique_elements, counts))
    total_count = np.sum(counts)

    # Normalize the note counts so it adds up to 1
    norm_note_counts = {note: count/total_count for note, count in note_counts.items()}

    # calculate entropy
    pitch_entropy = scipy.stats.entropy(list(norm_note_counts.values()), base= 2) 
    
    return {
        'song_name': x,
        'pitch_class_entropy': round(pitch_class_entropy, 2),
        'pitch_entropy': round(pitch_entropy, 2),
        'normalized_pitch_class_count': normalized_pitch_class_count,
        'norm_note_counts': norm_note_counts
    }


## Simple Range FUNCTION ------------
# Function that computes & saves range metrics: [song_name, rms_ratio, rms_ratio_bool, mean, median, st dev, 5 percentile, 95 percentile, difference, tv] 
def simple_range(f0, x, rms_ratio):
    f0[f0 == 0] = np.nan  # change all 0s to nans

    # Compute total variation
    tv = np.nanmean(np.abs(np.diff(f0)))

    # Determine rms_ratio_bool
    if rms_ratio >= 0.08:
        rms_ratio_bool = 1
    else:
        rms_ratio_bool = 0

    # Compute simple_range metrics
    if pd.isnull(f0).all():
        simple_range_metrics = [
            x,
            rms_ratio,
            rms_ratio_bool,
            0, 0, 0, 0, 0, 0, 0,
            None, None, None, None  # Placeholder for entropy values
        ]
    else:
        entropy_result = compute_entropy(f0, x)

        simple_range_metrics = [
            entropy_result['song_name'],
            rms_ratio,
            rms_ratio_bool,
            round(np.nanmean(f0), 1),
            round(np.nanmedian(f0)),
            round(np.nanstd(f0)),
            round(np.nanpercentile(f0, 5)),
            round(np.nanpercentile(f0, 95)),
            round(np.nanpercentile(f0, 95)) - round(np.nanpercentile(f0, 5)),
            round(tv, 2),
            entropy_result['pitch_class_entropy'],
            entropy_result['pitch_entropy'],
            entropy_result['normalized_pitch_class_count'],
            entropy_result['norm_note_counts']
        ]

    # Create a new row to append to the dataframe
    new_row = {
        'song_name': simple_range_metrics[0],
        'rms_ratio': simple_range_metrics[1],
        'rms_ratio_bool': simple_range_metrics[2],
        'mean': simple_range_metrics[3],
        'median': simple_range_metrics[4],
        'stdev': simple_range_metrics[5],
        '5th': simple_range_metrics[6],
        '95th': simple_range_metrics[7],
        'diff': simple_range_metrics[8],
        'tv': simple_range_metrics[9],
        'pitch_class_entropy': simple_range_metrics[10],
        'pitch_entropy': simple_range_metrics[11],
        'normalized_pitch_class_count': simple_range_metrics[12],
        'norm_note_counts': simple_range_metrics[13]
    }

    return new_row

## 2,3,4. PYIN & SONIFY & CALL SIMPLE RANGE FUNC ---------- 
early_simple_range_df = pd.DataFrame(columns=["song_name", "rms_ratio", "rms_ratio_bool", "mean", "median", "stdev", "5th", "95th", "diff", "tv","pitch_class_entropy", "pitch_entropy", "normalized_pitch_class_count", "norm_note_counts"])
vocal_counter=0
dne_counter = 0

for x in os.listdir(vocals_output_path): 
    if x.endswith('.clip_vocals.wav'): # x is vocal stem 
        # Paths & load signal
        x_no_extension = str(x[0:len(x)-4]) 
        specific_path = vocals_output_path + x
        if os.path.isfile(specific_path) and os.path.getsize(specific_path) > 0: # If non-0 size
            y,fs = librosa.load(specific_path, sr = 44100, mono=True) # load vocal stem
            
            # rms_ratio: compare RMS of vocals and full mix, note that value---
            vocals_rms = librosa.feature.rms(y=y) # vocals
            song_id = str(x[0:len(x)-16])                                                                                                
            y_mix, fs = librosa.load(msd_path + song_id[2] + '/' + song_id[3] + '/' + song_id[4] + '/' + song_id + '.clip.mp3', sr = 44100, mono=True) # load mix
            mix_rms = librosa.feature.rms(y= y_mix) # mix
            rms_ratio = np.nanmedian(vocals_rms[0])/np.nanmedian(mix_rms[0])

            if (os.path.isfile(pyin_path + x_no_extension + '.npy')): # If pyin.npy is already there ---
                f0 = np.load(pyin_path + x_no_extension + '.npy',allow_pickle=True)[0,:] #take the 0th row
                f0_nonzero = f0[f0 != 0]  # extract non-zero elements of f0
                f0_nonzero = 1200 * np.log2(f0_nonzero.astype(np.float) / 16.35)  # apply function to non-zero elements # Covert to cents relative to C0 = 16.35Hz
                f0[f0 != 0] = f0_nonzero  # replace non-zero elements in f0 with the result
                new_df = simple_range(f0, x, rms_ratio) 
                early_simple_range_df = early_simple_range_df.append(new_df, ignore_index=True) #append to end of early_simple_range_df
            
            else: # Run PYIN f0 estimate & save to .npy per song
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=70, fmax=900, sr=44100, n_thresholds=100 , switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan) # D2 to A5. 
                f0 = np.nan_to_num(f0) #change nans to 0s
                f0_nonzero = f0[f0 != 0]  # extract non-zero elements of f0
                f0_nonzero = 1200 * np.log2(f0_nonzero.astype(np.float) / 16.35)  # apply function to non-zero elements # Covert to cents relative to C0 = 16.35Hz
                f0[f0 != 0] = f0_nonzero  # replace non-zero elements in f0 with the result
                pyin_df = pd.DataFrame((f0, voiced_flag, voiced_probs))
                np.save(pyin_path + x_no_extension + '.npy', pyin_df) # 1 .npy per song. allow_pickle=True as is default      

                new_df = simple_range(f0, x, rms_ratio) 
                early_simple_range_df = early_simple_range_df.append(new_df, ignore_index=True) #append to end of early_simple_range_df
        else:
            dne_counter += 1

        vocal_counter+=1 # Do these regardless of if vs else
        if (vocal_counter == 10):
            print("-------- Nice, we scanned 10!")
        if (vocal_counter % 500 == 0):
            print("-------- Progress! vocal_counter: " + str(vocal_counter)) 

# At the end, write the df to simple_range.csv
print("-------- DNE or is 0kb: " + str(dne_counter))
print("writing csv...")
with open(processed_path + 'vocanalysis_entropy_split_0.csv', "w+") as f: # and here*
    early_simple_range_df.to_csv(f, index=False)
print("done :)")

