
# This file is to precompute the HCQTs of all songs used in validation and in testing. This is highly recommended, as it
# saves a large amount of time in both training and testing the model.

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import librosa
from pathlib import Path
import traceback


def MDBPrecomputedData(songName):
    try:
          print(songName)
          harmonics = [0.5, 1, 2, 3, 4, 5]
          wavSong,sr = librosa.load(f'{mdbWavPath}{songName}_MIX.wav',sr=44100)
                    
          hcqt = []

          base_fmin = librosa.note_to_hz('C2')

          for h in harmonics:
              cqt_h = librosa.cqt(
                  wavSong,
                  sr=sr,
                  n_bins=288,
                  bins_per_octave=48,
                  fmin=h * base_fmin,
                  hop_length=512
              )

              cqt_h=librosa.amplitude_to_db(np.abs(cqt_h),ref=np.max)
              cqt_h = np.maximum(cqt_h, -80.0)
              cqt_h = (cqt_h + 80) / 80
              hcqt.append(cqt_h)

          min_frames = min(h.shape[-1] for h in hcqt)
          hcqt = [h[:, :min_frames] for h in hcqt]
          hcqt = np.stack(hcqt, axis=0)
                   
          np.save(f"{storeHCQTPath}{songName}.npy",hcqt)
    except:
        print(f"\nERROR PROCESSING: {song}")
        traceback.print_exc()

        raise e

def ADC2004(songName):
    try:
          print(songName)
          harmonics = [0.5, 1, 2, 3, 4, 5]
          wavSong,sr = librosa.load(f'{adcWavPath}{songName}.wav',sr=44100)
                    
          hcqt = []

          base_fmin = librosa.note_to_hz('C2')

          for h in harmonics:
              cqt_h = librosa.cqt(
                  wavSong,
                  sr=sr,
                  n_bins=288,
                  bins_per_octave=48,
                  fmin=h * base_fmin,
                  hop_length=512
              )

              cqt_h=librosa.amplitude_to_db(np.abs(cqt_h),ref=np.max)
              cqt_h = np.maximum(cqt_h, -80.0)
              cqt_h = (cqt_h + 80) / 80
              hcqt.append(cqt_h)

          min_frames = min(h.shape[-1] for h in hcqt)
          hcqt = [h[:, :min_frames] for h in hcqt]
          hcqt = np.stack(hcqt, axis=0)
                   
          np.save(f"{storeHCQTPath}{songName}.npy",hcqt)
    except:
        print(f"\nERROR PROCESSING: {song}")
        traceback.print_exc()

        raise e

def Mirex05(songName):
    try:
          print(songName)
          harmonics = [0.5, 1, 2, 3, 4, 5]
          wavSong,sr = librosa.load(f'{mirexWavPath}{songName}.wav',sr=44100)
                    
        
          base_fmin = librosa.note_to_hz('C2')

          for h in harmonics:
              cqt_h = librosa.cqt(
                  wavSong,
                  sr=sr,
                  n_bins=288,
                  bins_per_octave=48,
                  fmin=h * base_fmin,
                  hop_length=512
              )

              cqt_h=librosa.amplitude_to_db(np.abs(cqt_h),ref=np.max)
              cqt_h = np.maximum(cqt_h, -80.0)
              cqt_h = (cqt_h + 80) / 80
              hcqt.append(cqt_h)

          min_frames = min(h.shape[-1] for h in hcqt)
          hcqt = [h[:, :min_frames] for h in hcqt]
          hcqt = np.stack(hcqt, axis=0)
                   
          np.save(f"{storeHCQTPath}{songName}.npy",hcqt)
    except:
        print(f"\nERROR PROCESSING: {song}")
        traceback.print_exc()

        raise e

if __name__ == "__main__":

    storeHCQTPath = '' # Write the path of where you want the precomputed HCQTs to be stored.
    # For example, in our experiments, we stored the HCQTs at '/mnt/SSD/processedData/precomputedHCQT/'

    mdbWavPath = ''

# The first half of the MDBSongs array is the MedleyDB songs used in testing. The second half is the songs used in the validation phase of training the model.

    MDBSongs = [
        "MusicDelta_Beatles",
        "TheSoSoGlos_Emergency",
        "InvisibleFamiliars_DisturbingWildlife",
        "HezekiahJones_BorrowedHeart",
        "MatthewEntwistle_Lontano",
        "TheDistricts_Vermont",
        "MusicDelta_Pachelbel",
        "MichaelKropf_AllGoodThings",
        "ChrisJacoby_PigsFoot",
        "MusicDelta_Shadows",
        "EthanHein_1930sSynthAndUprightBass",
        "Schumann_Mignon",

        "SecretMountains_HighHorse",
        "ClaraBerryAndWooldog_WaltzForMyVictims",
        "NightPanther_Fire",
        "MusicDelta_Gospel",
        "AvaLuna_Waterduct",
        "CelestialShore_DieForUs",
        "MusicDelta_ChineseHenan",
        "MusicDelta_Vivaldi",
        "MusicDelta_ChineseJiangNan",
        "MusicDelta_FusionJazz",
        "MusicDelta_ChineseDrama"
    ]

    mirexWavPath = ''
    root = Path('')
    mirexTrain=[]
    for a in root.rglob('*'):
        mirexTrain.append(a.stem)

    adcWavPath = ''
    root = Path('')
    adcTrain=[]
    for a in root.rglob('*'):
        adcTrain.append(a.stem)

    workers = 12

    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(MDBPrecomputedData, MDBSongs))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(Mirex05,mirexTrain))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(ADC2004,adcTrain))
    
        
    
        
