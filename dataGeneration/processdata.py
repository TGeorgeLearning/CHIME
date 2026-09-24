from concurrent.futures import ProcessPoolExecutor
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from pedalboard import PitchShift
from scipy.interpolate import interp1d
import pandas as pd



def vectorFreqToBin(freqs):
    bins = np.full(freqs.shape, -1, dtype=int)

    mask = freqs > 0

    bins[mask] = np.round(
        48 * np.log2(freqs[mask] / librosa.note_to_hz('C2'))
    ).astype(int)

    bins[(bins < 0) | (bins >= 288)] = -1

    return bins

def semitone_to_hz(s):
    s = np.asarray(s, dtype=float)
    return 440.0 * (2.0 ** ((s - 69.0) / 12.0))

def mirCSV(csvPath,fftLength,sr): # to make the target is our goal as well

  df=pd.read_csv(csvPath,header=None)
  frequencies = df[0].to_numpy(dtype=float)

  frequencies=semitone_to_hz(frequencies)
  binfreqs=vectorFreqToBin(frequencies)

  frame_times = np.arange(len(frequencies))*0.02


  hcqt_times = np.arange(fftLength) * 512 / sr

  interp_func = interp1d(frame_times, binfreqs, kind='nearest',fill_value=-1,bounds_error=False)
  binfreqs = interp_func(hcqt_times)
  return binfreqs.astype(np.int32)

def MIR1kData(songName):
          print(songName)
          harmonics = [0.5, 1, 2, 3, 4, 5]
          wavSong,sr = librosa.load(f'{MIR1kWavPath}{songName}.wav',sr=44100)

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

          targetData = mirCSV(f'{MIR1kGTLabel}{songName}.pv',(hcqt.shape[2]),sr)

          wholeData=[]
          wholeTarget=[]
          for i in range(0, hcqt.shape[2] - 86,12):

              wholeData.append(hcqt[:,:,i:i+86])
              wholeTarget.append(targetData[i:i+86])
              
          np.save(f"{trainXDataPath}{songName}.npy",wholeData)
          np.save(f"{trainYDataPath}{songName}.npy",wholeTarget)






def targetMDB(csvPath):
    df = pd.read_csv(csvPath, sep=',', header=None)
    frame_times = df[0].to_numpy(dtype=float)
    frequencies = df[1].to_numpy(dtype=float)
    return frame_times,frequencies


def mdbCSV(csvPath,fftLength,sr,shift=0): 
  frame_times, frequencies = targetMDB(csvPath)

  frequencies = frequencies * (2 ** (shift / 12))
  binfreqs = vectorFreqToBin(frequencies)

  hcqt_times = np.arange(fftLength) * 512 / sr

  interp_func = interp1d(frame_times, binfreqs, kind='nearest',fill_value=-1,bounds_error=False)

  binfreqs = interp_func(hcqt_times)

  return binfreqs.astype(np.int32)

def MDBData(songName):
  try:
    harmonics = [0.5, 1, 2, 3, 4, 5]
    wavSongOG,sr = librosa.load(f'{mdbWavPath}{songName}_MIX.wav',sr=44100)
    base_fmin = librosa.note_to_hz('C2') 
    for pitch in [-1,0]:
            print(songName,pitch)
            if (pitch!=0):
              effect = PitchShift(semitones=pitch)
              wavSong = effect(wavSongOG, sr)
            else:
              wavSong=wavSongOG.copy()

            hcqt = []

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
            
            targetData = mdbCSV(f'{mdbGTLabel}{songName}_MELODY2.csv',(hcqt.shape[2]),sr,shift=pitch)

            wholeData=[]
            wholeTarget=[]
            for i in range(0, hcqt.shape[2] - 86,36):
                wholeData.append(hcqt[:,:,i:i+86])
                wholeTarget.append(targetData[i:i+86])
              
            np.save(f"{trainXDataPath}{songName}{pitch}.npy",wholeData)
            np.save(f"{trainYDataPath}{songName}{pitch}.npy",wholeTarget)

  except Exception as e:
    print(e)
    raise
       
 





    
def midiToFreq(midiN):
   
  return 440.0 * (2 ** ((midiN - 69) / 12))

def generateMidi(mscorePath,fftLength,sr,shift): # to make the target is our goal as well
  pm = pretty_midi.PrettyMIDI(mscorePath)
  data=[-1] * fftLength
  frameTime=0
  floatNumber = hopSize/sr
  totalDura = 0
  index=0
  
  assert (len(pm.instruments)==1)

  notesArray=pm.instruments[0].notes
  startTime=notesArray[index].start
  endTime=notesArray[index].end
  noteMidi=notesArray[index].pitch

  for i in range(len(data)):
    numData=-1
    totalDura+=floatNumber
    if totalDura<startTime:
      pass
    elif (totalDura<endTime):
      targBin = freqToBin(midiToFreq(noteMidi+shift))
      if (targBin<288 and targBin>-1):
        numData=targBin
    if (totalDura+floatNumber>=endTime):
      index+=1
      if (index>=len(notesArray)):
        startTime=100000
      else:
        startTime=notesArray[index].start
        endTime=notesArray[index].end
        noteMidi=notesArray[index].pitch
    data[i]=numData


  return data

def ORCHSETData(songName):

          harmonics = [0.5, 1, 2, 3, 4, 5]
          base_fmin = librosa.note_to_hz('C2') 
          wavSongOG,sr = librosa.load(f'{ORCHSETWavPath}{songName}.wav',sr=44100)
          for pitch in [-2,0,1]:
            print(songName,pitch)
            
            if (pitch!=0):
              effect = PitchShift(semitones=pitch)
              wavSong = effect(wavSongOG, sr)
            else:
              wavSong=wavSongOG.copy()
              
            hcqt = []

            

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

            targetData = generateMidi(f'{ORCHSETGTLabel}{songName}.mid',(hcqt.shape[2]),sr,pitch)


            wholeData=[]
            wholeTarget=[]
            for i in range(0, hcqt.shape[2] - 86,24):

              wholeData.append(hcqt[:,:,i:i+86])
              wholeTarget.append(targetData[i:i+86])


            np.save(f"{trainXDataPath}{songName}{pitch}.npy",wholeData)
            np.save(f"{trainYDataPath}{songName}{pitch}.npy",wholeTarget)


if __name__ == "__main__":

  # For this section, you need to store the MedleyDB, ORCHSET, and MIR-1k audio and annotations in a folder. Replace the corresponding ground truth
  # label and audio paths with your own file paths

  # The for loops involving root are used to generate the list of songs used in training for convenience. The list for the MedleyDB training songs are shown separately as
  # the training dataset involves a subset of the MedleyDB songs, unlike ORCHSET and MIR-1k, where the entire dataset is used.

  # To specify where you want the processed training data to be stored, modify the variables "trainXDataPath" and "trainYDataPath". In our experiments, we stored the data 
  # in a folder with the path '/mnt/SSD/processedData/', so the variables were '/mnt/SSD/processedData/trainXData/' and '/mnt/SSD/processedData/trainYData/' respectively

    trainXDataPath = '/mnt/SSD/processedData/trainXData/'
    trainYDataPath  = '/mnt/SSD/processedData/trainYData/'

    mdbGTLabel=''
    mdbWavPath = ''

    MDBTrainSongs = [
    
    "MusicDelta_Rockabilly",
    "ClaraBerryAndWooldog_Boys",
    "Auctioneer_OurFutureFaces",
    "MusicDelta_Rock",
    "Mozart_BesterJungling",
    "AClassicEducation_NightOwl",
    "SweetLights_YouLetMeDown",
    "MusicDelta_Reggae",
    "LizNelson_ImComingHome",
    "MusicDelta_Britpop",
    "Schubert_Erstarrung",
    "Wolf_DieBekherte",
    "MusicDelta_Disco",
    "MusicDelta_Punk",
    "Meaxic_TakeAStep",
    "Meaxic_YouListen",
    "LizNelson_Rainfall",
    "DreamersOfTheGhetto_HeavyLove",
    "AlexanderRoss_GoodbyeBolero",
    "MusicDelta_Country1",
    "FacesOnFilm_WaitingForGa",
    "StevenClark_Bounty",
    "StrandOfOaks_Spacestation",
    "BrandonWebster_DontHearAThing",
    "TheScarletBrand_LesFleursDuMal",
    "MatthewEntwistle_DontYouEver",
    "MusicDelta_Hendrix",
    "ClaraBerryAndWooldog_TheBadGuys",
    "MusicDelta_80sRock",
    "Debussy_LenfantProdigue",
    "Mozart_DiesBildnis",
    "Snowmine_Curfews",
    "FamilyBand_Again",
    "PortStWillow_StayEven",
    "Handel_TornamiAVagheggiar",
    "AlexanderRoss_VelvetCurtain",
    "Creepoid_OldTree",
    "BrandonWebster_YesSirICanFly",
    "PurlingHiss_Lolita",
    "ClaraBerryAndWooldog_AirTraffic",
    "HeladoNegro_MitadDelMundo",
    "BigTroubles_Phantom",
    "ClaraBerryAndWooldog_Stella",
    "MusicDelta_Country2",
    "HopAlong_SisterCities",
    "MusicDelta_Grunge",
    "LizNelson_Coldwar",
    "MusicDelta_FreeJazz",
    "MusicDelta_ChineseXinJing",
    "MusicDelta_Zeppelin",
    "MusicDelta_ChineseChaoZhou",
    "MatthewEntwistle_TheFlaxenField",
    "JoelHelander_IntheAtticBedroom",
    "KarimDouaidy_Hopscotch",
    "MusicDelta_SpeedMetal",
    "MusicDelta_GriegTrolltog",
    "MusicDelta_FunkJazz",
    "Phoenix_ElzicsFarewell",
    "MatthewEntwistle_FairerHopes",
    "Phoenix_ScotchMorris",
    "CroqueMadame_Pilot",
    "AmarLal_SpringDay1",
    "MusicDelta_ChineseYaoZu",
    "Phoenix_LarkOnTheStrandDrummondCastle",
    "CroqueMadame_Oil",
    "MusicDelta_Beethoven",
    "AimeeNorwich_Flying",
    "AimeeNorwich_Child",
    "Phoenix_ColliersDaughter",
    "MusicDelta_BebopJazz",
    "ChrisJacoby_BoothShotLincoln",
    "Phoenix_SeanCaughlinsTheScartaglen",
    "MatthewEntwistle_ImpressionsOfSaturn",
    "Phoenix_BrokenPledgeChicagoReel",
    "MusicDelta_ModalJazz",
    "JoelHelander_Definition",
    "JoelHelander_ExcessiveResistancetoChange",
    "MatthewEntwistle_TheArch",
    "AmarLal_Rest",
    "MusicDelta_CoolJazz",
    "MusicDelta_LatinJazz",
    "KarimDouaidy_Yatora",
    "EthanHein_GirlOnABridge",
    "MusicDelta_SwingJazz",
    "MusicDelta_InTheHalloftheMountainKing"
    ]
    
    MIR1kWavPath = ''
    MIR1kGTLabel = ''
    root = Path('')
    MIR1kTrain=[]
    for a in root.rglob('*'):
        MIR1kTrain.append(a.stem)
    print(len(MIR1kTrain))

    ORCHSETGTLabel=''
    ORCHSETWavPath=''

    root = Path('')
    orchsetTrain=[]
    for a in root.rglob('*'):
        orchsetTrain.append(a.stem)

    workers = 16

    # Generates MedleyDB training data
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(MDBData, MDBTrainSongs))

    # Generates ORCHSET training data
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(ORCHSETData, orchsetTrain))
    
    # Generates MIR-1k training data
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(MIR1kData, MIR1kTrain))


        
