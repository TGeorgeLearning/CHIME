# All HCQTs were precomputed to increase testing speed. 

import mir_eval
bestoa=0

model.load_state_dict(torch.load("")) # Replace this with the HarmoLite checkpoint you want to evaluate
model.eval()

def targetFreq(csvPath):
   
    df = pd.read_csv(csvPath, sep=',', header=None)
    
    
    
    frame_times = df[0].to_numpy(dtype=float)
    frequencies = df[1].to_numpy(dtype=float)
    return frame_times,frequencies

def targetMirex(csvPath):
   
    df = pd.read_csv(csvPath, sep='\t', header=None)
    
    
    
    frame_times = df[0].to_numpy(dtype=float)
    frequencies = df[1].to_numpy(dtype=float)
    return frame_times,frequencies

def targetADC(csvPath):
    df = pd.read_csv(csvPath, sep=r"\s+",header=None)
 
    frame_times = df[0].to_numpy(dtype=float)
    frequencies = df[1].to_numpy(dtype=float)
    return frame_times,frequencies
   
MDBsongArr=[]
MDBvoiceArr=[]
MDBtimeArr=[]
MDBtargArr=[]
MDBtargTimeArr=[]

MIREXsongArr=[]
MIREXvoiceArr=[]
MIREXtimeArr=[]
MIREXtargArr=[]
MIREXtargTimeArr=[]

ADCsongArr=[]
ADCvoiceArr=[]
ADCtimeArr=[]
ADCtargArr=[]
ADCtargTimeArr=[]

vrT=0
vfaT=0
rcaT=0
rpaT=0
oaT=0
from scipy.interpolate import interp1d

base_fmin = librosa.note_to_hz('C2')

MDBTestSongs = [
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
    "Schumann_Mignon"]

mirexTrain = [
    'train02',
    'train11',
    'train04',
    'train09',
    'train10',
    'train13',
    'train03',
    'train07',
    'train06',
    'train12',
    'train08',
    'train01',
    'train05']

adcTrain = [
    'midi3',
    'jazz2',
    'opera_fem4',
    'midi2',
    'daisy4',
    'opera_male5',
    'daisy1',
    'jazz3',
    'pop4',
    'jazz4',
    'pop1',
    'pop2',
    'midi4',
    'opera_fem2',
    'midi1',
    'pop3',
    'daisy3',
    'jazz1',
    'daisy2',
    'opera_male3']


harmonics = [0.5, 1, 2, 3, 4, 5]
# The below file paths refer to where the test datasets are stored. The "Midi" paths refer to the ground truth annotations, and the "wav"
# paths refer to the audio files for the test datasets

mdbMidi='/home/tristan/datasets/medleyData/Melody/Melody2/'
mdbWav = '/home/tristan/datasets/medleyData/medleyWav/'

mirexMidi='/home/tristan/datasets/mirex05TrainFiles/midi/'
wavPath = '/home/tristan/datasets/mirex05TrainFiles/wavFiles/'

adcMidi='/home/tristan/datasets/ADC2004/midi/'
adcwavPath = '/home/tristan/datasets/ADC2004/wavFiles/'

for song in MDBTestSongs:
            print(song)

            hcqt = np.load(f'/mnt/SSD/processedData/compare16HopCQT/{song}.npy')
            window = 86
            hop = 43
            
            targetData, freq = targetFreq(f'{mdbMidi}{song}_MELODY2.csv')
            MDBtargTimeArr.append(targetData)
            MDBtargArr.append(freq)
            
            num_frames = hcqt.shape[2]
            num_classes = 288
            
            pitch_sum_logits = np.zeros((num_frames, num_classes), dtype=np.float32)
            voice_sum_logits = np.zeros(num_frames, dtype=np.float32)
            counts = np.zeros(num_frames, dtype=np.float32)
            
            for i in range(0, num_frames - window + 1, hop):
            
                x = torch.from_numpy(
                    hcqt[:, :, i:i+window]
                ).unsqueeze(0).float().to(gpu())
            
                pitch_logits, voice_logits = model(x)
            
                pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            
                voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
                pitch_sum_logits[i:i+window] += pitch_logits
                voice_sum_logits[i:i+window] += voice_logits
                counts[i:i+window] += 1
            
            last_start = num_frames - window
            
            x = torch.from_numpy(
                hcqt[:, :, last_start:last_start+window]
            ).unsqueeze(0).float().to(gpu())
            
            pitch_logits, voice_logits = model(x)
            
            pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
            pitch_sum_logits[last_start:last_start+window] += pitch_logits
            voice_sum_logits[last_start:last_start+window] += voice_logits
            counts[last_start:last_start+window] += 1
            
            avg_pitch_logits = pitch_sum_logits / counts[:, None]
            avg_voice_logits = voice_sum_logits / counts
            
            predPitch = np.argmax(avg_pitch_logits, axis=1)
            predictVal = base_fmin * (2.0 ** (predPitch / 48))
            
            probs = 1 / (1 + np.exp(-avg_voice_logits))

            MDBvoiceArr.append(probs)
            MDBsongArr.append(predictVal)
      
            
            pitchTime = np.arange(num_frames) * (512 / 44100)
            MDBtimeArr.append(pitchTime)
    

for song in mirexTrain:
            hcqt = np.load(f'/mnt/SSD/processedData/compare16HopCQT/{song}.npy')
    
            targetData,freq = targetMirex(f'{mirexMidi}{song}REF.txt')
            MIREXtargTimeArr.append(targetData)
            MIREXtargArr.append(freq)
            
            num_frames = hcqt.shape[2]
            num_classes = 288
            
            pitch_sum_logits = np.zeros((num_frames, num_classes), dtype=np.float32)
            voice_sum_logits = np.zeros(num_frames, dtype=np.float32)
            counts = np.zeros(num_frames, dtype=np.float32)
            
            for i in range(0, num_frames - window + 1, hop):
            
                x = torch.from_numpy(
                    hcqt[:, :, i:i+window]
                ).unsqueeze(0).float().to(gpu())
            
                pitch_logits, voice_logits = model(x)
            
                pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            
                voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
                pitch_sum_logits[i:i+window] += pitch_logits
                voice_sum_logits[i:i+window] += voice_logits
                counts[i:i+window] += 1
            
            last_start = num_frames - window
            
            x = torch.from_numpy(
                hcqt[:, :, last_start:last_start+window]
            ).unsqueeze(0).float().to(gpu())
            
            pitch_logits, voice_logits = model(x)
            
            pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
            pitch_sum_logits[last_start:last_start+window] += pitch_logits
            voice_sum_logits[last_start:last_start+window] += voice_logits
            counts[last_start:last_start+window] += 1
            
            avg_pitch_logits = pitch_sum_logits / counts[:, None]
            avg_voice_logits = voice_sum_logits / counts
            
            predPitch = np.argmax(avg_pitch_logits, axis=1)
            predictVal = base_fmin * (2.0 ** (predPitch / 48))
            
            probs = 1 / (1 + np.exp(-avg_voice_logits))

            MIREXvoiceArr.append(probs)
            MIREXsongArr.append(predictVal)

            
            pitchTime = np.arange(num_frames) * (512 / 44100)
            MIREXtimeArr.append(pitchTime)



for song in adcTrain:
                    
            hcqt = np.load(f'/mnt/SSD/processedData/compare16HopCQT/{song}.npy')
    

            targetData,freq = targetADC(f'{adcMidi}{song}REF.txt')
            ADCtargTimeArr.append(targetData)
            ADCtargArr.append(freq)
            
            num_frames = hcqt.shape[2]
            num_classes = 288
            
            pitch_sum_logits = np.zeros((num_frames, num_classes), dtype=np.float32)
            voice_sum_logits = np.zeros(num_frames, dtype=np.float32)
            counts = np.zeros(num_frames, dtype=np.float32)
            
            for i in range(0, num_frames - window + 1, hop):
            
                x = torch.from_numpy(
                    hcqt[:, :, i:i+window]
                ).unsqueeze(0).float().to(gpu())
            
                pitch_logits, voice_logits = model(x)
            
                pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            
                voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
                pitch_sum_logits[i:i+window] += pitch_logits
                voice_sum_logits[i:i+window] += voice_logits
                counts[i:i+window] += 1
            
            last_start = num_frames - window
            
            x = torch.from_numpy(
                hcqt[:, :, last_start:last_start+window]
            ).unsqueeze(0).float().to(gpu())
            
            pitch_logits, voice_logits = model(x)
            
            pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()
            voice_logits = voice_logits.squeeze(0).squeeze(-1).detach().cpu().numpy()
            
            pitch_sum_logits[last_start:last_start+window] += pitch_logits
            voice_sum_logits[last_start:last_start+window] += voice_logits
            counts[last_start:last_start+window] += 1
            
            avg_pitch_logits = pitch_sum_logits / counts[:, None]
            avg_voice_logits = voice_sum_logits / counts
            
            predPitch = np.argmax(avg_pitch_logits, axis=1)
            predictVal = base_fmin * (2.0 ** (predPitch / 48))
            
            probs = 1 / (1 + np.exp(-avg_voice_logits))

            ADCvoiceArr.append(probs)
            ADCsongArr.append(predictVal)
      
            
            pitchTime = np.arange(num_frames) * (512 / 44100)
            ADCtimeArr.append(pitchTime)



cutoff=0 # Replace this with the cutoff that was obtained during training

vr=[]
vfa=[]
rca=[]
rpa=[]
oa=[]
vrT=0
vfaT=0
rcaT=0
rpaT=0
oaT=0
for i in range(len(MDBtimeArr)):
    predictVal=MDBsongArr[i].copy()
    pitchTime=MDBtimeArr[i]
    voici = MDBvoiceArr[i]
    predVoicing = (voici > cutoff).astype(np.int32)
    predictVal[predVoicing == 0] = 0.0
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        MDBtargTimeArr[i],
        MDBtargArr[i],
        pitchTime,
        predictVal,
        est_voicing=predVoicing
        )
    
    vrval=mir_eval.melody.voicing_recall(ref_v, est_v)
    vfaval=mir_eval.melody.voicing_false_alarm(ref_v, est_v)
    rcaval = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    rpaval=mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    oaval=mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    
    vr.append(vrval)
    vfa.append(vfaval)
    rca.append(rcaval)
    rpa.append(rpaval)
    oa.append(oaval)
vrT+=sum(vr)/len(vr)
vfaT+=sum(vfa)/len(vfa)
rcaT+=sum(rca)/len(rca)
rpaT+=sum(rpa)/len(rpa)
oaT+=sum(oa)/len(oa)

print("---- FINAL MDB ----", cutoff)     
print(f"VR is {sum(vr)/len(vr)}")
print(f"VFA is {sum(vfa)/len(vfa)}")
print(f"RPA is {sum(rpa)/len(rpa)}")
print(f"RCA is {sum(rca)/len(rca)}")
print(f"OA is {sum(oa)/len(oa)}")

vr=[]
vfa=[]
rca=[]
rpa=[]
oa=[]

for i in range(len(MIREXtimeArr)):
    predictVal=MIREXsongArr[i].copy()
    pitchTime=MIREXtimeArr[i]
    voici = MIREXvoiceArr[i]
    predVoicing = (voici > cutoff).astype(np.int32)
    predictVal[predVoicing == 0] = 0.0
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        MIREXtargTimeArr[i],
        MIREXtargArr[i],
        pitchTime,
        predictVal,
        est_voicing=predVoicing
        )
    
    vrval=mir_eval.melody.voicing_recall(ref_v, est_v)
    vfaval=mir_eval.melody.voicing_false_alarm(ref_v, est_v)
    rcaval = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    rpaval=mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    oaval=mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    
    vr.append(vrval)
    vfa.append(vfaval)
    rca.append(rcaval)
    rpa.append(rpaval)
    oa.append(oaval)
vrT+=sum(vr)/len(vr)
vfaT+=sum(vfa)/len(vfa)
rcaT+=sum(rca)/len(rca)
rpaT+=sum(rpa)/len(rpa)
oaT+=sum(oa)/len(oa)

print("---- FINAL MIREX ----", cutoff)     
print(f"VR is {sum(vr)/len(vr)}")
print(f"VFA is {sum(vfa)/len(vfa)}")
print(f"RPA is {sum(rpa)/len(rpa)}")
print(f"RCA is {sum(rca)/len(rca)}")
print(f"OA is {sum(oa)/len(oa)}")

vr=[]
vfa=[]
rca=[]
rpa=[]
oa=[]

for i in range(len(ADCtimeArr)):
    predictVal=ADCsongArr[i].copy()
    pitchTime=ADCtimeArr[i]
    voici = ADCvoiceArr[i]
    predVoicing = (voici > cutoff).astype(np.int32)
    predictVal[predVoicing == 0] = 0.0
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        ADCtargTimeArr[i],
        ADCtargArr[i],
        pitchTime,
        predictVal,
        est_voicing=predVoicing
        )
    
    vrval=mir_eval.melody.voicing_recall(ref_v, est_v)
    vfaval=mir_eval.melody.voicing_false_alarm(ref_v, est_v)
    rcaval = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    rpaval=mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    oaval=mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    
    vr.append(vrval)
    vfa.append(vfaval)
    rca.append(rcaval)
    rpa.append(rpaval)
    oa.append(oaval)
vrT+=sum(vr)/len(vr)
vfaT+=sum(vfa)/len(vfa)
rcaT+=sum(rca)/len(rca)
rpaT+=sum(rpa)/len(rpa)
oaT+=sum(oa)/len(oa)

print("---- FINAL ADC ----", cutoff)     
print(f"VR is {sum(vr)/len(vr)}")
print(f"VFA is {sum(vfa)/len(vfa)}")
print(f"RPA is {sum(rpa)/len(rpa)}")
print(f"RCA is {sum(rca)/len(rca)}")
print(f"OA is {sum(oa)/len(oa)}")

print("---- FINAL ----", cutoff)     
print(f"VR is {vrT/3}")
print(f"VFA is {vfaT/3}")
print(f"RPA is {rpaT/3}")
print(f"RCA is {rcaT/3}")
print(f"OA is {oaT/3}")
