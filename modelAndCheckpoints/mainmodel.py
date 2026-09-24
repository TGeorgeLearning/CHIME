import torch
import torch.nn as nn
import torch.nn.functional as 

class CHIME(nn.Module): 

    def __init__(self,lr):
        super().__init__()
        self.lr=lr
        self.currEpoch=0
       
        self.finalLayer = nn.Sequential(
                                        nn.LazyConv2d(12,5,padding='same'),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),
            
                                        nn.LazyConv2d(6,(53,3),padding='same'),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),
            
                                        nn.LazyConv2d(4,(3,7),padding='same'),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),
            
                                        nn.LazyConv2d(2,1))
                                       
                                            
            

        self.pitch = nn.Sequential(
                                    nn.ReLU(),nn.Dropout(0.15),
                                        nn.LazyLinear(288))

        self.pitchgru = nn.Sequential(
            nn.ReLU(),nn.Dropout(0.1), nn.LazyLinear(288),
            nn.GRU(
            input_size=288,
            hidden_size=288,
            batch_first=True,
            bidirectional=True)) 



        
        self.analyze = nn.Sequential(
                                        nn.LazyConv2d(12,5,padding=(0,2),stride=(2,1)),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),
            
                                        nn.LazyConv2d(6,(53,3),padding=(0,1),stride=(3,1)),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),

                                        nn.LazyConv2d(6,(5,1),stride=(2,1)),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU(),
        
                                        nn.AdaptiveMaxPool2d((8,86)))

        
        self.final=nn.Sequential(nn.LazyLinear(16),nn.ReLU(),nn.Dropout(0.2),nn.LazyLinear(1))

        self.voicegru = nn.GRU(
            input_size=48,
            hidden_size=36,
            bidirectional=True,
            batch_first=True,num_layers=3,dropout=0.05)
        
            
    def forward(self, x):

        pitchX = self.finalLayer(x)
        
        B = x.shape[0]

        pitchX=pitchX.flatten(1,2)
        pitchX = pitchX.transpose(1, 2)
        lstmX,_=self.plstm(pitchX)
    
        pitchPred=self.pitch(lstmX)
        
        pitch_map = torch.softmax(pitchPred, dim=-1)
        pitch_map = pitch_map.permute(0, 2, 1)
        pitch_map = pitch_map.unsqueeze(1)
        
        combined = torch.cat([pitch_map, x], dim=1)    

        voiceX = self.analyze(combined)
        voiceX=voiceX.flatten(1,2)
        voiceX = voiceX.permute(0,2,1)

        finalVoice,_ = self.vlstm(voiceX)
        
        finalVoice=self.final(finalVoice)

        return [pitchPred,finalVoice]

        


    def loss(self, predicted, targets, averaged=True):
        
        voicePred = predicted[1]
        pitchPred = predicted[0]

        voiceTarget = (targets >= 0).float()
        pitchTarget = targets
        
        pitchLoss = F.cross_entropy(
            pitchPred.reshape(-1, pitchPred.shape[-1]),
            pitchTarget.reshape(-1),
            ignore_index=-1
        )

        finalLoss = pitchLoss

        if (self.currEpoch>2):
            voiceLoss = F.binary_cross_entropy_with_logits(
                voicePred.squeeze(-1),
                voiceTarget,
                pos_weight=torch.tensor(
                        [1.5],
                        dtype=torch.float32,
                        device=targets.device
                    ),
                reduction='mean'
            )
            
            finalLoss = 4*finalLoss + voiceLoss
      
            

        return finalLoss

            

    def trainStep(self,batch):    
        forwardRes = self.forward(*batch[:-1])
        l = self.loss(forwardRes,batch[-1])
        accuracy = self.valAccuracy(forwardRes,batch[-1])
        return l,accuracy  
    
    def computeValAccuracy(self):
        base_fmin = librosa.note_to_hz('C2') 
        MDBsongArr=[]
        MDBvoiceArr=[]
        MDBtimeArr=[]
        MDBtargArr=[]
        MDBtargTimeArr=[]
        for song in MDBValSongs: # This is defined below
            print(song)
        
            hcqt = np.load(f'{storeHCQTPath}{song}.npy') # The HCQTs should be precomputed for efficiency 
            window = 172
            hop = 86
            
            targetData, freq = targetFreq(f'{mdbMidi}{song}_MELODY2.csv')
            MDBtargTimeArr.append(targetData)
            MDBtargArr.append(freq)
            
            num_frames = hcqt.shape[2]
            num_classes = 288
            
            # Accumulators
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
            
        bestoa=0
        bestcutoff=0
        for cutoff in range(0,1001):
            cutoff=cutoff/1000
            vr=[]
            vfa=[]
            rca=[]
            rpa=[]
            oa=[]
           
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
   
            candidateOA = sum(oa)/len(oa)
            if (sum(vr)/len(vr) < 0.2):
                break
            if (candidateOA>bestoa):
                bestoa=candidateOA
                bestcutoff=cutoff
        print(f"Found best oa at cutoff {bestcutoff} and oa {bestoa}")
        return bestoa
    

    def config_optimiser(self):
      return torch.optim.Adam(self.parameters(),lr=self.lr) 
  
from torchinfo import summary
model = Flower(lr=0.001).to(gpu()) # Moved to GPU for summary to avoid device mismatch error
summary(model,input_size=(1,6,288,172))