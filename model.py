import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size=16, n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, input):
        if USE_CUDA:
            hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda()
            context = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda()
        else:
            hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
            context = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))

        return hidden, context
     
    def forward(self, input, input_masking):
        """
        input : B,T
        input_masking : B,T
        """
        hidden = self.init_hidden(input)
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        
        real_context = []
        
        for i, o in enumerate(output):
            # B,T,D
            real_length = input_masking[i].data.tolist().count(0)
            real_context.append(o[real_length-1])
            
        return output, torch.cat(real_context).view(input.size(0), -1).unsqueeze(1)

class Decoder(nn.Module):
    
    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        # self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size*2, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)   # Attention
        self.slot_out = nn.Linear(self.hidden_size*2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size*2, self.intent_size)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T
        output : B,1,D
        """
        hidden = hidden.squeeze(0).unsqueeze(2) # (1,B,D) -> (B,D,1)
        batch_size = encoder_outputs.size(0)    # B
        max_len = encoder_outputs.size(1)       # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))   # (B*T,D) -> (B*T,D)
        energies = energies.view(batch_size, max_len, -1)       # (B*T,D) -> B,T,D
        attn_energies = energies.bmm(hidden).transpose(1, 2)    # (B,T,D) * (B,D,1) -> (B,1,T)
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)       # PAD masking

        # print(attn_energies.shape)
        alpha = F.softmax(attn_energies, dim=1) # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)    # B,1,T * B,T,D => B,1,D
        
        return context
    
    def init_hidden(self, input):
        if USE_CUDA:
            hidden = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size)).cuda()
            context = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size)).cuda()
        else:
            hidden = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size))
            context = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size))

        return hidden, context
    
    def forward(self, input, context, encoder_outputs, encoder_maskings, training=True):
        """
        input : B,1
        context : B,1,D
        encoder_outputs : B,T,D
        """
        # Get the embedding of the current input word
        # B,1 -> B,1,D
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)    # B,T,D -> T,B,D
        length = encoder_outputs.size(1)            # T
        for i in range(length):
            aligned = aligns[i].unsqueeze(1)        # B,D -> B,1,D
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2), hidden)
            
            # Intent classification
            if i == 0:
                intent_hidden = hidden[0].clone()   # 1,B,D
                intent_context = self.Attention(intent_hidden, encoder_outputs, encoder_maskings)   # B,1,D
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)    # 1,B,D
                intent_score = self.intent_out(concated.squeeze(0)) # B,D
            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)                   # 1,B,D -> 1,B,2*D
            score = self.slot_out(concated.squeeze(0))      # B,slot_size
            # print(score.shape)
            softmaxed = F.log_softmax(score, dim=1)
            decode.append(softmaxed)
            _, input = torch.max(softmaxed, 1)              # B
            embedded = self.embedding(input.unsqueeze(1))   # B,1 -> B,1,D
            context = self.Attention(hidden[0], encoder_outputs, encoder_maskings)          # B,1,D

        slot_scores = torch.cat(decode, 1)          # B,slot_size*T

        return slot_scores.view(input.size(0)*length, -1), intent_score                     # B*T,slot_size