import os
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import BertModel, XLMRobertaModel

from .helpers import bert_subword_tokenize, merge_subword_encodings_for_words, merge_subword_encodings_for_sentences

# from character_bert_master import CharacterIndexer, CharacterBertModel
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.models import Transformer, Pooling


""" ##################### """
""" classification models """
""" ##################### """


def _get_simple_linear(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    )


class SimpleMLP(nn.Module):
    def __init__(self, out_dim: int, input_dim1: int, input_dim2: int = -1):
        super(SimpleMLP, self).__init__()

        """ layers """
        self.input1_layer1 = _get_simple_linear(input_dim1, 300, 0.25)
        self.combined_dim = 300
        if input_dim2 > 0:
            self.input2_layer1 = _get_simple_linear(input_dim2, 300, 0.25)
            self.combined_dim = 300 + 300
        self.combined_layer1 = _get_simple_linear(self.combined_dim, 128, 0.25)
        self.out_layer = nn.Linear(128, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input1, input2=None, targets=None):
        input1 = input1.to(self.device)
        input1 = self.input1_layer1(input1)
        combined = input1
        if input2 is not None:
            input2 = input2.to(self.device)
            input2 = self.input2_layer1(input2)
            combined = torch.cat((input1, input2), dim=-1)
        combined = self.combined_layer1(combined)
        logits = self.out_layer(combined)

        output_dict = {}
        output_dict.update({"logits": logits})

        """ compute loss with specified criterion """
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, input1, input2=None, targets=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(input1, input2, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            assert len(preds) == len(targets), print(len(preds), len(targets))
            acc_num = sum([i == j for i, j in zip(preds, targets)])
            output_dict.update({"acc_num": acc_num})  # dims: [1]
            output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class BertMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(BertMLP, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        self.config = AutoConfig.from_pretrained(self.pretrained_path, num_labels=self.out_dim)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_path, config=self.config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.config.hidden_size
        # print(self.config)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch: List[str], targets: list = None):

        # encoding = self.bert_tokenizer(text_batch, return_tensors='pt', padding='longest', return_attention_mask=True)
        encoding = self.bert_tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                       return_attention_mask=True, max_length=200)  # padding to longest
        # max_length is taken as default if unspecificed
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        # print(input_ids.shape)
        output_dict = {}
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        output_dict.update({"logits": logits})

        """ compute loss with specified criterion """
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class FusedBertMLP(nn.Module):
    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 fusion_n: int,
                 fusion_strategy: str = "concat"  # allowed: max_pool, mean_pool, concat
                 ):
        super(FusedBertMLP, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert
        self.fusion_n = fusion_n
        self.fusion_strategy = fusion_strategy

        """ BERT modules """
        self.config = AutoConfig.from_pretrained(self.pretrained_path, num_labels=self.out_dim)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.config)
        self.bert_model.train()
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.config.hidden_size
        print(self.config)

        """linear layers"""
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if self.fusion_strategy == "concat":
            self.linear = nn.Linear(self.fusion_n * self.bert_model_outdim, self.out_dim)
        elif self.fusion_strategy in ["max_pool", "mean_pool"]:
            self.linear = nn.Linear(self.bert_model_outdim, self.out_dim)
        else:
            raise Exception("allowed types for fusion_strategy are: [max_pool, mean_pool, concat]")
        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch: List[str], targets: list = None):
        # encoding = self.bert_tokenizer(text_batch, return_tensors='pt', padding='longest', return_attention_mask=True)
        encoding = self.bert_tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                       return_attention_mask=True, max_length=200)  # padding to longest
        # max_length is taken as default if unspecificed
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        # print(input_ids.shape)
        output_dict = {}
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        n_samples = int(len(text_batch) / self.fusion_n)
        pooler_output_reshaped = pooler_output.reshape(self.fusion_n, n_samples, self.bert_model_outdim)
        if self.fusion_strategy == "max_pool":
            fused_output = torch.max(pooler_output_reshaped, dim=0)[0]
        elif self.fusion_strategy == "mean_pool":
            fused_output = torch.mean(pooler_output_reshaped, dim=0)
        elif self.fusion_strategy == "concat":
            splits = torch.split(pooler_output_reshaped, 1, 0)
            fused_output = torch.cat([torch.squeeze(split_) for split_ in splits], dim=-1)
        logits = self.linear(self.dropout(fused_output))
        output_dict.update({"logits": logits})

        """ compute loss with specified criterion """
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


# NOTE: https://github.com/pytorch/pytorch/issues/43227
# `batch_lengths.cpu()` to be used instead of `batch_lengths` if torch==1.7.0
# if torch==1.4.0, use `batch_lengths` itself in pack_padded_sequence(...)
class CharLstmModel(nn.Module):

    def __init__(self, nembs, embdim, padding_idx, hidden_size, num_layers, bidirectional, output_combination):
        super(CharLstmModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # lstm module
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        self.lstm_model = nn.LSTM(embdim, hidden_size, num_layers, batch_first=True, dropout=0.3,
                                  bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output
        assert output_combination in ["end", "max", "mean"], print(
            'invalid output_combination; required one of {"end","max","mean"}')
        self.output_combination = output_combination

    def forward(self, batch_tensor, batch_lengths):

        batch_size = len(batch_tensor)
        # print("************ stage 2")

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        embs_packed = pack_padded_sequence(embs, batch_lengths, batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(embs_packed)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # [BS, max_seq_len, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError

        return source_encodings


class CharLstmLstmMLP(nn.Module):

    def __init__(self, nchars, char_emb_dim, char_padding_idx, padding_idx, output_dim):
        super(CharLstmLstmMLP, self).__init__()

        # charlstm module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,charlstm_outputdim]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, charlstm_outputdim]
        hidden_size, num_layers, bidirectional, output_combination = 128, 1, True, "end"
        self.charlstm_model = CharLstmModel(nchars, char_emb_dim, char_padding_idx, hidden_size, num_layers,
                                            bidirectional, output_combination)
        self.charlstm_model_outdim = self.charlstm_model.lstm_model_outdim

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 256, 2
        self.lstm_model = nn.LSTM(self.charlstm_model_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.output_combination = "end"
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(self.lstm_model_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self,
                batch_idxs: "List[pad_sequence]",
                batch_char_lengths: "List[tensor]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None
                ):

        batch_size = len(batch_idxs)
        output_dict = {}

        batch_idxs = [x.to(self.device) for x in batch_idxs]
        batch_char_lengths = [x.to(self.device) for x in batch_char_lengths]
        batch_lengths = batch_lengths.to(self.device)
        if aux_word_embs is not None:
            aux_word_embs = aux_word_embs.to(self.device)

        # charlstm
        charlstm_encodings = [self.charlstm_model(pad_sequence_, lens) for pad_sequence_, lens in
                              zip(batch_idxs, batch_char_lengths)]
        charlstm_encodings = pad_sequence(charlstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"charlstm_encodings": charlstm_encodings})

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        concatenated_encodings = charlstm_encodings
        if aux_word_embs is not None:
            concatenated_encodings = torch.cat((concatenated_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        concatenated_encodings = pack_padded_sequence(concatenated_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(concatenated_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"lstm_encodings": lstm_encodings})

        # dense
        # [BS, max_nwords, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError
        # [BS,self.lstm_model_outdim]->[BS,output_dim]
        logits = self.linear(self.dropout(source_encodings))
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self,
                batch_idxs: "List[pad_sequence]",
                batch_char_lengths: "List[tensor]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(batch_idxs, batch_char_lengths, batch_lengths, aux_word_embs, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


# NOTE
# This is just a base class for CharCNNLSTMMLP
# But that class isn't implemented yet!
class CharCNNModel(nn.Module):

    def __init__(self, nembs, embdim, padding_idx, filterlens, nfilters):
        super(CharCNNModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # Unsqueeze [BS, MAXSEQ, EMDDIM] as [BS, 1, MAXSEQ, EMDDIM] and send as input
        self.convmodule = nn.ModuleList()
        for length, n in zip(filterlens, nfilters):
            self.convmodule.append(
                nn.Sequential(
                    nn.Conv2d(1, n, (length, embdim), padding=(length - 1, 0), dilation=1, bias=True,
                              padding_mode='zeros'),
                    nn.ReLU()
                )
            )
        # each conv outputs [BS, nfilters, MAXSEQ, 1]

    def forward(self, batch_tensor, batch_lengths=None):
        batch_size = len(batch_tensor)

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # [BS, max_seq_len, emb_dim]->[BS, 1, max_seq_len, emb_dim]
        embs_unsqueezed = torch.unsqueeze(embs, dim=1)

        # [BS, 1, max_seq_len, emb_dim]->[BS, out_channels, max_seq_len, 1]->[BS, out_channels, max_seq_len]
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convmodule]

        # [BS, out_channels, max_seq_len]->[BS, out_channels]
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # cat( [BS, out_channels] )->[BS, sum(nfilters)]
        source_encodings = torch.cat(maxpool_conv_outputs, dim=1)
        return source_encodings


class ScLstmMLP(nn.Module):

    def __init__(self, screp_dim, padding_idx, output_dim):

        super(ScLstmMLP, self).__init__()

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 256, 2
        self.lstm_model = nn.LSTM(screp_dim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.output_combination = "end"
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(self.lstm_model_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self,
                batch_screps: "List[pad_sequence]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None
                ):

        batch_size = len(batch_screps)
        output_dict = {}

        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0).to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        if aux_word_embs is not None:
            aux_word_embs = aux_word_embs.to(self.device)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        concated_encodings = batch_screps
        if aux_word_embs is not None:
            concated_encodings = torch.cat((concated_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        concated_encodings = pack_padded_sequence(concated_encodings, batch_lengths,
                                                  batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(concated_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"lstm_encodings": lstm_encodings})

        # dense
        # [BS, max_nwords, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError
        # [BS,self.lstm_model_outdim]->[BS,output_dim]
        logits = self.linear(self.dropout(source_encodings))
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self,
                batch_screps: "List[pad_sequence]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(batch_screps, batch_lengths, aux_word_embs, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class SentenceBert(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(SentenceBert, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, targets=None):
        batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(text_batch, self.bert_tokenizer,
                                                                               max_len=200)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       targets: List = None):

        batch_size = len(batch_sentences)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        #
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_sentences(bert_seq_encodings, seq_splits)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        #
        # although no padding values are added anywhere, we just copy this snippet from WholeWordBert
        sentence_pooled_output = pad_sequence(bert_encodings_splitted,
                                              batch_first=True,
                                              padding_value=0
                                              ).to(self.device)  # [BS,self.bert_model_outdim]
        pooler_output = self.dropout(sentence_pooled_output)
        output_dict.update({"sentence_pooled_output": sentence_pooled_output})

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class SentenceBertForSemanticSimilarity(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(SentenceBertForSemanticSimilarity, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert
        self.fusion_n = 2

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim * 4, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, targets=None):
        batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(text_batch, self.bert_tokenizer,
                                                                               max_len=200)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       targets: List = None):

        batch_size = len(batch_sentences)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        #
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_sentences(bert_seq_encodings, seq_splits)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        #
        # although no padding values are added anywhere, we just copy this snippet from WholeWordBert
        sentence_pooled_output = pad_sequence(bert_encodings_splitted,
                                              batch_first=True,
                                              padding_value=0
                                              ).to(self.device)  # [BS,self.bert_model_outdim]

        n_samples = int(len(batch_sentences) / self.fusion_n)
        sentence_pooled_output = sentence_pooled_output.reshape(self.fusion_n, n_samples, self.bert_model_outdim)

        u, v = sentence_pooled_output[0], sentence_pooled_output[1]
        u_minus_v = torch.abs(u - v)
        u_product_v = torch.mul(u, v)
        concat_representation = torch.cat((u, v, u_minus_v, u_product_v), dim=-1)  # [BS/2, 4*self.bert_model_outdim]

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        pooler_output = self.dropout(concat_representation)
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class WholeWordBertMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(WholeWordBertMLP, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, targets=None):
        batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(text_batch, self.bert_tokenizer,
                                                                               max_len=200)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       targets: List = None):

        batch_size = len(batch_sentences)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        #
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', keep_terminals=False,
                                               device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             ).to(self.device)  # [BS,max_nwords,self.bert_model_outdim]
        output_dict.update({"bert_merged_encodings": bert_merged_encodings})
        #
        pooler_output = self.dropout(pooler_output)
        output_dict.update({"pooler_output": pooler_output})

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class WholeWordBertXXXInformedMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 n_lang_ids: int,
                 device: str,
                 token_type_pad_idx: int
                 ):
        super(WholeWordBertXXXInformedMLP, self).__init__()

        assert os.path.exists(pretrained_path), print()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.n_lang_ids = n_lang_ids
        self.token_type_pad_idx = token_type_pad_idx

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_config.type_vocab_size = self.n_lang_ids
        print(f"self.bert_config.type_vocab_size = {self.bert_config.type_vocab_size}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)

        model_type = self.pretrained_path.strip("/").split("/")[-1]
        if model_type in ["bert-base-cased", "bert-base-multilingual-cased"]:
            self.bert_model = BertModel(config=self.bert_config)
        elif model_type == "xlm-roberta-base":
            self.bert_model = XLMRobertaModel(config=self.bert_config)
        else:
            raise NotImplementedError("add conditions for more models")

        print(f"\nLoading weights from self.pretrained_path:{self.pretrained_path}")
        pretrained_dict = torch.load(f"{self.pretrained_path}/pytorch_model.bin", map_location=device)
        bert_model_state_dict = self.bert_model.state_dict()
        # 1. filter out unnecessary keys
        used_dict = {}
        for k, v in bert_model_state_dict.items():
            if "classifier.weight" in k or "classifier.bias" in k:
                print(k)
                continue
            if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
                used_dict[k] = pretrained_dict[k]
            elif ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
            elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
                used_dict[k] = pretrained_dict["bert." + k]
            elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
                used_dict[k] = pretrained_dict["roberta." + k]
        unused_dict = {k: v for k, v in bert_model_state_dict.items() if k not in used_dict}
        # 2. overwrite entries in the existing state dict
        bert_model_state_dict.update(used_dict)
        # 3. load the new state dict
        self.bert_model.load_state_dict(bert_model_state_dict)
        # 4. print unused_dict
        print("WARNING !!!")
        print(
            f"Following {len([*unused_dict.keys()])} keys are not updated from {self.pretrained_path}/pytorch_model.bin")
        print(f"  →→ {[*unused_dict.keys()]}")

        self.bert_model_outdim = self.bert_config.hidden_size

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, lang_ids_batch, targets=None):
        batch_sentences, batch_bert_dict, batch_splits, = \
            bert_subword_tokenize(text_batch, self.bert_tokenizer, max_len=200,
                                  batch_lang_sequences=lang_ids_batch, token_type_padding_idx=self.token_type_pad_idx)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       targets: List = None):

        batch_size = len(batch_sentences)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device),
            token_type_ids=batch_bert_dict["token_type_ids"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        #
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             ).to(self.device)  # [BS,max_nwords,self.bert_model_outdim]
        output_dict.update({"bert_merged_encodings": bert_merged_encodings})
        #
        pooler_output = self.dropout(pooler_output)
        output_dict.update({"pooler_output": pooler_output})

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch, lang_ids_batch, targets=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, lang_ids_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class WholeWordBertLstmMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(WholeWordBertLstmMLP, self).__init__()

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ LSTM modules """
        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 256, 2
        self.lstm_model = nn.LSTM(self.bert_model_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        """ output module """
        assert out_dim > 0
        self.output_combination = "end"
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.lstm_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, targets=None):
        batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(text_batch, self.bert_tokenizer,
                                                                               max_len=200)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       targets: List = None):

        batch_size = len(batch_sentences)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        batch_lengths = torch.tensor([len(x) for x in bert_encodings_splitted]).long().to(self.device)
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             ).to(self.device)  # [BS,max_nwords,self.bert_model_outdim]
        output_dict.update({"bert_merged_encodings": bert_merged_encodings})

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        intermediate_encodings = pack_padded_sequence(bert_merged_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"lstm_encodings": lstm_encodings})

        # dense
        # [BS, max_nwords, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError
        # [BS,self.lstm_model_outdim]->[BS,output_dim]
        logits = self.linear(self.dropout(source_encodings))
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class WholeWordBertScLstmMLP(nn.Module):

    def __init__(self,
                 # for sc representations
                 screp_dim: int,
                 # for lstm & others
                 out_dim: int,
                 # for bert
                 pretrained_path: str,
                 finetune_bert: bool):

        super(WholeWordBertScLstmMLP, self).__init__()

        """ parameters """
        self.screp_dim = screp_dim
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """  representation-1: ScLstm modules """
        # nothing is required

        """  representation-2: BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ concatenation & final lstm module """
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 256, 2
        self.lstm_model = nn.LSTM(self.screp_dim + self.bert_model_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        """ output module """
        assert out_dim > 0
        self.output_combination = "end"
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.lstm_model_outdim, self.out_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self,
                # for bert
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: List[List[int]],
                # for sc representations
                batch_screps: "List[pad_sequence]",
                # for lstm & others
                targets: "tensor" = None
                ):

        batch_size = len(batch_screps)
        output_dict = {}

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        batch_lengths = torch.tensor([len(x) for x in bert_encodings_splitted]).long().to(self.device)
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             ).to(self.device)  # [BS,max_nwords,self.bert_model_outdim]
        output_dict.update({"bert_merged_encodings": bert_merged_encodings})

        # for sc representations and concat
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0).to(self.device)
        concated_encodings = torch.cat((batch_screps, bert_merged_encodings), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        concated_encodings = pack_padded_sequence(concated_encodings, batch_lengths,
                                                  batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(concated_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"lstm_encodings": lstm_encodings})

        # dense
        # [BS, max_nwords, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError
        # [BS,self.lstm_model_outdim]->[BS,output_dim]
        logits = self.linear(self.dropout(source_encodings))
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self,
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: List[List[int]],
                batch_screps: "List[pad_sequence]",
                targets: "tensor" = None
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(batch_bert_dict, batch_splits, batch_screps, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class WholeWordBertCharLstmLstmMLP(nn.Module):

    def __init__(self,
                 # for CharLstm representations
                 nchars: int,
                 char_emb_dim: int,
                 char_padding_idx: int,
                 # for lstm & others
                 out_dim: int,
                 # for bert
                 pretrained_path: str,  # eg. "xlm-roberta-base", "bert-base-mulilingual-cased", etc.
                 freezable_pretrained_path: str = None,
                 # eg. "../checkpoints/arxiv-{dataset-name}/Hinglish/baseline/text_raw"
                 device=None
                 ):

        super(WholeWordBertCharLstmLstmMLP, self).__init__()

        """ parameters """
        self.nchars = nchars
        self.char_emb_dim = char_emb_dim
        self.char_padding_idx = char_padding_idx
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.freezable_pretrained_path = freezable_pretrained_path

        """ checks """
        if self.freezable_pretrained_path:
            print("running v2 of the model i suppose; bert weights will be frozen")
            self.finetune_bert = False
        else:
            print("bert weights are being finetuned")
            self.finetune_bert = True

        """ representation-1: CharLstm representation modules """
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,charlstm_outputdim]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, charlstm_outputdim]
        hidden_size, num_layers, bidirectional, output_combination = 128, 1, True, "end"
        self.charlstm_model = CharLstmModel(nchars, char_emb_dim, char_padding_idx, hidden_size, num_layers,
                                            bidirectional, output_combination)
        self.charlstm_model_outdim = self.charlstm_model.lstm_model_outdim

        """ representation-2: BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModel.from_pretrained(self.pretrained_path, config=self.bert_config)

        if self.freezable_pretrained_path:
            # use self.pretrained_path as template to load bert/roberta model
            # then load weights from your pretrained checkpoint
            print(f"\nLoading weights from {freezable_pretrained_path}")
            pretrained_dict = torch.load(os.path.join(freezable_pretrained_path, "model.pth.tar"),
                                         map_location=torch.device(device))["model_state_dict"]
            # pretrained_dict = {".".join(k.split(".")[1:]): v for k, v in pretrained_dict.items()}
            model_dict = self.bert_model.state_dict()
            # 1. filter out unnecessary keys
            used_dict = {}
            for k, v in model_dict.items():
                if "classifier.weight" in k or "classifier.bias" in k:
                    print(k)
                    continue
                if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
                    used_dict[k] = pretrained_dict[k]
                elif ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                    ".".join(k.split(".")[1:])].shape:
                    used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
                elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                    "bert." + ".".join(k.split(".")[1:])].shape:
                    used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
                elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                    "roberta." + ".".join(k.split(".")[1:])].shape:
                    used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
                elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
                    used_dict[k] = pretrained_dict["bert." + k]
                elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
                    used_dict[k] = pretrained_dict["roberta." + k]
                elif "bert_model." + k in pretrained_dict and v.shape == pretrained_dict["bert_model." + k].shape:
                    used_dict[k] = pretrained_dict["bert_model." + k]
            unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(used_dict)
            # 3. load the new state dict
            self.bert_model.load_state_dict(model_dict)
            # 4. print unused_dict
            print("WARNING !!!")
            print(
                f"Following {len([*unused_dict.keys()])} keys are not updated from {freezable_pretrained_path}/model.pth.tar")
            print(f"  →→ {[*unused_dict.keys()]}")

        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ concatenation & final lstm module """
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 256, 2
        self.lstm_model = nn.LSTM(self.charlstm_model_outdim + self.bert_model_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)
        self.lstm_model_outdim = hidden_size * 2 if bidirectional else hidden_size

        """ output module """
        assert out_dim > 0
        self.output_combination = "end"
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.lstm_model_outdim, self.out_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self,
                # for bert
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: List[List[int]],
                # for CharLstm representations
                batch_idxs: "List[pad_sequence]",
                batch_char_lengths: "List[tensor]",
                batch_lengths: "tensor",
                # for lstm & others
                targets: "tensor" = None
                ):

        batch_size = len(batch_idxs)
        output_dict = {}
        batch_idxs = [x.to(self.device) for x in batch_idxs]
        batch_char_lengths = [x.to(self.device) for x in batch_char_lengths]
        batch_lengths = batch_lengths.to(self.device)

        # charlstm
        charlstm_encodings = [self.charlstm_model(pad_sequence_, lens) for pad_sequence_, lens in
                              zip(batch_idxs, batch_char_lengths)]
        charlstm_encodings = pad_sequence(charlstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"charlstm_encodings": charlstm_encodings})

        # bert
        bert_outputs = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device)
        )
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        bert_encodings = self.bert_dropout(bert_encodings)
        bert_encodings_splitted = \
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)]
        batch_lengths = torch.tensor([len(x) for x in bert_encodings_splitted]).long().to(self.device)
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             ).to(self.device)  # [BS,max_nwords,self.bert_model_outdim]
        output_dict.update({"bert_merged_encodings": bert_merged_encodings})

        # concat
        concated_encodings = torch.cat((charlstm_encodings, bert_merged_encodings), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstm_model_outdim]
        concated_encodings = pack_padded_sequence(concated_encodings, batch_lengths,
                                                  batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstm_model(concated_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)
        output_dict.update({"lstm_encodings": lstm_encodings})

        # dense
        # [BS, max_nwords, self.lstm_model_outdim]->[BS, self.lstm_model_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstm_model_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError
        # [BS,self.lstm_model_outdim]->[BS,output_dim]
        logits = self.linear(self.dropout(source_encodings))
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self,
                # for bert
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: List[List[int]],
                # for CharLstm representations
                batch_idxs: "List[pad_sequence]",
                batch_char_lengths: "List[tensor]",
                batch_lengths: "tensor",
                # for lstm & others
                targets: "tensor" = None
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(batch_bert_dict, batch_splits,
                                       batch_idxs, batch_char_lengths, batch_lengths, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class CNNCharacterBertMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(CNNCharacterBertMLP, self).__init__()

        # """ tokenizer config path """
        # bert_base_uncased_ckpt = "../checkpoints/pretrained/bert-base-uncased"
        # if not os.path.exists(bert_base_uncased_ckpt):
        #     raise Exception("please download 'bert-base-uncased' using ./character_bert_master/download.py as: "
        #                     "python download.py --model='bert-base-uncased'")
        # logging.info(f"using config from 'bert-base-uncased' ({bert_base_uncased_ckpt}) to initialize bert tokenizer")

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = CharacterBertModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_config.hidden_size

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        """ others """
        self.indexer = CharacterIndexer()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch: List[str], targets: list = None):
        """
        See https://github.com/helboukkouri/character-bert#example-2-using-characterbert-for-binary-classification
            for more details
        """

        batch_size = len(text_batch)
        output_dict = {}

        tokenized_text = [self.bert_tokenizer.basic_tokenizer.tokenize(txt) for txt in text_batch]
        input_tensor = self.indexer.as_padded_tensor(tokenized_text, maxlen=200).to(self.device)

        # bert_outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        bert_outputs = self.bert_model(input_tensor)
        bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        pooler_output = self.dropout(pooler_output)
        output_dict.update({"pooler_output": pooler_output})

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


class SentenceTransformersBertMLP(nn.Module):

    def __init__(self,
                 out_dim: int,
                 pretrained_path: str,
                 finetune_bert: bool,
                 ):
        super(SentenceTransformersBertMLP, self).__init__()

        # """ tokenizer config path """
        # bert_base_uncased_ckpt = "../checkpoints/pretrained/bert-base-uncased"
        # if not os.path.exists(bert_base_uncased_ckpt):
        #     raise Exception("please download 'bert-base-uncased' using ./character_bert_master/download.py as: "
        #                     "python download.py --model='bert-base-uncased'")
        # logging.info(f"using config from 'bert-base-uncased' ({bert_base_uncased_ckpt}) to initialize bert tokenizer")

        """ parameters """
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.finetune_bert = finetune_bert

        """ BERT modules """
        # self.bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        # self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        # self.bert_model = CharacterBertModel.from_pretrained(self.pretrained_path, config=self.bert_config)
        try:
            tmp = Transformer(self.pretrained_path, max_seq_length=200)
            self.bert_model = SentenceTransformer(modules=[Transformer(self.pretrained_path, max_seq_length=200),
                                                           Pooling(tmp.get_word_embedding_dimension(),
                                                                   pooling_mode_cls_token=False,
                                                                   pooling_mode_mean_tokens=True)])
            del tmp
        except OSError:
            self.bert_model = SentenceTransformer(self.pretrained_path)
            self.bert_model.max_seq_length = 200
        if not self.finetune_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.bert_model_outdim = self.bert_model.get_sentence_embedding_dimension()

        """ output module """
        assert out_dim > 0
        self.dropout = nn.Dropout(p=0.25)
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.linear = nn.Linear(self.bert_model_outdim, out_dim)

        """ learning criterion """
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch: List[str], targets: list = None):
        """
        See https://github.com/helboukkouri/character-bert#example-2-using-characterbert-for-binary-classification
            for more details
        """

        batch_size = len(text_batch)
        output_dict = {}

        pooler_output = self.bert_model.encode(text_batch,
                                               batch_size=len(text_batch),
                                               convert_to_tensor=True,
                                               show_progress_bar=False)

        # tokenized_text = [self.bert_tokenizer.basic_tokenizer.tokenize(txt) for txt in text_batch]
        # input_tensor = self.indexer.as_padded_tensor(tokenized_text, maxlen=200).to(self.device)
        #
        # # bert_outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        # bert_outputs = self.bert_model(input_tensor)
        # bert_encodings, pooler_output = bert_outputs[0], bert_outputs[1]
        pooler_output = self.dropout(pooler_output)
        output_dict.update({"pooler_output": pooler_output})

        # [BS,self.bert_model_outdim]->[BS,output_dim]
        logits = self.linear(pooler_output)
        output_dict.update({"logits": logits})

        # loss
        if targets is not None:
            targets = torch.tensor(targets).to(self.device)
            loss = self.criterion(logits, targets)
            output_dict.update({"loss": loss})

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            if targets is not None:
                assert len(preds) == len(targets), print(len(preds), len(targets))
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict


""" ############## """
""" tagging models """
""" ############## """


class WholeWordBertForSeqClassificationAndTagging(nn.Module):

    def __init__(self,
                 sent_out_dim,
                 lang_out_dim,
                 pretrained_path):

        super(WholeWordBertForSeqClassificationAndTagging, self).__init__()

        assert sent_out_dim > 0
        assert lang_out_dim > 0

        """ bert """
        self.pretrained_path = pretrained_path
        self.bert_dropout = torch.nn.Dropout(0.25)
        self.dropout = torch.nn.Dropout(0.25)
        self.sent_out_dim = sent_out_dim
        self.config = AutoConfig.from_pretrained(self.pretrained_path, num_labels=self.sent_out_dim,
                                                 output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_path, config=self.config)
        self.bert_model.train()
        self.bert_model_outdim = self.bert_model.config.hidden_size
        # Uncomment to freeze BERT layers
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        """ dense layer """
        self.linear = nn.Linear(self.bert_model_outdim, lang_out_dim)

        """ learning criterions """
        self.ignore_index = -1
        self.sa_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.la_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, text_batch, sa_targets=None, lid_targets=None):
        max_len = 400
        batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(text_batch,
                                                                               self.bert_tokenizer,
                                                                               max_len=max_len)
        if lid_targets:
            for i, (txt, batch_split, lid_target) in enumerate(zip(text_batch, batch_splits, lid_targets)):
                if len(batch_split) != len(lid_target):
                    # print(f"\nINFO :: a sentences was found to have more tokrns than {max_len}. "
                    #       f"Found {len(txt.split(' '))} words but retained only {max_len}. "
                    #       f"Trimming target lang ids too! Sentence: {txt}")
                    lid_target = lid_target[:len(batch_split)]
                    lid_targets[i] = lid_target
                assert len(batch_split) == len(lid_target)
        return self.custom_forward(batch_sentences, batch_bert_dict, batch_splits, sa_targets, lid_targets)

    def custom_forward(self,
                       batch_sentences: List[str],
                       batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                       batch_splits: List[List[int]],
                       sa_targets: List = None,
                       lid_targets: List[List[int]] = None):

        # BS X self.sent_out_dim, Tuple of hidden states for each layer
        sa_logits, hidden_states = self.bert_model(
            input_ids=batch_bert_dict["input_ids"].to(self.device),
            attention_mask=batch_bert_dict["attention_mask"].to(self.device),
            # token_type_ids=batch_bert_dict["token_type_ids"].to(self.device),
        )

        output_dict = {"sa_logits": sa_logits, "loss": 0, "modified_sentences": batch_sentences}

        """ compute loss with specified criterion """
        if sa_targets is not None:
            sa_targets = torch.tensor(sa_targets).to(self.device)
            loss = self.sa_criterion(sa_logits, sa_targets)
            output_dict.update({"sa_loss": loss})
            output_dict["loss"] += loss

        bert_encodings = hidden_states[-1]

        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bert_model_outdim
        bert_merged_encodings = pad_sequence(
            [merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, mode='avg', device=self.device)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],
            batch_first=True,
            padding_value=0
        ).to(self.device)

        # dense
        # [BS,max_nwords,self.bert_model_outdim]->[BS,max_nwords,lang_output_dim]
        lid_logits = self.linear(self.dropout(bert_merged_encodings))
        output_dict["lid_logits"] = lid_logits

        if lid_targets is not None:
            batch_lid_targets = pad_sequence(
                [torch.tensor(lids[:len(sent.split(" "))]) for lids, sent in zip(lid_targets,
                                                                                 batch_sentences)], batch_first=True,
                padding_value=self.ignore_index).to(self.device)
            loss = self.la_criterion(lid_logits.reshape(-1, lid_logits.shape[-1]), batch_lid_targets.reshape(-1))
            output_dict.update({"lid_loss": loss})
            output_dict["loss"] += loss

        return output_dict

    def predict(self, text_batch: List[str], targets: list = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, targets)
            logits = output_dict["sa_logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})  # dims: [batch_size]
            assert len(preds) == len(targets), print(len(preds), len(targets))
            acc_num = sum([i == j for i, j in zip(preds, targets)])
            output_dict.update({"acc_num": acc_num})  # dims: [1]
            output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]
        if was_training:
            self.train()
        return output_dict

    def predict_lid(self, text_batch: List[str], targets: List[List[int]] = None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(text_batch, None, targets)
            logits = output_dict["lid_logits"]
            probs = F.softmax(logits, dim=-1)
            output_dict.update({"probs": probs.cpu().detach().numpy().tolist()})
            arg_max_prob = torch.argmax(probs, dim=-1).reshape(-1)
            preds = arg_max_prob.cpu().detach().numpy().tolist()
            output_dict.update({"preds": preds})

            if targets is not None:

                targets = pad_sequence([torch.tensor(lids[:len(sent.split(" "))])
                                        for lids, sent in zip(targets, output_dict["modified_sentences"])],
                                       batch_first=True,
                                       padding_value=self.ignore_index).reshape(-1).cpu().detach().numpy().tolist()
                new_preds, new_targets = [], []
                for i, j in zip(preds, targets):
                    if j != -1:
                        new_preds.append(i)
                        new_targets.append(j)
                preds = new_preds
                targets = new_targets
                output_dict.update({"preds": preds})
                output_dict.update({"targets": targets})
                acc_num = sum([i == j for i, j in zip(preds, targets)])
                output_dict.update({"acc_num": acc_num})  # dims: [1]
                output_dict.update({"acc": acc_num / len(targets)})  # dims: [1]

        if was_training:
            self.train()
        return output_dict
