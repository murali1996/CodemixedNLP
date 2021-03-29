import os
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ParameterList, Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel

from ..datasets import create_path
from .helpers import get_model_nparams, merge_subword_encodings_for_sentences, merge_subword_encodings_for_words

# from character_bert_master import CharacterIndexer, CharacterBertModel
# from modeling.character_cnn import CharacterCNN
# self.bert_model.word_embeddings = CharacterCNN(requires_grad=True, output_dim=self.bert_config.hidden_size)

""" ############## """
"""     helpers    """
""" ############## """


def load_bert_pretrained_weights(model, load_path, device):
    print(f"\nLoading weights from directory: {load_path}")
    pretrained_dict = torch.load(f"{load_path}/pytorch_model.bin", map_location=torch.device(device))
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    used_dict = {}
    for k, v in model_dict.items():
        if "classifier.weight" in k or "classifier.bias" in k:
            print(f"Ignoring to load '{k}' from custom pretrained model")
            continue
        if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
            used_dict[k] = pretrained_dict[k]
        elif ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict[".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
        elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict["bert." + ".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
        elif "bert." + ".".join(k.split(".")[3:]) in pretrained_dict \
                and v.shape == pretrained_dict["bert." + ".".join(k.split(".")[3:])].shape:
            used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[3:])]
        elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict["roberta." + ".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
        elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
            used_dict[k] = pretrained_dict["bert." + k]
        elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
            used_dict[k] = pretrained_dict["roberta." + k]
    unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(used_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. print unused_dict
    print(f"WARNING !!! Following {len([*unused_dict.keys()])} keys are not loaded from {load_path}/pytorch_model.bin")
    print(f"      →→ {[*unused_dict.keys()]}")
    return model


class BertAttributes:
    def __init__(self,
                 pretrained_name_or_path: str = "bert-base-multilingual-cased",
                 input_representation: str = "subwords",
                 output_representation: str = "cls",
                 freeze_bert: bool = False,
                 n_token_type_ids: int = None,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):
        assert input_representation in ["subwords", "charcnn"]
        assert output_representation in ["cls", "meanpool"]
        assert device in ["cpu", "cuda"]

        self.input_representation = input_representation
        self.output_representation = output_representation
        self.pretrained_name_or_path = pretrained_name_or_path
        self.finetune_bert = not freeze_bert
        self.n_token_type_ids = n_token_type_ids
        self.device = device

        # self.output_normalization = False
        # self.max_len = 200


class LstmAttributes:
    def __init__(self,
                 embdim: int,
                 hidden_size: int,
                 num_layers: int,
                 n_embs: int = None,
                 padding_idx: int = None,
                 input_representation: str = "char",
                 output_representation: str = "end",
                 freeze_lstm: bool = False,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):
        assert input_representation in ["fasttext", "sc", "char"]
        allowed_output_representation = ["", "end", "max", "meanpool"]
        assert output_representation in allowed_output_representation, print(
            f'invalid output_representation {output_representation}; required one of {allowed_output_representation}')
        assert device in ["cpu", "cuda"]

        self.input_representation = input_representation
        self.output_representation = output_representation
        self.embdim = embdim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_embs = n_embs
        self.padding_idx = padding_idx
        self.bidirectional = True
        self.finetune_lstm = not freeze_lstm
        self.device = device

        # self.output_normalization = False
        # self.max_len = 200


class TaskAttributes:
    def __init__(self,
                 name: str,  # "classification" or "seq_tagging"
                 nlabels: int,
                 adapter_type: str = None,  # "lstm" is only supported for now
                 ignore_index: int = -1,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):
        assert adapter_type in ["lstm", ] if adapter_type else adapter_type is None

        self.name = name
        self.nlabels = nlabels
        self.adapter_type = adapter_type
        self.ignore_index = ignore_index
        self.device = device

        # self.loss_weightage = None


""" ############## """
"""  base models   """
""" ############## """


class VanillaBiLSTM(nn.Module):

    def __init__(self, config):
        super(VanillaBiLSTM, self).__init__()

        self.embdim = config.get("embdim")
        self.hidden_size = config.get("hidden_size")
        self.num_layers = config.get("num_layers")
        self.n_embs = config.get("n_embs", None)
        self.padding_idx = config.get("padding_idx", None)
        self.bidirectional = config.get("bidirectional", True)
        self.output_representation = config.get("output_representation", "end")

        # Embeddings
        self.embeddings = None
        if self.n_embs:
            assert self.padding_idx, print("padding idx is required for initializing embedding matrix")
            self.embeddings = nn.Embedding(self.n_embs, self.embdim, padding_idx=self.padding_idx)
            # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
            self.embeddings.weight.requires_grad = True

        # lstm module
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        self.lstm = nn.LSTM(self.embdim,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            dropout=0.3,
                            bidirectional=self.bidirectional)
        self.outdim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

    def _forward(self, batch_tensor, batch_lengths):

        batch_size = len(batch_tensor)

        # Embeddings
        if self.embeddings:
            # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
            embs = self.embeddings(batch_tensor)
        else:
            # [BS, max_seq_len, emb_dim]
            embs = batch_tensor

        # lstm module
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.outdim]
        embs_packed = pack_padded_sequence(embs, batch_lengths, batch_first=True, enforce_sorted=False)
        last_hidden_state, (h_n, c_n) = self.lstm(embs_packed)
        last_hidden_state, _ = pad_packed_sequence(last_hidden_state, batch_first=True, padding_value=0)

        # output
        # [BS, max_seq_len, self.outdim]->[BS, self.outdim]
        if self.output_representation == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            pooler_output = last_hidden_state[range(last_hidden_state.shape[0]), last_seq_idxs, :]
        elif self.output_representation == "max":
            pooler_output, _ = torch.max(last_hidden_state, dim=1)
        elif self.output_representation == "meanpool":
            sum_ = torch.sum(last_hidden_state, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.outdim)
            assert sum_.size() == lens_.size()
            pooler_output = torch.div(sum_, lens_)
        else:
            pooler_output = None

        # return {"last_hidden_state": last_hidden_state, "pooler_output": pooler_output}
        return last_hidden_state, pooler_output

    def forward(self, batch_tensor, batch_lengths, batch_char_lengths=None):
        """
        :param batch_tensor:
        :param batch_lengths:
        :param batch_char_lengths: if not None, implies this is a charecter LSTM
        :return: last_hidden_state, pooler_output
        """

        if batch_char_lengths:

            assert self.embeddings is not None

            last_hidden_state, pooler_output = list(zip(
                *[self._forward(pad_sequence_, lens_) for pad_sequence_, lens_ in
                  zip(batch_tensor, batch_char_lengths)]))
            # padding is required because each item in `pooler_output` above is of dim [nwords_in_sentence, lstm_dim]
            # no need to put on gpu specially  if one of the inputs to pad_sequence is on gpu
            last_hidden_state = pad_sequence(pooler_output, batch_first=True, padding_value=0)
            pooler_output = None  # not consistent with outputs of other input_representation's or BERT model's

        else:

            # the `batch_tensor` here can be either a word-level embeddings or idxs
            last_hidden_state, pooler_output = self._forward(batch_tensor, batch_lengths)

        return last_hidden_state, pooler_output


""" ################ """
""" resource classes """
""" ################ """


class Resource:
    def __init__(self):
        super(Resource, self).__init__()
        self.name = None
        self.attr = None
        self.optimizer = None
        self.requires_bert_optimizer = False

    def features_to_device(self, feats_dict):
        # inputs to device
        device = self.attr.device
        for k in feats_dict:
            v = feats_dict[k]
            if v is None:
                continue
            if isinstance(v, list):
                v = [x.to(device) for x in v]
            elif isinstance(v, torch.Tensor):
                v = v.to(device)
            else:
                raise NotImplementedError(f"{type(v)}")
            feats_dict.update({k: v})
        return feats_dict


class EncoderResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 attr: Union[BertAttributes, LstmAttributes],
                 config=None,
                 model=None):
        super(EncoderResource, self).__init__()

        self.name = name
        self.attr = attr
        self.config = config
        self.model = model
        self.outdim = None

        self.dropout_hidden_state = nn.Dropout(p=0.25)
        self.dropout_pooler = nn.Dropout(p=0.25)

    def init_bert(self):
        attr = self.attr
        pretrained_name_or_path = attr.pretrained_name_or_path
        bert_config = AutoConfig.from_pretrained(pretrained_name_or_path)
        if attr.input_representation == "subwords":
            # tag informed modeling
            if attr.n_token_type_ids and attr.n_token_type_ids > bert_config.type_vocab_size:
                # load from config and then map embeddings manually from diretory `pretrained_name_or_path`
                if not os.path.exists(pretrained_name_or_path):
                    raise Exception("when using tag informed modeling by inputting `n_token_type_ids`, "
                                    "you mst specify a directory of downloaded model weights and not name")
                print(f"upping type_vocab_size of BERT to {attr.n_token_type_ids} from {bert_config.type_vocab_size}")
                bert_config.type_vocab_size = attr.n_token_type_ids
                bert_model = AutoModel.from_config(config=bert_config)
                bert_model = load_bert_pretrained_weights(bert_model, pretrained_name_or_path, attr.device)
            else:
                bert_model = AutoModel.from_pretrained(pretrained_name_or_path)
        elif attr.input_representation == "charcnn":
            # bert_model = CharacterBertModel.from_pretrained(pretrained_name_or_path, config=config)
            raise NotImplementedError
        else:
            raise ValueError
        if not attr.finetune_bert:
            for param in bert_model.parameters():
                param.requires_grad = False
        bert_model.to(attr.device)
        self.outdim = bert_config.hidden_size
        self.config = bert_config
        self.model = bert_model
        self.requires_bert_optimizer = True
        return

    def init_lstm(self):
        attr = self.attr
        lstm_model = VanillaBiLSTM(config=attr.__dict__)
        if not attr.finetune_lstm:
            for param in lstm_model.parameters():
                param.requires_grad = False
        lstm_model.to(attr.device)
        self.outdim = lstm_model.outdim
        self.config = None
        self.model = lstm_model
        return

    def forward(self, tokenizer_output):

        features = tokenizer_output["features"]
        features = self.features_to_device(features)
        last_hidden_state, pooler_output = self.model(**features)

        # TODO (BERT): implement this as part of a `VanillaBERT` class like `VanillaBiLSTM` ?
        if self.name == "bert":

            # modify pooler_output if required
            if self.attr.output_representation == "meanpool":
                pooler_output = [
                    merge_subword_encodings_for_sentences(bert_seq_encodings, seq_splits)
                    for bert_seq_encodings, seq_splits in zip(last_hidden_state, tokenizer_output.get("batch_splits"))
                ]
                pooler_output = pad_sequence(pooler_output, batch_first=True, padding_value=0).to(self.attr.device)

            # modify last_hidden_state (todo (BERT): make this if required)
            last_hidden_state = [
                merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, device=self.attr.device)
                for bert_seq_encodings, seq_splits in zip(last_hidden_state, tokenizer_output.get("batch_splits"))
            ]
            last_hidden_state = pad_sequence(last_hidden_state, batch_first=True, padding_value=0).to(self.attr.device)

        if last_hidden_state is not None:
            last_hidden_state = self.dropout_hidden_state(last_hidden_state)

        if pooler_output is not None:
            pooler_output = self.dropout_pooler(pooler_output)

        return last_hidden_state, pooler_output


class TaskResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 attr: TaskAttributes,
                 adapter_layer=None,
                 mlp_layer=None,
                 criterion=None
                 ):
        super(TaskResource, self).__init__()

        self.name = name
        self.attr = attr
        self.adapter_layer = adapter_layer
        self.mlp_layer = mlp_layer
        self.criterion = criterion

        if attr.nlabels < 2:
            raise NotImplementedError("Coming Soon: Support for regression.")

        if attr.nlabels == 2:
            print("WARNING: Fitting 2 label problem with nn.CrossEntropyLoss instead of nn.BCEWithLogitsLoss")

    def init_lstm(self, indim):
        attr = self.attr
        self.adapter_layer = VanillaBiLSTM({"embdim": indim, "hidden_size": 256, "num_layers": 2})
        self.mlp_layer = nn.Linear(self.adapter_layer.outdim, attr.nlabels)
        self.adapter_layer.to(attr.device)
        self.mlp_layer.to(attr.device)
        self.set_criterion()
        return

    def init_mlp(self, indim):
        attr = self.attr
        self.adapter_layer = None
        self.mlp_layer = nn.Linear(indim, attr.nlabels)
        self.mlp_layer.to(attr.device)
        self.set_criterion()
        return

    def set_criterion(self):
        attr = self.attr
        if attr.name == "classification":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif attr.name == "seq_tagging":
            self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=attr.ignore_index)
        else:
            raise ValueError

    def forward(self, last_hidden_state, pooler_output, batch_lengths):

        if self.attr.adapter_type == "lstm":
            feats = {
                "batch_tensor": last_hidden_state,
                "batch_lengths": batch_lengths
            }
            feats = self.features_to_device(feats)
            _last_hidden_state, _pooler_output = self.adapter_layer(**feats)
            return last_hidden_state, pooler_output
        else:
            return last_hidden_state, pooler_output


class ConcatResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 inpdims: list,
                 outdim: int = None,  # projections only available for `weighted_ensemble`
                 initial_scalar_parameters: List[float] = None,
                 trainable: bool = True,
                 ):
        super(ConcatResource, self).__init__()

        assert name in ["concat", "weighted_ensemble"]

        self.name = name
        self.inpdims = inpdims
        self.outdim = outdim
        self.initial_scalar_parameters = initial_scalar_parameters

        self.n_concat = len(inpdims)

        if self.name == "concat":
            self.outdim = sum(inpdims)

        elif self.name == "weighted_ensemble":

            if self.n_concat < 2:
                raise Exception("Cannot use `weighted_ensemble` with less than 2 encoders")

            # init projections
            self.projections_last_hidden_state, self.projections_pooler_output = None, None
            if all([x == inpdims[0] for x in self.inpdims]):
                self.outdim = inpdims[0]
            else:
                if not self.outdim:
                    self.outdim = min(self.inpdims)
                self.projections_last_hidden_state, self.projections_pooler_output = nn.ModuleList(), nn.ModuleList()
                for inpdim in inpdims:
                    self.projections_last_hidden_state.append(nn.Linear(inpdim, self.outdim))
                    self.projections_pooler_output.append(nn.Linear(inpdim, self.outdim))

            # init weights
            if not self.initial_scalar_parameters:
                self.initial_scalar_parameters = [0.0] * self.n_concat
            self.scalar_parameters = ParameterList(
                [
                    Parameter(
                        torch.FloatTensor([self.initial_scalar_parameters[i]]), requires_grad=trainable
                    )
                    for i in range(self.n_concat)
                ]
            )

        else:
            raise ValueError

        self.to("cpu" if not torch.cuda.is_available() else "cuda")

    def _weighted_ensemble(self, tensors_list=None, projections=None):

        if tensors_list is None:
            return tensors_list

        # projections (if required)
        if projections is not None:
            tensors = [
                proj(tt) for proj, tt in zip(projections, tensors_list)
            ]
        else:
            tensors = tensors_list

        # weighted sum
        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return sum(pieces)

    def forward(self, last_hidden_state_list, pooler_output_list):

        last_hidden_state, pooler_output = None, None

        if last_hidden_state_list:

            if len(last_hidden_state_list) == 1:
                last_hidden_state = last_hidden_state_list[0]
            else:
                if self.name == "concat":
                    last_hidden_state = torch.cat(last_hidden_state_list, dim=-1)
                elif self.name == "weighted_ensemble":
                    last_hidden_state = self._weighted_ensemble(last_hidden_state_list,
                                                                self.projections_last_hidden_state)

        if pooler_output_list:

            if len(pooler_output_list) == 1:
                pooler_output = pooler_output_list[0]
            else:
                if self.name == "concat":
                    pooler_output = torch.cat(pooler_output_list, dim=-1)
                elif self.name == "weighted_ensemble":
                    pooler_output = self._weighted_ensemble(pooler_output_list,
                                                            self.projections_pooler_output)

        return last_hidden_state, pooler_output


""" ############## """
""" unified model  """
""" ############## """


class Model(nn.Module):
    def __init__(self,
                 encoders: Union[str, List[str]],
                 encoder_attributes: Union[dict, List[dict]],
                 task_attributes: Union[dict, List[dict]],
                 n_fields_fusion: int = None,
                 fusion_strategy: str = None,
                 encodings_merge_type: str = "concat",
                 ):
        """
        :param encoders: ["bert", "lstm"] or ["bert"] or "bert",  see docs for details
        :param encoder_attributes: [{...}, {...}], see docs for details
        :param task_attributes: [{...}, {"ignore_index": ignore_index, ...}], see docs for details
        :param n_fields_fusion: number of fusable fileds in the batch
                eg. 2 for a bs=16, where first 8 are for normalized and next 8 for raw texts
        :param fusion_strategy: how to fuse the final encodings before computing logits
        """
        super(Model, self).__init__()

        self.single_encoder = False
        if isinstance(encoders, str):
            assert isinstance(encoder_attributes, dict)
            encoders = [encoders, ]
            encoder_attributes = [encoder_attributes, ]
            self.single_encoder = True

        self.single_task = False
        if isinstance(task_attributes, dict):
            task_attributes = [task_attributes, ]
            self.single_task = True

        if len(encoders) != len(encoder_attributes):
            msg = f"number of model attributes ({len(encoder_attributes)}) " \
                  f"must be same as number of models ({len(encoders)})"
            raise ValueError(msg)
        if "bert" in encoders and encoders[0] != "bert":
            raise Exception("`bert` model is expected to be placed at beginning of `encoders` list")

        if n_fields_fusion:
            # assert batch_size % n_fields_fusion == 0, \
            #     print(f"n_fields_fusion ({n_fields_fusion}) must be a factor of batch_size ({batch_size}) ")
            assert fusion_strategy in ["concat", "meanpool", "maxpool"]
        self.n_fields_fusion = n_fields_fusion
        self.fusion_strategy = fusion_strategy

        enc_attr = []
        for name, model_attribute in zip(encoders, encoder_attributes):
            if name == "bert":
                attributes = BertAttributes(**model_attribute)
            elif name == "lstm":
                attributes = LstmAttributes(**model_attribute)
            else:
                msg = f'`encoders` expected to be among the list items {["bert", "lstm"]} but found {name}'
                raise ValueError(msg)
            enc_attr.append(attributes)

        self.encoder_resources = []
        for name, attr in zip(encoders, enc_attr):
            encoder_resource = EncoderResource(name=name, attr=attr)
            if name == "bert":
                encoder_resource.init_bert()
            elif name == "lstm":
                encoder_resource.init_lstm()
            self.encoder_resources.append(encoder_resource)

        self.concat_resource = (
            ConcatResource(name=encodings_merge_type, inpdims=[x.outdim for x in self.encoder_resources])
        )
        outdim = self.concat_resource.outdim

        assert outdim > 0
        outdim = outdim * self.n_fields_fusion if self.n_fields_fusion else outdim
        self.task_resources = []
        for attributes in task_attributes:
            attr = TaskAttributes(**attributes)
            task_resource = TaskResource(name=attr.name, attr=attr)
            if attr.adapter_type == "lstm":
                task_resource.init_lstm(outdim)
            else:
                task_resource.init_mlp(outdim)
            self.task_resources.append(task_resource)

    def all_resources(self, return_dict=False):
        if return_dict:
            return {
                "encoder_resources": self._get_encoders_resources(),
                "task_resources": self._get_tasks_resources(),
                "concat_resources": self._get_concats_resources(),
            }
        else:
            return [*self.encoder_resources, *self.task_resources, self.concat_resource]

    def _get_encoders_resources(self):
        return [*self.encoder_resources]

    def _get_tasks_resources(self):
        return [*self.task_resources]

    def _get_concats_resources(self):
        return [self.concat_resource, ]

    def get_model_nparams(self):
        all_resc = self.all_resources()
        return get_model_nparams(all_resc)

    def save_state_dicts(self, path):
        encoder_resc = self._get_encoders_resources()
        for i, resource in enumerate(encoder_resc):
            st_dict = resource.state_dict()
            save_name = "pytorch_model.bin"
            create_path(os.path.join(path, f"encoder_{i}"))
            torch.save(st_dict, os.path.join(path, f"encoder_{i}", save_name))

        task_resc = self._get_tasks_resources()
        for i, resource in enumerate(task_resc):
            st_dict = resource.state_dict()
            save_name = "pytorch_model.bin"
            create_path(os.path.join(path, f"task_{i}"))
            torch.save(st_dict, os.path.join(path, f"task_{i}", save_name))

        concat_resc = self._get_concats_resources()
        for i, resource in enumerate(concat_resc):
            st_dict = resource.state_dict()
            save_name = "pytorch_model.bin"
            create_path(os.path.join(path, f"concat_{i}"))
            torch.save(st_dict, os.path.join(path, f"concat_{i}", save_name))
        return

    def load_state_dicts(self, path, map_location="cpu" if not torch.cuda.is_available() else "cuda"):
        encode_resc = self._get_encoders_resources()
        for i, resource in enumerate(encode_resc):
            save_name = "pytorch_model.bin"
            existing_dict = torch.load(os.path.join(path, f"encoder_{i}", save_name), map_location)
            resource.load_state_dict(existing_dict)

        task_resc = self._get_tasks_resources()
        for i, resource in enumerate(task_resc):
            save_name = "pytorch_model.bin"
            existing_dict = torch.load(os.path.join(path, f"task_{i}", save_name), map_location)
            resource.load_state_dict(existing_dict)

        concat_resc = self._get_concats_resources()
        for i, resource in enumerate(concat_resc):
            save_name = "pytorch_model.bin"
            existing_dict = torch.load(os.path.join(path, f"concat_{i}", save_name), map_location)
            resource.load_state_dict(existing_dict)
        return

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        for resc in self.all_resources():
            resc.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self,
                inputs: Union[dict, List[dict]],
                targets: Union[List[Union[List[int], List[List[int]]]], Union[List[int], List[List[int]]]] = None,
                ):
        """
        :param inputs: each item is an output of tokenizer method in `.helpers`
        :param targets: target idxs to compute loss using nn.CrossEntropyLoss
        :return: a dict of outputs and loss
        """

        output_dict = {}

        # init
        if True:

            if self.single_encoder:
                assert isinstance(inputs, dict)
                inputs = [inputs, ]
            assert len(inputs) == len(self.encoder_resources)

            batch_size = None
            for inp in inputs:
                if not batch_size:
                    batch_size = inp["batch_size"]
                    continue
                assert batch_size == inp["batch_size"], \
                    print(f"all `inputs` must have same batch size; found {batch_size} and {inp['batch_size']}")

            batch_lengths = inputs[0]["batch_lengths"],  # every tokenizer in `.helpers` returns this !!,

        # input encodings
        last_hidden_state_list, pooler_output_list = [], []
        for i, (tokenizer_output, enc_resc) in enumerate(zip(inputs, self.encoder_resources)):
            last_hidden_state, pooler_output = enc_resc(tokenizer_output)
            last_hidden_state_list.append(last_hidden_state)
            pooler_output_list.append(pooler_output)
        # make them void if cannot be used for concat (special cases like char-lstm that doesn't gibe pooler_output)
        if any([x is None for x in last_hidden_state_list]):
            last_hidden_state_list = None
        if any([x is None for x in pooler_output_list]):
            pooler_output_list = None

        # concat
        last_hidden_state, pooler_output = self.concat_resource(last_hidden_state_list, pooler_output_list)

        def do_fuse(tensor, startegy):
            if startegy == "maxpool":
                fused_output = torch.max(tensor, dim=0)[0]
            elif startegy == "meanpool":
                fused_output = torch.mean(tensor, dim=0)
            elif startegy == "concat":
                splits = torch.split(tensor, 1, 0)
                fused_output = torch.cat([torch.squeeze(_split) for _split in splits], dim=-1)
            else:
                raise ValueError
            return fused_output

        # fusion and logits
        logits, loss = [], None
        for i, task_resc in enumerate(self.task_resources):

            _last_hidden_state, _pooler_output = task_resc(last_hidden_state, pooler_output, batch_lengths)

            if self.n_fields_fusion:

                assert batch_size % self.n_fields_fusion == 0
                n_samples = int(batch_size / self.n_fields_fusion)
                if _pooler_output is not None:
                    _pooler_output = _pooler_output.reshape(self.n_fields_fusion, n_samples, *_pooler_output.shape[1:])
                    _pooler_output = do_fuse(_pooler_output, startegy=self.fusion_strategy)
                if _last_hidden_state is not None:
                    _last_hidden_state = \
                        _last_hidden_state.reshape(self.n_fields_fusion, n_samples, *_last_hidden_state.shape[1:])
                    _last_hidden_state = do_fuse(_last_hidden_state, startegy=self.fusion_strategy)

            if task_resc.name == "classification":
                _logits = task_resc.mlp_layer(_pooler_output)
                logits.append(_logits)
                if targets:
                    _target = targets[i]
                    _target = torch.as_tensor(_target).to(task_resc.attr.device)
                    _loss = task_resc.criterion(_logits, _target)
                    loss = _loss if not loss else loss + _loss

            elif task_resc.name == "seq_tagging":
                _logits = task_resc.mlp_layer(_last_hidden_state)
                logits.append(_logits)
                if targets:
                    _target = targets[i]
                    _batch_lengths = inputs[0]["batch_lengths"]  # every tokenizer in `.helpers` returns this !!
                    _target = [torch.tensor(t[:b]) for t, b in zip(_target, _batch_lengths)]
                    _target = pad_sequence(_target, padding_value=task_resc.attr.ignore_index, batch_first=True)
                    _target = torch.as_tensor(_target).to(task_resc.attr.device)
                    #
                    _logits = _logits.reshape(-1, _logits.shape[-1])
                    _target = _target.reshape(-1)
                    _loss = task_resc.criterion(_logits, _target)
                    loss = _loss if not loss else loss + _loss

        output_dict.update(
            {
                "loss": loss,
                "logits": logits[0] if self.single_task else logits
            }
        )

        return output_dict

    def predict(self,
                inputs: Union[dict, List[dict]],
                targets: Union[List[Union[List[int], List[List[int]]]], Union[List[int], List[List[int]]]] = None,
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(inputs, targets)

            logits = output_dict["logits"]
            if self.single_task:
                logits = [logits, ]

            probs_list, preds_list, targets_list, acc_num_list, acc_list = [], [], [], [], []
            for i, _logits in enumerate(logits):
                probs = F.softmax(_logits, dim=-1)
                probs_list.append(probs.cpu().detach().numpy().tolist())
                # dims: [batch_size] if task=classification, [batch_size, mxlen] if task=seq_tagging
                argmax_probs = torch.argmax(probs, dim=-1)

                if targets is not None:
                    _targets = targets[i]
                    if self.task_resources[i].name == "classification":
                        #
                        preds = argmax_probs.cpu().detach().numpy().tolist()
                        assert len(preds) == len(_targets), print(len(preds), len(_targets))
                        #
                        acc_num = sum([m == n for m, n in zip(preds, _targets)])
                        acc_num_list.append(acc_num)  # dims: [1]
                        acc_list.append(acc_num / len(_targets))  # dims: [1]
                        preds_list.append(preds)
                        targets_list.append(_targets)
                    elif self.task_resources[i].name == "seq_tagging":
                        #
                        preds = argmax_probs.reshape(-1).cpu().detach().numpy().tolist()
                        ignore_index = self.task_resources[i].attr.ignore_index
                        _batch_lengths = inputs[0]["batch_lengths"]  # every tokenizer in `.helpers` returns this !!
                        _targets = [torch.tensor(t[:b]) for t, b in zip(_targets, _batch_lengths)]
                        _targets = pad_sequence(_targets, padding_value=ignore_index, batch_first=True)
                        _targets = _targets.reshape(-1).cpu().detach().numpy().tolist()
                        assert len(preds) == len(_targets), print(len(preds), len(_targets))
                        #
                        new_preds, new_targets = [], []
                        for ii, jj in zip(preds, _targets):
                            if jj != ignore_index:
                                new_preds.append(ii)
                                new_targets.append(jj)
                        acc_num = sum([m == n for m, n in zip(new_preds, new_targets)])
                        acc_num_list.append(acc_num)  # dims: [1]
                        acc_list.append(acc_num / len(new_targets))  # dims: [1]
                        preds_list.append(new_preds)
                        targets_list.append(new_targets)

        output_dict.update(
            {
                "logits": logits[0] if self.single_task else logits,
                "probs": probs_list[0] if self.single_task else probs_list,
                "preds": preds_list[0] if self.single_task else preds_list,
                "targets": targets_list[0] if self.single_task else targets_list,
                "acc_num": acc_num_list[0] if self.single_task else acc_num_list,
                "acc": acc_list[0] if self.single_task else acc_list,
            }
        )

        if was_training:
            self.train()

        return output_dict
