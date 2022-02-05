import torch
import torch.nn as nn
from transformers import logging
from transformers import VisualBertModel, AutoModel, ViTModel, GPT2LMHeadModel
from transformers.models.vit.modeling_vit import PatchEmbeddings
from torch import repeat_interleave as repeat
from collections import OrderedDict
logging.set_verbosity(40)

# TODO: Idea ==> V+L Encoder (MLM) -> GPT2 Decoder (CLM)


class LaVGPT2(nn.Module):
    """ Language-Vision GPT-2 Transformer"""

    def __init__(self, vocab_size=30522, model_name=None, im_size=224, patch_size=16, vit_patch_emb=True, ckpt=None):
        """
        Builds upon VisualBERT transformer as backbone, with GPT-2 (one-layer) for generating text
        """
        super().__init__()

        # Transformer
        self.transformer = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        if ckpt:
            self._load_transformer_backbone(ckpt)
        self.transformer.resize_token_embeddings(vocab_size)

        self.word_embed = self._get_word_emb_layer()
        self.vit_patch_emb = vit_patch_emb
        h_dim = self._get_hidden_dim()

        # Image Args
        self.visual_dim = h_dim
        self.im_size = im_size
        self.patch_size = patch_size
        self.num_patches = (self.im_size // self.patch_size) ** 2

        # Patch + Position Embeddings
        self.visual_position = nn.Parameter(torch.randn([1, self.num_patches, self.visual_dim]))
        self.visual_embed = self._get_visual_emb_layer()
        # self.object_embed = self.visual_embed.projection

        # Remove visual projection layer
        self.transformer.embeddings.visual_projection = nn.Identity()

        # MLP Classifier
        self.mlp_cls = nn.Linear(h_dim, 1)

    def _get_hidden_dim(self):
        assert hasattr(self, 'transformer'), 'Called before `transformer` is defined!'

        return self.transformer.config.hidden_size

    def _get_word_emb_layer(self):
        return self.transformer.embeddings.word_embeddings

    def _get_visual_emb_layer(self):
        if self.vit_patch_emb:
            vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            visual_emb_layer = vit.embeddings.patch_embeddings
        else:
            visual_emb_layer = PatchEmbeddings(self.im_size, self.patch_size, 3, self.visual_dim)

        return visual_emb_layer

    def load_weights(self, checkpoint):
        state_dict = checkpoint['model_state_dict']

        state_dict_new = OrderedDict()

        for k, v in state_dict.items():
            k_ = k.replace('module.', '')
            state_dict_new[k_] = v

        self.load_state_dict(state_dict_new)

    def _load_transformer_backbone(self, checkpoint):
        state_dict = checkpoint['model_state_dict']

        transformer_wts = OrderedDict()

        for k, v in state_dict.items():
            # only consider transformer layers
            if 'transformer' in k:
                k_ = k.replace('module.', '').replace('transformer.', '')
                transformer_wts[k_] = v

        self.transformer.load_state_dict(transformer_wts, strict=False)

    def forward(self, batch):
        # Image
        image = batch['image']                                              # [B, 3, H, W]

        # batch_size, num_choices, text_length
        B, C, L = batch['input_ids'].shape

        # Patch special tokens
        patch_ids = batch['patch_ids']                                      # [B, P]
        P = self.num_patches

        # Patches (Linear + Position + Placeholder)
        visual_embeddings = self.visual_embed(image)                        # [B, P, D]
        visual_embeddings += self.visual_position
        visual_embeddings += self.word_embed(patch_ids)

        # Repeat image across all Q-A pairs
        visual_embeddings = repeat(visual_embeddings, C, dim=0)             # [B*C, P, D]

        vis_attn_mask = torch.ones([B*C, P], device=image.device)           # [B*C, P]

        # Visual Inputs
        visual_inputs = dict(visual_embeds=visual_embeddings,
                             visual_attention_mask=vis_attn_mask)           # [B*C, P]

        # Text Inputs
        text_inputs = dict(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           token_type_ids=batch['token_type_ids'])

        text_inputs = {k: v.view(-1, L) for k, v in text_inputs.items()}    # [B*C, L]

        # Concat Inputs
        inputs = {**text_inputs, **visual_inputs}                           # [B*C, P]

        # Transformer
        x = self.transformer(**inputs)                                      # [B, L, D]

        # [CLS]
        x = x.last_hidden_state[:, 0, :]                                    # [B, D]

        # Score
        logits = self.mlp_cls(x).view(B, C)                                 # [B, C]

        return logits


class GPT2(nn.Module):

    def __init__(self, model_name='gpt2', vocab_size=50257, eos_id=50256):
        """
        GPT-2 based Text-only model

        :param model_name: supports models from `AutoModel` class
        """
        super().__init__()

        # Transformer
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=eos_id)

        # Expand Vocab
        self.transformer.resize_token_embeddings(vocab_size)

    def load_weights(self, checkpoint):
        state_dict = checkpoint['model_state_dict']

        state_dict_new = OrderedDict()

        for k, v in state_dict.items():
            k_ = k.replace('module.', '')
            state_dict_new[k_] = v

        self.load_state_dict(state_dict_new)

    def generate(self, **kwargs):
        return self.transformer.generate(**kwargs)

    def forward(self, inputs):
        # Text inputs
        batch = dict(input_ids=inputs['input_ids'],             # [B, L]
                     labels=inputs['label'],                    # [B, L]
                     attention_mask=inputs['attention_mask'])   # [B, L]

        # Transformer
        outputs = self.transformer(**batch)

        return outputs.loss


if __name__ == '__main__':
    import torch as t

    # Args
    _B, _C, _L, _P = 2, 4, 16, 196
    _H, _W = (224, 224)
    _h, _w = (16, 16)
    gpu = 'cuda:0'

    inp_ids = t.randint(100, [_B, _L]).to(gpu)
    type_ids = t.zeros([_B, _L], dtype=t.long).to(gpu)
    attn_mask = t.ones([_B, _L], dtype=t.float).to(gpu)
    label_ids = t.ones([_B, _L], dtype=t.long).to(gpu) * -100

    p_ids = t.randint(100, [_B, _P]).to(gpu)
    img = t.rand([_B, 3, _H, _W]).to(gpu)

    # Input
    b = {  # 'image': img,
           # 'patch_ids': p_ids,
         'input_ids': inp_ids,
         'label': label_ids,
         'token_type_ids': type_ids,
         'attention_mask': attn_mask}

    m = GPT2()
    m.to('cuda:0')

    loss = m(b)
    print(loss)
