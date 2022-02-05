import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from os.path import join as osj
from shapely.geometry import Point
from transformers import GPT2Tokenizer
from shapely.geometry.polygon import Polygon
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from utils import Timer, read_json
from PIL import ImageFile
#import ipdb
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Gender Neutral Names for Text-only setup.
NAMES = ['Alex', 'Ash', 'Jesse', 'Sam', 'Avery',
         'Jaime', 'Kai', 'Lee', 'Jody', 'Cal',
         'Adrian', 'Corey', 'Drew', 'Morgan',
         'Logan', 'Skyler', 'Noah', 'Pat', 'Taylor']

# Person Tokens
PERSON_TOKENS = ['[person 1]', '[person 2]', '[person 3]',
                 '[person 4]', '[person 5]', '[person 6]',
                 '[person 7]', '[person 8]', '[person 9]',
                 '[person 10]', '[person 11]', '[person 12]']


class VisualCOMETDataset(Dataset):
    """ VisualCOMET Dataset """

    # Image Stats
    mean = (0.2593, 0.2324, 0.2114)
    std = (0.1840, 0.1739, 0.1675)

    # Vocab
    special_tokens = dict(image='<|image|>',
                          event='<|event|>',
                          place='<|place|>',
                          inference='<|infer|>',
                          before='<|before|>',
                          intent='<|intent|>',
                          after='<|after|>')

    def __init__(self, data_dir, split, max_text_len, tok_name,
                 im_size=224, patch_size=16, text_only=False):
        """
        Preprocess & Load Dataset

        Workflow:
            raw[row] --> load()
                           └── preprocess(row)  \n
                           └── to_individual_records(row)

            data[rec] -->   __getitem__(i)    --> model_inputs
                              └── process_visual() + process_text()

        :param str data_dir: path to dataset directory
        :param str split: 'train' or 'val' set
        :param int max_text_len: max text sequence length
        :param str tok_name: model tokenizer name
        :param bool text_only: exclude visual input
        :param int im_size: desired image size (after resize)
        :param int patch_size: image patch size
        """
        # Data Args
        self.data_dir = data_dir
        self.split = split
        self.with_image = not text_only
        self.H, self.W = (im_size, im_size)
        self.h, self.w = (patch_size, patch_size)
        self.num_patches = (im_size // patch_size) ** 2

        # Tokenizer
        tok_name = self._tokenizer_name(tok_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tok_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Max Text length
        self.max_text_len = max_text_len

        # Add special tokens
        self._add_additional_tokens()

        # Image Transforms
        self.transforms = Compose([Resize((self.H, self.W)),
                                   ToTensor(),
                                   Normalize(self.mean, self.std)])
        # Load Data
        self.data = self._load()

    def __len__(self):
        return len(self.data)

    def get_tokenizer_len(self):
        return len(self.tokenizer)

    def _add_additional_tokens(self):
        tokens = PERSON_TOKENS
        tokens += list(self.special_tokens.values())

        additional_vocab = {'additional_special_tokens': tokens}

        self.tokenizer.add_special_tokens(additional_vocab)

    @staticmethod
    def _tokenizer_name(name):
        # For visual-bert, we use the bert tokenizer
        if 'visualbert' in name:
            name = 'bert-base-uncased'
        return name

    def _load(self):
        """
        Processes raw dataset json file.

        Output:
            {
                'img_path': str,
                'seg_path': str,
                'event': str,
                'place': str,
                'inference': str,
                'inference_type': str,
            }

        :returns: list of image paths, questions & answers.
        :rtype: list[dict]
        """
        # Load data
        path = osj(self.data_dir, 'viscomet', f'{self.split}_annots.json')

        cols = ['img_fn', 'metadata_fn', 'place',
                'event', 'intent', 'before', 'after']

        raw_data = pd.read_json(path).to_dict(orient='records')

        # Process samples
        data = []
        for row in tqdm(raw_data):
            record = self._preprocess(row)
            records = self._to_individual_records(record)

            data += records

        return data

    def _preprocess(self, row: dict) -> dict:
        """
        Processes raw dataset.

        :param row: single sample from raw dataset
        :return: processed record
        """
        # Text
        def _insert_name(txt: str) -> str:
            """
            Text-only setup: inserts names in place of digits.
            """
            n = len(NAMES)

            text = []
            for w in txt.split():
                word = NAMES[int(w) % n] if w.isdigit() else w
                text.append(word)

            # list to string
            text = ' '.join(text)
            return text

        # Image + Text
        def _special_tokens2txt(txt_in: list, objects: list) -> list:
            """
            Converts 2D index (list) to comma-separated elements (int),
            and replaces each object idx (int) as follows:
                - If 'person', inserts the special token
                - Else, simply inserts the object name.

            Example:
                In: ['Is', [0], 'holding', 'a', [1]]; objects = ['person', 'book', 'person'] \n
                Out: ['Is', 'person', '[person 0]', 'holding', 'a', 'book', '[object]']
            """

            n_toks = self.num_person_tokens
            text = []
            for e in txt_in:
                # If element is list (obj idxs)
                if type(e) is list:
                    toks = []
                    for idx in e:
                        obj_tag = objects[idx]
                        if obj_tag == 'person':
                            toks += [obj_tag, PERSON_TOKENS[idx % n_toks]]
                        else:
                            toks += [obj_tag, self.object_token]
                        toks += [',']

                    text += toks[:-1]  # drop last ','
                else:
                    text += [e]
            return text

        # Parse Raw Data
        event = row['event']
        place = row['place']
        intent_lst = row['intent']
        before_lst = row['before']
        after_lst = row['after']

        # Image + Text
        if self.with_image:
            # Image
            img_path = osj(self.data_dir, 'images', row['img_fn'])
            annot_path = osj(self.data_dir, 'images', row['metadata_fn'])

            # Segmentation Mask
            seg_path = annot_path

        # Text
        else:
            # Ignore Visual
            img_path = ''
            seg_path = ''

            # Process Text
            event = _insert_name(event)
            place = _insert_name(place)
            intent_lst = [_insert_name(intent) for intent in intent_lst]
            before_lst = [_insert_name(before) for before in before_lst]
            after_lst = [_insert_name(after) for after in after_lst]

        record = dict(img_path=img_path,
                      seg_path=seg_path,
                      event=event,
                      place=place,
                      intent_lst=intent_lst,
                      before_lst=before_lst,
                      after_lst=after_lst)

        return record

    def _to_individual_records(self, record):
        """
        Splits raw dataset sample to individual records along inferences.

        e.g. {E, P, [B1, B2], [A1, A2, A3]}
        --> [{E, P, B1}, {E, P, B2}, {E, P, A1}, {E, P, A2}, {E, P, A3}]

        :return: list of records
        :rtype: list[dict]
        """
        record_new = dict(img_path=record['img_path'],
                          seg_path=record['seg_path'],
                          event=record['event'],
                          place=record['place'])

        def _prepare(infer_lst, infer_type) -> list:
            """
            Given list of inferences, prepares individual records.
            """
            _recs = []
            for infer in infer_lst:
                # Create new record
                _new = deepcopy(record_new)
                _new['inference'] = infer
                _new['infer_type'] = infer_type

                _recs.append(_new)

            return _recs

        records = []
        records += _prepare(record['before_lst'], infer_type=self.special_tokens['before'])
        records += _prepare(record['intent_lst'], infer_type=self.special_tokens['intent'])
        records += _prepare(record['after_lst'], infer_type=self.special_tokens['after'])

        return records

    def _process_text(self, event, place, infer_type, inference) -> dict:
        """
        Tokenizes `text`:

        '<|place|> `place` <|event|> `event` <|infer_type|> <|infer|> `inference` <|endoftext|>'

        Where, `infer_type` = before | intent | after

        :returns: {'input_ids', 'attention_mask'}
        """
        event_tok = self.special_tokens['event']
        place_tok = self.special_tokens['place']
        inference_tok = self.special_tokens['inference']
        end_tok = self.tokenizer.eos_token

        # Text
        text = f'{place_tok} {place} '
        text += f'{event_tok} {event} '
        text += f'{infer_type} '
        text += f'{inference_tok} {inference} '
        text += f'{end_tok}'

        # Tokenize
        text = self.tokenizer(text=text,
                              padding='max_length',
                              max_length=self.max_text_len,
                              truncation=True)
        # To Tensor
        text = {k: torch.tensor(v) for k, v in text.data.items()}

        return text

    def _process_visual(self, img_path, seg_path, obj_tags):
        """
        Given image & objects generates patch to object mapping.

        :returns: image, patch2object (tokens-ids)
        """
        def _convert_obj_idxs_to_tokens(obj_idxs: list, tags: list) -> list:
            """
            Insert special tokens in place of object indexes. See example below.

            In: obj_idxs=[-1, -1, 0, 1, -1, 2]; tags=['person', 'person', 'book']
            Out: obj_toks=['[BG]', '[BG]', '[person 0]', '[person 1]', '[BG]', '[object]']
            """
            n_toks = self.num_person_tokens

            # Replace object idxs with special tokens
            obj_toks = []
            for idx in obj_idxs:
                # Background
                if idx == -1:
                    tok = self.background_token

                # Person
                elif tags[idx] == 'person':
                    tok = PERSON_TOKENS[idx % n_toks]

                # Object
                else:
                    tok = self.object_token

                obj_toks += [tok]

            return obj_toks

        def _filter_seg(seg_candidates):
            """
            Returns largest segmentation mask for a given object.

            If num-points < 3, return None (should be a polygon)
            """
            # If no candidates
            if len(seg_candidates) == 0:
                return None

            # Select best candidate
            _lens = [len(s) for s in seg_candidates]
            _idx = _lens.index(max(_lens))

            seg = seg_candidates[_idx]
            if len(seg) < 3:
                return None

            return seg

        # Read image
        image = Image.open(img_path).convert('RGB')

        # Read segmentation mask
        seg_masks = read_json(seg_path)['segms']

        # Patch-to-Object mapping
        patch2object = [-1] * self.num_patches      # default token is -1 [background]

        for obj_idx, seg_mask in enumerate(seg_masks):
            # Check & Select segmentation mask
            seg_mask = _filter_seg(seg_mask)

            if seg_mask:
                patch2object = self._patch_ids_from_seg_mask(image, seg_mask, obj_idx, patch2object)

        # Convert object ids to tokens
        patch2object = _convert_obj_idxs_to_tokens(patch2object, obj_tags)

        # Compute word token IDs
        patch2object = ' '.join(patch2object)
        patch2object = self.tokenizer(patch2object, add_special_tokens=False)

        patch2object = patch2object.data['input_ids']

        # Transform image
        image = self.transforms(image)

        return image, patch2object

    def _patch_ids_from_seg_mask(self, img: Image, seg_mask: list, obj_idx: int, patch2obj: list) -> list:
        """
        Given image & segmentation mask, returns a mapping from patch idxs to object id.
        A patch is mapped to the object if its center lies inside the object segmentation mask.

        Resizes image dims to the min dim before generating patch grids.

        patch2object: object-idx if patch inside object-segmask, else -1 [background]

        :param img: original image
        :param seg_mask: polygon coordinates of the object (2D list)
        :param obj_idx: object ID corresponding to `obj_tags`
        :param patch2obj: stores mapping from patch idx to object idx.
        :return: Patch IDs inside segmentation mask
        :rtype: list[int]
        """

        def _point_in_polygon(polygon: Polygon, point: list):
            point = Point(point)
            return polygon.contains(point)

        # Patch per dim
        num_patch_per_dim = self.H // self.h

        # Image dims
        w, h = img.size
        scale = max(w, h) / min(w, h)

        # Compute new size
        im_size = w = h = min(w, h)
        scale_dim = np.argmax((w, h))

        # Rescale Mask
        seg_mask = [(x / scale, y) if scale_dim == 0 else (x, y / scale) for x, y in seg_mask]

        # Patch size (rescaled)
        p_size = im_size // num_patch_per_dim

        # Create Patches
        x = np.linspace(0, im_size - p_size, num_patch_per_dim)
        y = np.linspace(0, im_size - p_size, num_patch_per_dim)
        xs, ys = np.meshgrid(x, y)

        xs, ys = xs.astype(int), ys.astype(int)

        # Segmentation Mask as Polygon
        seg_polygon = Polygon(seg_mask)

        # Iterate over patches and assign object idx
        p = 0
        for i in range(num_patch_per_dim):
            for j in range(num_patch_per_dim):
                x1 = xs[i, j]
                y1 = ys[i, j]
                x2 = x1 + p_size
                y2 = y1 + p_size

                patch = [x1, y1, x2, y2]
                center = [(x1 + x2) // 2, (y1 + y2) // 2]

                # Check if patch (cx, cy) lies inside & hasn't been assigned to an object
                if _point_in_polygon(seg_polygon, center) and patch2obj[p] == -1:
                    patch2obj[p] = obj_idx
                p += 1

        return patch2obj

    def _create_label(self, input_ids: list) -> list:
        """
        Generates labels from tokenized text ids.
        """
        def _token2id(tok):
            return self.tokenizer.convert_tokens_to_ids(tok)

        labels = [-100] * self.max_text_len

        # Inference start & end tokens
        infer_tok_id = _token2id(self.special_tokens['inference'])
        eos_tok_id = self.tokenizer.eos_token_id

        # Start index
        start_idx = self.max_text_len
        if infer_tok_id in input_ids:
            start_idx = input_ids.index(infer_tok_id) + 1

        # End index
        end_idx = self.max_text_len
        if eos_tok_id in input_ids:
            end_idx = input_ids.index(eos_tok_id) + 1

        # Set labels corresponding to the Inference (ignore rest)
        labels[start_idx: end_idx] = input_ids[start_idx: end_idx]

        return labels

    def __getitem__(self, idx):
        record = self.data[idx]

        # Visual inputs
        img_path = record['img_path']
        seg_path = record['seg_path']

        # Text inputs
        event = record['event']
        place = record['place']
        inference = record['inference']
        infer_type = record['infer_type']

        # Image + Patch
        image, patch_ids = torch.zeros(1), torch.zeros(1)
        # if self.with_image:
        #     image, patch_ids = self._process_visual(img_path, seg_path, obj_tags)

        # Tokenize
        text = self._process_text(event, place, infer_type, inference)

        # Label
        inp_ids = text['input_ids'].tolist()

        label = self._create_label(inp_ids)
        label = torch.tensor(label)

        # Model Input
        model_input = {'image': image,                              # [C, H, W]
                       'patch_ids': patch_ids,                      # [P]
                       'input_ids': text['input_ids'],              # [L]
                       'attention_mask': text['attention_mask'],    # [L]
                       'label': label}                              # [L]

        return model_input


class VisCOMETEvaluationDataset(VisualCOMETDataset):
    """
    For Evaluating VisualCOMET dataset, via NLG metrics.
    """

    def _to_individual_records(self, record):
        """
        Splits raw dataset sample to individual records along inference type.

        e.g. {E, P, [B1, B2], [A1, A2, A3]}
        --> [{E, P, [B1, B2]}, {E, P, [A1, A2, A3]}]

        :return: list of records
        :rtype: list[dict]
        """
        record_new = dict(img_path=record['img_path'],
                          seg_path=record['seg_path'],
                          event=record['event'],
                          place=record['place'])

        def _prepares(infer_lst, infer_type) -> dict:
            """
            Prepares list of inferences.
            """
            # Create new record
            rec = deepcopy(record_new)
            rec['inferences'] = infer_lst
            rec['infer_type'] = infer_type

            return rec

        records = [
            _prepares(record['before_lst'], infer_type=self.special_tokens['before']),
            _prepares(record['intent_lst'], infer_type=self.special_tokens['intent']),
            _prepares(record['after_lst'], infer_type=self.special_tokens['after'])
        ]
        return records

    def _process_text(self, event, place, infer_type, inference) -> dict:
        """
        Provides prompt for text generation.

        Tokenizes `text`:

        '<|place|> `place` <|event|> `event` <|infer_type|> <|infer|>'

        Where, `infer_type` = before | intent | after

        :returns: {'input_ids', 'attention_mask'}
        """
        event_tok = self.special_tokens['event']
        place_tok = self.special_tokens['place']
        inference_tok = self.special_tokens['inference']
        end_tok = self.tokenizer.eos_token

        # Text
        text = f'{place_tok} {place} '
        text += f'{event_tok} {event} '
        text += f'{infer_type} '
        text += f'{inference_tok} '
        # text += f'{end_tok}'

        # Tokenize
        text = self.tokenizer(text=text,
                              padding='max_length',
                              max_length=self.max_text_len,
                              truncation=True)
        # To Tensor
        text = {k: torch.tensor(v) for k, v in text.data.items()}

        return text

    @staticmethod
    def pack(inferences: list) -> str:
        """
        To pass variable no. of lists, we convert
        list of inferences to a string

        e.g. ['A1', 'A2', 'A3'] --> 'A1 | A2 | A3'
        """
        return ' | '.join(inferences)

    @staticmethod
    def unpack(inferences: str) -> list:
        """
        For evaluation, we need to recover from string to list.

        e.g. 'A1 | A2 | A3' --> ['A1', 'A2', 'A3']
        """
        return inferences.split('|')

    def __getitem__(self, idx):
        record = self.data[idx]

        # Visual inputs
        img_path = record['img_path']
        seg_path = record['seg_path']

        # Text inputs
        event = record['event']
        place = record['place']
        infer_type = record['infer_type']
        inferences = record['inferences']

        # Image + Patch
        image, patch_ids = torch.zeros(1), torch.zeros(1)
        # if self.with_image:
        #     image, patch_ids = self._process_visual(img_path, seg_path, obj_tags)

        # Tokenize
        text = self._process_text(event, place, infer_type, inferences)

        # Encode Inferences
        references = self.pack(inferences)

        # Model Input
        model_input = {'image': image,                              # [C, H, W]
                       'patch_ids': patch_ids,                      # [P]
                       'input_ids': text['input_ids'],              # [L]
                       'attention_mask': text['attention_mask'],    # [L]
                       'references': references,
                       'infer_type': infer_type}

        return model_input


class MoviePreTrainDataset(Dataset):
    """
    Movie Script Dataset for Causal-LM Pre-training.
    """
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class ATOMICDataset(Dataset):
    
    RELATIONS = {"xEffect":"so, X", "xWant":"so, X wants", "xNeed":"X needed",
             "xIntent":"because X wants","HinderedBy":"but not if","xReact":"so, X feels",
             "xAttr":"X is seen as"}
    
    def __init__(self, data_dir, split, max_text_len, tok_name):
        self.data_dir = data_dir
        self.split = split

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tok_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Max Text length
        self.max_text_len = max_text_len
        

        # Load Data
        self.data = self._load()
        
    def _create_label(self, input_ids: list) -> list:
        """
        Generates labels from tokenized text ids.
        """
        labels = [-100] * self.max_text_len
        eos_tok_id = self.tokenizer.eos_token_id
                
        # End index
        end_idx = self.max_text_len
        if eos_tok_id in input_ids:
            end_idx = input_ids.index(eos_tok_id) + 1

        labels[:end_idx] = input_ids[:end_idx]
        
        return labels
    
    def __getitem__(self,idx):
        record = self.data[idx]
        
        text = self.tokenizer(text = record, padding='max_length', max_length=self.max_text_len)
                
        text = {k: torch.tensor(v) for k, v in text.data.items()}

        # Label
        inp_ids = text['input_ids'].tolist()

        label = self._create_label(inp_ids)
        label = torch.tensor(label)

        # Model Input
        model_input = { 'input_ids': text['input_ids'],             
                       'attention_mask': text['attention_mask'],
                       'label': label}

        return model_input
    
    def __len__(self):
        return len(self.data)

    def get_tokenizer_len(self):
        return len(self.tokenizer)
    
    def _load(self):
        # Load data

        raw_data = pd.read_json(self.data_dir, lines=True)
        
        df = raw_data[raw_data['split']== self.split]
        
        df['relation'] = df['relation'].apply(lambda x: self.RELATIONS[x])
        df['sentence'] = df['head'] + " " + df['relation'] + " " + df['tail']
        
        df = df[['sentence']]
    
        # Process samples
        data = df['sentence'].values.tolist()

        return data


if __name__ == '__main__':
    # ** Dataset Testing **
    d_dir = '../../Datasets/VCR/'

    #ds = VisCOMETEvaluationDataset(d_dir, split='val', max_text_len=64, tok_name='gpt2', text_only=True)
    ds = ATOMICDataset(data_dir = "C:/Users/acer/Downloads/projects_symbolic-knowledge-decoding_ATOMIC10X.jsonl",split = 'val', max_text_len = 64, tok_name = "gpt2")
    #ipdb.set_trace()
    #seq_lens = [len(ds.tokenizer.tokenize(d)) for d in ds.data]
    #print(np.mean(seq_lens));print(np.std(seq_lens));print(sorted(seq_lens, reverse=True)[:200]);

    dl = DataLoader(ds, batch_size=32)
    b = next(iter(dl))

    # for in_ids, refs in zip(b['input_ids'], b['references']):
    #     print(ds.tokenizer.decode(in_ids))
    #     print(refs)
    #     print()

    # for attention in b['attention_mask']:
    #     print(attention)
    
    # for l in b['label']:
    #     print(l)

    for in_ids in b['input_ids']:
        print(in_ids)
        print(ds.tokenizer.decode(in_ids))

    print('Done!')