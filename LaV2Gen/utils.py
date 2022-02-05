import os
import sys
import json
import time
import torch
import numpy as np
from PIL import Image
from jury import Jury
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage


@torch.inference_mode()
def compute_eval_loss(model, dataloader, device, size):
    """
    For the given model, computes loss on validation/test set.

    :param model: model to evaluate
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: loss
    """
    model.eval()
    losses = []

    eval_samples = 0

    # Evaluate on mini-batches
    for batch in dataloader:
        # Load batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward Pass
        loss = model(batch)

        # Loss
        losses.append(loss)

        # Samples evaluated
        eval_samples += dataloader.batch_size

        if eval_samples >= size:
            break

    loss = torch.stack(losses).mean()

    return loss


@torch.inference_mode()
def compute_eval_metrics(model, dataloader, device, size=None):
    """
    For the given model, computes NLG metrics on validation/test set.

    :param model: model to evaluate
    :param dataloader: validation/test set Dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return metrics {'bleu', 'meteor', ...}
    :rtype: dict
    """
    num_seqs = 5
    ds = dataloader.dataset
    tokenizer = dataloader.dataset.tokenizer

    def _prepare_true(refs):
        """Prepare references"""
        # Repeat Interleave
        refs_ = []
        for r in refs:
            refs_ += [r] * num_seqs

        # Decode ('a | b | c' --> ['a', 'b', 'c'])
        refs_ = [ds.unpack(ref) for ref in refs_]

        return refs_

    scores = dict(bleu_1=[], bleu_2=[], meteor=[])
    metric = Jury()

    if size is None:
        size = len(ds)

    model.eval()
    eval_samples = 0

    for batch in dataloader:
        # Load batch to device
        batch = {k: v.to(device) if type(v) is torch.Tensor else v for k, v in batch.items()}

        # Ground Truth
        true = _prepare_true(batch['references'])

        # Prediction
        pred = generate_text(model,
                             tokenizer,
                             batch['input_ids'],
                             batch['attention_mask'],
                             num_seqs=num_seqs,
                             out_len=10)

        # Compute scores
        score = metric(predictions=pred, references=true)
        scores['bleu_1'] += [score['bleu_1']['score']]
        scores['bleu_2'] += [score['bleu_2']['score']]
        scores['meteor'] += [score['meteor']['score']]

        # Samples evaluated
        eval_samples += dataloader.batch_size

        if eval_samples >= size:
            break

    # Compute Mean over batches
    results = {metric_name: np.mean(score) for metric_name, score in scores.items()}

    # We define 'Accuracy' as BLEU 1
    results['accuracy'] = results['bleu_1']
    print('\n\n{}\n\n'.format(results))

    return results


def generate_text(model, tokenizer, input_ids, attention_mask, out_len=10, num_seqs=5) -> list:
    """
    Generate Text during inference, from input prompt.

    :param model: model
    :param tokenizer: tokenizer
    :param input_ids: prompt text -- [B, L]
    :param attention_mask: mask over pad tokens -- [B, L]
    :param out_len: length of generated text (after prompt)
    :param num_seqs: no. of sentences to generate
    :return: generated text [B * num_seqs]
    """
    inp_len = input_ids.shape[1]
    max_len = inp_len + out_len

    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             do_sample=True,
                             top_k=50,
                             max_length=max_len,
                             top_p=0.9,
                             num_return_sequences=num_seqs)

    inputs = input_ids.repeat_interleave(num_seqs, dim=0).tolist()

    text_generated = []

    for inp, out in zip(inputs, outputs):
        # Token ids to Text
        inp_txt = tokenizer.decode(inp, skip_special_tokens=True)
        out_txt = tokenizer.decode(out, skip_special_tokens=True)

        # Clip out the input prompt from the generated text
        inp_txt = inp_txt.split()
        out_txt = out_txt.split()
        out_txt = out_txt[len(inp_txt):]

        # list -> str
        out_txt = ' '.join(out_txt)

        if len(out_txt) == 0:
            out_txt = 'this is a dummy output'  # prevent div by zero

        text_generated.append(out_txt)

    return text_generated


# ---------------------------------------------------------------------------
def setup_logger(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def csv2list(v, cast=str):
    assert type(v) == str, 'Converts: comma-separated string --> list of strings'
    return [cast(s.strip()) for s in v.split(',')]


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false', 't', 'f', '1', '0'], 'Option requires: "true" or "false"'
    return v in ['true', 't', '1']


def read_json(path):
    with open(path) as json_data:
        data = json.load(json_data)

    return data


def _compute_and_save_mean_image(data, save_path, n_images=8000):
    """
    Computes the mean image given a list of images and saves to disk.
    """
    from multiprocessing import Pool, cpu_count
    _num_proc = cpu_count()

    paths_all = [d['img_path'] for d in data]
    paths_all = list(set(paths_all))
    paths_all = paths_all[:n_images]

    # Split into batches (2D list)
    paths_b = np.array_split(paths_all, _num_proc)
    paths_b = [list(batch) for batch in paths_b]

    # Parallel compute
    t = Timer('Mean Image:')
    with Pool(_num_proc) as p:
        mean_images = p.map_async(_get_mean_img, paths_b).get()
    t.end()

    # Average mean-image over batches
    mean_image = torch.stack([mean for mean in mean_images]).mean(0)

    # To PIL image
    mean_image = ToPILImage()(mean_image)

    # Save to disk
    mean_image.save(save_path)


def _get_mean_img(paths):
    """Computes Image mean (spatial dims)"""
    images = []

    # Fetch Images
    for path in paths:
        # Read
        img = Image.open(path).convert('RGB')

        # Resize
        img = img.resize((224, 224))

        # ToTensor: [0, 1]
        img = ToTensor()(img).half()

        images.append(img)

    # Compute Mean
    mean_image = torch.stack(images).mean(0)

    return mean_image


def intersection_over_union(box_1: list, box_2: list):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    # compute area of bboxes
    area_1 = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    area_2 = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    # compute intersection & union
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    union = area_1 + area_2 - intersection

    iou = intersection / float(union)
    return iou


class Timer:
    def __init__(self, tag=''):
        self.tag = tag
        self.start = time.time()

    def end(self):
        time_taken = time.time() - self.start
        print('{} completed in {:.2f} secs'.format(self.tag, time_taken))


if __name__ == '__main__':
    # Explore Text Generation
    from dataloader import VisCOMETEvaluationDataset, DataLoader
    from model import GPT2

    d_dir = '../../Datasets/VCR/'

    dset = VisCOMETEvaluationDataset(d_dir, 'val', 64, 'gpt2', text_only=True)
    dl = DataLoader(dset, batch_size=2)
    b = next(iter(dl))

    m = GPT2(vocab_size=dset.get_tokenizer_len())

    # preds = generate_text(m, dset.tokenizer, b['input_ids'], b['attention_mask'])

    res = compute_eval_metrics(m, dl, 'cpu', size=64)

    '''
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from model import GPT2

    MAX_LEN = 80
    inp_txt = ["I don't know about you, but there's only one thing I want to do after a long day of work",
               "If you want to travel by plane, you should"]

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token_id=50256)
    tokenizer.pad_token = tokenizer.eos_token
    # model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    model = GPT2()

    # encode context the generation is conditioned on
    inp = tokenizer(inp_txt[0], return_tensors='pt', padding='max_length', max_length=32, add_special_tokens=False)
    # inp = tokenizer(inp_txt[1], return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches 50
    output = model.generate(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], max_length=MAX_LEN)

    print("Greedy:\n" + 100 * '-')
    out_txt = tokenizer.decode(output[0], skip_special_tokens=True)

    print(out_txt)
    print(len(out_txt.split()))
    '''

