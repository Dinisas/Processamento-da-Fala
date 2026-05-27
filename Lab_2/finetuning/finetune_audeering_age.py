"""End-to-end fine-tuning of audeering/wav2vec2-large-robust-24-ft-age-gender for age regression."""

import argparse
import copy
import csv
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LAB2_DIR))

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import mean_absolute_error
from torch.autograd import Function
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    logging as tf_logging,
)
from transformers.modeling_outputs import ModelOutput

from pf_tools import create_submission_file


AUD_MODEL_ID = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
AGE_MIN, AGE_MAX = 20.0, 90.0


def select_device_str():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_partition_dataframe(datadir, partition):
    """Load <datadir>/<partition>/info.csv into a DataFrame with columns:
    wav, gender, age (numeric or NaN), fileid (basename), abs_path."""
    info_path = datadir / partition / 'info.csv'
    df = pd.read_csv(info_path)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')      # '?' -> NaN
    df['fileid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['abs_path'] = (datadir / partition / 'wav' / df['wav']).astype(str)
    return df


class W2V2LazyDataset(torch.utils.data.Dataset):
    """Lazy torch Dataset that loads audio + runs the feature extractor on
    __getitem__ (rather than eagerly building a HF Arrow table).

    Eager Dataset.from_list works fine up to ~10k 10-second clips, but at
    big_train_falar scale (38k+ clips) the cumulative element count blows
    past int32 (~2.1B) and the underlying PyArrow ListArray offsets
    overflow with `Python int too large to convert to C long`. This class
    sidesteps the Arrow path entirely — HF Trainer accepts any torch
    Dataset, so this is a drop-in replacement.

    Records have:
      - input_values : list[float]  (variable length, collator pads)
      - labels       : float        (age in years; 0.0 for evl)
      - speaker_idx  : int          (only when `has_speaker=True`)
      - gender_idx   : int          (only when `has_gender=True`; 0=M, 1=F)
    """

    def __init__(self, df, feat_ext, duration=10.0,
                 has_labels=True, has_speaker=False, has_gender=False):
        self.df = df.reset_index(drop=True)
        self.fe = feat_ext
        self.duration = duration
        self.has_labels = has_labels
        self.has_speaker = has_speaker
        self.has_gender = has_gender
        self.max_samples = int(16000 * duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(
            row['abs_path'], sr=16000, mono=True, duration=self.duration,
        )
        inputs = self.fe(
            audio, sampling_rate=16000,
            max_length=self.max_samples,
            truncation=True, padding=False,
        )
        rec = {'input_values': inputs['input_values'][0]}
        if self.has_labels and not pd.isna(row['age']):
            rec['labels'] = float(row['age'])
        else:
            rec['labels'] = 0.0
        if self.has_speaker:
            rec['speaker_idx'] = int(row['speaker_idx'])
        if self.has_gender:
            # 0 = M, 1 = F. Anything else (NaN, '?', ...) flagged as -1 so the
            # collator can mask it out of the gender loss (rare on lab train).
            g = str(row.get('gender', '')).strip()
            rec['gender_idx'] = 0 if g == 'M' else (1 if g == 'F' else -1)
        return rec


def build_dataset(df, feat_ext, duration=10.0, has_labels=True,
                  has_speaker=False, has_gender=False, desc='loading'):
    """Returns a W2V2LazyDataset (no audio is loaded yet — it happens on
    __getitem__). The `desc` arg is kept for API compatibility but no progress
    bar is shown because there's nothing to iterate at construction time."""
    return W2V2LazyDataset(
        df, feat_ext, duration=duration,
        has_labels=has_labels, has_speaker=has_speaker, has_gender=has_gender,
    )


def freeze_lower_layers(model, n_top_layers_to_train):
    """Freeze all but the top N transformer layers of the wav2vec2 backbone.
    The CNN feature encoder is frozen separately via `freeze_feature_encoder`.
    Returns the number of layers frozen (purely for logging)."""
    layers = model.wav2vec2.encoder.layers
    n_total = len(layers)
    n_to_freeze = max(n_total - n_top_layers_to_train, 0)
    for i, layer in enumerate(layers):
        if i < n_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False
    return n_to_freeze, n_total


class W2V2RegressionCollator:
    """Pad variable-length `input_values` to the longest in the batch and
    attach float `labels`. Avoids the wasteful 'pad-everything-to-10s'
    approach when the batch happens to contain short clips.

    If every record in the batch carries `speaker_idx` (the speaker-adversarial
    train fold), also attach a `speaker_labels` long tensor. Dev/evl batches
    have no speaker_idx and skip this — compute_loss conditionally adds the
    speaker CE term only when speaker_labels is present."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, features):
        input_features = [{'input_values': f['input_values']} for f in features]
        batch = self.fe.pad(input_features, padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor(
            [f['labels'] for f in features], dtype=torch.float32
        )
        if all('speaker_idx' in f for f in features):
            batch['speaker_labels'] = torch.tensor(
                [f['speaker_idx'] for f in features], dtype=torch.long
            )
        if all('gender_idx' in f for f in features):
            batch['gender_labels'] = torch.tensor(
                [f['gender_idx'] for f in features], dtype=torch.long
            )
        return batch


def make_compute_metrics(y_mean=0.0, y_std=1.0):
    """Factory that returns a compute_metrics callable. The model is trained
    on normalised targets (zero mean, unit std on the training fold), so its
    raw outputs live near 0; here we denormalise them before computing MAE/MSE
    against the raw-age reference labels."""
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        # In the speaker-adversarial setup eval_pred.predictions can be a
        # tuple (age_logits, speaker_logits) if the speaker_logits weren't
        # filtered out by config.keys_to_ignore_at_inference. Defensively
        # pick the first element (age_logits) in that case.
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds_norm = preds.squeeze(-1)
        preds = preds_norm * y_std + y_mean
        refs = eval_pred.label_ids                     # raw ages, never normalised
        return {
            'mae': float(mean_absolute_error(refs, preds)),
            'mse': float(np.mean((preds - refs) ** 2)),
        }
    return compute_metrics


class GradientReversalFunction(Function):
    """Forward: identity. Backward: multiplies incoming gradient by `-lambda`.

    Standard DANN trick — placed between the shared backbone and the
    adversarial speaker classifier head. The speaker head receives the
    normal forward signal and trains to classify speakers; the backbone
    receives the NEGATED speaker gradient, so it learns features that
    confuse the speaker head while still solving age regression."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


@dataclass
class AgeSpeakerOutput(ModelOutput):
    """Forward output of Wav2Vec2AgeSpeaker.

    `logits` is the age scalar (same key the parent RegressionTrainer reads
    via `outputs.logits`); `speaker_logits` is the per-speaker classification
    output added for adversarial training. Eval / compute_metrics only look
    at `.logits`, so they keep working unchanged."""
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    speaker_logits: torch.FloatTensor = None


class Wav2Vec2AgeSpeaker(Wav2Vec2ForSequenceClassification):
    """Wav2Vec2ForSequenceClassification + a speaker-adversarial head.

    Two heads on the same pooled features:
      - age regressor (parent's `self.classifier`, single scalar)
      - speaker classifier (2-layer MLP, `n_speakers` outputs)

    A GradientReversalFunction sits between the pooled features and the
    speaker head. During backward, it flips the sign of the gradient
    flowing to the backbone (and scales by `self.adv_lambda`). The speaker
    head itself trains normally.

    `n_speakers` is read from `config.n_speakers` (set by the caller before
    `.from_pretrained`)."""

    def __init__(self, config):
        super().__init__(config)
        n_speakers = int(getattr(config, 'n_speakers', 1))
        cps = config.classifier_proj_size
        self.speaker_classifier = nn.Sequential(
            nn.Linear(cps, cps),
            nn.ReLU(),
            nn.Linear(cps, n_speakers),
        )
        self.adv_lambda = 0.0      # set per-step by SpeakerAdversarialTrainer
        self.config.n_speakers = n_speakers
        # Tell HF Trainer's prediction_step to skip the speaker classifier
        # output when gathering inference predictions. Without this, the
        # eval pred is a (age_logits, speaker_logits) tuple and compute_metrics
        # breaks on `.squeeze(-1)` because tuples have no squeeze.
        existing = list(getattr(self.config, 'keys_to_ignore_at_inference', None) or [])
        if 'speaker_logits' not in existing:
            existing.append('speaker_logits')
        self.config.keys_to_ignore_at_inference = existing
        # Initialise the new head with the same init scheme as the rest.
        self.post_init()

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        speaker_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Re-implement the parent's forward up to the pooled features so we
        # can branch the speaker head off the same intermediate tensor.
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

        # Attention-aware mean pool over time (matches parent's behaviour).
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask,
            )
            hidden_states = hidden_states.masked_fill(
                ~padding_mask.unsqueeze(-1), 0.0,
            )
            denom = padding_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
            pooled = hidden_states.sum(dim=1) / denom

        age_logits = self.classifier(pooled)

        # Speaker head through gradient-reversal: forward is identity, backward
        # multiplies the backbone-bound gradient by -lambda.
        pooled_rev = GradientReversalFunction.apply(pooled, self.adv_lambda)
        speaker_logits = self.speaker_classifier(pooled_rev)

        return AgeSpeakerOutput(
            loss=None,
            logits=age_logits,
            speaker_logits=speaker_logits,
        )


@dataclass
class AgeGenderOutput(ModelOutput):
    """Forward output of Wav2Vec2AgeGender. Same .logits convention as the
    parent so existing eval / inference paths work unchanged; the gender
    branch lives on a separate key and is only consumed during training."""
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    gender_logits: torch.FloatTensor = None


class Wav2Vec2AgeGender(Wav2Vec2ForSequenceClassification):
    """Wav2Vec2ForSequenceClassification + an auxiliary gender head.

    Two heads on the same pooled features:
      - age regressor (parent's `self.classifier`, single scalar)
      - gender classifier (2-layer MLP, 2-way M/F)

    Both heads see the SAME pooled features — no gradient reversal, no
    adversary. The auxiliary gender loss adds gradient signal to the
    backbone, encouraging it to produce features that disentangle the
    gender axis from the age axis. This is multi-task learning, the
    inverse of adversarial. Training loss is age_loss + lambda * gender_CE.

    Eval time forward passes don't receive gender_labels, so the gender
    branch is computed but its CE term is skipped by the trainer."""

    def __init__(self, config):
        super().__init__(config)
        cps = config.classifier_proj_size
        self.gender_classifier = nn.Sequential(
            nn.Linear(cps, cps),
            nn.ReLU(),
            nn.Linear(cps, 2),
        )
        # Hide gender_logits from HF Trainer's prediction collection so
        # eval_pred.predictions stays a single array (age logits) — same
        # trick used in Wav2Vec2AgeSpeaker.
        existing = list(getattr(self.config, 'keys_to_ignore_at_inference', None) or [])
        if 'gender_logits' not in existing:
            existing.append('gender_logits')
        self.config.keys_to_ignore_at_inference = existing
        self.post_init()

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        gender_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Replicate parent's forward up to the pooled features so we can
        # branch the gender head off the same intermediate tensor.
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask,
            )
            hidden_states = hidden_states.masked_fill(
                ~padding_mask.unsqueeze(-1), 0.0,
            )
            denom = padding_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
            pooled = hidden_states.sum(dim=1) / denom

        age_logits = self.classifier(pooled)
        gender_logits = self.gender_classifier(pooled)

        return AgeGenderOutput(
            loss=None,
            logits=age_logits,
            gender_logits=gender_logits,
        )


def _age_loss(logits, target, loss_type='mse', tau=0.5):
    """Loss dispatch for the age head.

    Why support more than MSE: MSE-trained regressors shrink predictions
    toward the conditional mean — diagnose_4_91.py measured a -0.37 slope
    of (residual vs true_age) for the 4.91 model, i.e. ~37% regression
    toward the training mean. L1 / quantile losses don't have this
    shrinkage property (they target conditional medians/quantiles instead
    of means), so they should produce a flatter residual curve at the
    age tails.

    Loss formulas (all operate on normalised target):
      mse      : (yhat - y)^2                 -> conditional mean
      l1       : |yhat - y|                   -> conditional median (= quantile@0.5)
      quantile : max(tau*(y-yhat), (tau-1)*(y-yhat))   -> conditional quantile@tau
    """
    if loss_type == 'mse':
        return nn.functional.mse_loss(logits, target)
    if loss_type == 'l1':
        return nn.functional.l1_loss(logits, target)
    if loss_type == 'quantile':
        err = target - logits
        return torch.mean(torch.maximum(tau * err, (tau - 1.0) * err))
    raise ValueError(f'unknown loss_type: {loss_type!r}')


class RegressionTrainer(Trainer):
    """Single-logit regression with target normalisation + selectable loss.

    Without normalisation, raw ages (~50, std ~10) make MSE start at ~2500
    on a fresh random head; gradients explode, default clipping caps the
    step size, and the model never escapes "predict near 0". Normalising
    targets to ~N(0, 1) keeps initial loss on the order of 1 and lets the
    head actually move. Set `trainer.y_mean` / `trainer.y_std` after init.

    Also: set `trainer.loss_type` ('mse' / 'l1' / 'quantile') and
    `trainer.tau` (only used by quantile)."""
    y_mean = 0.0
    y_std = 1.0
    loss_type = 'mse'
    tau = 0.5

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop('labels').float()
        # Strip speaker_labels if present — vanilla RegressionTrainer ignores
        # them (only SpeakerAdversarialTrainer below uses them).
        inputs.pop('speaker_labels', None)
        normalised = (labels - self.y_mean) / self.y_std
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = _age_loss(logits, normalised, self.loss_type, self.tau)
        return (loss, outputs) if return_outputs else loss


class SpeakerAdversarialTrainer(RegressionTrainer):
    """RegressionTrainer + adversarial speaker classification loss.

    Total loss = MSE(age) + CE(speaker).  The lambda used to scale the
    speaker-gradient flowing back into the backbone lives in
    `model.adv_lambda` (set per-step via the GradientReversalFunction). We
    linearly ramp it from 0 to `adv_lambda_max` over the first
    `adv_warmup_steps` optimizer steps — so the backbone first learns to
    predict age before the adversary starts pushing back. The speaker head
    itself trains at full strength regardless of lambda.

    Eval-time forward passes don't carry `speaker_labels` (dev/evl speakers
    are unseen), so the speaker-CE term is skipped automatically."""

    adv_lambda_max = 0.05
    adv_warmup_steps = 200

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop('labels').float()
        speaker_labels = inputs.pop('speaker_labels', None)
        normalised = (labels - self.y_mean) / self.y_std

        # Ramp lambda only on the backbone-bound branch (the GRL).
        if model.training:
            step = self.state.global_step
            lam = self.adv_lambda_max * min(
                1.0, step / max(1, self.adv_warmup_steps)
            )
        else:
            lam = 0.0
        # Trainer wrapping might give us a DataParallel wrapper; reach inside.
        target = model.module if hasattr(model, 'module') else model
        target.adv_lambda = lam

        outputs = model(**inputs)
        age_logits = outputs.logits.squeeze(-1)
        age_loss = _age_loss(age_logits, normalised, self.loss_type, self.tau)

        if speaker_labels is not None and outputs.speaker_logits is not None:
            ce = nn.functional.cross_entropy(
                outputs.speaker_logits, speaker_labels.long(),
            )
            # The GRL has already flipped the sign on the backbone side and
            # scaled by lambda — here we just sum (NOT subtract).
            loss = age_loss + ce
        else:
            loss = age_loss

        return (loss, outputs) if return_outputs else loss


class MultiTaskTrainer(RegressionTrainer):
    """RegressionTrainer + auxiliary gender CE loss.

    Total loss = L_age(normalised) + mt_lambda * CE(gender_logits, gender_labels).

    No gradient reversal — gender is a HELPER task, not an adversary. The
    auxiliary loss adds gradient signal to the backbone and pushes it to
    encode features that separate the gender axis from the age axis, which
    empirically helps age MAE by 0.1-0.3.

    Eval-time forward passes don't include gender_labels (we never have
    them on dev/evl through the eval pipeline), so the gender CE term is
    automatically skipped and only the age loss is computed."""

    mt_lambda = 0.5  # set after init by main()

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop('labels').float()
        gender_labels = inputs.pop('gender_labels', None)
        # The vanilla trainer strips speaker_labels; do the same defensively.
        inputs.pop('speaker_labels', None)
        normalised = (labels - self.y_mean) / self.y_std

        outputs = model(**inputs)
        age_logits = outputs.logits.squeeze(-1)
        age_loss = _age_loss(age_logits, normalised, self.loss_type, self.tau)

        # Optionally add gender CE when the labels are present in the batch
        # AND the model returned gender_logits. Mask out rows where the gender
        # label is -1 (unknown / non-binary; rare on lab train but defensive).
        if (gender_labels is not None
                and getattr(outputs, 'gender_logits', None) is not None):
            mask = gender_labels >= 0
            if mask.any():
                gl = outputs.gender_logits[mask]
                gy = gender_labels[mask].long()
                gender_loss = nn.functional.cross_entropy(gl, gy)
                loss = age_loss + self.mt_lambda * gender_loss
            else:
                loss = age_loss
        else:
            loss = age_loss

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run(args):
    datadir = LAB2_DIR / 'lab2_data'
    device = select_device_str()
    print(f'device  : {device}')
    print(f'trainset: {args.trainset}')
    print(f'epochs  : {args.epochs}   batch: {args.batch_size}  '
          f'grad_accum: {args.grad_accum}   '
          f'effective batch: {args.batch_size * args.grad_accum}')
    print(f'lr      : {args.lr}')

    # ----- pre-load manifest (needed to size the speaker head if --speaker-adv) -----
    manifest = None
    speaker_to_idx = None
    n_speakers = 0
    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
        if not manifest_path.is_absolute():
            manifest_path = LAB2_DIR / manifest_path
        with open(manifest_path) as f:
            manifest = json.load(f)
        if args.speaker_adv:
            if 'uuid_to_speaker' not in manifest:
                raise ValueError(
                    '--speaker-adv requires a manifest with `uuid_to_speaker` '
                    '(produced by build_speaker_split_real.py from a partition '
                    'that has ground-truth speaker_ids, e.g. big_train_falar). '
                    'The clustering-based manifest lacks this mapping.'
                )
            train_inner_uuids = set(manifest['train_inner_uuids'])
            train_speakers = sorted({
                str(manifest['uuid_to_speaker'][u])
                for u in train_inner_uuids
                if u in manifest['uuid_to_speaker']
            })
            speaker_to_idx = {s: i for i, s in enumerate(train_speakers)}
            n_speakers = len(speaker_to_idx)
            print(f'  speaker-adversarial: {n_speakers} unique train speakers '
                  f'will be classified by the adversary head')
    elif args.speaker_adv:
        raise ValueError('--speaker-adv requires --split-manifest '
                         '(speaker IDs come from the manifest)')

    # ----- model -----
    print(f'\n-- loading audeering wav2vec2 (regression head) --')
    t0 = time.time()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(AUD_MODEL_ID)
    tf_logging.set_verbosity_error()        # silence expected unexpected-keys

    # SpecAugment kwargs — apply random time + frequency masking to the
    # backbone's hidden states during training as regularization. These
    # are read by the Wav2Vec2 model on every forward pass when
    # apply_spec_augment=True. Off by default; turn on with --spec-augment.
    spec_kwargs = {}
    if args.spec_augment:
        spec_kwargs = dict(
            apply_spec_augment=True,
            mask_time_prob=args.spec_time_prob,
            mask_time_length=args.spec_time_length,
            mask_feature_prob=args.spec_feature_prob,
            mask_feature_length=args.spec_feature_length,
        )
        print(f'  SpecAugment ON: '
              f'time_prob={args.spec_time_prob} time_length={args.spec_time_length} '
              f'feature_prob={args.spec_feature_prob} feature_length={args.spec_feature_length}')

    if args.speaker_adv:
        config = Wav2Vec2Config.from_pretrained(
            AUD_MODEL_ID, num_labels=1, problem_type='regression',
            **spec_kwargs,
        )
        config.n_speakers = n_speakers
        model = Wav2Vec2AgeSpeaker.from_pretrained(AUD_MODEL_ID, config=config)
        print(f'  Wav2Vec2AgeSpeaker  head: {n_speakers} train speakers, '
              f'lambda_max={args.adv_lambda:.3f}, warmup_steps={args.adv_warmup_steps}')
    elif args.multi_task:
        config = Wav2Vec2Config.from_pretrained(
            AUD_MODEL_ID, num_labels=1, problem_type='regression',
            **spec_kwargs,
        )
        model = Wav2Vec2AgeGender.from_pretrained(AUD_MODEL_ID, config=config)
        print(f'  Wav2Vec2AgeGender  head: aux gender (M/F), '
              f'mt_lambda={args.mt_lambda}')
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUD_MODEL_ID,
            num_labels=1,
            problem_type='regression',
            **spec_kwargs,
        )
    tf_logging.set_verbosity_warning()

    model.freeze_feature_encoder()
    if args.top_layers is not None:
        n_frozen, n_total_layers = freeze_lower_layers(model, args.top_layers)
        print(f'  frozen lower transformer layers: {n_frozen}/{n_total_layers}  '
              f'(training top {args.top_layers} + regression head)')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  loaded in {time.time()-t0:.1f}s')
    print(f'  trainable params: {n_trainable/1e6:.1f} M / {n_total/1e6:.1f} M '
          f'({100*n_trainable/n_total:.1f}%)')

    # ----- data -----
    print(f'\n-- building HF datasets (librosa + feature extractor) --')
    df_train = load_partition_dataframe(datadir, args.trainset)
    df_dev   = load_partition_dataframe(datadir, 'dev')
    df_evl   = load_partition_dataframe(datadir, 'evl')
    print(f'  {args.trainset}: {len(df_train)}   dev: {len(df_dev)}   evl: {len(df_evl)}')

    # Speaker-disjoint internal split: when a manifest is loaded (above), replace
    # the default train+dev partitions with the manifest's speaker-disjoint fold.
    # evl is left untouched.
    if manifest is not None:
        manifest_part = manifest.get('partition') or manifest.get('trainset')
        if manifest_part and manifest_part != args.trainset:
            print(f'  WARNING: manifest was built from {manifest_part!r} '
                  f'but --trainset is {args.trainset!r}; UUIDs may not match.')

        train_inner = set(manifest['train_inner_uuids'])
        dev_inner   = set(manifest['dev_inner_uuids'])
        if train_inner & dev_inner:
            raise ValueError('manifest train_inner and dev_inner overlap; '
                             'something is wrong with the split file')

        # df_train currently holds the full --trainset partition; carve it.
        base = df_train
        df_train = base[base['fileid'].isin(train_inner)].reset_index(drop=True)
        if args.use_lab_dev:
            # --use-lab-dev: train on train_inner (so adversary speaker labels
            # still work) but evaluate on lab dev (the 117-row official metric)
            # instead of dev_inner. The manifest is used purely for the
            # speaker-adversarial labels + train carving; dev_inner is ignored.
            # df_dev is whatever load_partition_dataframe(datadir, 'dev')
            # already produced above — keep it.
            missing_t = len(train_inner) - len(df_train)
            print(f'  [split-manifest + --use-lab-dev] '
                  f'train_inner: {len(df_train)} '
                  f'(missing UUIDs: {missing_t}) | '
                  f'dev: lab dev ({len(df_dev)} rows)')
        else:
            df_dev = base[base['fileid'].isin(dev_inner)].reset_index(drop=True)
            missing_t = len(train_inner) - len(df_train)
            missing_d = len(dev_inner) - len(df_dev)
            print(f'  [split-manifest] speaker-disjoint fold')
            print(f'    train_inner: {len(df_train)}   '
                  f'dev_inner: {len(df_dev)}'
                  + (f'   (missing UUIDs: train_inner={missing_t} '
                     f'dev_inner={missing_d})' if (missing_t or missing_d) else ''))
        if len(df_train) == 0 or len(df_dev) == 0:
            raise RuntimeError('split-manifest produced an empty fold; '
                               'check that --trainset matches the manifest.')

        # Attach speaker_idx for adversarial training (train only).
        if args.speaker_adv:
            u2s = manifest['uuid_to_speaker']
            df_train['speaker_idx'] = df_train['fileid'].map(
                lambda u: speaker_to_idx[str(u2s[u])]
            ).astype(int)
            print(f'    speaker_idx assigned: {df_train["speaker_idx"].nunique()} '
                  f'unique speaker indices in train_inner')

    # Target normalisation stats computed on the training fold only. Raw
    # labels stay raw inside the Dataset; the RegressionTrainer normalises
    # them just before computing MSE, and compute_metrics / final inference
    # denormalise predictions back to age in years.
    train_ages = [float(a) for a in df_train['age'].tolist() if not pd.isna(a)]
    y_mean = float(np.mean(train_ages))
    y_std  = float(np.std(train_ages) + 1e-8)
    print(f'  target normalisation: y_mean={y_mean:.2f}  y_std={y_std:.2f}')

    # Lazy torch Datasets — no Arrow table, no int32-offset overflow. Audio
    # is loaded on __getitem__; collator handles per-batch padding.
    ds = {
        'train': build_dataset(df_train, fe, duration=args.duration,
                               has_labels=True,
                               has_speaker=args.speaker_adv,
                               has_gender=args.multi_task,
                               desc=f'{args.trainset:>10s}'),
        'dev':   build_dataset(df_dev,   fe, duration=args.duration,
                               has_labels=True,  desc=f'{"dev":>10s}'),
        'evl':   build_dataset(df_evl,   fe, duration=args.duration,
                               has_labels=False, desc=f'{"evl":>10s}'),
    }

    # ----- training -----
    suffix = f'_{args.run_id}' if args.run_id else ''
    output_dir = (LAB2_DIR / 'lab2_data' / args.trainset / 'models'
                  / f'finetune_audeering_age{suffix}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # If --max-epochs is set, use it as the ceiling and let EarlyStopping decide
    # the real stopping point. Otherwise fall back to the fixed --epochs schedule.
    use_early_stop = args.max_epochs is not None
    n_epochs = args.max_epochs if use_early_stop else args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 4),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=n_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=10,
        logging_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_mae',
        greater_is_better=False,
        report_to=[],                       # no wandb / tensorboard
        # MPS can be picky with workers; CUDA benefits from parallel
        # audio loading (lazy dataset loads from disk on every __getitem__).
        dataloader_num_workers=(2 if device == 'cuda' else 0),
        remove_unused_columns=False,
        # Without this, HF Trainer auto-detects label_names by scanning the
        # model.forward signature for parameters containing 'label'. With
        # the speaker-adv head, that picks up BOTH 'labels' and
        # 'speaker_labels' — and on dev batches (no speaker_labels) it
        # treats has_labels=False, skipping loss/compute_metrics entirely.
        # Pin it to the actual label name.
        label_names=['labels'],
        # only enable fp16 on CUDA; MPS / CPU stick with fp32
        fp16=(device == 'cuda' and not args.no_fp16),
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    callbacks = []
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.patience,
            early_stopping_threshold=0.01,
        ))

    if args.speaker_adv:
        trainer_cls = SpeakerAdversarialTrainer
    elif args.multi_task:
        trainer_cls = MultiTaskTrainer
    else:
        trainer_cls = RegressionTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        data_collator=W2V2RegressionCollator(fe),
        compute_metrics=make_compute_metrics(y_mean, y_std),
        callbacks=callbacks,
    )
    trainer.y_mean = y_mean
    trainer.y_std  = y_std
    trainer.loss_type = args.loss
    trainer.tau = args.tau
    if args.multi_task:
        trainer.mt_lambda = float(args.mt_lambda)
    if args.speaker_adv:
        trainer.adv_lambda_max    = float(args.adv_lambda)
        trainer.adv_warmup_steps  = int(args.adv_warmup_steps)

    if use_early_stop:
        print(f'\n-- training (max {n_epochs} epochs, '
              f'early-stop patience={args.patience}) --')
    else:
        print(f'\n-- training ({n_epochs} epochs, no early stopping) --')
    train_t0 = time.time()
    trainer.train()
    train_dt = time.time() - train_t0
    print(f'\n  training finished in {train_dt:.1f}s ({train_dt/60:.1f} min)')

    # ----- evaluation (denormalise model outputs to age in years) -----
    print('\n-- final dev prediction --')
    dev_pred = trainer.predict(ds['dev'])
    dev_hyp = np.clip(
        dev_pred.predictions.squeeze(-1) * y_std + y_mean,
        AGE_MIN, AGE_MAX,
    )
    dev_ref = np.asarray(dev_pred.label_ids, dtype=float)
    dev_mae = float(mean_absolute_error(dev_ref, dev_hyp))
    print(f'  best-model dev MAE: {dev_mae:.3f}')

    print('\n-- evl prediction --')
    evl_pred = trainer.predict(ds['evl'])
    evl_hyp = np.clip(
        evl_pred.predictions.squeeze(-1) * y_std + y_mean,
        AGE_MIN, AGE_MAX,
    )

    # ----- submission CSV (same pkl format the sweep script uses) -----
    resdir = output_dir / 'final'
    resdir.mkdir(parents=True, exist_ok=True)
    with open(resdir / 'dev.pkl', 'wb') as f:
        pickle.dump({'hyp': dev_hyp,
                     'fileids': np.asarray(df_dev['fileid'].tolist())}, f)
    with open(resdir / 'evl.pkl', 'wb') as f:
        pickle.dump({'hyp': evl_hyp,
                     'fileids': np.asarray(df_evl['fileid'].tolist())}, f)
    sub_path = LAB2_DIR / f'g{args.group}_{args.trainset}_finetune_audeering_age{suffix}.csv'
    create_submission_file(str(resdir), str(sub_path))
    print(f'  submission -> {sub_path.name}')

    # ----- per-epoch trajectory (the actual insight) -----
    print('\n========= DEV MAE TRAJECTORY =========')
    train_losses, dev_maes = {}, {}
    for entry in trainer.state.log_history:
        if 'eval_mae' in entry and 'epoch' in entry:
            dev_maes[round(entry['epoch'], 4)] = entry['eval_mae']
        elif 'loss' in entry and 'epoch' in entry and 'eval_mae' not in entry:
            # Keep the most recent train-loss log per epoch (overwrites within epoch).
            train_losses[round(entry['epoch'], 4)] = entry['loss']

    print(f'{"epoch":>6s}  {"train_loss":>12s}  {"dev MAE":>8s}')
    print('-' * 34)
    all_epochs = sorted(set(list(train_losses.keys()) + list(dev_maes.keys())))
    for epoch in all_epochs:
        tl = train_losses.get(epoch)
        dm = dev_maes.get(epoch)
        tl_s = f'{tl:12.4f}' if tl is not None else '           -'
        dm_s = f'{dm:8.3f}' if dm is not None else '       -'
        print(f'{epoch:6.2f}  {tl_s}  {dm_s}')

    # Verdict — what to do with the result.
    sorted_items = sorted(dev_maes.items())
    sorted_devs = [v for _, v in sorted_items]
    if sorted_devs:
        baseline = 6.0   # frozen-feature ceiling on this dataset
        first, last = sorted_devs[0], sorted_devs[-1]
        best = min(sorted_devs)
        best_epoch = min(sorted_items, key=lambda kv: kv[1])[0]
        n_epochs_done = len(sorted_devs)
        stop_msg = ''
        if use_early_stop and n_epochs_done < n_epochs:
            stop_msg = (f' (early-stopped at epoch {n_epochs_done} '
                        f'of ceiling {n_epochs})')
        print(f'\n  frozen-feature reference (sweep ensemble): {baseline:.2f}')
        print(f'  this run: first epoch dev = {first:.3f}, '
              f'final = {last:.3f}, best = {best:.3f} '
              f'@ epoch {best_epoch:g}{stop_msg}')
        if best < baseline - 0.5:
            print('  -> meaningful improvement. Fine-tuning works for our data; '
                  'rent the GPU and scale up.')
        elif best < baseline - 0.1:
            print('  -> small improvement. Worth scaling once on a GPU; '
                  'try more epochs and larger batch.')
        else:
            print('  -> no clear gain over frozen features. Debug LR / freeze '
                  'policy before spending on GPU.')

    # CSV trajectory for offline plotting / comparison.
    csv_path = LAB2_DIR / f'finetune_results{suffix}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'dev_mae'])
        for epoch in all_epochs:
            w.writerow([
                epoch,
                f'{train_losses[epoch]:.6f}' if epoch in train_losses else '',
                f'{dev_maes[epoch]:.6f}'     if epoch in dev_maes     else '',
            ])
    print(f'\nTrajectory CSV: {csv_path.name}')

    # Free GPU memory before the next grid cell (no-op when run stand-alone).
    del trainer, model, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dev_mae


# ---------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------

# Grid cells. Each row: (run_id, lr, epochs, top_layers, seed).
# `epochs` is interpreted as the EarlyStopping ceiling when --grid is run
# with --max-epochs; otherwise it's the fixed number of epochs.
GRID = [
    # ----- Phase A: seed sweep at the winning recipe (uncertainty on 5.19) -----
    ('A_lr5e5_s42',  5e-5, 40, None,  42),
    ('A_lr5e5_s7',   5e-5, 40, None,   7),
    ('A_lr5e5_s13',  5e-5, 40, None,  13),
    ('A_lr5e5_s21',  5e-5, 40, None,  21),
    ('A_lr5e5_s99',  5e-5, 40, None,  99),

    # ----- Phase B: LR neighbourhood at seed 42 -----
    ('B_lr3e5_s42',  3e-5, 40, None,  42),
    ('B_lr4e5_s42',  4e-5, 40, None,  42),
    ('B_lr6e5_s42',  6e-5, 40, None,  42),
    ('B_lr7e5_s42',  7e-5, 40, None,  42),
    ('B_lr8e5_s42',  8e-5, 40, None,  42),

    # ----- Phase D: lower LR with high ceiling (1e-5 deserved its fair shot) -----
    ('D_lr1e5_s42',  1e-5, 60, None,  42),
    ('D_lr2e5_s42',  2e-5, 60, None,  42),
]


def run_grid(args):
    print(f'\n{"="*80}')
    print(f'GRID SEARCH — {len(GRID)} runs')
    print(f'{"="*80}')
    for i, (run_id, lr, epochs, top_n, seed) in enumerate(GRID, 1):
        print(f'  {i:>2d}. {run_id:<14s}  lr={lr:.0e}  epochs={epochs}  '
              f'top_layers={top_n}  seed={seed}')

    grid_csv = LAB2_DIR / f'grid_results_{args.trainset}.csv'
    results = []

    for i, (run_id, lr, epochs, top_n, seed) in enumerate(GRID, 1):
        print(f'\n\n{"="*80}')
        print(f'GRID CELL {i}/{len(GRID)}: {run_id}')
        print(f'  lr={lr}  epochs={epochs}  top_layers={top_n}  seed={seed}')
        print(f'{"="*80}')

        cell_args = copy.deepcopy(args)
        cell_args.lr = lr
        cell_args.epochs = epochs
        # In grid mode the per-cell `epochs` is used as the early-stopping
        # ceiling unless the user explicitly disabled early stopping via
        # --no-early-stop on the grid invocation.
        if getattr(args, 'no_early_stop', False):
            cell_args.max_epochs = None
        else:
            cell_args.max_epochs = epochs
        cell_args.patience = getattr(args, 'patience', 5)
        cell_args.top_layers = top_n
        cell_args.seed = seed
        cell_args.run_id = run_id
        cell_args.grid = False           # do not recurse

        t0 = time.time()
        try:
            dev_mae = run(cell_args)
            status = 'ok'
        except Exception as e:
            print(f'\n  -> {run_id} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            dev_mae = float('nan')
            status = f'failed:{type(e).__name__}'
        dt = time.time() - t0

        results.append({
            'run_id': run_id, 'lr': lr, 'epochs': epochs,
            'top_layers': top_n if top_n is not None else 'all',
            'seed': seed, 'dev_mae': dev_mae, 'time_s': dt, 'status': status,
        })

        # Write the running CSV after every cell so an interrupted run keeps its results.
        with open(grid_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run_id', 'lr', 'epochs', 'top_layers', 'seed',
                        'dev_mae', 'time_s', 'status'])
            for r in results:
                dm = f'{r["dev_mae"]:.4f}' if not np.isnan(r["dev_mae"]) else 'NaN'
                w.writerow([r['run_id'], r['lr'], r['epochs'], r['top_layers'],
                            r['seed'], dm, f'{r["time_s"]:.1f}', r['status']])

    # Final summary
    print(f'\n\n{"="*80}')
    print('GRID SUMMARY (sorted by dev MAE)')
    print(f'{"="*80}')
    valid = [r for r in results if not np.isnan(r['dev_mae'])]
    valid.sort(key=lambda r: r['dev_mae'])
    print(f'{"rank":>4s}  {"run_id":<14s}  {"lr":>8s}  {"ep":>3s}  '
          f'{"top":>5s}  {"seed":>4s}  {"dev MAE":>8s}  {"time":>8s}')
    print('-' * 70)
    for i, r in enumerate(valid, 1):
        print(f'{i:>4d}  {r["run_id"]:<14s}  {r["lr"]:>8.0e}  {r["epochs"]:>3d}  '
              f'{str(r["top_layers"]):>5s}  {r["seed"]:>4d}  '
              f'{r["dev_mae"]:>8.3f}  {r["time_s"]:>7.0f}s')
    failed = [r for r in results if np.isnan(r['dev_mae'])]
    if failed:
        print(f'\nFailed cells ({len(failed)}):')
        for r in failed:
            print(f'  {r["run_id"]:<14s}  {r["status"]}')
    print(f'\nGrid CSV: {grid_csv.name}')
    if valid:
        best = valid[0]
        print(f'\nBest: {best["run_id"]} -> dev MAE {best["dev_mae"]:.3f}.  '
              f'Submission: g{args.group}_{args.trainset}_finetune_audeering_age_'
              f'{best["run_id"]}.csv')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--trainset', default='train_small',
                        choices=('train_small', 'train', 'big_train_falar',
                                 'big_train_diversified', 'lab_train_old_boost',
                                 'lab_train_old_boost_n3', 'lab_train_old_boost_n4',
                                 'lab_train_old_boost_n5', 'lab_train_old_boost_n6',
                                 'lab_train_old_boost_n7', 'lab_train_old_boost_n8'),
                        help='training partition (default train_small). '
                             '"big_train_falar" is the FalAR-streamed expansion '
                             '(many utts per speaker, 300 speakers). '
                             '"big_train_diversified" is the SAA-style 1-utt-per-'
                             'speaker variant built by build_train_diversified.py. '
                             '"lab_train_old_boost" is lab train + extra FalAR '
                             'utts for 60+ speakers (build_old_speaker_boost.py).')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epochs (default 3). '
                             'Ignored when --max-epochs is given.')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='if set, enables early stopping. Training runs up '
                             'to this many epochs but stops once dev MAE stops '
                             'improving (see --patience). Recommended: 40.')
    parser.add_argument('--patience', type=int, default=5,
                        help='early-stop patience: number of consecutive eval '
                             'rounds without dev-MAE improvement (>=0.01) before '
                             'stopping. Only used with --max-epochs. Default 5.')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='per-device batch size (default 2 for MPS; 8-16 on a GPU)')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='gradient accumulation steps. '
                             'effective batch = batch_size * grad_accum (default 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='AdamW learning rate (default 5e-5)')
    parser.add_argument('--loss', default='mse',
                        choices=('mse', 'l1', 'quantile'),
                        help='age-head loss. "mse" targets the conditional '
                             'mean (default; produces regression-toward-mean '
                             'shrinkage at the tails). "l1" targets the '
                             'conditional median (= quantile at tau=0.5) and '
                             'directly optimises MAE. "quantile" uses pinball '
                             'loss with --tau for asymmetric targets.')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='quantile parameter for --loss quantile '
                             '(default 0.5 = median; 0.5 here is equivalent '
                             'to --loss l1 up to a factor of 2)')
    parser.add_argument('--spec-augment', action='store_true',
                        help='enable SpecAugment-style random time + feature '
                             'masking on the backbone hidden states during '
                             'training. Standard regularisation for SSL '
                             'speech models; typically saves 0.2-0.4 MAE on '
                             'small training sets. Off by default.')
    parser.add_argument('--spec-time-prob', type=float, default=0.05,
                        help='probability of starting a time mask per frame '
                             '(default 0.05). Higher = more aggressive masking.')
    parser.add_argument('--spec-time-length', type=int, default=10,
                        help='length (in time frames) of each time mask '
                             '(default 10)')
    parser.add_argument('--spec-feature-prob', type=float, default=0.05,
                        help='probability of starting a feature mask per '
                             'channel (default 0.05)')
    parser.add_argument('--spec-feature-length', type=int, default=64,
                        help='length (in feature channels) of each feature '
                             'mask (default 64)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='max audio duration in seconds (default 10)')
    parser.add_argument('--group', default='07',
                        help='student group prefix for submission CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='trade compute for memory — useful on small-VRAM GPUs')
    parser.add_argument('--no-fp16', action='store_true',
                        help='disable fp16 even on CUDA (debugging)')
    parser.add_argument('--top-layers', type=int, default=None,
                        help='if set, freeze all but the top N transformer '
                             'layers (in addition to the always-frozen CNN). '
                             'Reduces trainable params and memory. Default: '
                             'all 24 layers trainable.')
    parser.add_argument('--run-id', default=None,
                        help='unique tag appended to output paths (checkpoint '
                             'dir, submission CSV, trajectory CSV). Required '
                             'in --grid mode (set automatically).')
    parser.add_argument('--split-manifest', default=None,
                        help='path to a speaker-disjoint split JSON produced '
                             'by build_speaker_split.py. When set, the '
                             '--trainset partition is carved into '
                             'train_inner / dev_inner by the manifest, and '
                             'dev_inner replaces the default dev partition. '
                             'evl is left untouched. Use to measure the honest '
                             'speaker-disjoint dev MAE (Phase 3).')
    parser.add_argument('--speaker-adv', action='store_true',
                        help='speaker-adversarial finetuning (DANN). Requires '
                             '--split-manifest with a `uuid_to_speaker` field '
                             '(produced by build_speaker_split_real.py from a '
                             'partition with ground-truth speaker_ids, e.g. '
                             'big_train_falar). Adds a gradient-reversal '
                             'speaker-classifier head that pushes the '
                             'backbone toward speaker-invariant features '
                             'while still solving age regression.')
    parser.add_argument('--multi-task', action='store_true',
                        help='multi-task learning: train an auxiliary gender '
                             '(M/F) classification head jointly with the age '
                             'regression head. The two heads share the same '
                             'pooled backbone features; total loss is '
                             'age_loss + mt_lambda * CE(gender). No gradient '
                             'reversal — gender is a HELPER, not adversary. '
                             'Empirically improves age MAE by 0.1-0.3 because '
                             'the auxiliary loss helps the backbone produce '
                             'features that disentangle gender from age.')
    parser.add_argument('--mt-lambda', type=float, default=0.5,
                        help='weight of the gender CE term in multi-task loss '
                             '(default 0.5). Higher = backbone caters more '
                             'to gender; too high will start to hurt age.')
    parser.add_argument('--use-lab-dev', action='store_true',
                        help='when --split-manifest is set, use lab dev (the '
                             '117-row official metric) as the eval set instead '
                             'of dev_inner from the manifest. Training data is '
                             'still carved by train_inner_uuids so adversarial '
                             'speaker labels keep working. Lets you compare '
                             'speaker-adv runs directly against the 4.91-style '
                             'lab dev baseline.')
    parser.add_argument('--adv-lambda', type=float, default=0.05,
                        help='maximum lambda for the gradient-reversal layer '
                             '(default 0.05). Higher = stronger speaker-'
                             'invariance pressure on the backbone.')
    parser.add_argument('--adv-warmup-steps', type=int, default=200,
                        help='optimizer steps to linearly ramp lambda from 0 '
                             'to --adv-lambda (default 200). Lets the backbone '
                             'first learn the age task before the adversary '
                             'starts pushing back.')
    parser.add_argument('--grid', action='store_true',
                        help='run the predefined GRID of fine-tune configs. '
                             'Cells use early stopping; the per-cell `epochs` '
                             'field is treated as the early-stop ceiling.')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='in --grid mode, disable early stopping and run '
                             'each cell for its fixed number of epochs.')
    args = parser.parse_args()
    if args.grid:
        run_grid(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
