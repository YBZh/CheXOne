from __future__ import annotations
import numpy as np
import torch
# from RadEval.nlg.bertscore.bertscore import BertScore
import torch.nn as nn
from radgraph import RadGraph
# from RadEval.factual.f1chexbert import F1CheXbert
from sklearn.preprocessing import StandardScaler
# from RadEval.nlg.bleu.bleu import Bleu
from bert_score import BERTScorer
# from RadEval import RadCliQv1

from contextlib import contextmanager
import torch


from rouge_score import rouge_scorer
from six.moves import zip_longest


class Rouge(nn.Module):
    def __init__(self, rouges, **kwargs):
        super().__init__()
        rouges = [r.replace('rougel', 'rougeL') for r in rouges]
        self.scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        self.rouges = rouges

    def forward(self, refs, hyps):
        scores = []
        for target_rec, prediction_rec in zip_longest(refs, hyps):
            if target_rec is None or prediction_rec is None:
                raise ValueError("Must have equal number of lines across target and "
                                 "prediction.")
            scores.append(self.scorer.score(target_rec, prediction_rec))
        f1_rouge = [s[self.rouges[0]].fmeasure for s in scores]
        return np.mean(f1_rouge), f1_rouge


class Rouge1(Rouge):
    def __init__(self, **kwargs):
        super(Rouge1, self).__init__(rouges=['rouge1'])

# ----------------------------- Bleu Scorer -----------------------------
#!/usr/bin/env python

# bleu_scorer.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# Modified by:
# Hao Fang <hfang@uw.edu>
# Tsung-Yi Lin <tl483@cornell.edu>

'''Provides:
cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
'''

import copy
import sys, math, re
from collections import defaultdict

import six
from six.moves import xrange as range


def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return (len(words), counts)

def cook_refs(refs, eff=None, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in six.iteritems(counts):
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen))/len(reflen)

    ## lhuang: N.B.: leave reflen computaiton to the very end!!

    ## lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return (reflen, maxcounts)

def cook_test(test, reflen_refmaxcounts, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''

    reflen, refmaxcounts = reflen_refmaxcounts
    testlen, counts = precook(test, n, True)

    result = {}

    # Calculate effective reference sentence length.

    if eff == "closest":
        result["reflen"] = min((abs(l-testlen), l) for l in reflen)[1]
    else: ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0,testlen-k+1) for k in range(1,n+1)]

    result['correct'] = [0]*n
    for (ngram, count) in six.iteritems(counts):
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result

class BleuScorer(object):
    """Bleu scorer.
    """

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

        self._score = None ## need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        '''return (bleu, len_ratio) pair'''
        return (self.fscore(option=option), self.ratio(option=option))

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''

        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None ## need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):

        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens))/len(reflens)
        elif option == "closest":
            reflen = min((abs(l-testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15 ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None: ## need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess','correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        /(float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1./(k+1)))
            ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1/ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1./(k+1)))
        ratio = (self._testlen + tiny) / (self._reflen + small) ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1/ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list

class Bleu(nn.Module):
    def __init__(self, n=4, **kwargs):
        # default compute Blue score up to 4
        super().__init__()
        self._n = n

    def forward(self, gts, res):
        return self.compute_score(gts, res)

    def compute_score(self, gts, res):
        res = {i: [v] for i, v in enumerate(res)}
        gts = {i: [v] for i, v in enumerate(gts)}
        bleu_scorer = BleuScorer(n=self._n)

        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score[self._n-1], scores[self._n-1]

    def method(self):
        return "Bleu"


# ----------------------------- BertScore -----------------------------
class BertScore(nn.Module):
    def __init__(self,
                 model_type='distilbert-base-uncased',
                 num_layers=5,
                 rescale_with_baseline=True,
                 idf=False,
                 model_local_dir: str | None = None,
                 local_files_only: bool | None = None,
                 cache_dir = '/home/zhangyabin/.cache/huggingface/hub/models--distilbert-base-uncased',
                 ):
        super(BertScore, self).__init__()
        
        # Normalize num_layers (bert-score accepts an int; guard against None)
        effective_layers = 5 if (num_layers is None) else num_layers
        # Resolve offline/local flags lazily to avoid requiring os at module import
        import os as _os
        offline_env = (
            str(_os.environ.get('TRANSFORMERS_OFFLINE', '')).lower() in {'1','true','yes'}
            or str(_os.environ.get('HF_HUB_OFFLINE', '')).lower() in {'1','true','yes'}
        )
        resolved_local_files_only = offline_env if (local_files_only is None) else local_files_only
        # If user passes a local dir as model_type, treat as local
        if model_local_dir is None:
            try:
                import os as _os2
                if _os2.path.isdir(model_type):
                    model_local_dir = model_type
            except Exception:
                pass
        resolved_model_id = model_local_dir if model_local_dir else model_type
        # import ipdb; ipdb.set_trace()
        # Avoid downloading baseline when offline
        use_rescale_baseline = bool(rescale_with_baseline and not resolved_local_files_only)

        def _force_model_fp32_on_cpu(bert_scorer_obj):
            model_attr = getattr(bert_scorer_obj, '_model', None)
            if model_attr is None:
                return
            # bert-score may hold a single model or a list; support both
            models = model_attr if isinstance(model_attr, (list, tuple)) else [model_attr]
            for m in models:
                try:
                    m.to(dtype=torch.float32, device='cpu')
                except Exception:
                    # Fallback: iterate params/buffers
                    for p in m.parameters(recurse=True):
                        p.data = p.data.float().cpu()
                    for b in m.buffers(recurse=True):
                        if b is not None:
                            b.data = b.data.float().cpu()

        def _total_params(model_attr) -> int:
            if model_attr is None:
                return 0
            if isinstance(model_attr, (list, tuple)):
                return sum(sum(p.numel() for p in m.parameters()) for m in model_attr)
            return sum(p.numel() for p in model_attr.parameters())

        # 尝试多种初始化方法
        # 方法1: 直接使用 BERTScorer (with retry and explicit dtype control)
        for attempt in range(1):
            try:
                from bert_score import BERTScorer
                from transformers import AutoModel, AutoTokenizer
                
                print(f"Method 1 attempt {attempt + 1}/3: Trying BERTScorer with {model_type}")
                print(f"Current torch_dtype: {torch.get_default_dtype()}")
                print(f"Current AMP state: {torch.is_autocast_enabled()}")
                
                # 保存当前环境状态
                original_dtype = torch.get_default_dtype()
                original_amp_state = torch.is_autocast_enabled()
                
                try:
                    # 强制设置为 float32 并在 CPU/CUDA 上禁用 autocast（上下文可重入，退出后自动恢复）
                    torch.set_default_dtype(torch.float32)
                    from contextlib import ExitStack
                    with ExitStack() as stack:
                        # Disable both CUDA and CPU autocast to avoid bfloat16 weight creation
                        try:
                            stack.enter_context(torch.autocast("cuda", enabled=False))
                        except Exception:
                            pass
                        try:
                            stack.enter_context(torch.autocast("cpu", enabled=False))
                        except Exception:
                            pass
                        # If offline is requested, set env so BERTScorer's internal AutoModel loads locally
                        if resolved_local_files_only:
                            _os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        if cache_dir:
                            _os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
                        self.bert_scorer = BERTScorer(
                            model_type=resolved_model_id,
                            num_layers=effective_layers,
                            batch_size=2,
                            nthreads=4,
                            all_layers=False,
                            idf=idf,
                            device='cuda',
                            lang='en',
                            rescale_with_baseline=use_rescale_baseline,
                            baseline_path=None
                        )
                    # Force model to CPU float32 explicitly (in case upstream respected ambient autocast)
                    # _force_model_fp32_on_cpu(self.bert_scorer)
                    
                    # 验证模型
                    if hasattr(self.bert_scorer, '_model') and self.bert_scorer._model is not None:
                        total_params = _total_params(self.bert_scorer._model)
                        if total_params > 0:
                            print(f"✅ BERTScorer initialized successfully using Method 1 (BERTScorer) with {model_type}")
                            return
                        else:
                            print(f"⚠️ Method 1 attempt {attempt + 1}: Model loaded but has no parameters")
                    else:
                        print(f"⚠️ Method 1 attempt {attempt + 1}: Model not loaded properly")
                        
                finally:
                    # 恢复默认 dtype（autocast 上下文会在 with 退出时自动恢复）
                    torch.set_default_dtype(original_dtype)
                
            except Exception as e:
                print(f"❌ Method 1 attempt {attempt + 1} failed: {e}")
                print(f"Exception type: {type(e).__name__}")
                if attempt < 2:  # 不是最后一次尝试
                    import time
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    import traceback
                    print(f"Method 1 final traceback: {traceback.format_exc()}")
            pass

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=64,
        )
        return torch.mean(f).item(), f.tolist()
        # # 使用 bert_score 库的底层函数来计算分数
        # from bert_score.utils import bert_cos_score_idf
        
        # with torch.no_grad():
        #     # 计算 BERTScore
        #     # 创建一个默认的 IDF 字典，为所有 token 提供默认值 1.0
        #     from collections import defaultdict
        #     idf_dict = defaultdict(lambda: 1.0)
            
        #     preds = bert_cos_score_idf(
        #         model=self.bert_scorer._model,
        #         refs=refs,
        #         hyps=hyps,
        #         tokenizer=self.bert_scorer._tokenizer,
        #         idf_dict=idf_dict,
        #         verbose=False,
        #         batch_size=self.bert_scorer.batch_size,
        #         device=self.bert_scorer.device,
        #         all_layers=False
        #     )
            
        #     if preds is None:
        #         raise ValueError("bert_cos_score_idf returned None")
            
        #     # preds 的形状是 [batch_size, 3]，其中最后一维是 [P, R, F1]
        #     P = preds[:, 0]
        #     R = preds[:, 1] 
        #     F1 = preds[:, 2]
            
        #     return torch.mean(F1).item(), F1.tolist()

# ----------------------------- F1CheXbert -----------------------------
#!/usr/bin/env python
"""CheXbert evaluation utilities – **device‑safe end‑to‑end**

This is a drop‑in replacement for your previous `f1chexbert.py` **and** for the helper
`SemanticEmbeddingScorer`.  All tensors – model weights *and* inputs – are created on
exactly the same device so the             ``Expected all tensors to be on the same device``
run‑time error disappears.  The public API stays identical, so the rest of your
pipeline does not need to change.
"""



import os
import warnings
import logging
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    BertModel,
    BertTokenizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.metrics._classification import _check_targets
from sklearn.utils.sparsefuncs import count_nonzero
from huggingface_hub import hf_hub_download
from appdirs import user_cache_dir

# -----------------------------------------------------------------------------
# GLOBALS & UTILITIES
# -----------------------------------------------------------------------------

CACHE_DIR = user_cache_dir("chexbert")
warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Helper ----------------------------------------------------------------------

def _generate_attention_masks(batch_ids: torch.LongTensor) -> torch.FloatTensor:
    """Create a padding mask: 1 for real tokens, 0 for pads."""
    # batch_ids shape: (B, L)
    lengths = (batch_ids != 0).sum(dim=1)  # (B,)
    max_len = batch_ids.size(1)
    idxs = torch.arange(max_len, device=batch_ids.device).unsqueeze(0)  # (1, L)
    return (idxs < lengths.unsqueeze(1)).float()  # (B, L)

# -----------------------------------------------------------------------------
# MODEL COMPONENTS
# -----------------------------------------------------------------------------

class BertLabeler(nn.Module):
    """BERT backbone + 14 small classification heads (CheXbert)."""

    def __init__(self, *, device: Union[str, torch.device]):
        super().__init__()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # 1) Backbone on *CPU* first – we'll move to correct device after weights load
        config = AutoConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(config)

        hidden = self.bert.config.hidden_size
        # 13 heads with 4‑way logits, + 1 head with 2‑way logits
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2))

        self.dropout = nn.Dropout(0.1)

        # 2) Load checkpoint weights directly onto CPU first -------------------
        ckpt_path = hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            cache_dir=CACHE_DIR,
        )
        state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=True)

        # 3) NOW move the entire module (recursively) to `self.device` ----------
        self.to(self.device)

        # freeze ---------------------------------------------------------------
        for p in self.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------------------
    # forward helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def cls_logits(self, input_ids: torch.LongTensor) -> List[torch.Tensor]:
        """Returns a list of logits for each head (no softmax)."""
        attn = _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return [head(cls_repr) for head in self.linear_heads]

    @torch.no_grad()
    def cls_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Returns pooled [CLS] representations (B, hidden_size)."""
        attn = _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        return outputs.last_hidden_state[:, 0]  # (B, hidden)

# -----------------------------------------------------------------------------
# F1‑CheXbert evaluator
# -----------------------------------------------------------------------------

class F1CheXbert(nn.Module):
    """Generate CheXbert labels + handy evaluation utilities."""

    CONDITION_NAMES = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    NO_FINDING = "No Finding"
    TARGET_NAMES = CONDITION_NAMES + [NO_FINDING]

    TOP5 = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    def __init__(
        self,
        *,
        refs_filename: str | None = None,
        hyps_filename: str | None = None,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        # Resolve device -------------------------------------------------------
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

        # HuggingFace tokenizer (always CPU, we just move tensors later) -------
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # backbone + heads ------------------------------------------------------
        self.model = BertLabeler(device=self.device).eval()

        # indices for the TOP‑5 label subset -----------------------------------
        self.top5_idx = [self.TARGET_NAMES.index(n) for n in self.TOP5]

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(self, reports: Sequence[str]) -> List[np.ndarray]:
        """Return list[np.ndarray] of pooled [CLS] vectors for each report."""
        # Tokenise *as a batch* for efficiency
        encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.to(self.device)
        # (B, hidden)
        cls = self.model.cls_embeddings(input_ids)
        return [v.cpu().numpy() for v in cls]

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        """Return 14‑dim binary vector for the given report."""
        input_ids = self.tokenizer(report, truncation=True, max_length=512, return_tensors="pt").input_ids.to(self.device)
        preds = [head.argmax(dim=1).item() for head in self.model.cls_logits(input_ids)]

        binary = []
        if mode == "rrg":
            for c in preds:
                binary.append(1 if c in {1, 3} else 0)
        elif mode == "classification":
            for c in preds:
                if c == 1:
                    binary.append(1)
                elif c == 2:
                    binary.append(0)
                elif c == 3:
                    binary.append(-1)
                else:
                    binary.append(0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return binary

    # ---------------------------------------------------------------------
    # Full evaluator – unchanged logic but simplified I/O
    # ---------------------------------------------------------------------

    def forward(self, hyps: List[str], refs: List[str]):
        """Return (accuracy, per‑example‑accuracy, full classification reports)."""
        # Reference labels -----------------------------------------------------
        if self.refs_filename and os.path.exists(self.refs_filename):
            with open(self.refs_filename) as f:
                refs_chexbert = [eval(line) for line in f]
        else:
            refs_chexbert = [self.get_label(r) for r in refs]
            if self.refs_filename:
                with open(self.refs_filename, "w") as f:
                    f.write("\n".join(map(str, refs_chexbert)))

        # Hypothesis labels ----------------------------------------------------
        hyps_chexbert = [self.get_label(h) for h in hyps]
        if self.hyps_filename:
            with open(self.hyps_filename, "w") as f:
                f.write("\n".join(map(str, hyps_chexbert)))

        # TOP‑5 subset arrays --------------------------------------------------
        refs5 = [np.array(r)[self.top5_idx] for r in refs_chexbert]
        hyps5 = [np.array(h)[self.top5_idx] for h in hyps_chexbert]

        # overall accuracy -----------------------------------------------------
        accuracy = accuracy_score(refs5, hyps5)
        _, y_true, y_pred = _check_targets(refs5, hyps5)
        pe_accuracy = (count_nonzero(y_true - y_pred, axis=1) == 0).astype(float)

        # full classification reports -----------------------------------------
        cr = classification_report(refs_chexbert, hyps_chexbert, target_names=self.TARGET_NAMES, output_dict=True)
        cr5 = classification_report(refs5, hyps5, target_names=self.TOP5, output_dict=True)

        return accuracy, pe_accuracy, cr, cr5

_GLOBAL_BERTSCORER_CACHE = {}

def clear_gpu_memory():
    """全局GPU内存清理函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 强制垃圾回收
        import gc
        gc.collect()

def radcliq_bertscore(refs, hyps, model):
    """
    Computes BERTScore for each pair of reference and hypothesis.

    Returns:
        np.ndarray of shape (N,) with the BERTScore F1 values per pair.
    """
    # https://github.com/rajpurkarlab/CXR-Report-Metric/blob/9c9ecad39be6cb2be8e75be1d1c50ef8888a3e40/CXRMetric/run_eval.py#L103
    # 初始化 BertScore
    scorer = model

    with torch.no_grad():
        _, scores = scorer(refs, hyps)
    
    # 显式释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.array([float(s) for s in scores])


def compute_f1(test_set, retrieved_set):
    """Helper to compute F1 between two sets of items."""
    tp = len(test_set & retrieved_set)
    fp = len(retrieved_set) - tp
    fn = len(test_set) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def extract_entities(output):
    """Extracts set of (tokens, label) tuples from RadGraph output."""
    return {(tuple(ent["tokens"]), ent["label"]) for ent in output.get("entities", {}).values()}


def extract_relations(output):
    """Extracts set of (src, tgt, relation) tuples from RadGraph output."""
    rels = set()
    entities = output.get("entities", {})
    for ent in entities.values():
        src = (tuple(ent["tokens"]), ent["label"])
        for rel_type, tgt_idx in ent.get("relations", []):
            tgt_ent = entities.get(tgt_idx)
            if tgt_ent:
                tgt = (tuple(tgt_ent["tokens"]), tgt_ent["label"])  
                rels.add((src, tgt, rel_type))
    return rels


def radcliq_radgraph_scores(refs, hyps, model):
    """
    Computes entity and relation F1 via RadGraph for each report pair and returns their average.

    Returns:
        np.ndarray of shape (N,) with (entity_f1 + relation_f1)/2 per pair.
    """
    rad = model
    with torch.no_grad():
        gt_outputs = rad(refs)
        pred_outputs = rad(hyps)
    
    scores = []
    for i in range(len(refs)):
        gt_out = gt_outputs.get(str(i), {})
        pred_out = pred_outputs.get(str(i), {})

        ents_gt = extract_entities(gt_out)
        ents_pred = extract_entities(pred_out)
        rels_gt = extract_relations(gt_out)
        rels_pred = extract_relations(pred_out)

        ent_f1 = compute_f1(ents_gt, ents_pred)
        rel_f1 = compute_f1(rels_gt, rels_pred)
        scores.append((ent_f1 + rel_f1) / 2)
    
    # 显式释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.array(scores)


def semantic_embedding_scores(refs, hyps, model):
    """
    Computes per-pair cosine similarity between embeddings from CheXbert labeler.

    Returns:
        np.ndarray of shape (N,) with cosine similarities per pair.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) must be same length")
    
    labeler = model

    # Batch processing to avoid OOM for long refs/hyps
    batch_size = 256
    gt_embs_list = []
    pred_embs_list = []
    
    for i in range(0, len(refs), batch_size):
        refs_batch = refs[i:min(i+batch_size, len(refs))]
        hyps_batch = hyps[i:min(i+batch_size, len(hyps))]
        
        with torch.no_grad():
            gt_embs = labeler.get_embeddings(refs_batch)
            pred_embs = labeler.get_embeddings(hyps_batch)
        
        # Ensure output is always 2D, even for last batch of size 1
        gt_embs_list.append(np.vstack(gt_embs) if len(gt_embs) > 1 else np.array(gt_embs))
        pred_embs_list.append(np.vstack(pred_embs) if len(pred_embs) > 1 else np.array(pred_embs))
        
        # 在每个批次后清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    gt_embs = np.vstack(gt_embs_list)
    pred_embs = np.vstack(pred_embs_list)
    
    # https://github.com/rajpurkarlab/CXR-Report-Metric/blob/9c9ecad39be6cb2be8e75be1d1c50ef8888a3e40/CXRMetric/run_eval.py#L126
    dot = np.einsum("nd,nd->n", gt_embs, pred_embs)
    norms = np.linalg.norm(gt_embs, axis=1) * np.linalg.norm(pred_embs, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(norms > 0, dot / norms, 0.0)
    
    # 最终清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return sims



def radcliq_scores(refs, hyps, bert_scorer, radgraph_scorer, chexbert_scorer):
    """
    Computes BERTScore, RadGraph score, and semantic embedding similarity for each ref-hyp pair.

    Args:
        refs: List of reference report strings.
        hyps: List of hypothesis report strings.
        device: Device for embedding model ('cpu' or 'cuda').
        bert_model: HuggingFace model name for BERTScore.
        radgraph_model: Model name for RadGraph inference.

    Returns:
        Dict with keys 'bertscore', 'radgraph', 'semantic', each mapping to a numpy array of shape (N,).
    """
    
    # BERTScore
    bert_scores = radcliq_bertscore(refs, hyps, model=bert_scorer)


    
    # RadGraph
    rad_scores = radcliq_radgraph_scores(refs, hyps, model=radgraph_scorer)
    
    # Semantic embeddings
    sem_scores = semantic_embedding_scores(refs, hyps, model=chexbert_scorer)

    # BLEU
    # bleu_scorer = Bleu()
    # bleu_scores = bleu_scorer(refs, hyps)[1]
    # bert_scores = 0.0
    # rad_scores = 0.0
    # sem_scores = 0.0
    bleu_scores = 0.0

    # 最终清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'bertscore': bert_scores,
        'radgraph': rad_scores,
        'semb_score': sem_scores,
        'bleu_score': bleu_scores
    }



class CompositeMetric:
    def __init__(self):
        scaler = StandardScaler(with_mean=True, with_std=True)
        # learnt parameters, infered from 
        # https://github.com/rajpurkarlab/CXR-Report-Metric/blob/main/CXRMetric/run_eval.py#L219
        scaler.mean_            = np.array([0.53792312, 0.61757256, 0.76479421, 0.44738335])
        scaler.scale_           = np.array([0.30282584, 0.22430938, 0.25394391, 0.29892717])
        scaler.var_             = np.array([0.09170349, 0.05031470, 0.06448751, 0.08935745])
        scaler.n_samples_seen_  = 160       # integer
        scaler.n_features_in_   = 4         # integer

        self.scaler = scaler
        self.coefs  = np.array([
                        -3.77083683e-01,   # radgraph weight
                        -3.70300100e-01,   # bertscore weight
                        -2.52616218e-01,   # s-emb weight
                        4.31504841e-12,   # bleu weight
                        2.46655256e-10    # intercept / bias
                    ])
        self.cols   = ["radgraph", "bertscore", "semb_score", "bleu_score"]

        # from transformers.modeling_utils import set_zero3_state
        # with set_zero3_state():  ## 参数load 不正确，有些parameters 是 0, 可能是有些gpu 上面的parameters 没有load 下来
        #########################################################
        # import deepspeed
        # ds_config = "swift/llm/ds_config/zero3.json"
        # with deepspeed.zero.Init(config_dict_or_path=ds_config):  ## 这个不行

        self.bert_scorer = BertScore(
            model_type='distilbert-base-uncased',
            rescale_with_baseline=True,
            idf=False,
            num_layers=None,
            local_files_only=None,
            cache_dir=None,
        )
        self.radgraph_scorer = RadGraph(model_type='radgraph')
        self.chexbert_scorer = F1CheXbert(device='cuda')
        self.rouge1_scorer = Rouge1()


        # Set requires_grad=False for all parameters in the three models
        for param in self.bert_scorer.bert_scorer._model.parameters():
            param.requires_grad = False
        for param in self.radgraph_scorer.model.parameters():
            param.requires_grad = False
        for param in self.chexbert_scorer.model.parameters():
            param.requires_grad = False
        # import ipdb; ipdb.set_trace()

        print('bert_scorer', self.bert_scorer.bert_scorer._model.embeddings.word_embeddings.weight)
        print('radgraph_scorer', self.radgraph_scorer.model._endpoint_span_extractor._span_width_embedding.weight) ## cuda 0
        print('chexbert_scorer', self.chexbert_scorer.model.bert.embeddings.word_embeddings.weight) ## cuda 0
        # print(self.chexbert_scorer.model.bert.embeddings.word_embeddings.weight) ## cuda 0

    def clear_cache(self):
        """显式清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            self.clear_cache()
        except:
            pass


    def predict_rouge1(self, refs, hyps):
        return self.rouge1_scorer(refs, hyps)


    def _build_matrix(self, metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Stack features in the canonical column order."""
        return np.column_stack([metrics[c] for c in self.cols])

    def predict(self, refs, hyps) -> np.ndarray:
        """
        Args
        ----
        metrics : dict returned by `radcliq_scores`

        Returns
        -------
        np.ndarray of shape (N,) – RadCliQ-v1 score for each ref/hyp pair.
        """

        metrics = radcliq_scores(refs, hyps, self.bert_scorer, self.radgraph_scorer, self.chexbert_scorer)
        # print('get the metrics')

        X = self._build_matrix(metrics)
        # print('get the X')
        Xn = self.scaler.transform(X)
        # print('get the Xn')
        # Append bias term
        Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        scores = Xn @ self.coefs
        # print('get the scores')
        mean_score = scores.mean()
        # 对 mean_score 进行 1- (2 * (sigmoid(x) - 0.5)) 处理后输出
        processed_score = 1 - (2 * (1 / (1 + np.exp(-mean_score)) - 0.5))
        return processed_score, scores
        # 在返回前清理显存
        # self.clear_cache()
        # large_better_score = 1/scores.mean()
        # return 1/scores.mean(), scores
