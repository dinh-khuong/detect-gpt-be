import random
import torch as tch
from torch.distributions import Categorical
import torch.nn.functional as F
import string
from transformers import (
    PreTrainedConfig,
    RobertaModel,
    RobertaTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
)
import numpy as np
from textblob import TextBlob
import textstat
import joblib
import math

def pertubating_text_logits(logits: tch.Tensor, target_ids: tch.Tensor, prob_temp: float = 0.8, sampling_temp: float = 1.2) -> tuple[tch.Tensor, tch.Tensor]:
    perterbated_probs = F.softmax(logits / sampling_temp, dim=-1)
    org_probs = F.log_softmax(logits / prob_temp, dim=-1)

    org_token_probs = tch.gather(org_probs, 2, target_ids).squeeze(-1)
    dist = Categorical(probs=perterbated_probs[0])
    n_sample = 5000
    perterbated_samples = dist.sample((n_sample, )).unsqueeze(-1)

    perterbated_token_probs = tch.gather(org_probs.expand(n_sample, -1, -1), 2, perterbated_samples).squeeze(-1)

    # ep = 1e-10
    # [1, probs], [n_sample, probs]
    return org_token_probs, perterbated_token_probs.mean(dim=-1)

class FastGptDetect:
    def __init__(self) -> None:
        # model_name = "roberta-large-mnli"
        model_name = "roberta-base"
        print("load model: " + model_name)
        self.configuration: PreTrainedConfig = AutoConfig.from_pretrained(
            model_name
        )
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            model_name, config=self.configuration
        )
        self.model: RobertaModel = AutoModelForMaskedLM.from_pretrained(
            model_name, config=self.configuration
        )

    def _samples_probs(self, inputs, sample: float) -> tuple[tch.Tensor, tch.Tensor]:
        # perterbating text
        inputs_input_ids: tch.Tensor = inputs['input_ids']
        masked_inputs = inputs_input_ids.clone()
        n = math.ceil(inputs_input_ids.shape[-1] * sample)

        masked_indexes = random.sample(range(inputs_input_ids.shape[-1]), n)
        masked_indexes.sort()
        for i in masked_indexes:
            masked_inputs[0][i] = self.tokenizer.mask_token_id

        with tch.no_grad():
            masked_logits = self.model(input_ids=masked_inputs, attention_mask=inputs['attention_mask'], use_cache=False).logits[:, masked_indexes, :]

        target_ids = inputs_input_ids[:, masked_indexes]

        return masked_logits, target_ids


    def get_text_logits(self, input: str, sample: float) -> tuple[tch.Tensor, tch.Tensor]:
        logits = []
        target_ids = []
        inputs = self.tokenizer(
            input,
            # max_length=configuration.max_position_embeddings - 2,
            return_tensors="pt"
        )
        max_position_token = self.configuration.max_position_embeddings - 2
        idx = 0
        while idx < inputs.input_ids.shape[1]:
            inputs_current = {
                'input_ids': inputs.input_ids[:, idx:idx+max_position_token],
                'attention_mask': inputs.attention_mask[:, idx:idx+max_position_token],
            }
            masked_logist, masked_target_ids = self._samples_probs(inputs_current, sample)
            logits.append(masked_logist)
            target_ids.append(masked_target_ids)

            idx += inputs_current['input_ids'].shape[1]

        logits = tch.cat(logits, dim=1)
        target_ids = tch.cat(target_ids, dim=-1)

        # [change_seq, logit], [1, change_seq]
        return logits, target_ids

    def sample_perturbate_text(self, input: str, sample: float, temps: list[tuple[float, float]]):
        org_probs = []
        perturbated_probs = []

        logits, target_ids = self.get_text_logits(input, sample)
        target_ids = target_ids.unsqueeze(-1)

        for prob_temp, sampling_temp in temps:
            org_token_probs, perturbated_token_probs = pertubating_text_logits(logits, target_ids, prob_temp=prob_temp, sampling_temp=sampling_temp)
            org_probs.append(org_token_probs)
            perturbated_probs.append(perturbated_token_probs)

        # [feat, probs], [feat, n_sample, probs]
        # [feat, probs], [feat, probs]
        return list(zip(org_probs, perturbated_probs))


def get_sampling_discrepancy(org_x: tch.Tensor, perturbation_x: tch.Tensor):
    r"""
        org_x: [seq]
        perturbation_x: [n_samples, seq]
    """
    # ep = 1e-10
    org_x_log = org_x
    perturbation_x_log = perturbation_x

    multitude = perturbation_x_log.mean()
    org_mul = org_x_log.mean()
    # perplexity = (-org_mul).exp()

    perturbation_std = perturbation_x_log.std()
    # score = (org_x_log - multitude).mean() / perturbation_std
    score = (org_mul - multitude) / perturbation_std
    # score = (org_x_log - multitude).mean()
    return score, perturbation_std


def calculate_ttr(text_tokens):
    """Calculates the Type-Token Ratio (TTR) for a given list of tokens."""
    word_count = len(text_tokens)
    unique_word_count = len(set(text_tokens))
    if word_count == 0:
        return 0
    return unique_word_count / word_count


def calc_punctuation(input_string, punctuation=string.punctuation):
    count = sum(1 for char in input_string if char in punctuation)
    return count


class GPTChecker:
    def __init__(self):
        self.fast_gpt = FastGptDetect()
        self.model = joblib.load("models/model_gradient.pkl")

    def predict(self, text: str):
        # org_log, per_log = self.fast_gpt.log_perterbate(text, 0.2, [(0.8, 20.0)])[0]
        # fast_gpt = get_sampling_discrepancy(org_log, per_log)

        tokens = self.fast_gpt.tokenizer(text)
        ttr = calculate_ttr(tokens)
        punct = calc_punctuation(text, string.punctuation) / len(text)
        text_blob = TextBlob(text)
        polarity = text_blob.polarity
        subjectivity = text_blob.subjectivity
        flesch = textstat.flesch_reading_ease(text)
        gunning_fog = textstat.gunning_fog(text)

        base_features = [
            ttr,
            punct,
            polarity,
            subjectivity,
            flesch,
            gunning_fog,
        ]

        logs_1 = self.fast_gpt.sample_perturbate_text(text, 0.2, [
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
            (0.8, 1.0), (0.8, 1.2), (1.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.8, 30),
        ])
        
        fast_gpts_1 = [get_sampling_discrepancy(log[0], log[1]) for log in logs_1]
        
        # 1. DO NOT use .mean(axis=0). Keep it as a 14x12 matrix.
        np_pertur_dist_1 = np.concatenate(np.array(fast_gpts_1)).reshape(14, 12)

        # 2. Build a batch matrix of 14 instances
        # For each of the 14 shapes, we append the 6 base features to its 12 perturbation values
        X_batch = []
        for row in np_pertur_dist_1:
            # Combined feature vector length: 6 + 12 = 18 elements
            combined_features = base_features + row.tolist()
            X_batch.append(combined_features)
            
        X_batch = np.array(X_batch) # Shape is now exactly (14, 18)
        # 'ttr', 'punct', 'polarity', 'subjectivity', 'flesch', 'gunning_fog'
        predictions = np.mean(self.model.predict_proba(X_batch)[:, 1])

        #'fast_gpt', 'ttr', 'puct', 'polarity', 'subjectivity', 'flesch', 'gunning_fog'
        return {"fast_gpt": predictions * 100, "polarity": polarity, "subjectivity": subjectivity}
        # return self.model.predict(np.array([[fast_gpt, ttr, puct, polarity, subjectivity, gunning_fog]]))

