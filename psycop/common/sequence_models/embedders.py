"""
Rewrite to dict[str, vector] instead of list[dict[str, value]]
"""

from copy import copy

import numpy as np
import torch
from torch import nn


class BEHRTEmbedder:
    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        max_sequence_length: int,
    ):
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        self.is_fitted: bool = False

        self.diagnosis_embeddings = None
        self.age_embeddings = None
        self.segment_embeddings = None
        self.position_embeddings = None

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def initialize_embeddings_layers(
        self,
        n_diagnosis_codes: int,
        n_age_bins: int,
        max_position_embeddings: int,
    ) -> None:
        self.n_diagnosis_codes = n_diagnosis_codes
        self.n_age_bins = n_age_bins
        self.n_segments = 2 + 1  # +1 for padding
        self.max_position_embeddings = max_position_embeddings

        self.diagnosis_embeddings = nn.Embedding(n_diagnosis_codes, self.d_model)
        self.age_embeddings = nn.Embedding(n_age_bins, self.d_model)
        self.segment_embeddings = nn.Embedding(self.n_segments, self.d_model)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, self.d_model
        ).from_pretrained(
            embeddings=self._init_position_embeddings(
                max_position_embeddings, self.d_model
            )
        )

    def forward(
        self,
        diagnosis_ids: torch.Tensor,
        age_ids: torch.Tensor | None,
        segment_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        age: bool = True,
    ):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

        if segment_ids is None:
            segment_ids = torch.zeros_like(diagnosis_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(diagnosis_ids)
        if position_ids is None:
            position_ids = torch.zeros_like(diagnosis_ids)

        assert (
            len(diagnosis_ids.shape) == 2
        ), "diagnosis_ids must be (batch / patients, sequence length)"
        assert (
            len(age_ids.shape) == 2
        ), "age_ids must be (batch / patients, sequence length)"
        assert (
            len(segment_ids.shape) == 2
        ), "segment_ids must be (batch / patients, sequence length)"
        assert (
            len(position_ids.shape) == 2
        ), "position_ids must be (batch / patients, sequence length)"

        word_embed = self.diagnosis_embeddings(diagnosis_ids)
        segment_embed = self.segment_embeddings(segment_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.position_embeddings(position_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_position_embeddings(self, max_position_embeddings: int, d_model: int):
        def even_code(pos, idx):  # type: ignore
            return np.sin(pos / (10000 ** (2 * idx / d_model)))

        def odd_code(pos, idx):  # type: ignore
            return np.cos(pos / (10000 ** (2 * idx / d_model)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embeddings, d_model), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embeddings):
            for idx in np.arange(0, d_model, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embeddings):
            for idx in np.arange(1, d_model, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

    def collate_fn(self, patients: list[Patient]):
        """
        Handles padding and indexing by converting each to an index tensor

        E.g. Age -> Age bin (index)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

        patient_sequences = [self.collate_patient(p) for p in patients]
        # padding
        max_seq_len = max([len(p) for p in patient_sequences])
        assert max_seq_len <= self.max_sequence_length
        padded_sequences = [
            self.pad_sequence(seq, max_seq_len) for seq in patient_sequences
        ]

        assert all(len(seq) == max_seq_len for seq in padded_sequences)

        return padded_sequences

    def add_position_and_segment(self, events: list[dict]) -> list[dict]:
        # add position and segment
        for i, e_input in enumerate(events):
            e_input["position"] = torch.tensor(i)
            is_even = i % 2 == 0
            e_input["segment"] = torch.tensor(is_even)
        return events

    def pad_sequence(
        self, sequence: list[dict[str, torch.Tensor]], max_seq_len: int
    ) -> list[dict[str, torch.Tensor]]:
        padding_event = {
            "age": torch.tensor(self.age2idx["PAD"]),
            "diagnosis": torch.tensor(self.diagnosis2idx["PAD"]),
            "is_padding": torch.tensor(1),
        }
        while len(sequence) < max_seq_len:
            _padding_event = copy(padding_event)
            sequence.append(_padding_event)

    def filter_events(self, events: list[Event]) -> list[Event]:
        filtered_events = []
        for event in events:
            if event.type == "diagnosis":
                filtered_events.append(event)
        return filtered_events

    def collate_patient(self, patient: Patient) -> list[dict[str, torch.Tensor]]:
        events = patient.events
        events = self.filter_events(events)
        event_inputs = [self.collate_event(event) for event in events]

        # reduce to max sequence length
        return event_inputs[
            : self.max_sequence_length
        ]  # take the first max_sequence_length events (probably better to use the last max_sequence_length events)

    def get_patient_age(self, event: Event) -> int:
        raise NotImplementedError

    def collate_event(self, event: Event) -> dict[str, torch.Tensor]:
        age = self.get_patient_age(event)
        age_idx = self.age2idx.get(age, self.age2idx["UNK"])
        diagnosis_idx = self.diagnosis2idx.get(
            event.diagnosis, self.diagnosis2idx["UNK"]
        )

        return {
            "age": torch.tensor(age_idx),
            "diagnosis": torch.tensor(diagnosis_idx),
            "is_padding": torch.tensor(0),
        }

    def fit(self, patients: list):  # type: ignore
        """
        Is not dependent on patient data.
        """
        patient_events: list[list[Event]] = [
            self.filter_events(p.events) for p in patients
        ]
        events: list[Event] = [e for p in patient_events for e in p]
        diagnosis_codes: list[str] = [e.diagnosis for e in events]
        n_diagnosis_codes: int = len(set(diagnosis_codes)) + 2  # UNK + Padding

        # create dianosis2idx mapping
        self.diagnosis2idx = {d: i for i, d in enumerate(set(diagnosis_codes))}
        self.diagnosis2idx["UNK"] = len(self.diagnosis2idx)
        self.diagnosis2idx["PAD"] = len(self.diagnosis2idx)

        ages: list[int] = [self.get_patient_age(e) for e in events]
        n_age_bins = len(set(ages)) + 2  # UNK + PAD

        # create age2idx mapping
        self.age2idx = {a: i for i, a in enumerate(set(ages))}
        self.age2idx["UNK"] = len(self.age2idx)
        self.age2idx["PAD"] = len(self.age2idx)

        max_position_embeddings = max([len(p.events) for p in patients])
        max_position_embeddings = max(max_position_embeddings, self.max_sequence_length)

        self.initialize_embeddings_layers(
            n_diagnosis_codes=n_diagnosis_codes,
            n_age_bins=n_age_bins,
            max_position_embeddings=max_position_embeddings,
        )

        self.is_fitted = True
