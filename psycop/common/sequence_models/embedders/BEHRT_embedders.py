"""
Rewrite to dict[str, vector] instead of list[dict[str, value]]
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from psycop.common.data_structures import TemporalEvent
from psycop.common.data_structures.patient import PatientSlice
from psycop.common.sequence_models.dataset import PatientSliceDataset

from ..registry import Registry
from .interface import EmbeddedSequence, PatientSliceEmbedder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BEHRTVocab:
    age: dict[int | str, int]  # str because must allow keys "UNK" and "PAD"
    diagnosis: dict[str, int]
    is_padding: dict[str, int] = field(default_factory=lambda: {"PAD": 1})
    segment: dict[str, int] = field(default_factory=lambda: {"PAD": 0})
    position: dict[str, int] = field(default_factory=lambda: {"PAD": 0})


class BEHRTEmbedder(nn.Module, PatientSliceEmbedder):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float,
        max_sequence_length: int,
        map_diagnosis_codes: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        self.map_diagnosis_codes = map_diagnosis_codes
        self.is_fitted: bool = False

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        self.icd2caliber = self.load_icd_to_caliber_mapping()

    @staticmethod
    def load_icd_to_caliber_mapping() -> dict[str, str]:
        with open(  # noqa: PTH123
            "psycop/common/sequence_models/embedders/diagnosis_code_mapping.json",
        ) as fp:
            mapping = json.load(fp)

        return mapping

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
            max_position_embeddings,
            self.d_model,
        ).from_pretrained(
            embeddings=self._init_position_embeddings(
                max_position_embeddings,
                self.d_model,
            ),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> EmbeddedSequence:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

        diagnosis_ids = inputs["diagnosis"]
        age_ids = inputs["age"]
        segment_ids = inputs["segment"]
        position_ids = inputs["position"]

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

        embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return EmbeddedSequence(embeddings, inputs["is_padding"] == 1)

    def _init_position_embeddings(
        self,
        max_position_embeddings: int,
        d_model: int,
    ) -> torch.Tensor:
        def even_code(pos, idx):  # type: ignore # noqa: ANN001, ANN202
            return np.sin(pos / (10000 ** (2 * idx / d_model)))

        def odd_code(pos, idx):  # type: ignore # noqa: ANN001, ANN202
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

    def collate_patient_slices(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> dict[str, torch.Tensor]:
        """
        Handles padding and indexing by converting each to an index tensor

        E.g. Age -> Age bin (index)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before use")

        patient_sequences_ids = [self.collate_patient_slice(p) for p in patient_slices]
        # padding
        padded_sequences_ids = self.pad_sequences(patient_sequences_ids)

        return padded_sequences_ids

    def add_position_and_segment(self, events: list[dict]) -> list[dict]:  # type: ignore
        # add position and segment
        for i, e_input in enumerate(events):
            e_input["position"] = torch.tensor(i)
            is_even = i % 2 == 0
            e_input["segment"] = torch.tensor(int(is_even))
        return events

    def pad_sequences(
        self,
        sequences: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_seq_len = max([len(p["age"]) for p in sequences])
        assert max_seq_len <= self.max_sequence_length

        keys = sequences[0].keys()
        padded_sequences: dict[str, torch.Tensor] = {}

        for key in keys:
            key_to_subvocab = {
                "age": self.vocab.age,
                "diagnosis": self.vocab.diagnosis,
                "segment": self.vocab.segment,
                "position": self.vocab.position,
                "is_padding": self.vocab.is_padding,
            }

            if key not in key_to_subvocab:
                raise ValueError(f"Key {key} not in {key_to_subvocab.keys()}")

            vocab = key_to_subvocab[key]
            pad_idx = vocab["PAD"]

            padded_sequences[key] = pad_sequence(
                [p[key] for p in sequences],
                batch_first=True,
                padding_value=pad_idx,
            )

        return padded_sequences

    def A_diagnoses_only(self, events: Sequence[TemporalEvent]) -> list[TemporalEvent]:
        filtered_events = []
        for event in events:
            if event.source_type == "diagnosis" and event.source_subtype == "A":
                filtered_events.append(event)
        return filtered_events

    def map_icd10_to_caliber(
        self,
        diagnosis_code: str,
    ) -> str | None:
        """Map diagnoses-codes to Caliber.

        To handle different levels of granularity, map to the most specific Caliber code available. E.g. for code E12, if both E1 and E12 exist in Caliber, map to E12. If only E1 exists, map to E1.


        Caliber codes are documented here: https://www.sciencedirect.com/science/article/pii/S2589750019300123?via%3Dihub
        """
        while len(diagnosis_code) > 2:  # only attempt codes with at least 3 characters
            if diagnosis_code in self.icd2caliber:
                return self.icd2caliber[diagnosis_code]
            diagnosis_code = diagnosis_code[:-1]
        return None

    def A_diagnoses_to_caliber(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> list[PatientSlice]:
        """
        1. Filter to only keep A-diagnoses.
        2. Map to Caliber codes
        3. Drop any PatientSlice without any events after filtering and mapping
        """
        assert all(
            e.source_type == "diagnosis"
            for ps in patient_slices
            for e in ps.temporal_events
        ), "PatientSlice.temporal_events must only include diagnosis codes"

        patient_events = [
            PatientSlice(p.patient, self.A_diagnoses_only(p.temporal_events))
            for p in tqdm(patient_slices)
        ]
        if not self.map_diagnosis_codes:
            return patient_events

        # map diagnosis codes
        _patient_slices = []
        n_filtered = 0
        for p in tqdm(patient_events):
            temporal_events = []
            for e in p.temporal_events:
                value = self.map_icd10_to_caliber(e.value)  # type: ignore
                if value is not None:
                    event = TemporalEvent(
                        timestamp=e.timestamp,
                        source_type=e.source_type,
                        source_subtype=e.source_subtype,
                        value=value,
                    )
                    temporal_events.append(event)

            if temporal_events:
                _patient_slices.append(PatientSlice(p.patient, temporal_events))
            else:
                n_filtered += 1

        log.warning(
            f"Lost {n_filtered} patients ({round(n_filtered / len(patient_slices) * 100, 2)}%) after filtering and mapping diagnosis codes.",
        )

        return _patient_slices

    def add_cls_token_to_sequence(self, events: list[dict]) -> list[dict]:  # type: ignore
        # add cls token to start of sequence
        cls_token = {
            "age": torch.tensor(self.vocab.age["CLS"]),
            "diagnosis": torch.tensor(self.vocab.diagnosis["CLS"]),
            "is_padding": torch.tensor(0),
        }
        return [cls_token, *events]

    def collate_patient_slice(
        self,
        patient_slice: PatientSlice,
    ) -> dict[str, torch.Tensor]:
        events = patient_slice.temporal_events
        events = self.A_diagnoses_only(events)
        event_inputs = [self.collate_event(event, patient_slice) for event in events]

        event_inputs = self.add_cls_token_to_sequence(event_inputs)

        # reduce to max sequence length
        # take the first max_sequence_length events (probably better to use the last max_sequence_length events)
        # but this is the same as the original implementation
        event_inputs = event_inputs[: self.max_sequence_length]

        event_inputs = self.add_position_and_segment(event_inputs)

        # convert to tensor
        output: dict[str, torch.Tensor] = {}
        for key in event_inputs[0]:
            output[key] = torch.stack([e[key] for e in event_inputs])

        return output

    def get_patient_age(self, event: TemporalEvent, date_of_birth: datetime) -> int:
        age = event.timestamp - date_of_birth
        return int(age.days // 365.25)

    def collate_event(
        self,
        event: TemporalEvent,
        patient_slice: PatientSlice,
    ) -> dict[str, torch.Tensor]:
        age = self.get_patient_age(event, patient_slice.patient.date_of_birth)
        diagnosis: str = event.value  # type: ignore

        age2idx = self.vocab.age
        diagnosis2idx = self.vocab.diagnosis

        age_idx: int = age2idx.get(age, age2idx["UNK"])
        diagnosis_idx: int = diagnosis2idx.get(diagnosis, diagnosis2idx["UNK"])

        return {
            "age": torch.tensor(age_idx),
            "diagnosis": torch.tensor(diagnosis_idx),
            "is_padding": torch.tensor(0),
        }

    def fit(
        self,
        patient_slices: Sequence[PatientSlice],
        add_mask_token: bool = True,
    ):
        patient_slices = self.A_diagnoses_to_caliber(patient_slices)

        diagnosis_codes: list[str] = [
            str(e.value) for p in patient_slices for e in p.temporal_events
        ]

        # create dianosis2idx mapping
        diagnosis2idx = {
            d: i
            for i, d in enumerate(set(diagnosis_codes))
            if d is not None  # type: ignore
        }
        diagnosis2idx["UNK"] = len(diagnosis2idx)
        diagnosis2idx["PAD"] = len(diagnosis2idx)
        diagnosis2idx["CLS"] = len(diagnosis2idx)
        if add_mask_token:
            diagnosis2idx["MASK"] = len(diagnosis2idx)

        self.mask_token_id = diagnosis2idx["MASK"]

        ages: list[int] = [
            self.get_patient_age(e, ps.patient.date_of_birth)
            for ps in patient_slices
            for e in ps.temporal_events
        ]

        # create age2idx mapping
        age2idx: dict[str | int, int] = {a: i for i, a in enumerate(set(ages))}
        age2idx["UNK"] = len(age2idx)
        age2idx["PAD"] = len(age2idx)
        age2idx["CLS"] = len(age2idx)

        self.vocab = BEHRTVocab(age=age2idx, diagnosis=diagnosis2idx)

        n_diagnosis_codes = len(diagnosis2idx)
        n_age_bins = len(age2idx)

        self.initialize_embeddings_layers(
            n_diagnosis_codes=n_diagnosis_codes,
            n_age_bins=n_age_bins,
            max_position_embeddings=self.max_sequence_length,
        )

        self.is_fitted = True


@Registry.embedders.register("behrt_embedder")
def create_behrt_embedder(
    d_model: int,
    dropout_prob: float,
    max_sequence_length: int,
    patient_slices: Sequence[PatientSlice] | PatientSliceDataset,
) -> BEHRTEmbedder:
    embedder = BEHRTEmbedder(
        d_model=d_model,
        dropout_prob=dropout_prob,
        max_sequence_length=max_sequence_length,
    )

    if isinstance(patient_slices, PatientSliceDataset):
        patient_slices = patient_slices.patient_slices

    log.info("Fitting Embedding Module")
    embedder.fit(patient_slices=patient_slices)
    return embedder
