import numpy as np
from torch import nn
import torch


class BEHRTEmbedder:
    def __init__(
        self,
        d_model: int,
        n_diagnosis_codes: int,
        n_age_bins: int,
        n_segments: int,
        max_position_embeddings: int,
        dropout_prob: float,
    ):
        self.d_model = d_model
        self.diagnosis_embeddings = nn.Embedding(n_diagnosis_codes, d_model)
        self.age_embeddings = nn.Embedding(n_age_bins, d_model)
        self.segment_embeddings = nn.Embedding(n_segments, d_model)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, d_model
        ).from_pretrained(
            embeddings=self._init_position_embeddings(max_position_embeddings, d_model)
        )

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        diagnosis_ids: int,
        age_ids: int | None,
        segment_ids: int | None,
        position_ids: int | None,
        age: bool = True,
    ):
        if segment_ids is None:
            segment_ids = torch.zeros_like(diagnosis_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(diagnosis_ids)
        if position_ids is None:
            position_ids = torch.zeros_like(diagnosis_ids)

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

    def collate_fn(self):
        raise NotImplementedError

    def fit(self, patients: list):  # type: ignore
        """
        Is not dependent on patient data.
        """
        return
