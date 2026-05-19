from typing import Any, Literal

from tango.common import Lazy
from tango.integrations.transformers import Tokenizer

from microbert2.microbert.tasks.mlm import MLMTask
from microbert2.microbert.tasks.task import MicroBERTTask


@MicroBERTTask.register("microbert2.microbert.tasks.tlm.TLMTask")
class TLMTask(MLMTask):
    def __init__(
        self,
        dataset: dict[Literal["train", "dev", "test"], list[dict[str, Any]]],
        tokenizer: Lazy[Tokenizer],
        proportion: float = 0.2,
        mlm_probability: float = 0.15,
        mlm_mask_replace_prob: float = 1.0,
        mlm_random_replace_prob: float = 0.0,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            mlm_mask_replace_prob=mlm_mask_replace_prob,
            mlm_random_replace_prob=mlm_random_replace_prob,
        )
        self._proportion = proportion

    @property
    def slug(self) -> str:
        return "tlm"

    @property
    def universal(self) -> bool:
        return False

    @property
    def inst_proportion(self) -> float:
        return self._proportion
