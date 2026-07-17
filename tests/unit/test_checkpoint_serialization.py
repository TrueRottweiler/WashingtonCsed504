import torch

from cv_search.checkpoints import load_checkpoint, save_checkpoint
from cv_search.storage import StudyStorage
from cv_search.types import TrialResult


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, None, None, {"epoch": 2})
    assert load_checkpoint(path, model, optimizer)["epoch"] == 2


def test_result_serialization(tmp_path):
    storage = StudyStorage(tmp_path / "study")
    result = TrialResult(
        study_name="s",
        adapter="x",
        profile="simple",
        stage="proxy",
        rung=None,
        trial_id="1",
        status="completed",
        architecture_id="a",
        config={"x": 1},
        budget={"kind": "steps", "value": 1},
        seed=1,
        device="cpu",
        validation_accuracy=0.5,
    )
    storage.append(result)
    loaded = storage.load()
    assert loaded[0].config == {"x": 1}
