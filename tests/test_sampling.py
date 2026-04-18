import csv
from pathlib import Path

from natural_reward_seeking.data.wildjailbreak import load_wildjailbreak_train_rows, sample_rows_by_category


def test_balanced_sampling_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "wildjailbreak"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    path = train_dir / "train.tsv"
    fieldnames = ["vanilla", "adversarial", "completion", "data_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for idx in range(3):
            writer.writerow({"vanilla": f"vh {idx}", "adversarial": "", "completion": "", "data_type": "vanilla_harmful"})
            writer.writerow({"vanilla": f"vb {idx}", "adversarial": "", "completion": "", "data_type": "vanilla_benign"})
            writer.writerow({"vanilla": "", "adversarial": f"ah {idx}", "completion": "", "data_type": "adversarial_harmful"})
            writer.writerow({"vanilla": "", "adversarial": f"ab {idx}", "completion": "", "data_type": "adversarial_benign"})

    rows = load_wildjailbreak_train_rows(root)
    sample_a, manifest_a = sample_rows_by_category(rows, sample_per_category=2, seed=7)
    sample_b, manifest_b = sample_rows_by_category(rows, sample_per_category=2, seed=7)
    assert [row["prompt_id"] for row in sample_a] == [row["prompt_id"] for row in sample_b]
    assert manifest_a["sampled_count"] == 8
    assert manifest_a["manifest_rows"] == manifest_b["manifest_rows"]

