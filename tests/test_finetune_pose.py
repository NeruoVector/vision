"""Dataset conversion and held-out split contracts; no synthetic optimizer evidence."""

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "finetune_pose.py"))
convert_person = MODULE["convert_person"]
select_ids = MODULE["select_ids"]
AnnotationError = MODULE["AnnotationError"]


def person(**changes):
    result = {
        "id": 7,
        "image_id": 1,
        "category_id": 1,
        "iscrowd": 0,
        "num_keypoints": 2,
        "bbox": [10, 20, 50, 100],
        "keypoints": [25, 50, 2, 40, 70, 1] + [0, 0, 0] * 15,
    }
    result.update(changes)
    return result


def test_coco17_pose_coordinates_class_and_visibility():
    row = list(map(float, convert_person(person(), 100, 200).split()))
    assert len(row) == 56
    assert row[:5] == pytest.approx([0, 0.35, 0.35, 0.5, 0.5])
    assert row[5:11] == pytest.approx([0.25, 0.25, 2, 0.4, 0.35, 1])
    assert row[11:] == [0] * 45


def test_crowds_nonpersons_and_zero_joint_people_are_excluded():
    assert convert_person(person(iscrowd=1), 100, 200) is None
    assert convert_person(person(category_id=2), 100, 200) is None
    assert convert_person(person(num_keypoints=0, keypoints=[99, 99, 0] * 17), 100, 200) is None


def test_box_clipping_preserves_valid_keypoints():
    row = list(
        map(
            float,
            convert_person(
                person(bbox=[-10, -20, 120, 240], keypoints=[100, 0, 2] * 17), 100, 200
            ).split(),
        )
    )
    assert row[1:5] == [0.5, 0.5, 1, 1]
    assert row[5:8] == [1, 0, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"keypoints": [0, 0, 0]},
        {"keypoints": [0, 0, 3] * 17},
        {"bbox": [10, 20, -1, 5]},
        {"bbox": [101, 201, 10, 20]},
        {"keypoints": [float("nan"), 0, 2] * 17},
        {"keypoints": [150, -10, 2] * 17},
    ],
)
def test_malformed_annotations_fail_instead_of_becoming_negatives(changes):
    with pytest.raises(AnnotationError):
        convert_person(person(**changes), 100, 200)


def test_split_is_deterministic_image_disjoint_and_order_independent():
    source = list(range(100))
    selection = select_ids(source, 40)
    assert select_ids(source[::-1], 40) == selection
    assert select_ids(source, 40, seed=99) != selection
    validation, heldout = selection[:20], selection[20:]
    assert not set(validation) & set(heldout)
    assert len(set(validation + heldout)) == 40


def test_selection_does_not_silently_repeat_or_truncate_images():
    with pytest.raises(AnnotationError):
        select_ids([1, 1, 2], 3)
    with pytest.raises(AnnotationError):
        select_ids([1], 0)


def test_coco_horizontal_flip_mapping_preserves_joint_semantics():
    flip = MODULE["FLIP"]
    assert sorted(flip) == list(range(17))
    assert [flip[i] for i in flip] == list(range(17))
    assert flip[5] == 6 and flip[15] == 16


def test_prepare_real_zip_conversion_and_content_disjointness(tmp_path, monkeypatch):
    import argparse
    import json
    import zipfile

    from PIL import Image

    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as compressed:
        for official, ids in (("train2017", [1, 2, 3]), ("val2017", [11, 12, 13, 14])):
            images = [
                {"id": i, "file_name": f"{i:012}.jpg", "width": 100, "height": 200} for i in ids
            ]
            annotations = [person(id=i, image_id=i) for i in ids]
            annotations.extend(
                person(id=i + 100, image_id=i, num_keypoints=0, keypoints=[0, 0, 0] * 17)
                for i in ids
            )
            compressed.writestr(
                f"annotations/person_keypoints_{official}.json",
                json.dumps({"images": images, "annotations": annotations}),
            )

    def fake_download(job):
        _, target = job
        image_id = int(target.stem)
        Image.new("RGB", (100, 200), (image_id * 10, 0, 0)).save(target)

    prepare = MODULE["prepare"]
    monkeypatch.setitem(prepare.__globals__, "download_image", fake_download)
    output = tmp_path / "dataset"
    prepare(
        argparse.Namespace(
            annotations=archive,
            output=output,
            train_images=3,
            val_images=2,
            test_images=2,
            download_workers=2,
            image_cache=None,
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["train"]["people"] == 3
    assert summary["train"]["excluded_zero_joint_people"] == 3
    assert summary["val"]["labeled_joints"] == 4
    manifests = {
        s: json.loads((output / f"{s}.json").read_text()) for s in ("train", "val", "test")
    }
    assert not {r["image_id"] for r in manifests["val"]} & {
        r["image_id"] for r in manifests["test"]
    }
    assert len(list((output / "images/train").glob("*.jpg"))) == 3
    assert len(next((output / "labels/train").glob("*.txt")).read_text().split()) == 56
    with pytest.raises(FileExistsError):
        prepare(argparse.Namespace(annotations=archive, output=output))
    monkeypatch.setitem(prepare.__globals__, "download_image", MODULE["download_image"])
    cached_output = tmp_path / "cached_dataset"
    prepare(
        argparse.Namespace(
            annotations=archive,
            output=cached_output,
            train_images=3,
            val_images=2,
            test_images=2,
            download_workers=2,
            image_cache=output / "images",
        )
    )
    cached_image = next((cached_output / "images/train").glob("*.jpg"))
    assert cached_image.is_symlink() and cached_image.readlink().is_absolute()
    assert (cached_output / "train.json").read_bytes() == (output / "train.json").read_bytes()
