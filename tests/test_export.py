from roa_processor.io.export import ensure_output_dirs


def test_replace_existing_output_folder_removes_stale_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output = data_dir / "processed"
    output.mkdir()
    (output / "stale.txt").write_text("old run", encoding="utf-8")

    resolved = ensure_output_dirs(
        "processed",
        base_dir=data_dir,
        replace_existing=True,
    )

    assert resolved == output.resolve()
    assert not (output / "stale.txt").exists()
    assert (output / "figures").is_dir()
