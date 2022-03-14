from src import helpers


def test_load_metadata():
    metadata = helpers.load_metadata(["data"], ["raw"])
    assert isinstance(metadata, dict)
    assert "data" in metadata
    assert "training" in  metadata["data"]
    assert "test" in  metadata["data"]    
    assert "delimiter" in  metadata["data"]


