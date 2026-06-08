import pytest
from unittest.mock import patch


def test_config_loads_from_env():
    """Verify SHConfig reads credentials from environment."""
    with patch.dict("os.environ", {
        "SH_CLIENT_ID": "test_id",
        "SH_CLIENT_SECRET": "test_secret",
        "SH_INSTANCE_ID": "test_instance",
    }):
        from src.satellite_fetcher import SentinelHubFetcher
        fetcher = SentinelHubFetcher()
        assert fetcher.config.sh_client_id == "test_id"
        assert fetcher.config.sh_client_secret == "test_secret"


def test_config_missing_credentials_raises():
    """Verify clear error when credentials missing."""
    with patch.dict("os.environ", {}, clear=True):
        from src.satellite_fetcher import SentinelHubFetcher
        with pytest.raises(ValueError, match="Sentinel Hub credentials"):
            SentinelHubFetcher()


def test_check_new_scenes_returns_date():
    """Verify check_new_scenes returns latest scene date."""
    from unittest.mock import MagicMock
    from src.satellite_fetcher import SentinelHubFetcher, LOMBOK_BBOX

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    # Mock Catalog API response
    mock_scene = {"properties": {"datetime": "2026-06-07T02:30:00Z"}}
    with patch("sentinelhub.SentinelHubCatalog") as MockCatalog:
        mock_client = MagicMock()
        mock_client.search.return_value = [mock_scene]
        MockCatalog.return_value = mock_client

        result = fetcher.check_new_scenes(LOMBOK_BBOX, days_back=7)
        assert result is not None
        assert "2026-06-07" in result


def test_fetch_sentinel1_returns_numpy():
    """Verify fetch_sentinel1 returns VV/VH numpy arrays."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher, LOMBOK_BBOX

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    # Mock 3D array: (height, width, bands)
    mock_data = np.zeros((256, 256, 2), dtype=np.float32)

    with patch("sentinelhub.SentinelHubRequest") as MockReq:
        instance = MagicMock()
        instance.get_data.return_value = [mock_data]
        MockReq.return_value = instance

        vv, vh = fetcher.fetch_sentinel1(LOMBOK_BBOX, "2026-06-07")
        assert vv.shape == (256, 256)
        assert vh.shape == (256, 256)


def test_fetch_sentinel2_returns_numpy():
    """Verify fetch_sentinel2 returns Green/NIR numpy arrays."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher, LOMBOK_BBOX

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    # Mock 3D array: (height, width, bands)
    mock_data = np.zeros((512, 512, 2), dtype=np.float32)

    with patch("sentinelhub.SentinelHubRequest") as MockReq:
        instance = MagicMock()
        instance.get_data.return_value = [mock_data]
        MockReq.return_value = instance

        green, nir = fetcher.fetch_sentinel2(LOMBOK_BBOX, "2026-06-07")
        assert green.shape == (512, 512)
        assert nir.shape == (512, 512)


def test_save_rasters_writes_files(tmp_path):
    """Verify save_rasters writes GeoTIFF to processed dir."""
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)

    rasters = {
        "s1_vv": np.zeros((64, 64), dtype=np.float32),
        "s1_vh": np.zeros((64, 64), dtype=np.float32),
        "s2_green": np.zeros((64, 64), dtype=np.float32),
        "s2_nir": np.zeros((64, 64), dtype=np.float32),
        "date": "2026-06-07",
    }

    with patch("src.satellite_fetcher.PROCESSED_DIR", tmp_path):
        fetcher.save_rasters(rasters)

    assert (tmp_path / "sentinel1_reproj.tif").exists()
    assert (tmp_path / "sentinel2_reproj.tif").exists()
