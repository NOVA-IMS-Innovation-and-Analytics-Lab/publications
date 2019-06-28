"""
Test the data module.
"""

from requests import head

from ..data import FETCH_URLS, Datasets


def test_url_connections():
    """Test the urls connections."""
    for name, url in FETCH_URLS.items():
        assert head(url).status_code in (200, 301)
