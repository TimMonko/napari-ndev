import time


def test_import_time():
    start_time = time.time()

    end_time = time.time()

    import_time = end_time - start_time

    assert import_time < 1.0, 'napari_ndev took too long to import'
