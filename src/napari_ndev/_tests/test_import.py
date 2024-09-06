import time

def test_import_time():
    
    start_time = time.time()
    
    import napari_ndev
    
    end_time = time.time()
    
    import_time = end_time - start_time
    
    print(f"napari_ndev was imported in {import_time: .3f} seconds")
    
    assert import_time < 1.0, "napari_ndev took too long to import"