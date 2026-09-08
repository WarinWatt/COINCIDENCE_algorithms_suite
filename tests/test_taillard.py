from coin.problems.flowshop import taillard_catalog, taillard_instance

def test_catalog_contains_all_published_instances_and_sizes():
    catalog=taillard_catalog()
    assert len(catalog)==120
    assert catalog[0]=={"id":"ta001","jobs":20,"machines":5,"seed":873654221}
    assert catalog[-1]=={"id":"ta120","jobs":500,"machines":20,"seed":28837162}

def test_ta001_matches_or_library_verification_matrix():
    instance=taillard_instance("ta001")
    assert instance.processing_times.shape==(20,5)
    assert instance.processing_times[0].tolist()==[54,79,16,66,58]
    assert instance.processing_times[-1].tolist()==[94,77,40,31,28]

def test_generation_is_deterministic_and_supports_largest_instance():
    assert (taillard_instance(11).processing_times==taillard_instance("TA011").processing_times).all()
    assert taillard_instance("ta120").processing_times.shape==(500,20)
