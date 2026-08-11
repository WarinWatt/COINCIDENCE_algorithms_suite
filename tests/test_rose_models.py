import numpy as np

from coin.core import PermutationCoinAlgorithm
from coin.experiments import ExperimentConfiguration, ExperimentRunner
from coin.models import ROSE, ROSESingleRef, TemplateROSE, TemplateROSESingleRef, RoseConfig
from coin.problems.flowshop import SMALL_3X2, FlowShopProblem

SELECTED = np.asarray([[0,1,2,3,4],[0,2,1,3,4],[1,0,2,4,3],[0,1,2,4,3],[1,0,2,3,4]], dtype=np.int16)

def fitted(cls=ROSESingleRef, **overrides):
    values = dict(problem_size=5, population_size=40, selection_ratio=100, smoothing=.01, reference_selection="uniform")
    values.update(overrides); model = cls(RoseConfig(**values), seed=23)
    statistics, _ = model.statistics(SELECTED, np.arange(len(SELECTED))); model.update(statistics, None)
    return model

def test_relative_histogram_is_signed_symmetric_and_normalized():
    model = fitted(); np.testing.assert_allclose(model.relative.sum(axis=2), 1)
    np.testing.assert_array_equal(model.relative_counts[0,1], model.relative_counts[1,0][::-1])

def test_multimodal_distances_do_not_collapse_to_mean():
    left=np.asarray([0,2,3,4,1],dtype=np.int16); right=np.asarray([1,2,3,4,0],dtype=np.int16)
    model=ROSESingleRef(RoseConfig(5,selection_ratio=100,smoothing=.001),seed=1)
    statistics,_=model.statistics(np.vstack([left]*10+[right]*10),np.arange(20)); model.update(statistics,None)
    pair=model.pair_statistics(0,1)
    assert pair["count"]==20 and pair["mean"]==0 and pair["minimum"]==-4 and pair["maximum"]==4
    assert pair["observed_distance_values"]==2

def test_uniform_reference_selection_is_nearly_uniform():
    model=fitted(); counts={0:0,1:0,2:0}
    for _ in range(6000): counts[model._select_reference(4,[0,1,2])]+=1
    assert all(1800<value<2200 for value in counts.values())

def test_confidence_reference_prefers_lower_entropy():
    model=fitted(reference_selection="confidence"); n=model.config.problem_size
    model.relative[0,4]=.001; model.relative[0,4,n]=.992; model.relative[0,4]/=model.relative[0,4].sum()
    model.relative[1,4]=1/(2*n-1); choices=[model._select_reference(4,[0,1]) for _ in range(2000)]
    assert choices.count(0)>choices.count(1)

def test_single_reference_is_valid_and_reproducible():
    left,right=fitted(roll_mode="all"),fitted(roll_mode="all"); a,b=left.generate_population(),right.generate_population()
    assert np.array_equal(a,b) and all(sorted(row.tolist())==list(range(5)) for row in a)
    assert left.diagnostics()["relative_sampling_count"]>0

def test_template_single_reference_is_valid():
    model=fitted(TemplateROSESingleRef,template_enabled=True,template_sample_ratio=40); children=model.generate_population()
    assert all(sorted(row.tolist())==list(range(5)) for row in children)
    assert model.diagnostics()["template_fixed_positions"]==120

def test_node_weight_extremes_are_supported():
    for weight in (0.0,1.0): assert all(sorted(row.tolist())==list(range(5)) for row in fitted(node_weight=weight).generate_population())

def test_mean_reference_baselines_remain_available():
    assert isinstance(fitted(ROSE,reference_selection="mean"),ROSE)
    assert isinstance(fitted(TemplateROSE,reference_selection="mean",template_enabled=True),TemplateROSE)

def test_all_four_variants_use_exact_budget():
    config=ExperimentConfiguration(algorithms=("rose","rose_single_ref","template_rose","template_rose_single_ref"),objectives=("makespan",),population_size=10,evaluation_budget=30,maximum_generations=3,seeds=(13,))
    result=ExperimentRunner().run(SMALL_3X2,config)
    assert len(result.runs)==4 and all(run.evaluations==30 for run in result.runs)

def test_reusable_permutation_problem_interface():
    algorithm=PermutationCoinAlgorithm(ROSESingleRef(RoseConfig(3,population_size=8),seed=9),FlowShopProblem(SMALL_3X2.processing_times,("makespan",)))
    for _ in range(3): algorithm.step()
    assert algorithm.evaluations==24
