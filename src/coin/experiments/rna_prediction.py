"""Fair permutation-search comparison for RNA secondary structures."""
from __future__ import annotations
from dataclasses import dataclass
import math, time
import numpy as np
from coin.models import (CNBCoin, EdgeConfig, ReferenceEdgeCoin, PositionCoin, HybridChainCoin, HybridCoin,
                         StartNodeEdgeCoin, EHBSA, NHBSA, HBSAConfig)
from coin.problems.rna import Helix, enumerate_helices, helices_compatible, subset_to_dot_bracket, structure_pairs
from coin.problems.rna_numba import NumbaBatchRnaDecoder, pair_table_key, pair_table_to_dot_bracket
try:
    import RNA  # type: ignore
except ImportError:
    RNA = None

ALGORITHMS=("coin","nb_coin","cnb_coin","hybrid_coin","hybrid_chain","start_node_edge_coin",
            "ehbsa","ehbsa_wt","nhbsa","nhbsa_wt","ga_ox","ga_er","sarna")
PARAMETERS={
    "coin":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"edge priority"},
    "nb_coin":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"node/position"},
    "cnb_coin":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"chained node/position 0 to n"},
    "hybrid_coin":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"Hybrid Template: Node at position 0 + scattered 30–70%; Edge completion"},
    "hybrid_chain":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"Hybrid Chain: Node at position 0; Node-or-Edge choice at each following link"},
    "start_node_edge_coin":{"reward_selection_percent":20,"punishment_selection_percent":20,"learning_strength":5,"model":"start node + directed edge"},
    "ehbsa":{"selection_percent":50,"bias_ratio":0.005,"sampling":"without template","model":"edge histogram"},
    "ehbsa_wt":{"selection_percent":50,"bias_ratio":0.005,"sampling":"with punched template","template_sample_percent":50,"model":"edge histogram"},
    "nhbsa":{"selection_percent":50,"bias_ratio":0.005,"sampling":"without template","model":"position histogram"},
    "nhbsa_wt":{"selection_percent":50,"bias_ratio":0.005,"sampling":"with punched template","template_sample_percent":50,"model":"position histogram"},
    "ga_ox":{"elite_percent":20,"crossover":"OX","crossover_probability":1.0,"swap_mutation_probability":0.2},
    "ga_er":{"elite_percent":20,"crossover":"ERX","crossover_probability":1.0,"swap_mutation_probability":0.2},
    "sarna":{"schedule":"geometric","initial_temperature":"max(1, 0.1 × |initial ΔG|)","final_temperature_ratio":0.001,"mutation":"one random swap per move"},
}

class RnaExperimentCancelled(RuntimeError):
    pass

@dataclass(slots=True)
class FoldResult:
    permutation: np.ndarray; dot_bracket: str; energy: float; pair_count: int

class RnaPermutationEvaluator:
    def __init__(self,sequence:str,helices:list[Helix]):
        self.sequence,self.helices,self.evaluations=sequence,helices,0
        self.proxy_evaluations,self.exact_evaluations,self.cache_hits=0,0,0
        self._decoder=NumbaBatchRnaDecoder(helices,len(sequence)); self._energy_cache={}
        self._fold=RNA.fold_compound(sequence) if RNA is not None else None
    @property
    def energy_model(self): return "ViennaRNA Turner nearest-neighbor" if self._fold else "canonical-pair proxy"
    def decode(self,permutation):
        chosen=[]
        for index in permutation:
            candidate=self.helices[int(index)]
            if all(helices_compatible(candidate,old) for old in chosen): chosen.append(candidate)
        return chosen
    def one(self,permutation):
        _,results=self.population(np.asarray(permutation,dtype=np.int16)[None,:])
        return results[0]
    def population(self,population):
        population=np.ascontiguousarray(population,dtype=np.int16); decoded=self._decoder.decode(population)
        self.evaluations+=len(population); self.proxy_evaluations+=len(population); results=[]
        for index,row in enumerate(population):
            table=self._decoder.pair_table(decoded.selected[index]); key=pair_table_key(table)
            cached=self._energy_cache.get(key)
            if cached is None:
                structure=pair_table_to_dot_bracket(table)
                energy=float(self._fold.eval_structure(structure)) if self._fold else float(decoded.proxy_energy[index])
                self._energy_cache[key]=(structure,energy); self.exact_evaluations+=1
            else:
                structure,energy=cached; self.cache_hits+=1
            results.append(FoldResult(row.copy(),structure,energy,int(decoded.pair_count[index])))
        return np.asarray([r.energy for r in results]),results

    def screened_population(self,population,top_k,archive_indices=()):
        """Decode all candidates by Numba, then exact-score only proxy top-k/archive."""
        population=np.ascontiguousarray(population,dtype=np.int16); decoded=self._decoder.decode(population)
        self.proxy_evaluations+=len(population)
        chosen=set(int(x) for x in np.argsort(decoded.proxy_energy)[:max(1,min(int(top_k),len(population)))])
        chosen.update(int(x) for x in archive_indices)
        exact={}
        for index in sorted(chosen):
            table=self._decoder.pair_table(decoded.selected[index]); key=pair_table_key(table); cached=self._energy_cache.get(key)
            if cached is None:
                structure=pair_table_to_dot_bracket(table); energy=float(self._fold.eval_structure(structure)) if self._fold else float(decoded.proxy_energy[index])
                self._energy_cache[key]=(structure,energy); self.exact_evaluations+=1
            else: structure,energy=cached; self.cache_hits+=1
            exact[index]=FoldResult(population[index].copy(),structure,energy,int(decoded.pair_count[index]))
        return decoded.proxy_energy,exact

def _random_population(rng,size,n): return np.asarray([rng.permutation(n) for _ in range(size)],dtype=np.int16)
def _ox(a,b,rng):
    n=len(a); left,right=sorted(rng.choice(n,2,replace=False)); right+=1; child=np.full(n,-1,dtype=np.int16); child[left:right]=a[left:right]
    used=set(int(x) for x in child[left:right]); child[np.flatnonzero(child<0)]=[int(x) for x in b if int(x) not in used]; return child
def _erx(a,b,rng):
    n=len(a); neighbors={int(x):set() for x in a}
    for parent in (a,b):
        for i,x in enumerate(parent):
            if i: neighbors[int(x)].add(int(parent[i-1]))
            if i+1<n: neighbors[int(x)].add(int(parent[i+1]))
    current=int(a[rng.integers(n)]); child=[]
    while len(child)<n:
        child.append(current)
        for values in neighbors.values(): values.discard(current)
        candidates=neighbors[current]
        if candidates:
            minimum=min(len(neighbors[x]) for x in candidates); pool=sorted(x for x in candidates if len(neighbors[x])==minimum)
        else: pool=sorted(set(range(n))-set(child))
        if pool: current=int(pool[rng.integers(len(pool))])
    return np.asarray(child,dtype=np.int16)
def _mutate(row,rng,p=.2):
    if rng.random()<p:
        i,j=rng.choice(len(row),2,replace=False); row[i],row[j]=row[j],row[i]
def _record(history,generation,evaluations,energies): history.append({"generation":generation,"evaluations":evaluations,"best_energy":float(np.min(energies)),"mean_energy":float(np.mean(energies))})

def run_model(name,evaluator,*,population_size,generations,seed,reward_ratio=20,punishment_ratio=20,
              learning_strength=5,hbsa_selection_ratio=50,hbsa_bias_ratio=.005,
              hbsa_template_sample_ratio=50,mutation_probability=.2,cancel_callback=None):
    if name not in ALGORITHMS: raise ValueError(f"unknown RNA algorithm: {name}")
    n=len(evaluator.helices); rng=np.random.default_rng(seed); history=[]; best=None; started=time.perf_counter()
    if name in ("coin","nb_coin","cnb_coin","hybrid_coin","hybrid_chain","start_node_edge_coin","ehbsa","ehbsa_wt","nhbsa","nhbsa_wt"):
        if name in ("coin","nb_coin","cnb_coin","hybrid_coin","hybrid_chain","start_node_edge_coin"):
            config=EdgeConfig(problem_size=n,population_size=population_size,training_rate=learning_strength,
                              reward_ratio=reward_ratio,punishment_ratio=punishment_ratio,objective="min")
            factory={"coin":ReferenceEdgeCoin,"nb_coin":PositionCoin,"cnb_coin":CNBCoin,"hybrid_coin":HybridCoin,"hybrid_chain":HybridChainCoin,
                     "start_node_edge_coin":StartNodeEdgeCoin}[name]
            model=factory(config,seed=seed)
        else:
            model_class=EHBSA if name.startswith("ehbsa") else NHBSA
            model=model_class(HBSAConfig(problem_size=n,population_size=population_size,
                selection_ratio=hbsa_selection_ratio,bias_ratio=hbsa_bias_ratio,
                sampling_mode="wt" if name.endswith("_wt") else "wo",
                template_sample_ratio=hbsa_template_sample_ratio,objective="min"),seed=seed)
        for generation in range(1,generations+1):
            if cancel_callback and cancel_callback(): raise RnaExperimentCancelled("RNA experiment cancelled")
            population=model.generate_population(); energies,results=evaluator.population(population); candidate=results[int(np.argmin(energies))]
            if best is None or candidate.energy<best.energy: best=candidate
            model.update(*model.statistics(population,energies)); _record(history,generation,evaluator.evaluations,energies)
    elif name in ("ga_ox","ga_er"):
        population=_random_population(rng,population_size,n)
        for generation in range(1,generations+1):
            if cancel_callback and cancel_callback(): raise RnaExperimentCancelled("RNA experiment cancelled")
            energies,results=evaluator.population(population); candidate=results[int(np.argmin(energies))]
            if best is None or candidate.energy<best.energy: best=candidate
            order=np.argsort(energies); elite=population[order[:max(2,population_size//5)]]; new=[population[order[0]].copy()]; crossover=_ox if name=="ga_ox" else _erx
            while len(new)<population_size:
                a,b=elite[rng.integers(len(elite),size=2)]; child=crossover(a,b,rng); _mutate(child,rng,mutation_probability); new.append(child)
            population=np.asarray(new,dtype=np.int16); _record(history,generation,evaluator.evaluations,energies)
    else:
        current=rng.permutation(n).astype(np.int16); current_result=evaluator.one(current); best=current_result; budget=population_size*generations; t0=max(1.0,abs(current_result.energy)*.1); alpha=math.exp(math.log(.001)/max(1,budget-1))
        for move in range(1,budget):
            if cancel_callback and cancel_callback(): raise RnaExperimentCancelled("RNA experiment cancelled")
            candidate=current.copy(); i,j=rng.choice(n,2,replace=False); candidate[i],candidate[j]=candidate[j],candidate[i]; result=evaluator.one(candidate); temperature=t0*(alpha**move); delta=result.energy-current_result.energy
            if delta<=0 or rng.random()<math.exp(-delta/max(temperature,1e-12)): current,current_result=candidate,result
            if result.energy<best.energy: best=result
            if move%population_size==0: _record(history,move//population_size,evaluator.evaluations,[best.energy,current_result.energy])
    return {"algorithm":name,"evaluations":evaluator.evaluations,"proxy_evaluations":evaluator.proxy_evaluations,
            "exact_evaluations":evaluator.exact_evaluations,"energy_cache_hits":evaluator.cache_hits,
            "elapsed_seconds":time.perf_counter()-started,"energy_model":evaluator.energy_model,
            "best_energy":best.energy,"pair_count":best.pair_count,"dot_bracket":best.dot_bracket,
            "permutation":[int(x) for x in best.permutation],"history":history}

def _known_metrics(predicted,known):
    predicted_pairs=set(structure_pairs(predicted)); known_pairs=set(structure_pairs(known)); tp=len(predicted_pairs&known_pairs); fp=len(predicted_pairs-known_pairs); fn=len(known_pairs-predicted_pairs)
    sensitivity=tp/(tp+fn) if tp+fn else 1.0; ppv=tp/(tp+fp) if tp+fp else 1.0; f1=2*sensitivity*ppv/(sensitivity+ppv) if sensitivity+ppv else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"pair_distance":fp+fn,"sensitivity":sensitivity,"ppv":ppv,"f1":f1,
            "exact_pair_match":fp==0 and fn==0}

def compare(sequence,*,algorithms,population_size,generations,seed,min_stem_length=2,min_loop_length=3,
            max_candidate_helices=600,known_structure=None,reward_ratio=20,punishment_ratio=20,
            learning_strength=5,hbsa_selection_ratio=50,hbsa_bias_ratio=.005,
            hbsa_template_sample_ratio=50,mutation_probability=.2,progress_callback=None,cancel_callback=None):
    helices=enumerate_helices(sequence,min_stem_length=min_stem_length,min_loop_length=min_loop_length,max_helices=max_candidate_helices)
    if len(helices)<2: raise ValueError("sequence produced fewer than two candidate helices")
    results=[]
    for index,name in enumerate(algorithms):
        if progress_callback: progress_callback(index,len(algorithms),name,0)
        result=run_model(name,RnaPermutationEvaluator(sequence,helices),population_size=population_size,generations=generations,seed=seed,
                         reward_ratio=reward_ratio,punishment_ratio=punishment_ratio,learning_strength=learning_strength,
                         hbsa_selection_ratio=hbsa_selection_ratio,hbsa_bias_ratio=hbsa_bias_ratio,
                         hbsa_template_sample_ratio=hbsa_template_sample_ratio,mutation_probability=mutation_probability,
                         cancel_callback=cancel_callback)
        results.append(result)
        if progress_callback: progress_callback(index+1,len(algorithms),name,result["evaluations"])
    if known_structure:
        if len(known_structure)!=len(sequence): raise ValueError("known structure length differs from sequence")
        structure_pairs(known_structure)
        for result in results: result["known_structure_metrics"]=_known_metrics(result["dot_bracket"],known_structure)
    reference={}
    if RNA is not None:
        reference_started=time.perf_counter(); structure,energy=RNA.fold(sequence)
        reference={"algorithm":"viennarna_dp","dot_bracket":structure,"energy":float(energy),"elapsed_seconds":time.perf_counter()-reference_started,"method":"dynamic programming reference; not equal-budget population search"}
    if known_structure and reference: reference["known_structure_metrics"]=_known_metrics(reference["dot_bracket"],known_structure)
    if reference:
        reference_pairs=set(structure_pairs(reference["dot_bracket"]))
        for result in results:
            predicted_pairs=set(structure_pairs(result["dot_bracket"]))
            result["vienna_comparison"]={
                "energy_gap":result["best_energy"]-reference["energy"],
                "pair_distance":len(predicted_pairs-reference_pairs)+len(reference_pairs-predicted_pairs),
            }
    measures=[
        {"id":"turner_energy","label":"Turner free energy","direction":"minimize","unit":"kcal/mol","role":"search objective"},
        {"id":"pair_count","label":"Predicted base pairs","direction":"descriptive","unit":"pairs","role":"diagnostic"},
        {"id":"runtime","label":"Runtime","direction":"minimize","unit":"seconds","role":"computational cost"},
        {"id":"vienna_energy_gap","label":"Energy gap from ViennaRNA DP","direction":"minimize","unit":"kcal/mol","role":"post-hoc reference"},
        {"id":"vienna_pair_distance","label":"Base-pair distance from ViennaRNA DP","direction":"minimize","unit":"pairs","role":"post-hoc reference"},
    ]
    if known_structure:
        measures.extend([
            {"id":"sensitivity","label":"Known-pair sensitivity","direction":"maximize","unit":"ratio","role":"held-out accuracy"},
            {"id":"ppv","label":"Known-pair PPV","direction":"maximize","unit":"ratio","role":"held-out accuracy"},
            {"id":"f1","label":"Known-pair F1","direction":"maximize","unit":"ratio","role":"held-out accuracy"},
            {"id":"known_pair_distance","label":"Base-pair distance from known structure","direction":"minimize","unit":"pairs","role":"held-out accuracy"},
        ])
    return {"sequence":sequence,"helix_count":len(helices),"known_structure":known_structure,
            "objective":{"name":"Turner free energy","unit":"kcal/mol","direction":"minimize","count":1},
            "research_measures":measures,"parameters":PARAMETERS,"vienna_reference":reference,"results":results}
