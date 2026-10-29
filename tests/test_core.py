import json
import pytest
torch=pytest.importorskip("torch")
from spade.benchmark.benchmark_builder import build_examples, stratified_sample
from spade.benchmark.evaluation import parse_generation
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.sampling import acceptance_probability, expected_acceptance, residual_distribution, total_variation
from spade.decoding.speculative import exact_speculative_round

def sample_row():
    return {"id":1,"claim":"A test claim.","label":"Refuted","justification":"SECRET GOLD TEXT","questions":[{"question":"What?","answer":"Evidence."}]}

def test_gold_not_in_prompt_and_oracle_is_labeled():
    ex=build_examples([sample_row()],"oracle_qa")[0]
    assert "SECRET GOLD TEXT" not in ex["prompt"]
    # Allowed verdict vocabulary is necessarily prompt-visible; the row's gold
    # label exists only in the evaluation namespace.
    assert ex["evaluation"]["gold_label"]=="Refuted"
    assert ex["evidence"][0]["provenance"]=="oracle_qa"

def test_prompt_deterministic_and_hash_stable():
    a=build_prompt("claim",[],"retrieved"); assert a==build_prompt("claim",[],"retrieved")

def test_sampling_identities_and_residual():
    p=torch.tensor([.7,.3],dtype=torch.float64); q=torch.tensor([.4,.6],dtype=torch.float64)
    assert expected_acceptance(p,q)==pytest.approx(1-total_variation(p,q))
    assert acceptance_probability(p,q,0).item()==1
    r=residual_distribution(p,q); assert r.tolist()==pytest.approx([1,0])

def test_disjoint_and_equal_support_residual():
    assert residual_distribution(torch.tensor([1.,0.]),torch.tensor([0.,1.])).tolist()==[1.,0.]
    with pytest.raises(ValueError): residual_distribution(torch.tensor([.5,.5]),torch.tensor([.5,.5]))

def test_full_acceptance_and_rejection_rounds():
    p=torch.tensor([1.,0.]); q=torch.tensor([1.,0.]); g=torch.Generator().manual_seed(2)
    result=exact_speculative_round([0],[q],[p],generator=g,target_extra=p)
    assert result.committed==[0,0] and result.full_acceptance
    p0=torch.tensor([1.,0.]); q0=torch.tensor([0.,1.])
    rejected=exact_speculative_round([1],[q0],[p0],generator=g)
    assert rejected.committed==[0] and rejected.statuses==["rejected"]

def test_parse_verdict_strict_and_citations():
    r=parse_generation(json.dumps({"summary":"x","citations":["E1"],"verdict":"Refuted"}),{"E1"})
    assert r["parse_success"] and r["citation_ids_valid"]
    assert not parse_generation('{"verdict":"unknown"}',set())["parse_success"]

def test_stratified_sample_deterministic():
    rows=[{"label":"a","id":i} for i in range(8)]+[{"label":"b","id":i+8} for i in range(2)]
    assert stratified_sample(rows,5,7)==stratified_sample(rows,5,7)
