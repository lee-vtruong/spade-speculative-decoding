import json
import tempfile
from pathlib import Path
import pytest
torch=pytest.importorskip("torch")
from spade.benchmark.benchmark_builder import build_examples, build_manifest, stratified_sample
from spade.benchmark.averitec_loader import claim_id
from spade.benchmark.evidence_adapter import load_claim_only_knowledge_store
from spade.benchmark.evaluation import parse_generation
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.sampling import acceptance_probability, expected_acceptance, processed_distribution, residual_distribution, sample_distribution, total_variation
from spade.decoding.speculative import exact_speculative_round
from spade.decoding.hf_assisted import QwenPair
from spade.instrumentation.trace_schema import ROUND_FIELDS, TOKEN_FIELDS, validate_record
from spade.analysis.greedy_compare import compare_greedy_runs

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

def test_top_p_keeps_minimal_prefix_that_reaches_threshold():
    probs=torch.tensor([.5,.25,.25],dtype=torch.float64)
    out=processed_distribution(torch.log(probs),top_p=.5)
    assert out.tolist()==pytest.approx([1.,0.,0.])

@pytest.mark.parametrize("p_values,q_values",[
    ([.5,.5],[.5,.5]),
    ([.1,.9],[.9,.1]),
    ([0.,.4,.6],[.7,.3,0.]),
    ([0.,0.,1.],[1.,0.,0.]),
    ([1.-1e-12,1e-12],[1e-12,1.-1e-12]),
])
def test_exact_speculative_empirical_distribution(p_values,q_values):
    p=torch.tensor(p_values,dtype=torch.float64); q=torch.tensor(q_values,dtype=torch.float64)
    g=torch.Generator().manual_seed(9217); n=5000; counts=torch.zeros_like(p,dtype=torch.int64)
    for _ in range(n):
        proposal=sample_distribution(q,g)
        result=exact_speculative_round([proposal],[q],[p],generator=g,remaining=1)
        counts[result.committed[0]]+=1
    observed=counts.double()/n
    tolerance=torch.maximum(torch.full_like(p,.025),5*torch.sqrt(p*(1-p)/n))
    assert torch.all(torch.abs(observed-p)<=tolerance), (observed.tolist(),p.tolist())

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

@pytest.mark.parametrize("reject_at",[0,1,2])
def test_exact_round_first_middle_and_last_rejection(reject_at):
    q=torch.tensor([1.,0.],dtype=torch.float64); accepted_p=torch.tensor([1.,0.],dtype=torch.float64); rejected_p=torch.tensor([0.,1.],dtype=torch.float64)
    ps=[accepted_p]*reject_at+[rejected_p]+[accepted_p]*(2-reject_at)
    result=exact_speculative_round([0,0,0],[q]*3,ps,generator=torch.Generator().manual_seed(17))
    assert result.rejected_position==reject_at
    assert result.committed==[0]*reject_at+[1]
    assert result.statuses==["accepted"]*reject_at+["rejected"]+["discarded_after_rejection"]*(2-reject_at)

def test_exact_round_eos_and_partial_final_block_do_not_add_extra():
    p=torch.tensor([.5,.5],dtype=torch.float64); g=torch.Generator().manual_seed(6)
    eos=exact_speculative_round([0,1,0],[p,p,p],[p,p,p],generator=g,eos_token_id=1,target_extra=p)
    assert eos.committed==[0,1] and eos.extra_target is None
    one=torch.tensor([1.,0.],dtype=torch.float64)
    partial=exact_speculative_round([0],[one],[one],generator=g,remaining=1,target_extra=one)
    assert partial.committed==[0] and partial.extra_target is None

def test_parse_verdict_strict_and_citations():
    r=parse_generation(json.dumps({"summary":"x","citations":["E1"],"verdict":"Refuted"}),{"E1"})
    assert r["parse_success"] and r["citation_ids_valid"]
    assert not parse_generation('{"verdict":"unknown"}',set())["parse_success"]

def test_stratified_sample_deterministic():
    rows=[{"label":"a","id":i} for i in range(8)]+[{"label":"b","id":i+8} for i in range(2)]
    assert stratified_sample(rows,5,7)==stratified_sample(rows,5,7)

def test_official_averitec_shape_gets_stable_non_gold_id():
    row={"claim":"A claim without an explicit ID", "claim_date":"2020", "speaker":"source", "label":"Refuted", "justification":"private"}
    other={**row,"label":"Supported","justification":"different gold text"}
    assert claim_id(row).startswith("avc-")
    assert claim_id(row)==claim_id(other)

def test_claim_only_store_excludes_gold_qa_and_other_query_types():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
        row={"claim":"A claim", "label":"Supported", "justification":"secret"}
        store=Path(temp)/"store"; store.mkdir()
        entries=[
            {"claim_id":"0","type":"gold","query":"_","url":"g","url2text":["gold text"]},
            {"claim_id":"0","type":"question","query":"Where?","url":"q","url2text":["qa text"]},
            {"claim_id":"0","type":"claim","query":"A claim","url":"safe","url2text":["claim retrieved text"]},
            {"claim_id":"0","type":"gpt_question","query":"Who?","url":"gpt","url2text":["generated text"]},
        ]
        (store/"0.json").write_text("\n".join(json.dumps(x) for x in entries),encoding="utf-8")
        mapped,audit=load_claim_only_knowledge_store([row],store,top_k=8)
        assert [x["text"] for x in mapped[claim_id(row)]]==["claim retrieved text"]
        assert audit["query_types_seen"]["question"]==1 and audit["query_type_allowlist"]==["claim"]

def test_manifest_tracks_ids_hashes_and_token_lengths():
    rows=[sample_row()]; examples=build_examples(rows,"oracle_qa")
    manifest=build_manifest(dataset_path="data.json",dataset_hash="abc",split="dev",seed=3,mode="oracle_qa",top_k=8,examples=examples,token_lengths=[37])
    assert manifest["selected_claim_ids"]==[claim_id(rows[0])]
    assert manifest["statistics"]["input_token_count"]["mean"]==37
    assert manifest["prompt_hashes"][claim_id(rows[0])]==examples[0]["prompt_hash"]

def test_oracle_adapter_supports_nested_averitec_answers():
    ex=build_examples([{**sample_row(),"questions":[{"question":"Q?","answers":[{"answer":"A.","source_url":"https://example.test"}]}]}],"oracle_qa")[0]
    assert ex["evidence"][0]["text"]=="Q? A."
    assert ex["evidence"][0]["source_url"]=="https://example.test"

class _TinyTokenizer:
    eos_token_id=2; pad_token_id=0
    def apply_chat_template(self, messages, **kwargs): return messages[0]["content"]
    def __call__(self, text, **kwargs): return {"input_ids":torch.tensor([[0,1]])}
    def decode(self, ids, **kwargs): return "".join(str(i) for i in ids)

class _TinyModel:
    def __init__(self): self.embedding=torch.nn.Embedding(3,2)
    def get_input_embeddings(self): return self.embedding
    def __call__(self, input_ids, use_cache=False):
        logits=torch.tensor([0.,2.,-4.]).view(1,1,3).expand(1,input_ids.shape[1],3)
        return type("Output",(),{"logits":logits})()

def _tiny_pair():
    pair=QwenPair.__new__(QwenPair); pair.target_tokenizer=_TinyTokenizer(); pair.draft_tokenizer=_TinyTokenizer()
    pair.target=_TinyModel(); pair.draft=_TinyModel(); return pair

def test_hf_adapter_tiny_greedy_and_exact_sampling_are_bounded():
    pair=_tiny_pair()
    greedy=pair.generate("claim",mode="B1",max_new_tokens=3,draft_length=2)
    exact=pair.generate("claim",mode="B2",max_new_tokens=3,draft_length=2,seed=7)
    assert greedy.token_ids==[1,1,1]
    assert len(exact.token_ids)==3
    assert exact.timings["decode_s"]>=0
    assert all(t["target_probability"]==1.0 for t in greedy.token_traces if t["status"]=="extra_target_token")
    for row in greedy.rounds + exact.rounds: validate_record(row,ROUND_FIELDS)
    for row in greedy.token_traces + exact.token_traces: validate_record(row,TOKEN_FIELDS)

def test_greedy_run_comparison_checks_claim_prompt_and_tokens():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
        tmp=Path(temp); b0,b1=tmp/"b0",tmp/"b1"; b0.mkdir(); b1.mkdir()
        (b0/"config_resolved.json").write_text(json.dumps({"mode":"B0","target":"t","dtype":"bf16","max_new_tokens":4}))
        (b1/"config_resolved.json").write_text(json.dumps({"mode":"B1","target":"t","dtype":"bf16","max_new_tokens":4}))
        record={"claim_id":"c1","prompt_hash":"abc","token_ids":[1,2,3]}
        (b0/"predictions.jsonl").write_text(json.dumps(record)+"\n")
        (b1/"predictions.jsonl").write_text(json.dumps(record)+"\n")
        assert compare_greedy_runs(b0,b1)["all_match"]
        (b1/"predictions.jsonl").write_text(json.dumps({**record,"token_ids":[1,9]})+"\n")
        report=compare_greedy_runs(b0,b1)
        assert not report["all_match"] and report["comparisons"][0]["first_mismatch_position"]==1
