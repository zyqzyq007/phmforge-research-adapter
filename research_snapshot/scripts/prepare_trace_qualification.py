"""Freeze a bounded execution-trace replay on already opened PHM evidence."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import shutil

P=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
A=Path('/data/yyf-zyq/PHM_20261010/20261025T012600Z_local347')
now=datetime.datetime.now(datetime.timezone.utc)
R=Path('/srv/yyf-zyq/PHM_20261010')/(now.strftime('%Y%m%dT%H%M%SZ')+'_trace352')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
R.mkdir(exist_ok=False)
for n in ['code','runtime/tmp','runtime/mpl']:(R/n).mkdir(parents=True)
receipt=json.loads((A/'ARCHIVE_VERIFIED.json').read_text())
assert sha(A/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']
M={x['path']:x for x in json.loads((A/'SOURCE_MANIFEST.json').read_text())['files']}
shutil.copytree(A/'common_tools',R/'common_tools')
for q in (R/'common_tools').glob('*.py'):assert sha(q)==M['common_tools/'+q.name]['sha256']
shutil.copy2(P/'scripts/phm_execution_trace.py',R/'common_tools/phm_execution_trace.py')
f=R/'common_tools/program_training_entry.py';text=f.read_text()
old="        runpy.run_path(args.program,run_name='__main__');status='completed'"
assert text.count(old)==1
text=text.replace(old,"        from phm_execution_trace import ProgramTrace\n        with ProgramTrace(Path(args.program).parent,args.audit.with_name('execution_trace.json')):\n            runpy.run_path(args.program,run_name='__main__');status='completed'")
f.write_text(text)
for n in ['qualify_program_trace.py','program_revision_runtime.py','controlled_feedback.py']:shutil.copy2(P/'scripts'/n,R/'code'/n)
spec=[('baseline','reference/p00','subjects/local_revision_1/independent_packet/p00'),
 ('global_update','subjects/local_revision_1/programs/p04','subjects/local_revision_1/independent_packet/p04'),
 ('dual_lda','subjects/ordinary_coding_2/programs/p03','subjects/ordinary_coding_2/independent_packet_359687e1097a/p03'),
 ('guarded_lda','subjects/ordinary_coding_2/programs/p04','subjects/ordinary_coding_2/independent_packet_b48e2d1e1172/p04')]
programs=[];bindings={}
for ident,base,check in spec:
    row=dict(id=ident,source=base+'/source',artifacts=base+'/full_fit/artifacts',packets=[])
    for name in M:
        if name.startswith(row['source']+'/') or name.startswith(row['artifacts']+'/'):bindings[name]=M[name]['sha256']
    for role,folder,n in [('development',base+'/development',8),('check',check,4)]:
        for i in range(n):
            item=dict(role=role,input=folder+'/input'+str(i)+'.npz',expected=folder+'/prediction'+str(i)+'/predictions.json')
            for name in [item['input'],item['expected']]:assert name in M;bindings[name]=M[name]['sha256']
            row['packets'].append(item)
    programs.append(row)
a=A/'subjects/ordinary_coding_2/programs/p03';b=A/'subjects/ordinary_coding_2/programs/p04'
def parts(p):return [ast.dump(n,include_attributes=False) for n in ast.parse((p/'source/program.py').read_text()).body if not (isinstance(n,ast.FunctionDef) and n.name=='predict_acquisition')]
assert parts(a)==parts(b) and sha(a/'full_fit/artifacts/model.joblib')==sha(b/'full_fit/artifacts/model.joblib')
save(R/'STATIC_GUARD_PROOF352.json',dict(status='static_preconditions_verified_before_replay',
    all_AST_except_predict_function_identical=True,model_artifact_SHA_identical=True,
    model_sha256=sha(a/'full_fit/artifacts/model.joblib'),
    branch_signature_mapping={'old_not1':'outerfalse,innernotexecuted','old1_new3':'outertrue,firsttrue,secondnotexecuted','old1_new2':'outertrue,firstfalse,secondtrue','old1_new1':'outertrue,firstfalse,secondfalse'},
    scope='Introducedguardconditionsonly;notlatentmodelbehavior',hypothesis='Changedguardoutcomesmayalreadybedeterminedbypairpredictions.',new_fits=0,new_predictions=0,new_LLM_calls=0))
save(R/'RELATED_WORK352.json',dict(retrieved_utc=now.isoformat(),novelty_established=False,
    review_scope='ThreequeryphrasingsplusISSTA/ICSEofficialvenuesandprimaryimplementation;limitedreview.',papers=[
    dict(title='RippleGUItester: Change-Aware Exploratory Testing',venue='ISSTA2026Researchpapersofficialprogram',url='https://arxiv.org/html/2603.03121',venue_url='https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/149/RippleGUItester-Change-Aware-Exploratory-Testing',code='https://github.com/RippleTester/RippleGUItesting',read_scope='officialabstract,primaryHTMLmethodoutlineandoraclediscussion',relevance='Changeimpact/prepostdifferentialtestingalreadyexist;GUIintentoracleisnotPHMlabel.'),
    dict(title='TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble',venue='preprint,noacceptedvenueasserted',url='https://arxiv.org/html/2610.10242v1',read_scope='abstract,introduction,relatedwork,AppendixA',relevance='Executiondefineddifferentiationandfailurememoryalreadyexist;generatedassertionscannotreplacePHMlabels.'),
    dict(title='SIKG: A Semantic Impact Knowledge Graph Approach for Adaptive Test Selection and Prioritization',venue='ICSE2026Posters,notmainresearchpaper',url='https://conf.researchr.org/details/icse-2026/icse-2026-posters/4/SIKG-A-Semantic-Impact-Knowledge-Graph-Approach-for-Adaptive-Test-Selection-and-Prio',read_scope='officialposterabstractonly',relevance='Semanticchange-awaretestselectionhaspriorart;no reproductionclaim.')]))
cfg=dict(iteration='A352',frozen_utc=now.isoformat(),purpose='Executiontracequalification,notmethodefficacyornewAgentfailure',archive=str(A),manifest_sha256=receipt['manifest_sha256'],programs=programs,source_bindings=bindings,
    budget=dict(new_LLM_calls=0,new_fits=0,actual_prediction_processes_max=48,prediction_wall_seconds=900,predict_seconds=45,parallel_programs=4,numeric_threads_each=2,disk_MiB=512),
    inputs='All12alreadyopenedlegaldevelopment/checkacquisitionsforeach4fixedprograms;9windows/group;label-freepackets;archivedweights,noretraining.',
    sequence='Allbindings/labelisolationfirst;onebaselinepreflightcountsinthe48;then4programsparallel.Noretry.',
    pass_rule='All48tracedpredictionsmatcheachof9archivedoutputs;nonemptytrace;0predictionfit/optimizer;withinbudget.',
    new_measurement_gate='DochangedPythonbranchesadddistinctionsbeyondpairpredictionsorconstantvalid-inputflow?Onlypositiveevidencejustifiesqueryexperiment;noautomaticLLMbatch.',
    predeclared_negative='Ifglobalreplacementpathsconstantandguardstatesdeduciblefromold/newlabels,dropbranchcoveragealoneasmainmechanism.',
    no_final_reads=True,no_A338_performance_reads=True,physical_source_clusters=1,independent_confirmation_scenarios=0,
    limits=['Alreadyopeneddiscoveryonly','Pythonmainthread,nolatentC-modelcoverage','Historicaltimeisnotmatchedcostcontrol','Fourprogramsarequalificationexamples,notfuturecandidatebankrestriction'])
save(R/'PROTOCOL352.json',cfg)
save(R/'ENGINE_CONFIG.json',dict(input_root=str(A/'private_inputs'),common_tools=str(R/'common_tools'),fit_seconds=120,predict_seconds=45,program_seconds=900))
save(R/'PREPARATION_ERROR352.json',dict(kind='BeforeexecutionliteralpreparationscriptSyntaxError',message="Reservedkeywordpassusedasdictkeyword;Pythonrejectedwholeinputbeforeanyfilesystemmutation.",new_fits=0,new_predictions=0,new_LLM_calls=0,initial_freeze_or_science_execution_started=False))
shutil.copy2(Path(__file__),R/'code/prepare_trace_qualification.py')
for q in [*(R/'code').glob('*.py'),*(R/'common_tools').glob('*.py')]:compile(q.read_text(),str(q),'exec')
names=[q for d in ['code','common_tools'] for q in (R/d).rglob('*') if q.is_file()]+[R/n for n in ['PROTOCOL352.json','ENGINE_CONFIG.json','STATIC_GUARD_PROOF352.json','RELATED_WORK352.json']]
save(R/'CODE_FREEZE352.json',{str(q.relative_to(R)):sha(q) for q in names})
save(R/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4',usage='Pinnedreadonlydeps;actualmodelinputs/source/weightsareSHA-boundA347archivereferences.')))
D=P/'results/trace352';D.mkdir(exist_ok=True)
for n in ['PROTOCOL352.json','STATIC_GUARD_PROOF352.json','RELATED_WORK352.json']:shutil.copy2(R/n,D/n)
value=dict(utc=now.isoformat(),status='frozen_before_first_prediction',run=str(R),new_fits=0,new_predictions=0,new_LLM_calls=0)
save(D/'current_run.json',value);print(json.dumps(value))
