#!/usr/bin/env python3
"""Concept-Tree Guided Graph + Vector RAG over an EXISTING MiniRAG index.

Independent research implementation. Does NOT change MiniRAG or original index.

Build: chunk embeddings -> hierarchical topical Concept tree; map chunks to leaf
concepts. Existing MiniRAG GraphML nodes/edges retain Entity-Chunk provenance.
QA: question embedding -> Concept tree -> entity matching in selected concepts
-> one-hop knowledge graph expansion -> vector reranking of sourced chunks
-> local LLM answer. The supplied gold answers are NEVER read into retrieval.

Requirements: numpy networkx scikit-learn sentence-transformers requests
For --backend hf also install torch transformers. For --label-backend ollama,
start Ollama and pull the chosen local labeling model. For --label-backend openai,
set OPENAI_API_KEY in .env (project root) or in the environment; only concept labels use the OpenAI API.
"""
from __future__ import annotations
import argparse
import base64
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np

SEP = '<SEP>'
DEFAULT_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'


def load_project_env():
    """Load .env without replacing explicitly exported environment variables."""
    candidates = [Path(__file__).resolve().parent / '.env', Path.cwd() / '.env']
    for env_path in dict.fromkeys(candidates):
        if not env_path.is_file():
            continue
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            if not os.environ.get('OPENAI_API_KEY'):
                raise RuntimeError('Found .env but python-dotenv is missing: pip install python-dotenv') from exc
            return
        load_dotenv(dotenv_path=env_path, override=False)



def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_json(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)


def read_vectors(path):
    raw = load_json(path)
    rows = raw['data']
    dim = int(raw['embedding_dim'])
    b = base64.b64decode(raw['matrix'])
    a = np.frombuffer(b, dtype='<f4')
    if a.size != len(rows) * dim:
        raise ValueError(f'Malformed vector store: {path}, expected {len(rows)}x{dim} float32 values')
    return rows, a.reshape(len(rows), dim).copy()


def normalized(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)


def norm_name(x):
    return str(x).strip().strip('"').strip().casefold()


def split_sources(v):
    return [s.strip() for s in str(v or '').split(SEP) if s.strip()]


def ollama_chat(url, model, messages, max_tokens=400, temperature=0):
    req = urllib.request.Request(
        url.rstrip('/') + '/api/chat',
        data=json.dumps({'model':model,'stream':False,'messages':messages,
                         'options':{'temperature':temperature,'num_predict':max_tokens,'num_ctx':8192}}).encode(),
        headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=300) as response:
        ans = json.load(response)
    return (ans.get('message') or {}).get('content','').strip()


def openai_label_chat(model, prompt, base_url='https://api.openai.com/v1'):
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set')
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'max_tokens': 60,
    }
    req = urllib.request.Request(
        base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    return (data['choices'][0]['message'].get('content') or '').strip()


def compact_title(text):
    first = str(text).split('\n', 1)[0].strip()
    if first.lower().startswith('title:'):
        return first[6:].strip()[:135]
    return first[:135]


def describe_cluster(docids, chunks, n=12, full_docs=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Only genuine document titles are passed to the labeler. A continuation
    # chunk's first line is not a title and previously produced misleading labels.
    full_docs = full_docs or {}
    titles = []
    seen_docs = set()
    for k in docids:
        if k not in chunks:
            continue
        doc_id = chunks[k].get('full_doc_id')
        if doc_id and doc_id in seen_docs:
            continue
        doc_text = (full_docs.get(doc_id) or {}).get('content', '')
        first = str(doc_text).split('\n', 1)[0].strip()
        if first.lower().startswith('title:'):
            title = first[6:].strip()[:135]
        else:
            first_chunk = str(chunks[k].get('content', '')).split('\n', 1)[0].strip()
            title = first_chunk[6:].strip()[:135] if first_chunk.lower().startswith('title:') else ''
        if title:
            titles.append(title)
            if doc_id:
                seen_docs.add(doc_id)
    if not titles:
        return 'Unlabeled concept', []
    try:
        v = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=3000, min_df=1)
        X = v.fit_transform(titles)
        terms = v.get_feature_names_out()
        scores = np.asarray(X.mean(axis=0)).ravel()
        ranked = [terms[i] for i in np.argsort(scores)[::-1][:6]]
    except ValueError:
        ranked = []
    # A deterministic representative description is available even without Ollama.
    title = ' / '.join(ranked[:3]) if ranked else 'News articles'
    examples = titles[:n]
    return title, examples


def label_concept(fallback, titles, args, parent=None, entities=None):
    if args.label_backend == 'keywords' or not titles:
        return fallback
    prompt = ('Assign a SHORT, meaningful English topical category label (2-5 words) '
              'for this group of news article titles. E.g. Sports, Technology and AI, '
              'Government and Politics, Business and Finance. Output ONLY the label. '
              'Do not fabricate a category unsupported by the titles.\n'
              f'Parent category (if any): {parent or "News"}\n'
              'Representative entities:\n' + ', '.join((entities or [])[:24]) + '\n' +
              'Article titles:\n' + '\n'.join('- '+t for t in titles[:12]))
    try:
        if args.label_backend == 'openai':
            response = openai_label_chat(args.label_model, prompt, args.openai_base_url)
        else:
            response = ollama_chat(args.ollama_url,args.label_model,
                                   [{'role':'user','content':prompt}],max_tokens=50)
        response = response.strip(' \"\'\n.:-')[:75]
        return response if response and '\n' not in response else fallback
    except Exception as e:
        if args.label_backend == 'openai':
            raise RuntimeError(f'OpenAI concept labeling failed: {e}') from e
        print(f'[WARN] Concept label API failed ({e}); using keyword label',file=sys.stderr)
        return fallback


def index_fingerprint(index):
    # Verify the tree against its source index without hashing multi-GB files.
    wanted=['graph_chunk_entity_relation.graphml','vdb_chunks.json','kv_store_text_chunks.json']
    ans={}
    for item in wanted:
        p=index/item
        if not p.exists():
            raise FileNotFoundError(f'Missing MiniRAG index file: {p}')
        st=p.stat()
        ans[item]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    return ans


def canonical_label(label):
    """Conservative lexical canonicalization: never guess synonym equivalence."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w ]+', ' ', str(label).casefold())).strip()


def harmonize_tree(tree, vectors, chunk_ids, centers):
    """Merge exact duplicate categories; keep ALL original chunks and graph links.

    Parent nodes are grouped globally under the root. Leaf labels are merged
    only within their parent, to avoid treating unrelated homonymous labels as
    interchangeable concepts. Each merged centroid is recomputed from original
    chunk vectors, not averaged from pre-normalized cluster centroids.
    """
    nodes=tree['nodes']
    root=tree['root']
    old_count=len(nodes)
    parents=list(nodes[root]['children'])
    by_name={}
    retained=[]
    merges=[]
    for pid in parents:
        if pid not in nodes:
            continue
        key=canonical_label(nodes[pid]['name'])
        if not key:
            key=pid
        if key in by_name:
            keeper=by_name[key]
            for child in nodes[pid]['children']:
                nodes[child]['parent']=keeper
                nodes[keeper]['children'].append(child)
            merges.append({'removed':pid,'kept':keeper,'label':nodes[keeper]['name'],'level':1})
            del nodes[pid]
        else:
            by_name[key]=pid
            retained.append(pid)
    nodes[root]['children']=retained
    # Merge ONLY leaves sharing the same lexical label AND parent.
    for pid in retained:
        unique=[]
        found={}
        for lid in list(nodes[pid]['children']):
            key=canonical_label(nodes[lid]['name']) or lid
            if key in found:
                keep=found[key]
                tree['leaves'][keep].extend(tree['leaves'].pop(lid))
                merges.append({'removed':lid,'kept':keep,'label':nodes[keep]['name'],'level':2})
                del nodes[lid]
            else:
                found[key]=lid
                unique.append(lid)
        nodes[pid]['children']=unique
        # A child with the same name as its parent is a redundant label.
        for lid in unique:
            if canonical_label(nodes[lid]['name']) == canonical_label(nodes[pid]['name']):
                nodes[lid]['name']='General '+nodes[pid]['name']
    tree['chunk_to_leaf']={}
    for lid,ids in tree['leaves'].items():
        ids=list(dict.fromkeys(ids))
        tree['leaves'][lid]=ids
        nodes[lid]['size']=len(ids)
        for cid in ids:
            if cid in tree['chunk_to_leaf']:
                raise ValueError('Chunk assigned to multiple leaves: '+cid)
            tree['chunk_to_leaf'][cid]=lid
    if len(tree['chunk_to_leaf'])!=len(chunk_ids):
        raise ValueError(f'Lost chunks in harmonization: {len(tree["chunk_to_leaf"])} != {len(chunk_ids)}')
    # Exact recomputation of every surviving centroid and aggregate size.
    positions={k:i for i,k in enumerate(chunk_ids)}
    new_centers={}
    for pid in retained:
        members=[]
        for lid in nodes[pid]['children']:
            ids=tree['leaves'][lid]
            members.extend(ids)
            new_centers[lid]=normalized(np.mean(vectors[[positions[k] for k in ids]],axis=0)).tolist()
        nodes[pid]['size']=len(members)
        new_centers[pid]=normalized(np.mean(vectors[[positions[k] for k in members]],axis=0)).tolist()
    centers.clear();centers.update(new_centers)
    tree['version']=2
    tree['harmonization']={
        'strategy':'exact_label_merge',
        'nodes_before':old_count,
        'nodes_after':len(nodes),
        'merged':merges,
        'caution':'Labels generated by the LLM are not validated semantic categories; identical names may combine heterogeneous clusters.'
    }
    return tree


def repair_tree(args):
    """Repair an already-built tree WITHOUT rerunning KMeans or Ollama.

    Read-only original index; writes a separate repaired directory by default.
    """
    index=Path(args.working_dir)
    source=Path(args.source_tree or index/'concept_tree_rag'/'concept_tree.json')
    tree=load_json(source)
    if tree['fingerprint'] != index_fingerprint(index):
        print('[WARN] Index file size/timestamp differs from the original tree. Verifying chunk IDs instead.', file=sys.stderr)
    rows,matrix=read_vectors(index/'vdb_chunks.json')
    ids=[r['__id__'] for r in rows if r['__id__'] in tree['chunk_to_leaf']]
    if set(ids)!=set(tree['chunk_to_leaf']):
        raise RuntimeError('Chunk IDs are missing or different; cannot safely repair this tree.')
    # A repaired tree is subsequently checked against the locally used index.
    tree['fingerprint']=index_fingerprint(index)
    idx={r['__id__']:i for i,r in enumerate(rows)}
    V=normalized(matrix[[idx[k] for k in ids]])
    centers={}
    harmonize_tree(tree,V,ids,centers)
    out=Path(args.tree_dir or index/'concept_tree_rag_v2')
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'concept_tree.json',tree)
    np.savez_compressed(out/'centroids.npz',**{k:np.array(v,dtype=np.float32) for k,v in centers.items()})
    print(f'[Repair] original tree: {source}')
    print(f'[Repair] merged {len(tree["harmonization"]["merged"])} duplicate nodes; '
          f'kept {len(tree["chunk_to_leaf"])} chunks; nodes now {len(tree["nodes"])}')
    print(f'[Repair] new tree: {out.resolve()}')


def build(args):
    from sklearn.cluster import KMeans
    index=Path(args.working_dir)
    chunks=load_json(index/'kv_store_text_chunks.json')
    full_docs=load_json(index/'kv_store_full_docs.json')
    records,matrix=read_vectors(index/'vdb_chunks.json')
    used=[r['__id__'] for r in records if r['__id__'] in chunks]
    if not used:
        raise RuntimeError('No overlap between chunk embeddings and chunk KV store')
    emb_by_id={r['__id__']:i for i,r in enumerate(records)}
    V=normalized(matrix[[emb_by_id[k] for k in used]])
    # Read original entity graph and associate entities with chunk concepts.
    graph=nx.read_graphml(index/'graph_chunk_entity_relation.graphml')
    chunk_ent=collections.defaultdict(collections.Counter)
    for entity,attr in graph.nodes(data=True):
        clean=norm_name(entity)
        for chunk_id in split_sources(attr.get('source_id')):
            if chunk_id in emb_by_id:
                chunk_ent[chunk_id][clean]+=1
    def frequent_entities(chunk_ids, limit=30):
        counts=collections.Counter()
        for ck in chunk_ids:
            counts.update(chunk_ent.get(ck,{}))
        # Drop short or generic graph artifacts.
        return [k for k,v in counts.most_common() if len(k)>=4 and not k.startswith('unknown')][:limit]
    if args.label_backend == 'openai' and not os.environ.get('OPENAI_API_KEY', '').strip():
        raise RuntimeError('OPENAI_API_KEY must be set when --label-backend openai')
    # Topic grouping uses original MiniRAG chunk embeddings, NO API calls.
    broad_n=min(args.broad, len(used))
    broad=KMeans(n_clusters=broad_n,random_state=args.seed,n_init=10).fit_predict(V)
    tree={'root':'news','nodes':{'news':{'name':'News','parent':None,'level':0,'children':[]}},
          'leaves':{},'chunk_to_leaf':{},'version':1,
          'embedding_model':args.embedding_model,
          'fingerprint':index_fingerprint(index),
          'build_args':{'broad':args.broad,'fine':args.fine,'seed':args.seed,
                        'label_backend':args.label_backend,'label_model':args.label_model}}
    centers={}
    for i in range(broad_n):
        bidx=np.where(broad==i)[0]
        ids=[used[j] for j in bidx]
        # Use representative chunks nearest the cluster centroid, not arbitrary
        # index order, when obtaining a human-readable label.
        center = normalized(np.mean(V[bidx], axis=0))
        representative_idx = bidx[np.argsort(V[bidx] @ center)[::-1]]
        ordered_ids = [used[z] for z in representative_idx]
        fallback,examples=describe_cluster(ordered_ids,chunks,full_docs=full_docs)
        bid=f'topic-{i:02d}'
        entities=frequent_entities(ids)
        lbl=label_concept(fallback,examples,args,entities=entities)
        tree['nodes'][bid]={'name':lbl,'parent':'news','level':1,'children':[],
                            'size':len(ids),'keywords':fallback,'example_titles':examples[:6],
                            'representative_entities':entities[:20]}
        tree['nodes']['news']['children'].append(bid)
        centers[bid]=normalized(np.mean(V[bidx],axis=0)).tolist()
        fine_n=min(args.fine,len(bidx))
        # avoid artificially splitting small groups into singleton concepts
        if len(bidx)<max(8,2*fine_n): fine_n=1
        if fine_n==1: fine=np.zeros(len(bidx),dtype=int)
        else: fine=KMeans(n_clusters=fine_n,random_state=args.seed,n_init=10).fit_predict(V[bidx])
        for j in range(fine_n):
            lidx=bidx[fine==j]
            keys=[used[k] for k in lidx]
            leaf_center = normalized(np.mean(V[lidx],axis=0))
            representative_leaf_idx = lidx[np.argsort(V[lidx] @ leaf_center)[::-1]]
            ordered_leaf_ids = [used[z] for z in representative_leaf_idx]
            lf,ex=describe_cluster(ordered_leaf_ids,chunks,full_docs=full_docs)
            lid=f'{bid}-sub-{j:02d}'
            leaf_entities=frequent_entities(keys)
            lname=label_concept(lf,ex,args,parent=lbl,entities=leaf_entities)
            tree['nodes'][lid]={'name':lname,'parent':bid,'level':2,
                                'children':[],'size':len(keys),'keywords':lf,'example_titles':ex[:5],
                                'representative_entities':leaf_entities[:20]}
            tree['nodes'][bid]['children'].append(lid)
            tree['leaves'][lid]=keys
            for k in keys: tree['chunk_to_leaf'][k]=lid
            centers[lid]=normalized(np.mean(V[lidx],axis=0)).tolist()
    # Merge categories labeled identically, and update memberships and centroids.
    harmonize_tree(tree, V, used, centers)
    folder=Path(args.tree_dir or index/'concept_tree_rag')
    folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'concept_tree.json',tree)
    np.savez_compressed(folder/'centroids.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in centers.items()})
    print(f'[Build] source chunks: {len(used)} | topics: {broad_n} | leaf concepts: {len(tree["leaves"])}')
    print(f'[Build] saved to: {folder.resolve()}')
    for bid in tree['nodes']['news']['children']:
        print(f'  {bid}: {tree["nodes"][bid]["name"]} ({tree["nodes"][bid]["size"]} chunks)')
        for lid in tree['nodes'][bid]['children']:
            print(f'    {lid}: {tree["nodes"][lid]["name"]} ({tree["nodes"][lid]["size"]})')


class ConceptRetriever:
    def __init__(self,args):
        self.args=args
        self.index=Path(args.working_dir)
        folder=Path(args.tree_dir or self.index/'concept_tree_rag')
        self.tree=load_json(folder/'concept_tree.json')
        if self.tree['fingerprint'] != index_fingerprint(self.index):
            raise RuntimeError('Index files changed since the tree was built. Re-run build.')
        if self.tree['embedding_model']!=args.embedding_model:
            raise RuntimeError('Embedding model differs from the model used to build the tree')
        self.centers=dict(np.load(folder/'centroids.npz'))
        from sentence_transformers import SentenceTransformer
        self.encoder=SentenceTransformer(args.embedding_model,device=args.embedding_device)
        self.chunks=load_json(self.index/'kv_store_text_chunks.json')
        rows,mat=read_vectors(self.index/'vdb_chunks.json')
        self.chunk_ids=[r['__id__'] for r in rows]
        self.chunk_ix={k:i for i,k in enumerate(self.chunk_ids)}
        self.chunk_mat=normalized(mat)
        del mat
        # Existing MiniRAG entity-name embeddings (same embedding model).
        erows,emat=read_vectors(self.index/'vdb_entities_name.json')
        self.entity_names=[norm_name(r.get('entity_name','')) for r in erows]
        self.entity_mat=normalized(emat)
        del emat
        self.graph=nx.read_graphml(self.index/'graph_chunk_entity_relation.graphml')
        # GraphML uses quoted strings as identifiers; normalize once.
        self.node_lookup={norm_name(k):k for k in self.graph.nodes}
        self.entity_chunks={}
        self.chunk_entities=collections.defaultdict(set)
        for ent,data in self.graph.nodes(data=True):
            name=norm_name(ent)
            src={x for x in split_sources(data.get('source_id')) if x in self.chunk_ix}
            self.entity_chunks[name]=src
            for ck in src: self.chunk_entities[ck].add(name)
        # Each entity may belong to many leaves, based on entity-source chunk links.
        self.entity_leaves={ent:{self.tree['chunk_to_leaf'][ck] for ck in sources
                                 if ck in self.tree['chunk_to_leaf']}
                            for ent,sources in self.entity_chunks.items()}
        self.last_diagnostic={}

    def embed(self,text):
        return normalized(self.encoder.encode([text],convert_to_numpy=True,show_progress_bar=False))[0]

    def _concepts(self,qvec):
        nodes=self.tree['nodes']
        bids=self.tree['nodes']['news']['children']
        ranked=sorted(bids,key=lambda k:float(np.dot(qvec,self.centers[k])),reverse=True)
        chosen_bids=ranked[:self.args.top_topics]
        leaves=[lid for b in chosen_bids for lid in nodes[b]['children']]
        ranked_leaves=sorted(leaves,key=lambda k:float(np.dot(qvec,self.centers[k])),reverse=True)
        picked=ranked_leaves[:self.args.top_leaves]
        return picked, [{'id':x,'label':nodes[x]['name'],
                         'score':round(float(np.dot(qvec,self.centers[x])),4)} for x in picked]

    def retrieve(self, question):
        """Independent research variant. Legacy retrieval available for ablation."""
        if self.args.retrieval_policy == 'legacy':
            return self._retrieve_legacy(question)
        return self._retrieve_enhanced(question)

    def _decompose_question(self, question, explicit_entities):
        """Deterministic, zero-API-cost requirement decomposition.

        Clause evidence requirements are formed without using ground truth or
        dataset question_type. This is deliberately conservative: weak fragments
        are ignored, and the full original question remains the primary query.
        """
        text = re.sub(r"\s+", " ", question).strip()
        parts = re.split(
            r"(?i)\s*[,;]\s*(?:while|whereas|and (?:the|a|an) |but |however,? |"
            r"in contrast to |compared (?:to|with) |subsequently |and subsequently )|"
            r"\s+(?:while|whereas|compared (?:to|with)|in contrast to)\s+",
            text,
        )
        # Date windows and publisher-specific clauses often make up different
        # pieces of evidence in MultiHop questions.
        clauses = [p.strip(' ,;?.') for p in parts
                   if len(p.strip()) >= 40 and len(p.split()) >= 7]
        # Avoid meaningless decomposition for short/single-fact questions.
        if len(clauses) < 2:
            clauses = []
        # Exact entities are the most reliable separable requirements.
        entities = list(dict.fromkeys(explicit_entities))[:3]
        facets = []
        for ent in entities:
            facets.append({'kind':'entity', 'label':ent,
                           'query':f'{ent}: {text}'})
        # Ensure the clauses are differentiated and don't overwhelm entities.
        for clause in clauses[:3]:
            if len(facets) >= self.args.max_facets:
                break
            facets.append({'kind':'clause', 'label':clause[:110],
                           'query':clause})
        # If there is no clear decomposition, original full query only.
        if not facets:
            facets = [{'kind':'whole', 'label':'whole_question', 'query':text}]
        return facets[:self.args.max_facets]

    def _coverage_rerank(self, ids, indices, scores, cosine, graph_score,
                         question, explicit, qvec):
        """Coverage-aware greedy selection from source-backed retrieved chunks.

        Selection uses only embeddings, source links, entity mentions, and
        document IDs. Does not inspect gold answers or QA judgments.
        """
        args=self.args
        # Candidates always begin with the same base retrieval as V3.
        base_order=np.argsort(-scores,kind='stable')
        facets=self._decompose_question(question,explicit)
        fvecs=normalized(self.encoder.encode(
            [f['query'] for f in facets],
            convert_to_numpy=True,show_progress_bar=False))
        facet_sim=(self.chunk_mat[indices] @ fvecs.T).astype(float)
        # Add a few high-quality per-facet candidates even if they weren't in
        # the global Top-60; cap by retrieval_top_k per facet.
        base=set(int(i) for i in base_order[:max(args.retrieval_top_k,args.top_k)])
        for j in range(len(facets)):
            best=np.argsort(-facet_sim[:,j],kind='stable')[:args.facet_candidate_k]
            base.update(int(i) for i in best)
        shortlist=sorted(base,key=lambda pos:(-float(scores[pos]),ids[pos]))
        # Prefetch source-based binary entity coverage once.
        facet_entity=[]
        for f in facets:
            ent=f['label'] if f['kind']=='entity' else None
            facet_entity.append(self.entity_chunks.get(ent,set()) if ent else set())
        chosen=[]
        chosen_docs=collections.Counter()
        covered=np.zeros(len(facets),dtype=float)
        selected_vec=[]
        per_chunk=[]
        max_base=max(float(scores[i]) for i in shortlist)
        for iteration in range(min(args.top_k,len(shortlist))):
            possibilities=[]
            for pos in shortlist:
                if pos in chosen: continue
                ck=ids[pos]
                doc=str(self.chunks[ck].get('full_doc_id') or ck)
                # Only include an entity-match bonus if the actual chunk is
                # linked to the corresponding entity in the existing graph.
                sim=np.maximum(0.0, facet_sim[pos])
                linked=np.array([1.0 if ck in cs else 0.0 for cs in facet_entity])
                requirement_strength=np.maximum(sim,linked)
                coverage_gain=float(np.mean(np.maximum(0.0,requirement_strength-covered)))
                redundancy=(max([float(self.chunk_mat[indices[pos]]@v) for v in selected_vec])
                            if selected_vec else 0.0)
                doc_penalty=max(0,chosen_docs[doc] - (args.max_chunks_per_doc-1))
                # Preserve relevance dominance. Coverage gain and redundancy
                # matter only among evidence with reasonable base relevance.
                score=(float(scores[pos])
                       + args.coverage_weight*coverage_gain
                       - args.redundancy_weight*max(0.0,redundancy)
                       - args.same_doc_penalty*doc_penalty)
                possibilities.append((score,float(scores[pos]),ids[pos],pos,
                                      requirement_strength,doc,coverage_gain,redundancy))
            if not possibilities: break
            possibilities.sort(key=lambda v:(-v[0],-v[1],v[2]))
            value,original,ck,pos,strength,doc,gain,redundancy=possibilities[0]
            chosen.append(pos)
            covered=np.maximum(covered,strength)
            chosen_docs[doc]+=1
            selected_vec.append(self.chunk_mat[indices[pos]])
            per_chunk.append({'chunk_id':ck,'greedy_score':round(float(value),5),
                              'base_score':round(float(original),5),
                              'coverage_gain':round(gain,5),
                              'redundancy':round(redundancy,5),
                              'covered_facets': [facets[j]['label'] for j,v in enumerate(strength)
                                                 if v>=args.facet_match_threshold]})
        diag={'query_facets':[{**f} for f in facets],
              'facet_candidate_total':len(shortlist),
              'facet_final_coverage':[{**facets[j], 'strength':round(float(v),5),
                                       'covered':bool(v>=args.facet_match_threshold)}
                                      for j,v in enumerate(covered)],
              'coverage_selection_trace':per_chunk,
              'rerank_hyperparameters':{
                  'coverage_weight':args.coverage_weight,
                  'redundancy_weight':args.redundancy_weight,
                  'same_doc_penalty':args.same_doc_penalty,
                  'facet_candidate_k':args.facet_candidate_k,
                  'facet_match_threshold':args.facet_match_threshold}}
        return chosen,diag

    def _retrieve_enhanced(self, question):
        """Concept-guided multi-entity graph retrieval and diverse vector ranking.

        - Three broad concepts / five leaves still determine the main search space.
        - Explicit question entities can contribute one-hop, cross-leaf evidence.
        - Retrieval top-k limits the INTERNAL ranked pool, while top-k limits
          final evidence sent to the answer model. These are different concepts.
        - No gold answers, query types, or reference evidence are consulted.
        """
        args = self.args
        qvec = self.embed(question)
        leaves, concepts = self._concepts(qvec)
        selected = set(leaves)
        main_pool = {ck for lid in leaves for ck in self.tree['leaves'].get(lid, ())
                     if ck in self.chunk_ix}
        if not main_pool:
            raise RuntimeError('No chunks in selected concept leaves')
        allowed = {ent for ent, ls in self.entity_leaves.items() if ls & selected}

        # Global exact-phrase matching is important: otherwise an entity missed
        # by the concept router is never considered, even when explicitly named.
        qtext = ' ' + re.sub(r'[^\w ]', ' ', question.casefold()) + ' '
        candidates = set(self.entity_chunks) | set(self.entity_names)
        explicit = []
        for ent in candidates:
            if len(ent) < 5 or len(ent.split()) > 6 or len(ent) > 90:
                continue
            phrase = ' ' + re.sub(r'[^\w ]', ' ', ent) + ' '
            if phrase in qtext:
                explicit.append(ent)
        # Prefer long, specific named entities rather than common generic terms.
        explicit.sort(key=lambda x: (-len(x.split()), -len(x), x))
        explicit = explicit[:args.top_entities]

        similarities = self.entity_mat @ qvec
        n = min(len(similarities), max(args.entity_pool, args.top_entities * 12))
        ranked_entities = (np.argsort(similarities)[-n:][::-1] if n else [])
        dense = []
        for i in ranked_entities:
            ent = self.entity_names[int(i)]
            if ent in allowed and ent not in explicit and ent not in dense:
                dense.append(ent)
            if len(dense) >= args.top_entities:
                break
        seeds = list(dict.fromkeys(explicit + dense))[:args.top_entities]
        graph_score = collections.defaultdict(float)
        edges = []
        cross_chunks = set()
        seed_chunks = collections.defaultdict(set)

        for ent in seeds:
            directly_linked = self.entity_chunks.get(ent, set())
            for ck in directly_linked:
                if ck not in self.chunk_ix:
                    continue
                if ck in main_pool or ent in explicit:
                    graph_score[ck] += 1.0
                    seed_chunks[ent].add(ck)
                    if ck not in main_pool:
                        cross_chunks.add(ck)

            original = self.node_lookup.get(ent)
            if original is None:
                continue
            neighbors = sorted(self.graph[original].items(),
                               key=lambda pair: float(pair[1].get('weight', 1) or 1),
                               reverse=True)
            for neighbor, edata in neighbors[:args.neighbors_per_entity]:
                target = norm_name(neighbor)
                weight = max(0.0, float(edata.get('weight', 1) or 1))
                increment = 0.25 * min(2.0, math.log1p(weight))
                for ck in split_sources(edata.get('source_id')):
                    if ck not in self.chunk_ix:
                        continue
                    # Cross-concept exploration is not a global search: only
                    # evidence connected to an explicit question entity crosses.
                    if ck in main_pool or ent in explicit:
                        graph_score[ck] += increment
                        seed_chunks[ent].add(ck)
                        if ck not in main_pool:
                            cross_chunks.add(ck)
                if len(edges) < 40:
                    edges.append({'from': ent, 'to': target, 'weight': weight,
                                  'description': str(edata.get('description', '')).split(SEP)[0][:180]})

        # Limit cross-leaf expansion independently of final answer evidence.
        # Prioritize vector similarity to the question, with a small graph bonus.
        if len(cross_chunks) > args.max_cross_chunks:
            ids = list(cross_chunks)
            indices = np.asarray([self.chunk_ix[ck] for ck in ids], dtype=int)
            scores = self.chunk_mat[indices] @ qvec
            keep = np.argsort(scores)[::-1][:args.max_cross_chunks]
            cross_chunks = {ids[int(i)] for i in keep}

        pool = main_pool | cross_chunks
        ids = sorted(pool)
        indices = np.asarray([self.chunk_ix[ck] for ck in ids], dtype=int)
        cosine = self.chunk_mat[indices] @ qvec
        scores = cosine + np.asarray([args.graph_weight * min(1.0, graph_score.get(ck, 0.0))
                                     for ck in ids], dtype=np.float32)
        # An independent internal candidate ceiling; never silently changes
        # --top-k (the final number of chunks shown to the LLM).
        k = min(len(ids), max(args.retrieval_top_k, args.top_k))
        chosen_positions, coverage_diagnostic = self._coverage_rerank(
            ids, indices, scores, cosine, graph_score, question, explicit, qvec)
        found = []
        for pos in chosen_positions:
            ck = ids[pos]
            leaf = self.tree['chunk_to_leaf'].get(ck)
            found.append({'chunk_id': ck,
                          'text': self.chunks[ck]['content'],
                          'concept': self.tree['nodes'][leaf]['name'] if leaf else '',
                          'leaf_id': leaf,
                          'selected_concept': leaf in selected,
                          'cross_concept': ck not in main_pool,
                          'cosine': round(float(cosine[pos]), 4),
                          'graph_score': round(float(graph_score.get(ck, 0)), 4),
                          'score': round(float(scores[pos]), 4),
                          'document_id': self.chunks[ck].get('full_doc_id')})
        self.last_diagnostic = {
            'retrieval_policy': 'v5_coverage', 'concepts': concepts,
            **coverage_diagnostic,
            'entity_seeds': seeds, 'explicit_entities': explicit,
            'graph_edges': edges, 'candidate_chunks': len(ids),
            'selected_concept_candidate_chunks': len(main_pool),
            'cross_concept_candidate_chunks': len(cross_chunks),
            'internal_ranked_chunks': k,
            'final_evidence_chunks': len(found),
            'final_unique_documents': len({r['document_id'] for r in found}),
            'final_cross_concept_chunks': sum(bool(r['cross_concept']) for r in found),
            'entity_source_chunk_counts': {ent: len(seed_chunks[ent]) for ent in seeds},
            'final_context_chars_raw': sum(len(str(r['text'])) for r in found)
        }
        return found

    def _retrieve_legacy(self,question):
        args=self.args
        q=self.embed(question)
        leaves,concepts=self._concepts(q)
        chosen=set(leaves)
        # Only entities whose *sourced chunks* belong to a selected concept subtree.
        allowed={name for name,ls in self.entity_leaves.items() if ls & chosen}
        # Dense entity matching uses the PRE-EXISTING index name vectors.
        sim=self.entity_mat@q
        # Large candidate pool before concept filtering prevents dropping valid entities.
        count=min(len(sim),max(args.entity_pool, args.top_entities*12))
        inds=np.argpartition(sim,-count)[-count:]
        inds=sorted(inds,key=lambda i:sim[i],reverse=True)
        chosen_entities=[]
        for i in inds:
            name=self.entity_names[i]
            if name in allowed and name not in chosen_entities:
                chosen_entities.append(name)
            if len(chosen_entities)>=args.top_entities: break
        # Exact entity matches within the question can rescue missed embedding matches.
        qtext=' '+re.sub(r'[^\w ]',' ',question.casefold())+' '
        phrase_matches=[]
        for ent in allowed:
            if len(ent)>=5 and len(ent.split())<=6 and (' '+ent+' ') in qtext:
                phrase_matches.append(ent)
        phrase_matches=sorted(phrase_matches,key=lambda e:(-len(e),e))[:args.top_entities]
        seeds=list(dict.fromkeys(phrase_matches+chosen_entities))[:args.top_entities]
        graph_chunk_scores=collections.defaultdict(float)
        graph_edges=[]
        for ent in seeds:
            for ck in self.entity_chunks.get(ent,()):
                if self.tree['chunk_to_leaf'].get(ck) in chosen:
                    graph_chunk_scores[ck]+=1.0
            original=self.node_lookup.get(ent)
            if original is None: continue
            # Preserve source graph edges. Expansion is ONE hop only.
            connections=sorted(self.graph[original].items(),
                               key=lambda pair:float(pair[1].get('weight',1) or 1),reverse=True)
            for neighbor,edata in connections[:args.neighbors_per_entity]:
                target=norm_name(neighbor)
                weight=max(0.0,float(edata.get('weight',1) or 1))
                w=0.25*min(2.0,math.log1p(weight))
                edge_src=[ck for ck in split_sources(edata.get('source_id'))
                          if ck in self.chunk_ix]
                # First favor evidence in the selected conceptual subgraphs.
                for ck in edge_src:
                    if self.tree['chunk_to_leaf'].get(ck) in chosen:
                        graph_chunk_scores[ck]+=w
                if len(graph_edges)<40:
                    graph_edges.append({'from':ent,'to':target,'weight':weight,
                                        'description':str(edata.get('description','')).split(SEP)[0][:180]})
        # Chunk candidates come from the selected subtree; graph acts as an evidence boost.
        pool=set()
        for leaf in leaves: pool.update(self.tree['leaves'][leaf])
        pool.update(graph_chunk_scores)
        if not pool:
            raise RuntimeError('Selected concept leaves returned no chunks')
        chunk_indices=np.fromiter((self.chunk_ix[c] for c in pool if c in self.chunk_ix),dtype=int)
        cos=self.chunk_mat[chunk_indices] @ q
        # Both scores are in known comparable ranges. Graph contributes to ranking,
        # not to writing claims absent from the retrieved text.
        scores=cos+np.array([args.graph_weight*min(1.0,graph_chunk_scores.get(self.chunk_ids[i],0))
                             for i in chunk_indices],dtype=np.float32)
        ranked=np.argsort(scores)[::-1][:args.top_k]
        found=[]
        for z in ranked:
            i=int(chunk_indices[z]);ck=self.chunk_ids[i]
            found.append({'chunk_id':ck,'text':self.chunks[ck]['content'],
                          'concept':self.tree['nodes'][self.tree['chunk_to_leaf'][ck]]['name'],
                          'cosine':round(float(cos[z]),4),
                          'graph_score':round(float(graph_chunk_scores.get(ck,0)),4),
                          'score':round(float(scores[z]),4),
                          'document_id':self.chunks[ck].get('full_doc_id')})
        self.last_diagnostic={'concepts':concepts,'entity_seeds':seeds,
                              'graph_edges':graph_edges,'candidate_chunks':len(chunk_indices)}
        return found


def build_balanced_context(evidence, max_context_chars, max_chunk_chars):
    """Fit every selected evidence item into a fixed *character* budget.

    Reserve space for headings and separators, share the remaining characters
    across all retrieved chunks, then redistribute unused space in rank order.
    This does not change retrieved chunks or their order; it only fixes a
    first-come-first-served truncation bias.
    """
    rows = list(evidence)
    if not rows:
        return '', []
    n = len(rows)
    headers = [f'[Evidence {i} | chunk_id={row["chunk_id"]}]\n' for i,row in enumerate(rows,1)]
    separator = '\n\n'
    reserved = sum(len(h) for h in headers) + len(separator) * (n - 1)
    available = int(max_context_chars) - reserved
    if available < n:
        raise ValueError(f'--max-context-chars={max_context_chars} too small to include {n} evidence headers and content')
    texts = [str(row.get('text') or '')[:max_chunk_chars] for row in rows]
    caps = [len(t) for t in texts]
    # Phase 1: equal minimum budget; Phase 2: redistribute leftover chars
    # to the higher-ranked evidence items that still have available text.
    base, remainder = divmod(available, n)
    allocations = [min(caps[i], base + (i < remainder)) for i in range(n)]
    rest = available - sum(allocations)
    for i in range(n):
        additional = min(rest, caps[i] - allocations[i])
        allocations[i] += additional
        rest -= additional
        if rest == 0:
            break
    excerpts = [headers[i] + texts[i][:allocations[i]] for i in range(n)]
    context = separator.join(excerpts)
    assert len(context) <= max_context_chars
    details = [{'chunk_id':row['chunk_id'], 'document_id':row.get('document_id'),
                'original_chars':len(str(row.get('text') or '')),
                'included_chars':allocations[i]} for i,row in enumerate(rows)]
    return context, details


def local_answer(args,question,evidence,system_prompt=None):
    """Same V5 HF answer logic, with stage-accurate timing (synchronizes MPS)."""
    timing = {'context_seconds': None, 'model_load_seconds': 0.0,
              'tokenization_seconds': None, 'transfer_seconds': None,
              'generation_seconds': None, 'decode_seconds': None,
              'generated_tokens': None, 'model_loaded_this_question': False}
    local_answer.last_timing = timing
    t = time.perf_counter()
    context, included = build_balanced_context(evidence,args.max_context_chars,args.max_chunk_chars)
    timing['context_seconds'] = time.perf_counter() - t
    local_answer.last_context_chars = len(context)
    local_answer.last_context_evidence = included
    local_answer.last_input_tokens = None
    system=('Answer the question using ONLY the provided evidence. '
            'When multiple documents are needed, combine information carefully. '
            'If evidence does not support an answer, respond "Insufficient information." '
            'Do not invent facts or cite nonexistent sources. Be concise.')
    if system_prompt is not None:
        system = system_prompt
    user=f'Question: {question}\n\nEvidence:\n'+context+'\n\nAnswer:'
    if args.backend=='ollama':
        return ollama_chat(args.ollama_url,args.qa_model,
                           [{'role':'system','content':system},{'role':'user','content':user}],
                           max_tokens=args.max_new_tokens)
    if args.backend == 'hf':
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not hasattr(local_answer, 'hf_cache'):
            t = time.perf_counter()
            preference = args.hf_device
            mps_ready = bool(getattr(torch.backends, 'mps', None) and
                             torch.backends.mps.is_built() and
                             torch.backends.mps.is_available())
            if preference == 'auto':
                target = 'cuda' if torch.cuda.is_available() else ('mps' if mps_ready else 'cpu')
            else:
                target = preference
            if target == 'mps' and not mps_ready:
                raise RuntimeError('MPS unavailable; install MPS-enabled PyTorch')
            if target == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            device = torch.device(target)
            dtype = torch.float16 if target in ('mps', 'cuda') else torch.float32
            print(f'[HF] Loading {args.qa_model} on {device} with {dtype}', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.qa_model, trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.qa_model, torch_dtype=dtype, trust_remote_code=False,
                    low_cpu_mem_usage=True)
            except ImportError as exc:
                raise RuntimeError('Install accelerate: pip install accelerate') from exc
            model = model.to(device).eval()
            actual_device = next(model.parameters()).device
            if actual_device.type != device.type:
                raise RuntimeError(f'Model on {actual_device}, expected {device}')
            local_answer.hf_cache = (tokenizer, model, device)
            timing['model_load_seconds'] = time.perf_counter() - t
            timing['model_loaded_this_question'] = True
            print(f'[HF] Model loaded in {timing["model_load_seconds"]:.2f}s', flush=True)
        tokenizer, model, device = local_answer.hf_cache
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}]
        t = time.perf_counter()
        # Model-level fairness: Qwen3 uses the same non-thinking chat template
        # as MiniRAG and Standard. Original system/user prompts are unchanged.
        chat_opts = {'enable_thinking': False} if args.qa_model == 'Qwen/Qwen3-4B' else {}
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, **chat_opts)
        inputs = tokenizer(prompt, return_tensors='pt')
        local_answer.last_input_tokens = int(inputs['input_ids'].shape[-1])
        timing['tokenization_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if device.type == 'mps': torch.mps.synchronize()
        elif device.type == 'cuda': torch.cuda.synchronize(device)
        timing['transfer_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
        if device.type == 'mps': torch.mps.synchronize()
        elif device.type == 'cuda': torch.cuda.synchronize(device)
        timing['generation_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        answer_ids = generated[0, inputs['input_ids'].shape[-1]:]
        timing['generated_tokens'] = int(answer_ids.shape[-1])
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        timing['decode_seconds'] = time.perf_counter() - t
        return answer
    raise ValueError(f'Unknown backend {args.backend}')


def qa(args):
    index=Path(args.working_dir)
    if args.qa_file:
        source=Path(args.qa_file)
    else:
        source=Path(args.data_dir)/'MultiHopRAG.json'
    questions=load_json(source)
    if not isinstance(questions,list): raise ValueError('QA input must be a JSON list')
    retriever=ConceptRetriever(args)
    output=Path(args.output or index/'results'/f'concept_tree_{args.backend}_{args.qa_model.replace("/","_")}.jsonl')
    output.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if output.exists() and not args.overwrite:
        for line in output.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    obj=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not obj.get('error') and str(obj.get('model_answer','')).strip():
                    done.add(obj['question_id'])
    if args.overwrite: output.write_text('',encoding='utf-8')
    elif output.exists():
        # Clean previous failed/empty/duplicate records before resumption.
        good = {}
        for line in output.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if not row.get('error') and str(row.get('model_answer','')).strip():
                    good[row['question_id']] = row
            except (ValueError, KeyError, TypeError):
                pass
        with output.open('w', encoding='utf-8') as cleaned:
            for row in good.values():
                cleaned.write(json.dumps(row, ensure_ascii=False)+'\n')
        done = set(good)
    last=min(len(questions),args.start+args.max_questions) if args.max_questions>0 else len(questions)
    print(f'Concept-tree QA: {source} | records={len(questions)} | [{args.start},{last})')
    print(f'Output: {output.resolve()} | already completed={len(done)}')
    with output.open('a',encoding='utf-8') as dst:
        for i in range(args.start,last):
            if i in done: continue
            item=questions[i]
            question=str(item.get('query') or item.get('question') or '')
            if not question: continue
            start=time.monotonic();answer='';evidence=[];error='';diagnostic={}
            try:
                evidence=retriever.retrieve(question)
                diagnostic=retriever.last_diagnostic.copy()
                answer=local_answer(args,question,evidence)
            except Exception as e:
                error=f'{type(e).__name__}: {e}'
                print(f'[ERROR] #{i}: {error}',file=sys.stderr)
            row={'question_id':i,'query':question,'gold_answer':item.get('answer',''),
                 'question_type':item.get('question_type','unknown'),
                 'model_answer':answer,'mode':'concept_tree_graph_vector',
                 'top_k':args.top_k,'llm_backend':args.backend,'llm_model':args.qa_model,
                 'embedding_model':args.embedding_model,
                 'concepts':diagnostic.get('concepts',[]),
                 'entity_seeds':diagnostic.get('entity_seeds',[]),
                 'retrieval_policy':args.retrieval_policy,
                 'candidate_chunks':diagnostic.get('candidate_chunks',0),
                 'retrieval_diagnostic':diagnostic,
                 'context_chars':getattr(local_answer,'last_context_chars',None),
                 'context_evidence_count':len(getattr(local_answer,'last_context_evidence',[])),
                 'context_evidence_details':getattr(local_answer,'last_context_evidence',[]),
                 'hf_input_tokens':getattr(local_answer,'last_input_tokens',None),
                 'retrieved_chunk_ids':[x['chunk_id'] for x in evidence],
                 'retrieval_scores':[{'chunk_id':x['chunk_id'],'cosine':x['cosine'],
                                       'graph_score':x['graph_score']} for x in evidence],
                 'latency_seconds':round(time.monotonic()-start,3),'error':error}
            dst.write(json.dumps(row,ensure_ascii=False)+'\n');dst.flush()
            if args.save_context:
                ctx=output.with_name(output.stem+'_contexts.jsonl')
                with ctx.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'question_id':i,'evidence':evidence,'diagnostic':diagnostic},ensure_ascii=False)+'\n')
            print(f'#{i} {"OK" if not error else "ERROR"} {row["latency_seconds"]:.1f}s '
                  f'concept={row["concepts"][0]["label"] if row["concepts"] else "-"}')
    print(f'Done: {output.resolve()}')


def make_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build',help='Create topical concept tree from saved MiniRAG chunk vectors')
    q=sub.add_parser('qa',help='Concept -> multi-entity graph -> diverse vector evidence -> local LLM')
    r=sub.add_parser('repair-tree',help='Merge repeated labels in an existing tree without rebuilding its clusters')
    for s in [b,q,r]:
        s.add_argument('--working-dir',required=True,help='Existing MiniRAG index folder; read-only')
        s.add_argument('--tree-dir',default=None,help='New tree directory (default working-dir/concept_tree_rag)')
        s.add_argument('--embedding-model',default=DEFAULT_EMBED)
        s.add_argument('--ollama-url',default='http://localhost:11434')
    r.add_argument('--source-tree',default=None,help='Existing concept_tree.json (default working-dir/concept_tree_rag/concept_tree.json)')
    b.add_argument('--broad',type=int,default=8,help='Number of top-level topical concepts')
    b.add_argument('--fine',type=int,default=4,help='Sub-concepts per top-level concept')
    b.add_argument('--seed',type=int,default=42)
    b.add_argument('--label-backend',choices=['keywords','ollama','openai'],default='keywords',
                   help='Generate category names via keyword extraction, Ollama, or OpenAI')
    b.add_argument('--openai-base-url',default='https://api.openai.com/v1',
                   help='OpenAI-compatible API endpoint for concept labeling')
    b.add_argument('--label-model',default='qwen2.5:3b')
    q.add_argument('--qa-file',default=None,help='MultiHopRAG.json with query,answer,question_type')
    q.add_argument('--data-dir',default='dataset/MultiHop-RAG')
    q.add_argument('--output',default=None)
    q.add_argument('--overwrite',action='store_true')
    q.add_argument('--backend',choices=['ollama','hf'],default='ollama')
    q.add_argument('--qa-model',default='qwen2.5:3b')
    q.add_argument('--embedding-device',default='cpu',choices=['cpu','cuda','mps'],help='SentenceTransformer retrieval embedding device')
    q.add_argument('--hf-device',default='auto',choices=['auto','mps','cuda','cpu'],help='Explicit device for local Hugging Face QA model; auto prefers CUDA, then MPS, then CPU')
    q.add_argument('--start',type=int,default=0)
    q.add_argument('--max-questions',type=int,default=0)
    q.add_argument('--top-topics',type=int,default=3)
    q.add_argument('--top-leaves',type=int,default=5)
    q.add_argument('--entity-pool',type=int,default=250)
    q.add_argument('--top-entities',type=int,default=8)
    q.add_argument('--neighbors-per-entity',type=int,default=12)
    q.add_argument('--top-k',type=int,default=5,help='Final chunks passed to the answer LLM (not MiniRAG QueryParam.top_k)')
    q.add_argument('--retrieval-policy',choices=['legacy','enhanced'],default='enhanced',
                   help='Legacy = previous Concept RAG; enhanced = multi-entity/cross-leaf/diverse evidence')
    q.add_argument('--retrieval-top-k',type=int,default=60,
                   help='Internal ranked candidate ceiling BEFORE selecting final --top-k chunks')
    q.add_argument('--max-cross-chunks',type=int,default=200,
                   help='Maximum source-backed, explicit-entity cross-concept candidate chunks')
    q.add_argument('--max-chunks-per-doc',type=int,default=2,
                   help='Soft cap per source document among final chunks; fills remaining slots if needed')
    q.add_argument('--max-facets',type=int,default=4)
    q.add_argument('--facet-candidate-k',type=int,default=12,
                   help='Extra source-backed candidate chunks per fact requirement')
    q.add_argument('--coverage-weight',type=float,default=0.18)
    q.add_argument('--redundancy-weight',type=float,default=0.06)
    q.add_argument('--same-doc-penalty',type=float,default=0.10)
    q.add_argument('--facet-match-threshold',type=float,default=0.45,
                   help='Diagnostic threshold only; no forced weak evidence')
    q.add_argument('--graph-weight',type=float,default=0.18)
    q.add_argument('--max-context-chars',type=int,default=14000)
    q.add_argument('--max-chunk-chars',type=int,default=3500)
    q.add_argument('--max-new-tokens',type=int,default=512)
    q.add_argument('--save-context',action='store_true')
    return p


def main():
    load_project_env()
    args=make_parser().parse_args()
    if args.command=='build':
        if args.broad<1 or args.fine<1: raise ValueError('--broad and --fine must be >=1')
        build(args)
    elif args.command=='repair-tree':
        repair_tree(args)
    else:
        if min(args.top_topics,args.top_leaves,args.top_k,args.retrieval_top_k,args.max_cross_chunks,args.max_chunks_per_doc)<1:
            raise ValueError('All retrieval count limits must be >=1')
        if args.max_facets < 1 or args.facet_candidate_k < 1:
            raise ValueError('Facet limits must be >=1')
        qa(args)

if __name__=='__main__':main()
