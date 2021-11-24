from typing import List, Optional
from pyserini.search import JSimpleSearcherResult, SimpleSearcher
from code.cqr import ConversationalQueryRewriter
from code.settings import PipelineSettings
from pygaggle.rerank.base import Query, Reranker, hits_to_texts
from pygaggle.rerank.transformer import MonoBERT

def bert_reranker(
    name_or_path: str = "castorini/monobert-large-msmarco-finetune-only",
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model = MonoBERT.get_model(name_or_path, device=device)
    tokenizer = MonoBERT.get_tokenizer(name_or_path)
    return MonoBERT(model, tokenizer)

def RRF(hlist: List[List[JSimpleSearcherResult]], k: int=60) -> List[JSimpleSearcherResult]:
    if (len(hlist) == 0):
        return []
    if (len(hlist) == 1):
        return hlist[0]
    d = {}
    for hlist_sub in hlist:
        for ind, h in enumerate(hlist_sub):
            sc = d.get(h.docid, (0.0, h))[0] + (1.0 / (k + ind))
            d[h.docid] = (sc, h)
    return [score_hit[1] for _, score_hit in sorted(iter(d.items()), key=lambda el: el[1][0], reverse=True)]

class Pipeline:
    def __init__(self, searcher: SimpleSearcher, cqr: List[ConversationalQueryRewriter], reranker: Reranker = bert_reranker(), settings: PipelineSettings = PipelineSettings()) -> None:
        self.searcher = searcher
        self.cqr = cqr
        self.reranker = reranker
        self.settings = settings
        self.top_k = settings.top_k
        self.early_fusion = settings.early_fusion
    
    def retrieve(self, query, context: Optional[str] = None) -> List[JSimpleSearcherResult]:
        qr_query: List[str] = []
        qr_results: List[List[JSimpleSearcherResult]] = []
        for qr in self.cqr:
            qr_query.append(qr.rewrite(query, context))
            qr_results.append(self.searcher.search(qr_query[-1], self.top_k))
        if (self.early_fusion or self.reranker is None):
            qr_results = RRF(qr_results)
        if (self.reranker is None):
            return qr_results[:self.top_k]
        if self.early_fusion:
            return self.rerank(qr_query[-1], qr_results[:self.top_k])
        else:
            li = list(zip(qr_query, qr_results))
            fin_results: List[List[JSimpleSearcherResult]] = []
            for qrq, qrr in li:
                fin_results.append(self.rerank(qrq, qrr))
            return RRF(fin_results)[:self.top_k]
        
    def rerank(self, query, hits) -> List[JSimpleSearcherResult]:
        reranked = self.reranker.rerank(Query(query), hits_to_texts(hits))
        reranked_scores = [r.score for r in reranked]

        # Reorder hits with reranker scores
        reranked = list(zip(hits, reranked_scores))
        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        reranked_hits = [r[0] for r in reranked]
        return reranked_hits

    def reset_history(self):
        for qr in self.cqr:
            qr.reset_history()
