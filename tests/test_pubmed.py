"""PubMed infrastructure tests with scripted E-utilities responses only."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl

import pytest

from quality_uncertainty_medrag import pubmed
from quality_uncertainty_medrag.loaders import load_retrieved_evidence
from quality_uncertainty_medrag.models import EvidenceType, MedicalQuestion


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, body):
        self.body = body.encode("utf-8") if isinstance(body, str) else body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class ScriptedOpener:
    def __init__(self, responses, clock):
        self.responses = list(responses)
        self.clock = clock
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout, self.clock()))
        assert self.responses, "Unexpected live request after scripted responses exhausted"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Response(response)


def params(request):
    return dict(parse_qsl(request.data.decode("ascii")))


def search_json(pmids=("202", "101")):
    return json.dumps({
        "esearchresult": {
            "count": str(len(pmids)), "retmax": str(len(pmids)), "retstart": "0",
            "idlist": list(pmids), "querytranslation": "translated PubMed query",
        }
    })


def article_xml(pmid, title, abstract, publication_type="Journal Article"):
    return f"""<PubmedArticle>
      <MedlineCitation Status="MEDLINE"><PMID Version="1">{pmid}</PMID>
        <Article PubModel="Print"><Journal><ISSN>1234-5678</ISSN>
          <JournalIssue><PubDate><Year>2024</Year><Month>Jan</Month><Day>2</Day></PubDate></JournalIssue>
          <Title>Example Biomedical Journal</Title><ISOAbbreviation>Ex Biomed J</ISOAbbreviation>
        </Journal><ArticleTitle>{title}</ArticleTitle>
        <Abstract><AbstractText>{abstract}</AbstractText></Abstract>
        <Language>eng</Language><PublicationTypeList>
          <PublicationType UI="D000001">{publication_type}</PublicationType>
        </PublicationTypeList></Article>
      </MedlineCitation><PubmedData><ArticleIdList>
        <ArticleId IdType="pubmed">{pmid}</ArticleId>
        <ArticleId IdType="doi">10.1000/{pmid}</ArticleId>
      </ArticleIdList></PubmedData>
    </PubmedArticle>"""


def fetch_xml():
    # XML order intentionally differs from ESearch's relevance order.
    return "<PubmedArticleSet>" + article_xml(
        "101", "First XML article", "First abstract", "Randomized Controlled Trial"
    ) + article_xml("202", "Second XML article", "Second abstract") + "</PubmedArticleSet>"


def make_question(question_id="q-1"):
    return MedicalQuestion(
        question_id=question_id,
        question="Diabetes mellitus and chronic kidney disease treatment?",
        option_labels=("A", "B"), options=("GoldOnlyOptionToken", "OtherOnlyOptionToken"),
        answer_index=0, metadata={"gold_marker": "MetadataOnlyToken"},
    )


def make_client(tmp_path, responses, **kwargs):
    clock = FakeClock()
    opener = ScriptedOpener(responses, clock)
    client = pubmed.PubMedClient(
        tmp_path / "cache", opener=opener, sleep=clock.sleep, clock=clock,
        **kwargs,
    )
    return client, opener, clock


def test_query_is_deterministic_and_uses_question_text_only(tmp_path):
    question = make_question()
    query = pubmed.build_pubmed_query(question.question)

    assert query == pubmed.build_pubmed_query(question.question)
    assert query.endswith("AND hasabstract")
    assert 'NOT "pubmed books"[sb]' in query
    assert "AND NOT" not in query
    assert "[Title/Abstract]" not in query
    assert " OR " not in query
    assert "chronic kidney disease" in query.lower()
    assert "diabetes" in query.lower()
    assert "GoldOnlyOptionToken".lower() not in query.lower()
    assert "OtherOnlyOptionToken".lower() not in query.lower()
    assert "MetadataOnlyToken".lower() not in query.lower()

    client, opener, _ = make_client(tmp_path, [search_json(())])
    pubmed.PubMedRetriever(client).retrieve(question, top_k=15)
    assert params(opener.calls[0][0])["term"] == query


@pytest.mark.parametrize("question_text", ["", "   ", None, 42])
def test_query_rejects_invalid_question_text(question_text):
    with pytest.raises(ValueError):
        pubmed.build_pubmed_query(question_text)


def test_client_uses_eutilities_post_and_preserves_search_metadata(tmp_path):
    client, opener, _ = make_client(tmp_path, [search_json()], email="research@example.org")

    result = client.search("diabetes[Title/Abstract] AND hasabstract", top_k=15)

    assert result["idlist"] == ["202", "101"]
    assert result["querytranslation"] == "translated PubMed query"
    request, timeout, _ = opener.calls[0]
    assert request.full_url.endswith("esearch.fcgi")
    assert request.get_method() == "POST"
    assert timeout == 30
    arguments = params(request)
    assert arguments["db"] == "pubmed"
    assert arguments["retmode"] == "json"
    assert arguments["retmax"] == "15"
    assert arguments["sort"] == "relevance"
    assert arguments["email"] == "research@example.org"
    assert "research@example.org" not in request.full_url


def test_retriever_preserves_search_order_metadata_and_loader_compatibility(tmp_path):
    client, opener, clock = make_client(tmp_path, [search_json(), fetch_xml()])
    retriever = pubmed.PubMedRetriever(client)

    evidence = retriever.retrieve(make_question(), top_k=15)

    assert isinstance(evidence, tuple)
    assert [item.doc_id for item in evidence] == ["202", "101"]
    assert [item.rank for item in evidence] == [1, 2]
    assert all(item.question_id == "q-1" for item in evidence)
    assert all(item.schema_version == "2.0" for item in evidence)
    assert all(item.source == "PubMed" for item in evidence)
    assert all(item.retrieval_score == 0.0 for item in evidence)
    assert evidence[0].evidence_type is EvidenceType.OTHER
    assert evidence[1].evidence_type is EvidenceType.RANDOMIZED_CONTROLLED_TRIAL
    assert evidence[0].metadata["title"] == "Second XML article"
    assert evidence[0].metadata["abstract"] == "Second abstract"
    assert evidence[0].metadata["journal"] == "Example Biomedical Journal"
    assert evidence[0].metadata["publication_date"]
    assert evidence[0].metadata["publication_types"] == ("Journal Article",)
    assert evidence[0].metadata["pubmed_metadata"]
    assert evidence[0].metadata["query"] == params(opener.calls[0][0])["term"]
    assert "Second XML article" in evidence[0].text
    assert "Second abstract" in evidence[0].text
    assert len(opener.calls) == 2
    assert opener.calls[1][0].full_url.endswith("efetch.fcgi")
    assert set(params(opener.calls[1][0])["id"].split(",")) == {"101", "202"}
    assert opener.calls[1][2] - opener.calls[0][2] >= 0.4
    assert clock.sleeps

    rows = [pubmed.evidence_to_json_record(item) for item in evidence]
    assert all("annotated_stances" not in row for row in rows)
    output = tmp_path / "retrieved.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_retrieved_evidence(output)["q-1"] == list(evidence)
    report = retriever.reports[0]
    assert report["question_id"] == "q-1"
    assert report["query"] == rows[0]["metadata"]["query"]
    assert report["retrieved_article_count"] == 2
    assert report["pmids"] == ["202", "101"]
    assert report["top_5"] == [
        {"pmid": "202", "title": "Second XML article"},
        {"pmid": "101", "title": "First XML article"},
    ]


def test_cache_reuses_articles_across_questions_and_client_instances(tmp_path):
    client, opener, _ = make_client(tmp_path, [search_json(), fetch_xml(), search_json()])
    first_retriever = pubmed.PubMedRetriever(client)
    first = first_retriever.retrieve(make_question(), top_k=15)
    other_question = MedicalQuestion(
        question_id="q-2", question="Acute kidney injury clinical question?",
        option_labels=("A", "B"), options=("first", "second"), answer_index=1,
    )
    second = first_retriever.retrieve(other_question, top_k=15)

    assert len(opener.calls) == 3
    assert sum(call[0].full_url.endswith("efetch.fcgi") for call in opener.calls) == 1
    assert [item.doc_id for item in first] == [item.doc_id for item in second]
    assert all(item.question_id == "q-2" for item in second)

    cached_client, cached_opener, _ = make_client(tmp_path, [])
    cached = pubmed.PubMedRetriever(cached_client).retrieve(make_question(), top_k=15)
    assert cached == first
    assert cached_opener.calls == []


def test_search_cache_distinguishes_top_k(tmp_path):
    client, opener, _ = make_client(tmp_path, [search_json(), search_json(("202",))])
    query = pubmed.build_pubmed_query(make_question().question)
    client.search(query, top_k=15)
    client.search(query, top_k=15)
    small = client.search(query, top_k=1)

    assert len(opener.calls) == 2
    assert small["idlist"] == ["202"]


def test_fetch_downloads_only_pmids_absent_from_record_cache(tmp_path):
    first_xml = "<PubmedArticleSet>" + article_xml("101", "Cached article", "Abstract one") + "</PubmedArticleSet>"
    second_xml = "<PubmedArticleSet>" + article_xml("202", "New article", "Abstract two") + "</PubmedArticleSet>"
    client, opener, _ = make_client(tmp_path, [first_xml, second_xml])

    client.fetch(["101"])
    records = client.fetch(["101", "202", "101"])

    assert set(records) == {"101", "202"}
    assert records["101"]["title"] == "Cached article"
    assert params(opener.calls[0][0])["id"] == "101"
    assert params(opener.calls[1][0])["id"] == "202"
    assert client.fetch([]) == {}
    assert len(opener.calls) == 2


@pytest.mark.parametrize("legacy_version", [None, 1])
def test_legacy_article_cache_is_reparsed_locally_preserving_fetch_time(tmp_path, legacy_version):
    xml = article_xml("101", "Previously cached article", "").replace(
        "</MedlineCitation>",
        '<OtherAbstract Type="NASA" Language="eng"><AbstractText>'
        'English fallback abstract.</AbstractText></OtherAbstract></MedlineCitation>',
    )
    legacy_record = pubmed.parse_pubmed_xml(xml)["101"]
    # A prior parser retained raw XML but did not extract OtherAbstract text.
    legacy_record["abstract"] = ""
    legacy_record["pubmed_metadata"]["abstract_sections"] = []
    legacy_record["pubmed_metadata"]["fetched_at_utc"] = "2026-09-26T12:00:00+00:00"
    if legacy_version is not None:
        legacy_record["parser_version"] = legacy_version
    cache_path = tmp_path / "cache" / "records" / "101.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(legacy_record), encoding="utf-8")
    client, opener, _ = make_client(tmp_path, [])

    upgraded = client.fetch(["101"])["101"]

    assert opener.calls == []
    assert upgraded["abstract"] == "English fallback abstract."
    assert upgraded["pubmed_metadata"]["abstract_source"] == "other_abstract"
    assert upgraded["pubmed_metadata"]["fetched_at_utc"] == "2026-09-26T12:00:00+00:00"
    assert upgraded["parser_version"] == pubmed.ARTICLE_CACHE_VERSION
    assert json.loads(cache_path.read_text(encoding="utf-8")) == upgraded
    assert client.fetch(["101"])["101"] == upgraded
    assert opener.calls == []


@pytest.mark.parametrize("response_xml", [
    "<PubmedArticleSet/>",
    "<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>101</PMID>"
    "<ArticleTitle>Unsupported book chapter</ArticleTitle></BookDocument>"
    "</PubmedBookArticle></PubmedArticleSet>",
])
def test_successful_unsupported_fetch_is_cached_across_clients_without_fake_evidence(tmp_path, response_xml):
    client, opener, _ = make_client(tmp_path, [response_xml])

    assert client.fetch(["101"]) == {}
    assert client.fetch(["101"]) == {}
    assert len(opener.calls) == 1
    unavailable_path = tmp_path / "cache" / "unavailable" / "101.json"
    cached = json.loads(unavailable_path.read_text(encoding="utf-8"))
    assert cached["pmid"] == "101"
    assert cached["reason"]
    assert cached["response_xml"] == response_xml
    assert cached["parser_version"] == pubmed.ARTICLE_CACHE_VERSION
    assert cached["fetched_at_utc"]
    assert not (tmp_path / "cache" / "records" / "101.json").exists()
    new_client, forbidden_opener, _ = make_client(tmp_path, [])
    assert new_client.fetch(["101"]) == {}
    assert forbidden_opener.calls == []

    # New parser versions can retry previously unsupported records.
    cached["parser_version"] = pubmed.ARTICLE_CACHE_VERSION - 1
    unavailable_path.write_text(json.dumps(cached), encoding="utf-8")
    current_response = "<PubmedArticleSet>" + article_xml("101", "Available article", "Abstract") + "</PubmedArticleSet>"
    upgraded_client, upgraded_opener, _ = make_client(tmp_path, [current_response])
    assert upgraded_client.fetch(["101"])["101"]["title"] == "Available article"
    assert len(upgraded_opener.calls) == 1


def test_empty_search_does_not_fetch_or_fabricate_evidence(tmp_path):
    client, opener, _ = make_client(tmp_path, [search_json(())])
    retriever = pubmed.PubMedRetriever(client)

    assert retriever.retrieve(make_question(), top_k=15) == ()
    assert len(opener.calls) == 1
    assert retriever.reports[0]["retrieved_article_count"] == 0
    assert retriever.reports[0]["pmids"] == []
    assert retriever.reports[0]["top_5"] == []


def test_partial_efetch_preserves_original_search_ranks(tmp_path):
    xml = "<PubmedArticleSet>" + article_xml("101", "Only available article", "Abstract") + "</PubmedArticleSet>"
    client, _, _ = make_client(tmp_path, [search_json(), xml])
    retriever = pubmed.PubMedRetriever(client)

    evidence = retriever.retrieve(make_question(), top_k=15)

    assert [item.doc_id for item in evidence] == ["101"]
    assert [item.rank for item in evidence] == [2]
    assert retriever.reports[0]["retrieved_article_count"] == 1


def http_error(status, *, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return HTTPError("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", status,
                     "scripted error", headers, io.BytesIO(b"scripted error"))


def test_http_429_respects_retry_after_and_failed_response_is_not_cached(tmp_path):
    client, opener, clock = make_client(tmp_path, [http_error(429, retry_after="2"), search_json()])

    result = client.search("diabetes AND hasabstract", top_k=15)

    assert result["idlist"] == ["202", "101"]
    assert len(opener.calls) == 2
    assert opener.calls[1][2] - opener.calls[0][2] >= 2.0
    assert any(seconds >= 2 for seconds in clock.sleeps)
    client.search("diabetes AND hasabstract", top_k=15)
    assert len(opener.calls) == 2


@pytest.mark.parametrize("failure", [http_error(503), URLError("offline"), TimeoutError("timed out")])
def test_transient_transport_failures_are_retried(tmp_path, failure):
    client, opener, clock = make_client(tmp_path, [failure, search_json()])

    assert client.search("diabetes AND hasabstract", top_k=15)["idlist"]
    assert len(opener.calls) == 2
    assert opener.calls[1][2] - opener.calls[0][2] >= 0.4
    assert clock.sleeps


def test_http_400_is_not_retried_or_cached(tmp_path):
    client, opener, _ = make_client(tmp_path, [http_error(400)])

    with pytest.raises(pubmed.PubMedError):
        client.search("diabetes AND hasabstract", top_k=15)

    assert len(opener.calls) == 1
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_retry_exhaustion_raises_without_caching(tmp_path):
    client, opener, _ = make_client(tmp_path, [http_error(503)] * 3, max_retries=2)

    with pytest.raises(pubmed.PubMedError):
        client.search("diabetes AND hasabstract", top_k=15)

    assert len(opener.calls) == 3
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("body", ["not JSON", '{"error": "Invalid database"}', '{"esearchresult": {"idlist": [null]}}'])
def test_invalid_search_response_raises_without_caching(tmp_path, body):
    client, opener, _ = make_client(tmp_path, [body])

    with pytest.raises(pubmed.PubMedError):
        client.search("diabetes AND hasabstract", top_k=15)

    assert len(opener.calls) == 1
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("body", ["not XML", "<eFetchResult><ERROR>Invalid database</ERROR></eFetchResult>"])
def test_invalid_fetch_response_raises_without_caching(tmp_path, body):
    client, opener, _ = make_client(tmp_path, [body])

    with pytest.raises(pubmed.PubMedError):
        client.fetch(["101"])

    assert len(opener.calls) == 1
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_cli_limits_questions_and_writes_only_retrieval_outputs(tmp_path, monkeypatch, capsys):
    questions = tmp_path / "questions.jsonl"
    clinical_topics = ("Diabetes kidney", "Migraine headache", "Asthma pulmonary", "Hypertension vascular")
    questions.write_text("".join(json.dumps({
        "id": f"q-{index}", "question": f"{topic} clinical question?",
        "options": {"A": "First choice", "B": "Second choice"}, "answer": "A",
    }) + "\n" for index, topic in enumerate(clinical_topics, start=1)), encoding="utf-8")
    output = tmp_path / "retrieved" / "evidence.jsonl"
    report = tmp_path / "retrieved" / "report.json"
    clock = FakeClock()
    opener = ScriptedOpener([search_json(), fetch_xml(), search_json(), search_json()], clock)
    real_client_type = pubmed.PubMedClient

    def client_factory(cache_dir, **kwargs):
        return real_client_type(cache_dir, opener=opener, sleep=clock.sleep, clock=clock, **kwargs)

    monkeypatch.setattr(pubmed, "PubMedClient", client_factory)

    assert pubmed.main([
        "--questions", str(questions), "--limit", "3", "--top-k", "15",
        "--output", str(output), "--report", str(report), "--cache-dir", str(tmp_path / "cache"),
    ]) == 0

    loaded = load_retrieved_evidence(output)
    assert list(loaded) == ["q-1", "q-2", "q-3"]
    assert all(len(evidence) == 2 for evidence in loaded.values())
    search_calls = [call for call in opener.calls if call[0].full_url.endswith("esearch.fcgi")]
    assert len(search_calls) == 3
    assert all(params(call[0])["retmax"] == "15" for call in search_calls)
    assert all("annotated_stances" not in json.loads(line) for line in output.read_text(encoding="utf-8").splitlines())
    saved_report = json.loads(report.read_text(encoding="utf-8"))
    printed_report = json.loads(capsys.readouterr().out)
    assert printed_report == saved_report
    assert "q-4" not in json.dumps(saved_report)
    assert "q-1" in json.dumps(saved_report)
