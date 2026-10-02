"""Offline tests for metadata parsing and conservative evidence type mapping."""

import json

import pytest

from quality_uncertainty_medrag.models import EvidenceType
from quality_uncertainty_medrag.pubmed_records import (
    evidence_type_from_publication_types,
    parse_pubmed_xml,
)


ARTICLE_XML = """<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID Version="1">12345</PMID>
    <Article>
      <Journal><ISSN>1234-5678</ISSN><JournalIssue>
        <PubDate><Year>2024</Year><Month>May</Month><Day>7</Day></PubDate>
      </JournalIssue><Title>Example Medical Journal</Title>
      <ISOAbbreviation>Ex Med J</ISOAbbreviation></Journal>
      <ArticleTitle>Treatment of <i>rare</i> disease.</ArticleTitle>
      <Abstract>
        <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Some <b>important</b> context.</AbstractText>
        <AbstractText Label="RESULTS" NlmCategory="RESULTS">Result &amp; follow-up.</AbstractText>
      </Abstract>
      <AuthorList><Author><LastName>Lee</LastName><ForeName>Alex</ForeName><Initials>A</Initials>
        <Identifier Source="ORCID">0000-0000</Identifier>
        <AffiliationInfo><Affiliation>Example University</Affiliation></AffiliationInfo>
      </Author></AuthorList>
      <Language>eng</Language>
      <PublicationTypeList><PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
        <PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
    </Article>
    <MeshHeadingList><MeshHeading><DescriptorName UI="D001" MajorTopicYN="Y">Disease</DescriptorName>
      <QualifierName UI="Q001" MajorTopicYN="N">therapy</QualifierName></MeshHeading></MeshHeadingList>
    <KeywordList><Keyword>treatment</Keyword></KeywordList>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">12345</ArticleId>
    <ArticleId IdType="doi">10.1234/example</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle>
</PubmedArticleSet>"""


def test_parse_preserves_article_and_structured_metadata():
    records = parse_pubmed_xml(ARTICLE_XML.encode())
    assert list(records) == ["12345"]
    record = records["12345"]
    assert record["pmid"] == "12345"
    assert record["title"] == "Treatment of rare disease."
    assert record["abstract"] == "BACKGROUND: Some important context.\nRESULTS: Result & follow-up."
    assert record["journal"] == "Example Medical Journal"
    assert record["publication_date"] == "2024-May-7"
    assert record["publication_types"] == ["Randomized Controlled Trial", "Journal Article"]
    metadata = record["pubmed_metadata"]
    assert metadata["doi"] == "10.1234/example"
    assert metadata["article_ids"][0] == {"id_type": "pubmed", "value": "12345"}
    assert metadata["abstract_sections"][0]["nlm_category"] == "BACKGROUND"
    assert metadata["abstract_source"] == "article_abstract"
    assert metadata["authors"][0]["LastName"] == "Lee"
    assert metadata["authors"][0]["affiliations"] == ["Example University"]
    assert metadata["languages"] == ["eng"]
    assert metadata["mesh_headings"][0]["descriptor_attributes"]["UI"] == "D001"
    assert metadata["keywords"] == ["treatment"]
    assert "<i>rare</i>" in metadata["raw_xml"]
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    ("date_xml", "expected"),
    [
        ("<MedlineDate>2020 Winter-2021 Spring</MedlineDate>", "2020 Winter-2021 Spring"),
        ("<Year>2024</Year>", "2024"),
        ("<Year>2024</Year><Month>02</Month>", "2024-02"),
        ("", None),
    ],
)
def test_publication_dates_preserve_supplied_precision(date_xml, expected):
    xml = ARTICLE_XML.replace("<Year>2024</Year><Month>May</Month><Day>7</Day>", date_xml)
    assert parse_pubmed_xml(xml)["12345"]["publication_date"] == expected


def test_article_date_is_available_when_journal_date_is_missing():
    xml = ARTICLE_XML.replace("<Year>2024</Year><Month>May</Month><Day>7</Day>", "")
    xml = xml.replace("<Language>eng</Language>", '<ArticleDate DateType="Electronic"><Year>2023</Year></ArticleDate>')
    record = parse_pubmed_xml(xml)["12345"]
    assert record["publication_date"] == "2023"
    assert record["pubmed_metadata"]["article_dates"] == [
        {"date_type": "Electronic", "fields": {"Year": "2023"}}
    ]


def test_missing_abstract_is_empty_and_not_fabricated():
    start = ARTICLE_XML.index("<Abstract>")
    end = ARTICLE_XML.index("</Abstract>") + len("</Abstract>")
    xml = ARTICLE_XML[:start] + ARTICLE_XML[end:]
    record = parse_pubmed_xml(xml)["12345"]
    assert record["abstract"] == ""
    assert record["pubmed_metadata"]["abstract_sections"] == []
    assert record["pubmed_metadata"]["abstract_source"] == "not_available"


def _without_primary_abstract(xml):
    start = xml.index("<Abstract>")
    end = xml.index("</Abstract>") + len("</Abstract>")
    return xml[:start] + xml[end:]


def _with_other_abstract(xml, other_abstract):
    return xml.replace("</MedlineCitation>", other_abstract + "</MedlineCitation>")


def test_primary_abstract_takes_precedence_over_english_other_abstract():
    xml = _with_other_abstract(ARTICLE_XML, """<OtherAbstract Type="PIP" Language="eng">
      <AbstractText>Alternative English summary.</AbstractText></OtherAbstract>""")
    record = parse_pubmed_xml(xml)["12345"]
    assert record["abstract"] == "BACKGROUND: Some important context.\nRESULTS: Result & follow-up."
    metadata = record["pubmed_metadata"]
    assert metadata["abstract_source"] == "article_abstract"
    assert metadata["abstract_source_attributes"] == {}
    assert metadata["other_abstracts"][0]["text"] == "Alternative English summary."
    assert metadata["primary_abstract_sections"] == metadata["abstract_sections"]


@pytest.mark.parametrize("language_attribute", [' Language="eng"', ""])
def test_other_abstract_fallback_preserves_sections_and_source(language_attribute):
    xml = _without_primary_abstract(ARTICLE_XML)
    xml = _with_other_abstract(xml, f"""<OtherAbstract Type="PIP"{language_attribute}>
      <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Original <i>English</i> context.</AbstractText>
      <AbstractText Label="RESULTS">Original results.</AbstractText>
      <CopyrightInformation>Original copyright.</CopyrightInformation></OtherAbstract>""")
    record = parse_pubmed_xml(xml)["12345"]
    assert record["abstract"] == "BACKGROUND: Original English context.\nRESULTS: Original results."
    metadata = record["pubmed_metadata"]
    assert metadata["abstract_source"] == "other_abstract"
    assert metadata["abstract_source_attributes"]["Type"] == "PIP"
    assert metadata["primary_abstract_sections"] == []
    other = metadata["other_abstracts"][0]
    assert other["text"] == record["abstract"]
    assert other["sections"] == metadata["abstract_sections"]
    assert other["sections"][0]["nlm_category"] == "BACKGROUND"
    assert other["copyright_information"] == "Original copyright."
    assert json.loads(json.dumps(record)) == record


def test_non_english_other_abstract_is_preserved_without_english_fallback():
    xml = _without_primary_abstract(ARTICLE_XML)
    xml = _with_other_abstract(xml, """<OtherAbstract Type="Publisher" Language="fre">
      <AbstractText Label="OBJECTIF">Texte original français.</AbstractText></OtherAbstract>""")
    record = parse_pubmed_xml(xml)["12345"]
    assert record["abstract"] == ""
    metadata = record["pubmed_metadata"]
    assert metadata["abstract_source"] == "not_available"
    assert metadata["abstract_sections"] == []
    assert metadata["other_abstracts"][0]["text"] == "OBJECTIF: Texte original français."
    assert metadata["other_abstracts"][0]["attributes"] == {"Type": "Publisher", "Language": "fre"}


def test_other_abstract_fallback_skips_non_english_and_empty_sections():
    xml = _without_primary_abstract(ARTICLE_XML)
    xml = _with_other_abstract(xml, """<OtherAbstract Type="Publisher" Language="ger">
      <AbstractText>Deutscher Text.</AbstractText></OtherAbstract>
      <OtherAbstract Type="PIP" Language="eng"><AbstractText>   </AbstractText></OtherAbstract>
      <OtherAbstract Type="PIP" Language="eng"><AbstractText>Available English text.</AbstractText></OtherAbstract>""")
    record = parse_pubmed_xml(xml)["12345"]
    assert record["abstract"] == "Available English text."
    assert len(record["pubmed_metadata"]["other_abstracts"]) == 3


def test_empty_set_book_and_incomplete_articles_are_skipped():
    assert parse_pubmed_xml("<PubmedArticleSet/>") == {}
    assert parse_pubmed_xml("""<PubmedArticleSet><PubmedBookArticle><BookDocument>
      <PMID>555</PMID></BookDocument></PubmedBookArticle>
      <PubmedArticle><MedlineCitation><Article><ArticleTitle>No PMID</ArticleTitle>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>""") == {}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("<PubmedArticleSet>", "Malformed"),
        ("<ERROR>API rate limit exceeded</ERROR>", "API rate limit exceeded"),
        ("<PubmedArticleSet><ERROR>Invalid PMID</ERROR></PubmedArticleSet>", "Invalid PMID"),
        ("<html>Unavailable</html>", "Unexpected"),
    ],
)
def test_invalid_or_error_response_is_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_pubmed_xml(payload)


@pytest.mark.parametrize(
    ("publication_types", "expected"),
    [
        (["Meta-Analysis"], EvidenceType.META_ANALYSIS),
        (["Systematic Review"], EvidenceType.SYSTEMATIC_REVIEW),
        (["Randomized Controlled Trial"], EvidenceType.RANDOMIZED_CONTROLLED_TRIAL),
        (["Journal Article", "Systematic Review", "Meta-Analysis"], EvidenceType.META_ANALYSIS),
        (["Randomized Controlled Trial", "Systematic Review"], EvidenceType.SYSTEMATIC_REVIEW),
        ([" meta-analysis "], EvidenceType.META_ANALYSIS),
        (["Practice Guideline"], EvidenceType.EVIDENCE_BASED_GUIDELINE),
        (["Guideline"], EvidenceType.EVIDENCE_BASED_GUIDELINE),
        (["Clinical Trial"], EvidenceType.OTHER),
        (["Observational Study"], EvidenceType.OTHER),
        (["Case Reports"], EvidenceType.OTHER),
        (["Journal Article"], EvidenceType.OTHER),
        ([], EvidenceType.OTHER),
    ],
)
def test_evidence_type_mapping_is_conservative(publication_types, expected):
    assert evidence_type_from_publication_types(publication_types) == expected
