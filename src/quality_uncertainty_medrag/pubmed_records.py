"""Parse PubMed EFetch XML without inferring article content or evidence types."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .pubmed_evidence_type import evidence_type_from_publication_types


def _text(element: ET.Element | None) -> str:
    """Preserve inline markup text while normalizing XML formatting whitespace."""

    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _date_fields(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    return {child.tag: _text(child) for child in element if _text(child)}


def _publication_date(fields: dict[str, str]) -> str | None:
    if fields.get("MedlineDate"):
        return fields["MedlineDate"]
    # Keep supplied precision and month spelling; never fill in absent dates.
    parts = [fields[name] for name in ("Year", "Month", "Day") if fields.get(name)]
    return "-".join(parts) if parts else None


def _abstract_sections(element: ET.Element | None) -> list[dict[str, Any]]:
    if element is None:
        return []
    return [
        {
            "label": node.get("Label"),
            "nlm_category": node.get("NlmCategory"),
            "text": _text(node),
        }
        for node in element.findall("AbstractText")
    ]


def _abstract_text(sections: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{section['label']}: {section['text']}" if section["label"] else section["text"]
        for section in sections
        if section["text"]
    )


def _authors(article: ET.Element) -> list[dict[str, Any]]:
    authors: list[dict[str, Any]] = []
    for author in article.findall("./AuthorList/Author"):
        record: dict[str, Any] = {}
        for field in ("LastName", "ForeName", "Initials", "Suffix", "CollectiveName"):
            value = _text(author.find(field))
            if value:
                record[field] = value
        record["affiliations"] = [
            _text(node) for node in author.findall("./AffiliationInfo/Affiliation")
        ]
        record["identifiers"] = [
            {"source": node.get("Source"), "value": _text(node)}
            for node in author.findall("Identifier")
        ]
        authors.append(record)
    return authors


def _mesh_headings(citation: ET.Element) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for heading in citation.findall("./MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        headings.append(
            {
                "descriptor": _text(descriptor),
                "descriptor_attributes": dict(descriptor.attrib) if descriptor is not None else {},
                "qualifiers": [
                    {"text": _text(node), "attributes": dict(node.attrib)}
                    for node in heading.findall("QualifierName")
                ],
            }
        )
    return headings


def _parse_article(element: ET.Element) -> dict[str, Any] | None:
    citation = element.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    article = citation.find("Article")
    if not pmid or article is None:
        return None

    primary_sections = _abstract_sections(article.find("Abstract"))
    sections = primary_sections
    abstract = _abstract_text(sections)
    other_abstracts = [
        {
            "attributes": dict(node.attrib),
            "sections": _abstract_sections(node),
            "text": _abstract_text(_abstract_sections(node)),
            "copyright_information": _text(node.find("CopyrightInformation")),
        }
        for node in citation.findall("OtherAbstract")
    ]
    abstract_source = "article_abstract" if abstract else "not_available"
    abstract_source_attributes: dict[str, str] = {}
    if not abstract:
        for other in other_abstracts:
            language = other["attributes"].get("Language", "").strip().casefold()
            if language in {"", "eng"} and other["text"]:
                sections = other["sections"]
                abstract = other["text"]
                abstract_source = "other_abstract"
                abstract_source_attributes = other["attributes"]
                break
    journal_date = _date_fields(article.find("./Journal/JournalIssue/PubDate"))
    article_dates = [
        {"date_type": node.get("DateType"), "fields": _date_fields(node)}
        for node in article.findall("ArticleDate")
    ]
    date_fields = journal_date or (article_dates[0]["fields"] if article_dates else {})
    article_ids = [
        {"id_type": node.get("IdType"), "value": _text(node)}
        for node in element.findall("./PubmedData/ArticleIdList/ArticleId")
    ]
    doi = next((record["value"] for record in article_ids if record["id_type"] == "doi"), None)
    publication_types = [
        _text(node) for node in article.findall("./PublicationTypeList/PublicationType")
        if _text(node)
    ]
    return {
        "pmid": pmid,
        "title": _text(article.find("ArticleTitle")),
        "abstract": abstract,
        "journal": _text(article.find("./Journal/Title")),
        "publication_date": _publication_date(date_fields),
        "publication_types": publication_types,
        "pubmed_metadata": {
            "raw_xml": ET.tostring(element, encoding="unicode"),
            "citation_attributes": dict(citation.attrib),
            "abstract_sections": sections,
            "primary_abstract_sections": primary_sections,
            "other_abstracts": other_abstracts,
            "abstract_source": abstract_source,
            "abstract_source_attributes": abstract_source_attributes,
            "journal_publication_date": journal_date,
            "article_dates": article_dates,
            "journal_issn": _text(article.find("./Journal/ISSN")),
            "journal_abbreviation": _text(article.find("./Journal/ISOAbbreviation")),
            "article_ids": article_ids,
            "doi": doi,
            "authors": _authors(article),
            "mesh_headings": _mesh_headings(citation),
            "keywords": [_text(node) for node in citation.findall("./KeywordList/Keyword")],
            "languages": [_text(node) for node in article.findall("Language")],
        },
    }


def parse_pubmed_xml(payload: bytes | str) -> dict[str, dict[str, Any]]:
    """Return PMID-keyed article records from an NCBI EFetch response.

    Empty article sets are valid. Book articles and incomplete article records
    are skipped rather than fabricating biomedical article content. Malformed
    XML, explicit API errors, and unexpected response formats raise ValueError.
    """

    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("Malformed PubMed XML response") from exc
    for node in root.iter():
        if node.tag.upper() == "ERROR":
            raise ValueError(f"PubMed API error: {_text(node)}")
    if root.tag not in {"PubmedArticleSet", "PubmedArticle"}:
        raise ValueError(f"Unexpected PubMed XML root: {root.tag}")
    elements = [root] if root.tag == "PubmedArticle" else root.findall("PubmedArticle")
    records: dict[str, dict[str, Any]] = {}
    for element in elements:
        record = _parse_article(element)
        if record is None:
            continue
        pmid = record["pmid"]
        if pmid in records:
            raise ValueError(f"Duplicate PMID in PubMed XML response: {pmid}")
        records[pmid] = record
    return records