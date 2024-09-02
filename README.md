# Reading lists from ACL Anthology tutorials

This repository contains the reading lists from the ACL Anthology tutorials.
Each list was manually curated from the published (PDF) paper describing the tutorial at the conference.

The instructions given to the tutorial presenters for preparing their reading lists are:

- ACL 2021: "Small reading list. It’s size should be such that it is reasonable to expect a trainee to read most of the recommended references before the tutorial (depending on their length, 4-10 seems a reasonable number). Preferably, at least 50% of the recommended papers should not be co-authored by the tutorial presenters."

- ACL 2022, 2023, 2024: "Reading list. Work that you expect the audience to read before the tutorial can be indicated by an asterisk. Recommended papers should provide breadth of authorship and include work by other authors, and work from other disciplines is welcome if relevant."

The reading lists are available in the `data` directory, organised by `year` and then by `conference` identified using the ACL bibkey (e.g. `2020.acl-tutorials`).
Each conference folder contains tutorials metadata ordered following ACL numbering conventions. (e.g. `2020.acl-tutorials.1.json`) aswell as a `*.proceedings.json` file containing metadata of the paper describing the conference.

```
# example of a conference proceeding file
{
    "2020.conference-name": {
        "title": "title",
        "editor": "editor",
        "year": 2020,
        "publisher": "publisher",
        "venues": [
            {
                "id": "id",
                "acronym": "acronym",
                "name": "name"
            }
        ],
        "url": "url",
        "tutorials": [
            "2020.conference-name.1",
            "2020.conference-name.2",
        ]   
    }
}
```

Tutorial files (e.g. `2020.acl-tutorials.1.json`) contains metadata of tutorial papers aswell as the article reading list (`readingList`). Reading lists objects are presented following the structure provided by the authors of the tutorial and indexed by their Semantic Scholar Academic Graph API `paperId` identifier. When articles where not present in the Semantic Scholar Academic Graph API, a `null` field is filled to preserve informations of list size and order.


```
# example of a tutorial file
{
    "2020.conference-name.1": {
        "title": "title",
        "author": "author",
        "year": 2020,
        "url": "url",
        "doi": "doi",
        "abstract": "abstract",
        "readingList": [
            {
                "sectionName": "Section 1",
                "subsectionName": null,
                "referencesIds": [
                    "0e661bd2cfe94ed58e4e2abc1409c75b98c2582c",
                    null, #Paper not found on Semantic Scholar
                ]
            },
            {
                "sectionName": "Section 2",
                "subsectionName": null,
                "referencesIds": [
                    "e8fa186444d98a39ee9139b1f5dd0c7618caef8f",
                    "f9acf607b858ac110c1bf83bf62835bcc1820e83",
                ]
            }
        ]
    }
}
```

Metadata of articles referenced in a reading list are located in `data/references_metadata.json`. Because some articles data are missing withing the Semantic Scholar Academic Graph API, the `data/references_missing_metadata.json` file is used to manually provide missing fields informations.

