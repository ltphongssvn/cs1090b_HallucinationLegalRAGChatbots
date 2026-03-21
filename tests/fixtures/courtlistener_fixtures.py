# tests/fixtures/courtlistener_fixtures.py
# Curated fixture suite for distribution-representative testing.
# Each fixture covers a distinct real-world data pattern from pile-of-law.

STANDARD_OPINION = {
    "text": "The court held that the defendant failed to establish a genuine issue of material fact sufficient to defeat summary judgment under Federal Rule of Civil Procedure 56.",
    "created_timestamp": "2022-01-15",
    "downloaded_timestamp": "2022-06-01",
    "url": "https://courtlistener.com/opinion/123456/smith-v-jones/",
}

WEIRDLY_FORMATTED_TEXT = {
    "text": (
        "\n\n   UNITED STATES DISTRICT COURT\n"
        "   SOUTHERN DISTRICT OF NEW YORK\n\n"
        "   -------------------------------------------x\n"
        "   IN RE: SOME MATTER\n"
        "   The Court, having reviewed the submissions,\n"
        "   hereby ORDERS as follows:   \n\n"
    ),
    "created_timestamp": "2021-03-10T09:15:00Z",
    "downloaded_timestamp": "2021-08-20",
    "url": "https://courtlistener.com/opinion/99999/in-re-some-matter/",
}

CONTENTS_FIELD_INSTEAD_OF_TEXT = {
    "contents": "Plaintiff's motion for summary judgment is GRANTED. The defendant has failed to raise any triable issue of fact on any element of the claim as required by the applicable standard of review.",
    "created_timestamp": "2020-05-01",
    "downloaded_timestamp": "2020-11-15",
    "url": "https://courtlistener.com/opinion/77777/plaintiff-v-defendant/",
}

MISSING_TIMESTAMPS = {
    "text": "The appeal is dismissed for lack of jurisdiction. The district court's order is affirmed without prejudice to refiling in the appropriate venue.",
    "created_timestamp": "",
    "downloaded_timestamp": "",
    "url": "https://courtlistener.com/opinion/55555/appeal-dismissed/",
}

MALFORMED_TIMESTAMPS = {
    "text": "Judgment entered in favor of plaintiff on all counts. Defendant ordered to pay damages as set forth in the accompanying order of the court.",
    "created_timestamp": "not-a-date",
    "downloaded_timestamp": "32/13/2022",
    "url": "https://courtlistener.com/opinion/44444/judgment-entered/",
}

EDGE_CASE_URL = {
    "text": "The court finds no merit in appellant's arguments and affirms the lower court's ruling in its entirety for the reasons stated in the record below.",
    "created_timestamp": "2019-07-04",
    "downloaded_timestamp": "2019-12-31",
    "url": "https://courtlistener.com/opinion/33333/appeal/?ref=index&page=1#section-2",
}

MINIMAL_VALID_TEXT = {
    "text": "Motion denied. See attached order for reasons.",  # exactly at edge — may be too short
    "created_timestamp": "2018-01-01",
    "downloaded_timestamp": "2018-06-01",
    "url": "https://courtlistener.com/opinion/11111/motion-denied/",
}

EXTRA_METADATA_FIELDS = {
    "text": "The court sustains the objection and strikes the testimony as inadmissible hearsay under Federal Rule of Evidence 802 as applied to the facts here.",
    "created_timestamp": "2023-02-14T14:00:00+05:30",
    "downloaded_timestamp": "2023-03-01",
    "url": "https://courtlistener.com/opinion/88888/hearsay-ruling/",
    "court_id": "ca9",
    "docket_number": "22-5678",
    "judges": ["Smith J.", "Jones J."],
    "precedential_status": "Published",
    "citations": ["42 F.3d 100", "55 F.3d 200"],
}

# All fixtures that must pass validate_row()
VALID_FIXTURES = [
    STANDARD_OPINION,
    WEIRDLY_FORMATTED_TEXT,
    CONTENTS_FIELD_INSTEAD_OF_TEXT,
    MISSING_TIMESTAMPS,
    MALFORMED_TIMESTAMPS,
    EDGE_CASE_URL,
    EXTRA_METADATA_FIELDS,
]

# Fixtures that are expected to fail validate_row()
INVALID_FIXTURES = [
    MINIMAL_VALID_TEXT,  # text is shorter than MIN_TEXT_LENGTH=50
]
