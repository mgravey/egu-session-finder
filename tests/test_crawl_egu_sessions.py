from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crawl_egu_sessions import (  # noqa: E402
    discovery_links,
    parse_session_page,
    parse_years,
    session_links,
    soup_from_bytes,
)


class CrawlEguSessionsTests(unittest.TestCase):
    def test_discovers_group_ids_and_allowed_special_interest_pages(self):
        html = b"""
        <input id="pg_checkbox_5859" name="pg_checkbox[]" value="5859">
        <label for="pg_checkbox_5859">Energy (ERE)</label>
        <input id="special" name="pg_checkbox[]" value="special"
          onclick="window.location.href='https://meetingorganizer.copernicus.org/EGU26/sessions-of-special-interest/EDI';">
        <label for="special">EDI</label>
        <a href="https://example.org/not-allowed">Other host</a>
        """
        links = discovery_links(
            soup_from_bytes(html),
            event_code="EGU26",
            current_url="https://meetingorganizer.copernicus.org/EGU26/sessionprogramme/pg-selection",
        )
        self.assertEqual(
            links["https://meetingorganizer.copernicus.org/EGU26/sessionprogramme/5859"],
            "Energy (ERE)",
        )
        self.assertEqual(
            links[
                "https://meetingorganizer.copernicus.org/EGU26/sessions-of-special-interest/EDI"
            ],
            "EDI",
        )
        self.assertEqual(len(links), 2)

    def test_session_links_deduplicate_fragments_and_keep_best_title(self):
        html = b"""
        <a href="/EGU26/session/55879#PICO">PICO</a>
        <a href="/EGU26/session/55879">Energy open session</a>
        <a href="/EGU25/session/123">Wrong event</a>
        """
        sessions = session_links(soup_from_bytes(html), "EGU26")
        self.assertEqual(list(sessions), ["55879"])
        self.assertEqual(sessions["55879"]["title_hint"], "Energy open session")
        self.assertEqual(
            sessions["55879"]["url"],
            "https://meetingorganizer.copernicus.org/EGU26/session/55879",
        )

    def test_parses_modern_description_title_and_abstract_assignments(self):
        html = b"""
        <html><head><title>Session ERE1.1</title></head><body>
        <div id="sessiondetails"><strong class="link-coloured">Energy open session</strong></div>
        <div id="modalSessionDescription55879">
          <h2 class="co-modal-default-modal-headline">ERE1.1 | Session description</h2>
          <div class="modal-content-container">Energy &amp; resources <b>for society</b>.</div>
        </div>
        <div>EGU26-12</div><a href="/EGU26/EGU26-99.html">EGU26-99</a>
        </body></html>
        """
        parsed = parse_session_page(
            html,
            event_code="EGU26",
            year=2026,
            session={
                "session_id": "55879",
                "url": "https://meetingorganizer.copernicus.org/EGU26/session/55879",
                "title_hint": "",
                "discovery_groups": ["Energy (ERE)"],
            },
        )
        self.assertEqual(parsed["session_code"], "ERE1.1")
        self.assertEqual(parsed["title"], "Energy open session")
        self.assertEqual(parsed["description"], "Energy & resources for society .")
        self.assertFalse(parsed["description_missing"])
        self.assertEqual(parsed["abstract_codes"], ["EGU26-12", "EGU26-99"])

    def test_parses_multiple_co_organising_divisions(self):
        html = b"""
        <html><head><title>Session HS7.2</title></head><body>
        <div id="sessiondetails"><h1>HS7.2</h1></div>
        <div class="co_mto_programme-session-block-cosponsoring-coorganizing">
          Co-organized by AS1/NP3, co-sponsored by WMO
        </div>
        </body></html>
        """
        parsed = parse_session_page(
            html,
            event_code="EGU26",
            year=2026,
            session={
                "session_id": "55706",
                "url": "https://meetingorganizer.copernicus.org/EGU26/session/55706",
                "title_hint": "Precipitation modelling",
                "discovery_groups": [
                    "Atmospheric Sciences (AS)",
                    "Hydrological Sciences (HS)",
                    "Nonlinear Processes in Geosciences (NP)",
                ],
            },
        )
        self.assertEqual(parsed["co_organizer_label"], "AS1/NP3")
        self.assertEqual(parsed["co_organizer_codes"], ["AS", "NP"])

    def test_parses_custom_element_description_and_heading_code(self):
        html = b"""
        <html><body><div id="sessiondetails">
          <div>ESSI1.1 | <strong class="link-coloured">AI for Earth</strong></div>
          <co-modal-default id="modalSessionDescription44904"
            title="ESSI1.1 | Session description">
            Spatial machine learning<br>for Earth science.
          </co-modal-default>
        </div></body></html>
        """
        parsed = parse_session_page(
            html,
            event_code="EGU23",
            year=2023,
            session={
                "session_id": "44904",
                "url": "https://meetingorganizer.copernicus.org/EGU23/session/44904",
                "title_hint": "AI for Earth",
                "discovery_groups": ["Earth & Space Science Informatics (ESSI)"],
            },
        )
        self.assertEqual(parsed["session_code"], "ESSI1.1")
        self.assertEqual(
            parsed["description"], "Spatial machine learning for Earth science."
        )

    def test_parses_legacy_heading_code_without_description(self):
        html = b"""
        <html><head><title>asset</title></head><body>
        <div id="sessiondetails"><h1>SSS5.7</h1></div>
        </body></html>
        """
        parsed = parse_session_page(
            html,
            event_code="EGU21",
            year=2021,
            session={
                "session_id": "38691",
                "url": "https://meetingorganizer.copernicus.org/EGU21/session/38691",
                "title_hint": "Soil session",
                "discovery_groups": ["Soil System Sciences (SSS)"],
            },
        )
        self.assertEqual(parsed["session_code"], "SSS5.7")
        self.assertIsNone(parsed["description"])
        self.assertTrue(parsed["description_missing"])

    def test_parses_2020_programme_block_code(self):
        html = b"""
        <html><body><div id="sessiondetails">
          <div class="co_mto_programme-session-block" data-id="34690">
            <div class="co_mto_programme-session-block-number"><span>US1</span></div>
          </div>
        </div></body></html>
        """
        parsed = parse_session_page(
            html,
            event_code="EGU2020",
            year=2020,
            session={
                "session_id": "34690",
                "url": "https://meetingorganizer.copernicus.org/EGU2020/session/34690",
                "title_hint": "Scientific integrity",
                "discovery_groups": ["Union Symposia (US)"],
            },
        )
        self.assertEqual(parsed["session_code"], "US1")

    def test_parse_year_ranges(self):
        self.assertEqual(parse_years("2020,2023-2026"), [2020, 2023, 2024, 2025, 2026])


if __name__ == "__main__":
    unittest.main()
