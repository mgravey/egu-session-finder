# SEO and outreach plan – EGU27 Session Finder

**Goal:** be the page EGU27 authors find when they wonder which session to
submit to, between the call opening (28 October 2026) and the abstract deadline
(**12 January 2027, 13:00 CET**). Traffic is short-lived and seasonal, so
discovery through EGU communities matters more than slow organic ranking.

Canonical URL: `https://mgravey.com/egu-session-finder/`

## 1. Done on the site

| Item | Where |
|---|---|
| Descriptive `<title>` and meta description with “EGU27”, “EGU General Assembly 2027”, “session”, “abstract” | all pages |
| Canonical URLs, `robots` meta, `lang="en"` | all pages |
| Open Graph and Twitter large-image cards with a 1200×630 preview | `web/public/og-image.png` |
| `WebApplication` + `Event` and `FAQPage` JSON-LD | `web/index.html` |
| FAQ answering real search questions (deadline, accuracy, privacy, official?) | home page |
| `robots.txt`, `sitemap.xml`, `404.html` | `web/public/` |
| Crawlable static text: headline numbers are in the HTML, not only in JS | home page |
| Fast first paint: model and catalogue load in a Web Worker after the page | `web/src/` |
| Self-hosted fonts, no third-party requests or trackers | `web/src/fonts.ts` |

## 2. Target queries

Primary: `EGU27 session`, `EGU 2027 sessions`, `EGU General Assembly 2027
session`, `which EGU session should I submit to`, `EGU abstract session finder`.
Secondary: `EGU27 abstract submission`, `EGU27 deadline`, `EGU session
programme 2027`, plus division-level phrases (`EGU27 hydrology sessions`).

Do not compete with the official programme for navigational queries; position
the site as the *matching* companion and always link to the official pages.

## 3. Launch checklist (first week of November 2026)

1. Verify `mgravey.com` in **Google Search Console** (domain property) and
   **Bing Webmaster Tools**; submit `sitemap.xml`; request indexing of the home page.
2. Validate structured data with Google’s Rich Results Test and the social
   cards with the LinkedIn Post Inspector and Mastodon/Bluesky previews.
3. Run Lighthouse (mobile) and keep Performance, Accessibility, SEO ≥ 90.
4. Link the tool from `mgravey.com` (home page and a short blog/project post).

## 4. Outreach (where EGU authors actually are)

- **Social:** a post with a 20-second screen recording on Bluesky, Mastodon
  (geoscience instances), LinkedIn, and X, tagged #EGU27 / #EGU2027. Repeat
  4 weeks and 1 week before the deadline.
- **EGU community:** ask division Early Career Scientist representatives and
  the EGU ECS team whether they would share it in their newsletters or
  channels; offer it to session conveners advertising their sessions (it brings
  them well-matched abstracts).
- **Mailing lists / groups:** relevant university and institute geoscience
  mailing lists, research-group Slack/Teams channels, and ECS networks
  (e.g. APECS for cryosphere, YHS for hydrology).
- **Credibility:** keep the method and backtest results public; this is what
  people will cite when sharing it. A short preprint or GitHub README is enough.

Always state clearly that it is independent and not an official EGU service.

## 5. Timeline

| When | Action |
|---|---|
| Early Nov 2026 | Launch, Search Console, first social posts, ECS contacts |
| Mid Dec 2026 | Second wave of posts (“4 weeks to the EGU27 deadline”) |
| 5–11 Jan 2027 | Final reminder posts; update the countdown copy if needed |
| After 12 Jan 2027 | Optionally re-crawl merged sessions; update the page to point to EGU28 |

## 6. Measuring success without tracking users

The site deliberately has no analytics. Use Search Console impressions and
clicks, and the GitHub repository traffic page. GitHub Pages provides no visitor
statistics, so if visit counts are wanted, add a privacy-respecting, cookie-less
counter (e.g. GoatCounter) and disclose it on the privacy page.
