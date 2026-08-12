use std::cmp::Ordering;
use std::collections::HashSet;

use crate::client::SearxngResult;

struct Authority {
    aliases: &'static [&'static str],
    domains: &'static [&'static str],
    github_repos: &'static [(&'static str, &'static str)],
}

const AUTHORITIES: &[Authority] = &[
    Authority {
        aliases: &["react"],
        domains: &["react.dev"],
        github_repos: &[("facebook", "react")],
    },
    Authority {
        aliases: &["next.js", "nextjs", "next js"],
        domains: &["nextjs.org"],
        github_repos: &[("vercel", "next.js")],
    },
    Authority {
        aliases: &["python"],
        domains: &["python.org"],
        github_repos: &[("python", "cpython")],
    },
    Authority {
        aliases: &["rust"],
        domains: &["rust-lang.org"],
        github_repos: &[("rust-lang", "rust")],
    },
    Authority {
        aliases: &["node.js", "nodejs", "node js"],
        domains: &["nodejs.org"],
        github_repos: &[("nodejs", "node")],
    },
    Authority {
        aliases: &["vue", "vue.js", "vuejs"],
        domains: &["vuejs.org"],
        github_repos: &[("vuejs", "core")],
    },
    Authority {
        aliases: &["angular"],
        domains: &["angular.dev", "angular.io"],
        github_repos: &[("angular", "angular")],
    },
    Authority {
        aliases: &["fastapi", "fast api"],
        domains: &["fastapi.tiangolo.com"],
        github_repos: &[("fastapi", "fastapi")],
    },
    Authority {
        aliases: &["django"],
        domains: &["djangoproject.com"],
        github_repos: &[("django", "django")],
    },
    Authority {
        aliases: &["laravel"],
        domains: &["laravel.com"],
        github_repos: &[("laravel", "framework")],
    },
    Authority {
        aliases: &["postgresql", "postgres"],
        domains: &["postgresql.org"],
        github_repos: &[("postgres", "postgres")],
    },
    Authority {
        aliases: &["openai"],
        domains: &["openai.com"],
        github_repos: &[],
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct Version(Vec<u32>);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum VersionTier {
    Mismatch,
    Neutral,
    Match,
}

#[derive(Clone, Debug, PartialEq)]
struct RankKey {
    version: VersionTier,
    authority: bool,
    lexical: usize,
    freshness: u32,
    upstream: f64,
}

struct QueryIntent<'a> {
    authority: Option<&'a Authority>,
    entity_tokens: Vec<String>,
    version: Option<Version>,
    terms: HashSet<String>,
}

pub fn rerank_technical<'a>(rows: &'a [SearxngResult], query: &str) -> Vec<&'a SearxngResult> {
    let intent = QueryIntent::parse(query);
    let mut ranked: Vec<(usize, &SearxngResult, RankKey)> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (index, row, rank_key(row, &intent)))
        .collect();

    ranked.sort_by(|a, b| compare_keys(&b.2, &a.2).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(_, row, _)| row).collect()
}

impl QueryIntent<'_> {
    fn parse(query: &str) -> QueryIntent<'static> {
        let normalized = normalize(query);
        let tokens = tokens(&normalized);
        let authority = AUTHORITIES
            .iter()
            .filter_map(|entry| {
                entry
                    .aliases
                    .iter()
                    .filter(|alias| contains_phrase(&normalized, alias))
                    .map(|alias| (entry, alias.len()))
                    .max_by_key(|(_, len)| *len)
            })
            .max_by_key(|(_, len)| *len)
            .map(|(entry, _)| entry);

        let (entity_tokens, version) = authority
            .and_then(|entry| parse_known_version(&tokens, entry))
            .map(|(entity, version)| (entity, Some(version)))
            .unwrap_or_else(|| parse_explicit_version(&tokens));

        let terms = tokens
            .iter()
            .filter(|token| is_search_term(token))
            .cloned()
            .collect();

        QueryIntent {
            authority,
            entity_tokens,
            version,
            terms,
        }
    }
}

fn rank_key(row: &SearxngResult, intent: &QueryIntent<'_>) -> RankKey {
    let title = normalize(row.title.as_deref().unwrap_or(""));
    let url = normalize(row.url.as_deref().unwrap_or(""));
    let content = normalize(row.content.as_deref().unwrap_or(""));
    let authority = intent
        .authority
        .is_some_and(|entry| is_official(row.url.as_deref().unwrap_or(""), entry));
    let versions = candidate_versions(&title, &url, &content, intent, authority);
    let version = intent
        .version
        .as_ref()
        .map_or(VersionTier::Neutral, |requested| {
            classify_versions(requested, &versions)
        });

    RankKey {
        version,
        authority,
        lexical: lexical_score(&intent.terms, &title, &url, &content),
        freshness: parse_date(row.published_date.as_deref()),
        upstream: row.score.unwrap_or(0.0),
    }
}

fn compare_keys(a: &RankKey, b: &RankKey) -> Ordering {
    a.version
        .cmp(&b.version)
        .then_with(|| a.authority.cmp(&b.authority))
        .then_with(|| a.lexical.cmp(&b.lexical))
        .then_with(|| a.freshness.cmp(&b.freshness))
        .then_with(|| {
            a.upstream
                .partial_cmp(&b.upstream)
                .unwrap_or(Ordering::Equal)
        })
}

fn parse_known_version(
    query_tokens: &[String],
    entry: &Authority,
) -> Option<(Vec<String>, Version)> {
    for alias in entry.aliases {
        let alias_tokens = tokens(alias);
        for start in phrase_positions(query_tokens, &alias_tokens) {
            let after = start + alias_tokens.len();
            if let Some(version) = version_near(query_tokens, after, true) {
                return Some((alias_tokens, version));
            }
        }
    }
    None
}

fn parse_explicit_version(tokens: &[String]) -> (Vec<String>, Option<Version>) {
    for (index, token) in tokens.iter().enumerate() {
        if let Some(version) = explicit_version(token).or_else(|| {
            (token == "version")
                .then(|| tokens.get(index + 1).and_then(|next| bare_version(next)))
                .flatten()
        }) {
            let end = index;
            let entity = tokens[..end]
                .iter()
                .rev()
                .filter(|token| is_search_term(token))
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return (entity, Some(version));
        }
    }
    (Vec::new(), None)
}

fn candidate_versions(
    title: &str,
    url: &str,
    content: &str,
    intent: &QueryIntent<'_>,
    is_authority: bool,
) -> Vec<Version> {
    let mut found = Vec::new();
    for text in [title, content] {
        let text_tokens = tokens(text);
        if is_authority {
            collect_explicit_versions(&text_tokens, &mut found);
        }
        for start in phrase_positions(&text_tokens, &intent.entity_tokens) {
            let after = start + intent.entity_tokens.len();
            for index in start.saturating_sub(2)..start {
                if let Some(version) = version_at(&text_tokens, index, true) {
                    found.push(version);
                }
            }
            for index in after..(after + 3).min(text_tokens.len()) {
                if let Some(version) = version_at(&text_tokens, index, true) {
                    found.push(version);
                }
            }
        }
    }
    if is_authority && let Ok(parsed) = url::Url::parse(url) {
        let mut previous = None;
        for segment in parsed.path_segments().into_iter().flatten() {
            let version = explicit_version(segment).or_else(|| {
                (segment.contains('.') || matches!(previous, Some("version" | "versions")))
                    .then(|| bare_version(segment))
                    .flatten()
            });
            if let Some(version) = version {
                found.push(version);
            }
            previous = Some(segment);
        }
    }
    found
}

fn collect_explicit_versions(tokens: &[String], output: &mut Vec<Version>) {
    for index in 0..tokens.len() {
        let version = explicit_version(&tokens[index]).or_else(|| {
            (tokens[index] == "version")
                .then(|| tokens.get(index + 1).and_then(|value| bare_version(value)))
                .flatten()
        });
        if let Some(version) = version {
            output.push(version);
        }
    }
}

fn version_near(tokens: &[String], index: usize, allow_bare: bool) -> Option<Version> {
    for cursor in index..(index + 3).min(tokens.len()) {
        if let Some(version) = version_at(tokens, cursor, allow_bare) {
            return Some(version);
        }
    }
    None
}

fn version_at(tokens: &[String], index: usize, allow_bare: bool) -> Option<Version> {
    let token = tokens.get(index)?;
    explicit_version(token)
        .or_else(|| {
            (token == "version")
                .then(|| tokens.get(index + 1).and_then(|v| bare_version(v)))
                .flatten()
        })
        .or_else(|| allow_bare.then(|| bare_version(token)).flatten())
}

fn explicit_version(token: &str) -> Option<Version> {
    token.strip_prefix('v').and_then(bare_version)
}

fn bare_version(token: &str) -> Option<Version> {
    if token.is_empty() || !token.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let parts: Option<Vec<u32>> = token.split('.').map(|part| part.parse().ok()).collect();
    let parts = parts?;
    if parts.is_empty() || parts.len() > 3 || (1900..=2099).contains(&parts[0]) {
        return None;
    }
    Some(Version(parts))
}

fn classify_versions(requested: &Version, candidates: &[Version]) -> VersionTier {
    if candidates
        .iter()
        .any(|candidate| version_matches(requested, candidate))
    {
        VersionTier::Match
    } else if candidates
        .iter()
        .any(|candidate| version_conflicts(requested, candidate))
    {
        VersionTier::Mismatch
    } else {
        VersionTier::Neutral
    }
}

fn version_matches(requested: &Version, candidate: &Version) -> bool {
    candidate.0.len() >= requested.0.len() && candidate.0[..requested.0.len()] == requested.0[..]
}

fn version_conflicts(requested: &Version, candidate: &Version) -> bool {
    let shared = requested.0.len().min(candidate.0.len());
    requested.0[..shared] != candidate.0[..shared]
}

fn is_official(raw_url: &str, entry: &Authority) -> bool {
    let Ok(url) = url::Url::parse(raw_url) else {
        return false;
    };
    let host = url
        .host_str()
        .unwrap_or("")
        .trim_start_matches("www.")
        .to_ascii_lowercase();
    if entry
        .domains
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
    {
        return true;
    }
    if host != "github.com" {
        return false;
    }
    let segments: Vec<String> = url
        .path_segments()
        .into_iter()
        .flatten()
        .take(2)
        .map(|segment| segment.to_ascii_lowercase())
        .collect();
    segments.len() == 2
        && entry
            .github_repos
            .iter()
            .any(|(owner, repo)| segments[0] == *owner && segments[1] == *repo)
}

fn lexical_score(terms: &HashSet<String>, title: &str, url: &str, content: &str) -> usize {
    let title_tokens: HashSet<String> = tokens(title).into_iter().collect();
    let url_tokens: HashSet<String> = tokens(url).into_iter().collect();
    let content_tokens: HashSet<String> = tokens(content).into_iter().collect();
    terms
        .iter()
        .map(|term| {
            usize::from(title_tokens.contains(term)) * 4
                + usize::from(url_tokens.contains(term)) * 2
                + usize::from(content_tokens.contains(term))
        })
        .sum()
}

fn parse_date(value: Option<&str>) -> u32 {
    let Some(date) = value.and_then(|value| value.get(..10)) else {
        return 0;
    };
    if date.as_bytes().get(4) != Some(&b'-') || date.as_bytes().get(7) != Some(&b'-') {
        return 0;
    }
    date.chars()
        .filter(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

fn normalize(value: &str) -> String {
    value.to_ascii_lowercase()
}

fn tokens(value: &str) -> Vec<String> {
    normalize(value)
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
        .map(|token| token.trim_matches('.'))
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn contains_phrase(value: &str, phrase: &str) -> bool {
    !phrase_positions(&tokens(value), &tokens(phrase)).is_empty()
}

fn phrase_positions(haystack: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, window)| (window == needle).then_some(index))
        .collect()
}

fn is_search_term(token: &str) -> bool {
    ![
        "a",
        "an",
        "and",
        "current",
        "documentation",
        "docs",
        "for",
        "how",
        "in",
        "latest",
        "official",
        "or",
        "the",
        "to",
        "version",
        "with",
    ]
    .contains(&token)
        && explicit_version(token).is_none()
        && bare_version(token).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(url: &str, title: &str, content: &str, score: f64) -> SearxngResult {
        SearxngResult {
            url: Some(url.into()),
            title: Some(title.into()),
            engine: Some("test".into()),
            content: Some(content.into()),
            score: Some(score),
            engines: Vec::new(),
            positions: Vec::new(),
            category: Some("general".into()),
            template: None,
            published_date: None,
            img_src: None,
            thumbnail_src: None,
            img_format: None,
            resolution: None,
        }
    }

    #[test]
    fn matching_version_then_official_source_dominate_upstream_score() {
        let rows = vec![
            row("https://blog.test/react-18", "React 18 guide", "", 10.0),
            row(
                "https://react.dev/reference",
                "React documentation",
                "",
                0.2,
            ),
            row(
                "https://react.dev/versions/19",
                "React 19 reference",
                "",
                0.1,
            ),
        ];
        let ranked = rerank_technical(&rows, "React 19 documentation");
        assert_eq!(ranked[0].title.as_deref(), Some("React 19 reference"));
        assert_eq!(ranked[1].title.as_deref(), Some("React documentation"));
        assert_eq!(ranked[2].title.as_deref(), Some("React 18 guide"));
    }

    #[test]
    fn arbitrary_docs_and_github_urls_are_not_official() {
        let intent = QueryIntent::parse("React 19 docs");
        let fake_docs = row(
            "https://docs.react-guide.test/v19",
            "React 19 docs",
            "",
            1.0,
        );
        let fake_repo = row("https://github.com/random/react", "React 19", "", 1.0);
        let official = row(
            "https://github.com/facebook/react/releases/tag/v19",
            "React 19",
            "",
            0.1,
        );
        assert!(!rank_key(&fake_docs, &intent).authority);
        assert!(!rank_key(&fake_repo, &intent).authority);
        assert!(rank_key(&official, &intent).authority);
    }

    #[test]
    fn version_is_bound_to_the_query_entity() {
        let rows = vec![
            row(
                "https://example.test/python",
                "Python asyncio with Django 5",
                "",
                9.0,
            ),
            row(
                "https://docs.python.org/3.12/library/asyncio.html",
                "asyncio",
                "",
                0.1,
            ),
        ];
        let intent = QueryIntent::parse("Python 3.12 asyncio documentation");
        assert_eq!(rank_key(&rows[0], &intent).version, VersionTier::Neutral);
        let ranked = rerank_technical(&rows, "Python 3.12 asyncio documentation");
        assert_eq!(
            ranked[0].url.as_deref(),
            Some("https://docs.python.org/3.12/library/asyncio.html")
        );
    }

    #[test]
    fn unrelated_explicit_version_does_not_create_a_mismatch() {
        let intent = QueryIntent::parse("Python 3.12 asyncio documentation");
        let unrelated = row(
            "https://example.test/asyncio",
            "Asyncio integration with Django v5",
            "",
            1.0,
        );
        assert_eq!(rank_key(&unrelated, &intent).version, VersionTier::Neutral);
    }

    #[test]
    fn numeric_url_segment_is_not_assumed_to_be_a_version() {
        let intent = QueryIntent::parse("React 19 documentation");
        let error_page = row(
            "https://react.dev/errors/404",
            "React error reference",
            "",
            1.0,
        );
        assert_eq!(rank_key(&error_page, &intent).version, VersionTier::Neutral);
    }

    #[test]
    fn ambiguous_hermes_version_is_not_bound_to_a_registered_authority() {
        let intent = QueryIntent::parse("Hermes Agent DeepSeek V4 support");
        assert!(intent.authority.is_none());
        assert_eq!(intent.version, Some(Version(vec![4])));
    }

    #[test]
    fn query_precision_controls_version_match() {
        assert!(version_matches(&Version(vec![4]), &Version(vec![4, 2])));
        assert!(!version_matches(&Version(vec![4, 1]), &Version(vec![4, 2])));
        assert!(!version_conflicts(&Version(vec![4, 1]), &Version(vec![4])));
        assert!(version_conflicts(
            &Version(vec![4, 1]),
            &Version(vec![4, 2])
        ));
    }

    #[test]
    fn newer_date_only_breaks_an_otherwise_equal_rank() {
        let mut old = row("https://a.test/react", "React guide", "", 1.0);
        old.published_date = Some("2024-01-01T00:00:00Z".into());
        let mut new = row("https://b.test/react", "React guide", "", 1.0);
        new.published_date = Some("2026-01-01T00:00:00Z".into());
        let rows = vec![old, new];
        let ranked = rerank_technical(&rows, "React documentation");
        assert_eq!(ranked[0].url.as_deref(), Some("https://b.test/react"));
    }

    #[test]
    fn equal_rank_preserves_upstream_order() {
        let rows = vec![
            row("https://a.test", "React", "", 1.0),
            row("https://b.test", "React", "", 1.0),
        ];
        let ranked = rerank_technical(&rows, "React docs");
        assert_eq!(ranked[0].url.as_deref(), Some("https://a.test"));
    }
}
