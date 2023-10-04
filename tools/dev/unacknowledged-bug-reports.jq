# Processor for `gh issue list` output that displays unacknowledged bug reports
# that are 2-12 months old. The date range is specific to OpenSSF best practices.

# `now` is in seconds since the unix epoch
def one_year_ago: now - (365 * 24 * 60 * 60);

def sixty_days_ago: now - (60 * 24 * 60 * 60);

def date_fmt: "%Y/%m/%d";

def make_pretty_date_range:
    (one_year_ago | strftime(date_fmt)) + " - " + (sixty_days_ago | strftime(date_fmt));

def make_issue_url: "https://github.com/mne-tools/mne-python/issues/\(.number)";

def get_dev_comments: .comments | map(select(.authorAssociation == "MEMBER"));


# main routine
map(
    select(
        (.createdAt > (one_year_ago | todate)) and
        (.createdAt < (sixty_days_ago | todate))
    ) +=
    { "devComments": . | get_dev_comments | length }
) |
{
    "range": make_pretty_date_range,
    "has_dev_comments": map(select(.devComments > 0)) | length,
    "no_dev_comments": map(select(.devComments == 0) and .state == "OPEN") | length,
    "unaddressed_bug_reports": map(select(.devComments == 0) | make_issue_url),
}
