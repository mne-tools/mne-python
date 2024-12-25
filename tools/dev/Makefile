.PHONY:

dev-activity: check_steering_committee.py
	python check_steering_committee.py

unacknowledged-bug-reports: unacknowledged-issues.jq bug-reports-12-to-2-months-old.json
	@echo "MUST acknowledge bug reports (OpenSSF criterion: 50%)"
	@jq -f unacknowledged-issues.jq bug-reports-12-to-2-months-old.json

unacknowledged-feature-requests: unacknowledged-issues.jq feature-requests-12-to-2-months-old.json
	@echo "SHOULD acknowledge feature requests (OpenSSF criterion: 50%)"
	@jq -f unacknowledged-issues.jq feature-requests-12-to-2-months-old.json

bug-reports-12-to-2-months-old.json:
	@echo "Querying GitHub REST API..."
	@gh issue list \
	-l bug \
	--json state,number,title,createdAt,comments > bug-reports-12-to-2-months-old.json

feature-requests-12-to-2-months-old.json:
	@echo "Querying GitHub REST API..."
	@gh issue list \
	-l enh \
	--json state,number,title,createdAt,comments > feature-requests-12-to-2-months-old.json

clean:
	@rm bug-reports-12-to-2-months-old.json || true

dep:
	cd ../../ && git grep -iI "\(deprecat\|futurewarning\)" -- ':!*.js' ':!*/conftest.py' ':!*/docs.py' ':!*/test_docs.py' ':!*/utils/__init__.pyi' mne/
