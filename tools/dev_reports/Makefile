.PHONY:

dev-activity: check_steering_committee.py
	python check_steering_committee.py

unacknowledged-bugs: unacknowledged-bug-reports.jq bug-reports-12-to-2-months-old.json
	@jq -f unacknowledged-bug-reports.jq bug-reports-12-to-2-months-old.json

bug-reports-12-to-2-months-old.json:
	@echo "Querying GitHub REST API..."
	@gh issue list \
	-l bug \
	--json state,number,title,createdAt,comments > bug-reports-12-to-2-months-old.json

clean:
	@rm bug-reports-12-to-2-months-old.json || true
