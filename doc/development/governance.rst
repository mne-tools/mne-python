.. _governance:

==================
Project Governance
==================

The purpose of this document is to formalize the governance process
used by the MNE-Python project in both ordinary and extraordinary
situations, and to clarify how decisions are made and how the various
elements of our community interact, including the relationship between
open source collaborative development and work that may be funded by
for-profit or non-profit entities.

The Project
===========

The MNE-Python Project (The Project) is an open source software project. The
goal of The Project is to develop open source software for analysis of
neuroscience data in Python. The Project is released under the BSD (or similar)
open source license, developed openly and is hosted publicly under the
``mne-tools`` GitHub organization.

The Project is developed by a team of distributed developers, called
Contributors. Contributors are individuals who have contributed code,
documentation, designs, or other work to the Project. Anyone can be a
Contributor. Contributors can be affiliated with any legal entity or
none. Contributors participate in the project by submitting, reviewing,
and discussing GitHub Pull Requests and Issues and participating in open
and public Project discussions on GitHub, Discourse, and other
channels. The foundation of Project participation is openness and
transparency.

The Project Community consists of all Contributors and Users of the
Project. Contributors work on behalf of and are responsible to the
larger Project Community and we strive to keep the barrier between
Contributors and Users as low as possible.

The Project is not a legal entity, nor does it currently have any formal
relationships with legal entities.

Governance Model
================

.. _leadership-roles:

Leadership Roles
^^^^^^^^^^^^^^^^

The MNE-Python leadership structure shall consist of the following groups.
A list of the current members of the respective groups is maintained at the
page :ref:`governance-people`.

Maintainer Team
---------------

The Maintainer Team is responsible for implementing changes to the software and
supporting the user community. Duties:

- Infrastructure/large-scale software decisions, in partnership with the Steering
  Council
- Reviewing and merging pull requests
- Responding to issues on GitHub
- Monitoring CI failures and addressing them
- Community maintenance: answering forum posts, holding office hours
- Community information: social media announcements (releases, new features, etc)
- Training new members of the Maintainer Team

*Note:* different permissions may be given to each maintainer based on the work they do
(e.g., GitHub repository triage/merge/admin permissions, social media account access,
Discord admin roles, forum admin rights). The role of maintainer does not confer these
automatically.

Steering Council
----------------

The Steering Council is responsible for guiding and shepherding the project on a
day-to-day basis. Duties:

- Obtaining funding (either by writing grants specifically for MNE development, or
  convincing others to include funds for MNE development in their research grants)
- Translating high-level roadmap guidance from the Advisory Board (e.g. “better support
  for OPMs”) into actionable roadmap items (e.g., “Add support for OPM manufacturers
  besides QuSpin, and add standard preprocessing routines for coreg and OPM-specific
  artifacts”)
- Coordination with the larger Scientific Python ecosystem
- Large-scale changes to the software (e.g., type hints, docdict, things that affect
  multiple submodules), in partnership with the Maintainer Team
- Infrastructure decisions (e.g., dependency version policy, release cadence, CI
  management, etc), in partnership with the Maintainer Team
- Any other governance task not mentioned elsewhere, and that falls outside of the
  responsibilities of other teams
- Attendance at Steering Council meetings (approx. every 2 weeks; time to be decided
  among SC members)
- Attendance at Advisory Board meetings (approx. every 1-2 years)
- Write funding proposals
- Communicate/coordinate with Maintainer Team

Members of the Steering Council shall additionally be considered as members of the
Maintainer Team, *ex officio*, and thus shall have the necessary rights and privileges
afforded to maintainers (passwords, merge rights, etc).

Chair of the Steering Council
-----------------------------

The Chair of the Steering Council is responsible for liaising between the Steering
Council and the community. Duties:

- Convening the Steering Council meetings
- Calling for votes when consensus fails
- Communicating important decisions (and the context for why those decisions were
  taken) to the community

External Advisory Board
-----------------------

The External Advisory Board is responsible for high-level roadmap and funding
guidance. Duties:

- Attendance at Advisory Board meetings (approx. every 1-2 years)
- Periodically communicating with Steering Council to impart guidance

Meetings
^^^^^^^^

Maintainer Meetings
-------------------

The Maintainer Team can decide if there should be maintainer meetings or not. These
could be either discussion meetings or social meetings to keep in touch with each other
(or something completely different!). Frequency and time could vary.

Steering Council Meetings
-------------------------

The Steering Council will have mandatory meetings every two weeks to discuss project
management and funding. The Steering Council may decide to change meeting time or
frequency at their discretion.

All-hands Meetings
------------------

At least once a year, all maintainers and Steering Council members should come together
in a (possibly virtual) meeting. Meeting time will be determined via poll. During this
meeting, any governance changes proposed since the prior meeting shall be discussed and
may be adopted by vote.

Population of Leadership Roles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maintainer Team
---------------

Appointment
~~~~~~~~~~~

- Once per year, before the annual all-hands meeting, the Maintainer Team shall assess
  their anticipated needs for the coming year. During the meeting, they shall decide how
  many new maintainers they want to add to their team. New maintainers are selected from
  applications by a vote of the currently-serving maintainers and the Steering Council.
- Ad-hoc additions are possible by initiative of the Steering Council for exceptional
  circumstances, e.g., hiring someone with grant funds to specifically do MNE
  maintenance. These ad-hoc additions do not necessarily require a vote from the
  Maintainer Team, as the job application procedures of the hiring institution are
  assumed to be sufficiently rigorous to avoid bias, nepotism, etc.

Duration of service
~~~~~~~~~~~~~~~~~~~

Appointment to the Maintainer Team is for an indefinite term.

Termination
~~~~~~~~~~~

Loss of maintainer status (and revocation of associated rights and
privileges, e.g., passwords, merge rights, etc) can occur under the following
circumstances:

- *Voluntary resignation*, at any time, for any reason.
- *Inactivity*. Once per year, before the annual all-hands meeting, the Steering Council
  shall assess maintainer activity for the preceding year. Any maintainers seeming to be
  inactive shall be contacted and given opportunity to dispute their inactivity (e.g.,
  by highlighting ways they have been working in the MNE community that may not be
  visible from simple metrics like GitHub or forum activity reports). Maintainers who do
  not dispute their inactivity (or fail to respond within 14 days, or longer at the
  discretion of the Steering Council) shall be removed from the Maintainer Team. In
  cases where the Steering Council and the maintainer still disagree about the alleged
  inactivity, removal may still occur by a two-thirds majority vote of the rest of the
  Maintainer Team.
- *Conduct*. At any time, a maintainer may be removed by unanimous vote of the
  code-of-conduct committee, for violations of our community guidelines (in accordance
  with the enforcement guidelines outlined therein).

*Reinstatement*. Maintainers who voluntarily resigned may be re-appointed on an ad-hoc
basis by a vote of the current Maintainer Team. Maintainers removed for inactivity may
re-apply to an annual call for new maintainers. Maintainers removed for reasons of
conduct may be reinstated only if their eligibility is allowed/restored by the
code-of-conduct committee in accordance with the enforcement section of our Community
Guidelines. In such cases the re-eligible former maintainer may re-apply through the
annual appointment process.

Steering Council
----------------

Appointment
~~~~~~~~~~~
A term on the Steering Council shall last approximately 2 years. Terms shall be
staggered such that no more than half of the seats shall be open for election in any
given year. Upon first constitution, the Steering Council decides which 50% of the
members shall be granted an initial three year term to initiate the staggering.

The Maintainer Team and current Steering Council will vote to fill the open seats on the
Steering Council. Candidates can be (self-)nominated from the current Maintainer Team
and Steering Council.

At any time, the Steering Council may increase the number of seats on the Council to
adapt to the Council’s workload and needs. New seats shall be filled in the same manner
as normal (re-)elections, i.e., by vote open to all members of the Maintainer Team and
Steering Council. Term length shall be set so as to maintain the 50/50 balance of
staggered re-election cycles as nearly as possible, and in cases where perfect balance
already exists, the term shall err towards being *longer*.

In the case of vacancies due to termination (see below), the Steering Council may call a
special election (following the same procedures as in a normal (re-)election), or may
choose to wait to fill the seat until the next scheduled election. For filled vacancies,
the term shall be the balance of the unserved term of the person vacating the seat,
unless the remaining time after the vacancy-filling election is 6 months or shorter, in
which case the term shall be for 2 years plus the remaining time on the vacant seat.

Termination
~~~~~~~~~~~

Loss of Steering Council status (and revocation of associated rights and privileges,
e.g., passwords, merge rights, etc) can occur under the following circumstances:

- *Voluntary resignation*, at any time, for any reason.
- *Conduct*. At any time, a member of the Steering Council may be removed by unanimous
  vote of the code-of-conduct committee, for violations of our community guidelines (in
  accordance with the enforcement guidelines outlined therein).

External Advisory Board
-----------------------

The External Advisory Board shall be populated by invitation from the Steering Council.
Anyone may propose individuals for potential invitation. Appointment and removal from
the External Advisory Board is determined by the Steering Council.

Decision Making Process
^^^^^^^^^^^^^^^^^^^^^^^

Announcement of Elections
-------------------------

All votes shall be open for at least ten days and shall be announced 14 days in advance
to all eligible voters by email. The voting deadline shall also be added to the core
team’s shared Google calendar. At least one reminder shall be sent out half-way through
the voting period.

Voting Mechanism
----------------

All elections shall be held as anonymous online votes using ElectionBuddy or a similar
service. Unless otherwise specified the mechanism shall be
`ranked choice voting <https://en.wikipedia.org/wiki/Instant-runoff_voting>`__
with a threshold of 50% + 1 vote. That means, everyone ranks those candidates (in order
of preference) that they could see filling the role in question. Note that it is
possible for a voter to reject all candidates by submitting a blank ballot, so that if a
single person is running for a seat it is still possible for them to fail to be elected
if enough voters cast blank ballots.

Voting for the Steering Council
-------------------------------

Votes for Steering Council membership shall be scheduled as-needed to address Steering
Council workload, and advertised to eligible candidates (i.e., the Maintainer Team) for
a minimum of 14 days, after which a vote of current maintainers and Steering Council
members shall be scheduled.

Voting for the Maintainer Team
------------------------------

Votes for additions to the Maintainer Team shall be scheduled promptly following the
annual all-hands meeting. The Maintainer Team shall advertise the open seats via online
MNE-Python channels. Applications (consisting of a short candidate statement) must be
open for a minimum of 14 days, after which a vote of the current maintainers and
Steering Council shall be scheduled. The Maintainer Team shall set up a confidential
submission system for applications (consisting of short candidate statements), such as a
dedicated email address, Google form, or similar confidential submission mechanism.

Institutional Partners and Funding
==================================

The leadership roles for the project are :ref:`defined above <leadership-roles>`. No
outside institution, individual, or legal entity has the ability to own,
control, usurp, or influence the project other than by participating in
the Project in one of those roles. However, because
institutions can be an important funding mechanism for the project, it
is important to formally acknowledge institutional participation in the
project. These are Institutional Partners.

An Institutional Contributor is any individual Project Contributor who
contributes to the project as part of their official duties at an
Institutional Partner. Likewise, an Institutional Project Leader is anyone
in a Project leadership role who contributes to the project as part
of their official duties at an Institutional Partner.

With these definitions, an Institutional Partner is any recognized legal
entity in any country that employs at least 1 Institutional Contributor or
Institutional Project Leader. Institutional Partners can be for-profit or
non-profit entities.

Institutions become eligible to become an Institutional Partner by
employing individuals who actively contribute to The Project as part of
their official duties. To state this another way, the only way for a
Partner to influence the project is by actively contributing to the open
development of the project, in equal terms to any other member of the
community of Contributors and Leaders. Merely using Project
Software in institutional context does not allow an entity to become an
Institutional Partner. Financial gifts do not enable an entity to become
an Institutional Partner. Once an institution becomes eligible for
Institutional Partnership, the Steering Council must nominate and
approve the Partnership.

If, at some point, an existing Institutional Partner stops having any
contributing employees, then a one year grace period commences. If, at
the end of this one-year period, they continue not to have any
contributing employees, then their Institutional Partnership will
lapse, and resuming it will require going through the normal process
for new Partnerships.

An Institutional Partner is free to pursue funding for their work on The
Project through any legal means. This could involve a non-profit
organization raising money from private foundations and donors or a
for-profit company building proprietary products and services that
leverage Project Software and Services. Funding acquired by
Institutional Partners to work on The Project is called Institutional
Funding. However, no funding obtained by an Institutional Partner can
override Project Leadership. If a Partner has funding to do MNE-Python work
and the Project Leadership decides to not pursue that work as a project, the
Partner is free to pursue it on their own. However, in this situation,
that part of the Partner’s work will not be under the MNE-Python umbrella and
cannot use the Project trademarks in any way that suggests a formal
relationship.

Institutional Partner benefits are:

- optional acknowledgement on the MNE-Python website and in talks
- ability to acknowledge their own funding sources on the MNE-Python
  website and in talks
- ability to influence the project through the participation of their
  Institutional Contributors and Institutional Project Leaders.
- invitation of the Council Members to MNE-Python Developer Meetings

A list of current Institutional Partners is maintained at the page
:ref:`supporting-institutions`.

Document History
================

https://github.com/mne-tools/mne-python/commits/main/doc/overview/governance.rst


Acknowledgements
================

Substantial portions of this document were adapted from the
`SciPy project's governance document
<https://github.com/scipy/scipy/blob/main/doc/source/dev/governance.rst>`_,
which in turn was adapted from
`Jupyter/IPython project's governance document
<https://github.com/jupyter/governance/blob/main/archive/governance.md>`_ and
`NumPy's governance document
<https://github.com/numpy/numpy/blob/master/doc/source/dev/governance/governance.rst>`_.

License
=======

To the extent possible under law, the authors have waived all
copyright and related or neighboring rights to the MNE-Python project
governance document, as per the `CC-0 public domain dedication / license
<https://creativecommons.org/publicdomain/zero/1.0/>`_.
