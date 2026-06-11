:orphan:

.. _funding:

Sponsors
========

Maintenance and development of MNE-Python is currently supported by the following funding agencies and partners:

.. rst-class:: list-unstyled current

- |nih| **National Institutes of Health:**
  `R01-NS104585 <https://reporter.nih.gov/project-details/10175064>`_
- |nsf| **US National Science Foundation:**
  `2449064 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=2449064>`_


Past sponsors
-------------

.. rst-class:: list-unstyled former

- |nih| National Institutes of Health:
  `R01-EB009048 <https://reporter.nih.gov/project-details/9053482>`_,
  `R01-EB006385 <https://reporter.nih.gov/project-details/8105475>`_,
  `R01-HD040712 <https://reporter.nih.gov/project-details/8511739>`_,
  `R01-NS044319 <https://reporter.nih.gov/project-details/6924553>`_,
  `R01-NS037462 <https://reporter.nih.gov/project-details/9083237>`_,
  `P41-EB015896 <https://reporter.nih.gov/project-details/9518908>`_,
  `P41-RR014075 <https://reporter.nih.gov/project-details/8098820>`_
- |nsf| US National Science Foundation:
  `0958669 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=0958669>`_,
  `1042134 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1042134>`_
- |erc| |erc_dk| European Research Council:
  `YStG-263584 <https://erc.easme-web.eu/?p=263584>`_,
  `YStG-676943 <https://erc.easme-web.eu/?p=676943>`_
- |doe| US Department of Energy: DE-FG02-99ER62764 (MIND)
- |anr| Agence Nationale de la Recherche:
  `14-NEUC-0002-01 <https://anr.fr/Project-ANR-14-NEUC-0002>`_,
  `11-IDEX-0003-02 <https://anr.fr/ProjetIA-11-IDEX-0003>`_
- |cds| |cds_dk| Paris-Saclay Center for Data Science:
  `PARIS-SACLAY <http://www.datascience-paris-saclay.fr>`_
- |google| Google:
  Summer of code (×8 years)
- |amazon| Amazon:
  AWS Research Grants
- |czi| Chan Zuckerberg Initiative:
  `EOSS2`_,
  `EOSS4`_


.. _supporting-institutions:

Supporting institutions
=======================

Some institutions support their employees’ contributions to MNE-Python as part of normal work duties. Current supporting institutions include:

.. rst-class:: list-unstyled current

{% for inst in current_institutions %}
{% if not inst.name.endswith("_dk") %}
- |{{ inst.name }}|{% if inst.name ~ "_dk" in current_sponsors_partners %} |{{ inst.name ~ "_dk" }}|{% endif %} `{{ inst.title }} <{{ inst.url }}>`_
{% endif %}
{% endfor %}


Past supporting institutions
----------------------------

.. rst-class:: centered

{% for inst in former_institutions %}|{{ inst.name }}| {% endfor %}

.. ↑↑↑ We need to do this roundabout approach of using substitutions here
..     (and defining them below in another loop) so that the resulting images end up
..     wrapped in a <p> tag, which can then get the `centerlast` class.

.. FUNDER SUBSTITUTION DEFS
{% for item in all_sponsors -%}

.. |{{ item.name }}| image:: ../_static/funding/{{ item.img }}
    :alt: {{ item.title }}
    {% if item.klass is defined %}:class: {{ item.klass }}{% endif %}
{% endfor %}

.. INSTITUTION SUBSTITUTION DEFS
{% for item in all_institutions %}

.. |{{ item.name }}| image:: ../_static/institution_logos/{{ item.img }}
    :alt: {{ item.title }}
    :target: {{ item.url }}
    :class: instlogo{% if item.klass is defined %} {{ item.klass }}{% endif %}
{% endfor %}
