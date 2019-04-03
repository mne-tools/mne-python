{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :special-members: __contains__,__getitem__,__iter__,__len__,__add__,__sub__,__mul__,__div__,__neg__,__hash__

   {% block methods %}
   {% endblock %}

.. include:: {{module}}.{{objname}}.examples
