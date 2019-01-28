:orphan:

.. _documentation:

Documentation
=============

.. raw:: html

    <style class='text/css'>
    .panel-title a {
        display: block;
        padding: 5px;
        text-decoration: none;
    }

    .plus {
        float: right;
        color: #212121;
    }

    .panel {
        margin-bottom: 3px;
    }

    .example_details {
        padding-left: 20px;
        margin-bottom: 10px;
    }
    </style>

    <script type="text/javascript">
    $(document).ready(function () {
        if(location.hash != null && location.hash != ""){
            $('.collapse').removeClass('in');
            $(location.hash + '.collapse').addClass('in');
        }
    });
    </script>

.. toctree::
    :maxdepth: 1

    workflows.rst
    tutorials.rst
    auto_examples/index.rst
    faq.rst
    python_reference.rst
    generated/commands.rst
    glossary.rst
    auto_tutorials/plot_configuration.rst
    whats_new.rst
    cited.rst

The **workflows** will get you started on defining which type of pipeline you should aim for in your data analysis: evoked data analysis in sensor space and in source space, time-frequency analysis in sensor space and in sensor space. Each step in a given workflow will link to the relevant tutorials and example codes.

The **tutorials** provide in-depth reviews of specific analyses (e.g. preprocessing, epoching, decoding) with narrative documentation and comments detailed ordered by topicx`.

The **code examples** provides you elaborate snippet codes that go through importing your raw data to the outcome of a planned analysis. The mne-study template drives you through the full pipeline of analysis on sample data.
You can also find a gallery of these examples in the :ref:`examples gallery <sphx_glr_auto_examples>`.

The **reference documentation** provides an overview of the code-level functinoality and documents.


XXXX Stuff to sort
==================

.. toctree::
    :maxdepth: 1

    tutorials/philosophy.rst
    manual/cookbook.rst
    workflows.rst
