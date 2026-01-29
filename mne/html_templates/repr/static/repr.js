// must be `var` (not `const`) because this can get embedded multiple times on a page
var toggleVisibility = (className) => {

    const elements = document.querySelectorAll(`.${className}`);

    elements.forEach(element => {
        if (element.classList.contains("mne-repr-section-header")) {
            return  // Don't collapse the section header row
        }
        element.classList.toggle("mne-repr-collapsed");
    });

    // trigger caret to rotate
    var sel = `.mne-repr-section-header.${className} > th.mne-repr-section-toggle > button`;
    const button = document.querySelector(sel);
    button.classList.toggle("collapsed");

    // adjust tooltip
    sel = `tr.mne-repr-section-header.${className}`;
    const secHeadRow = document.querySelector(sel);
    secHeadRow.classList.toggle("collapsed");
    secHeadRow.title = secHeadRow.title === "Hide section" ? "Show section" : "Hide section";
}
