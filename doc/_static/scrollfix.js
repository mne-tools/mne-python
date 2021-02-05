/* this prevents in-text anchors from being hidden under the navbar when the */
/* page scrolls to them (citation/equation backlinks are worst affected).    */
/* adapted from https://github.com/copyleft-org/copyleft-guide/commit/476a42bf0d737e13a561dbaf6f4e1e91a333e80d */
$(document).ready(function() {
    if ($(".navbar.fixed-top").length > 0) {
        var navHeight = $(".navbar").height();
        var shiftWindow = function() {
            var ourURL = document.URL;
            if ( (ourURL.search(/#[a-z]+\d\d\d\d[a-z]?$/) > 0) ||  // footbib item (bibtex key)
                 (ourURL.search("#id") > 0) ||                     // footbib backreference
                 (ourURL.search("#equation") > 0) ) {              // numbered equation
                scrollBy(0, -navHeight - 12);
            };
        };
        if (location.hash) shiftWindow();
        window.addEventListener("hashchange", shiftWindow);
    };
});
