function anchorScrollFix() {
    // this prevents anchors from being hidden under the navbar when the page
    // scrolls to them (citation/equation backlinks are worst affected).
    // adapted from https://github.com/copyleft-org/copyleft-guide/commit/476a42bf0d737e13a561dbaf6f4e1e91a333e80d
    if ($(".navbar.fixed-top").length > 0) {
        var navHeight = $(".navbar").height();
        var shiftWindow = function() {
            if (document.URL.search(/#[a-z]+/) > 0) {
                scrollBy(0, -navHeight - 12);
            };
        };
        if (location.hash) shiftWindow();
        window.addEventListener("hashchange", shiftWindow);
    };
};

function navbarShadow() {
    var addRemoveShadow = function() {
        if ($('.navbar-toggler').css('display') == 'none') {
            $('#navbar-menu').removeClass('shadow-lg');
        } else {
            $('#navbar-menu').addClass('shadow-lg');
        };
    };
    window.addEventListener('resize', addRemoveShadow);
};

$(document).ready( () => {
    anchorScrollFix();
    navbarShadow();
});
