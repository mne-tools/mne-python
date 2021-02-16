$(document).ready(function () {
    if(location.hash != null && location.hash != ""){
        $('.collapse').removeClass('show');
        $(location.hash + '.collapse').addClass('show');
    }
});
