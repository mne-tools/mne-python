$(document).ready(function () {
    if(location.hash != null && location.hash != ""){
        $('.collapse').removeClass('in');
        $(location.hash + '.collapse').addClass('in');
    }
});
