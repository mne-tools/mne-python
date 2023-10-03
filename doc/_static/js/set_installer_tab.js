/* inspired by https://tobiasahlin.com/blog/move-from-jquery-to-vanilla-javascript/ */

function documentReady(callback) {
    if (document.readyState != "loading") callback();
    else document.addEventListener("DOMContentLoaded", callback);
}

function setTabs() {
    var platform = "linux";
    if (navigator.userAgent.indexOf("Win") !== -1) {
        platform = "windows";
    }
    if (navigator.userAgent.indexOf("Mac") !== -1) {
        // there's no good way to distinguish intel vs M1 in javascript so we
        // just default to showing the first of the 2 macOS tabs
        platform = "macos-intel";
    }
    let all_tab_nodes = document.querySelectorAll(
        '.platform-selector-tabset')[0].children;
    let input_nodes = [...all_tab_nodes].filter(
        child => child.nodeName === "INPUT");
    let tab_label_nodes = [...document.querySelectorAll('.sd-tab-label')];
    let correct_label = tab_label_nodes.filter(
        // label.id is drawn from :name: property in the rST, which must
        // be unique across the whole site (*sigh*)
        label => label.id.startsWith(platform))[0];
    let input_id = correct_label.getAttribute('for');
    let correct_input = input_nodes.filter(node => node.id === input_id)[0];
    correct_input.checked = true;
}

documentReady(setTabs);
