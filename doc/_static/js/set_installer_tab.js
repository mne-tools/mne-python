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
        // just default to showing the most modern macOS installer
        platform = "macos-apple";
    }
    var platform_short = platform.split("-")[0];

    let tab_label_nodes = [...document.querySelectorAll('.sd-tab-label')];

    let install_tab_nodes = document.querySelectorAll(
        '.install-selector-tabset')[0].children;
    let install_input_nodes = [...install_tab_nodes].filter(
        child => child.nodeName === "INPUT");
    let install_label = tab_label_nodes.filter(
        // label.id is drawn from :name: property in the rST, which must
        // be unique across the whole site (*sigh*)
        label => label.id.startsWith(`install-${platform}`))[0];
    let install_id = install_label.getAttribute('for');
    let install_input = install_input_nodes.filter(node => node.id === install_id)[0];
    install_input.checked = true;

    let uninstall_tab_nodes = document.querySelectorAll(
        '.uninstall-selector-tabset')[0].children;
    let uninstall_input_nodes = [...uninstall_tab_nodes].filter(
        child => child.nodeName === "INPUT");
    let uninstall_label = tab_label_nodes.filter(
        label => label.id.startsWith(`uninstall-${platform_short}`))[0];
    let uninstall_id = uninstall_label.getAttribute('for');
    let uninstall_input = uninstall_input_nodes.filter(node => node.id === uninstall_id)[0];
    uninstall_input.checked = true;
}

function setAlert() {
    for (let button of document.querySelectorAll('.install-download-button')) {
        button.addEventListener('click', function() {
            alert = document.querySelectorAll('.install-download-alert')[0];
            alert.style.display = 'block';
        });
    }
}

documentReady(setTabs);
documentReady(setAlert);
