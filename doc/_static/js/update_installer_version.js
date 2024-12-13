/* inspired by https://tobiasahlin.com/blog/move-from-jquery-to-vanilla-javascript/ */

function documentReady(callback) {
    if (document.readyState != "loading") callback();
    else document.addEventListener("DOMContentLoaded", callback);
}

async function getRelease() {
    result = await fetch("https://api.github.com/repos/mne-tools/mne-installers/releases/latest");
    data = await result.json();
    return data;
}
async function warnVersion() {
    data = await getRelease();
    // Take v1.5.1 for example and change to 1.5
    ids = ["linux-installers", "macos-intel-installers", "macos-apple-installers", "windows-installers"];
    warn = false;
    ids.forEach((id) => {
        label_id = document.getElementById(id);
        // tab is immediately after label
        children = [].slice.call(label_id.parentNode.children);
        div = children[children.indexOf(label_id) + 1];
        a = div.children[0].children[0];  // div->p->a
        ending = a.href.split("-").slice(-1)[0];  // Should be one of: ["macOS_Intel.pkg", "macOS_M1.pkg", "Linux.sh", "Windows.exe"]
        data["assets"].every((asset) => {
            // find the matching asset
            if (!asset["browser_download_url"].endsWith(ending)) {
                return true;  // continue
            }
            old_stem = a.href.split("/").slice(-1)[0];
            new_stem = asset["browser_download_url"].split("/").slice(-1)[0];
            a.href = asset["browser_download_url"];
            // also replace the command on Linux
            if (ending === "Linux.sh") {
                code = document.getElementById("codecell0");
            }
            if (!warn) {
                // MNE-Python-1.5.1_0-Linux.sh to 1.5 for example
                old_ver = old_stem.split("-").slice(2)[0].split("_")[0].split(".").slice(0, 2).join(".");
                new_ver = new_stem.split("-").slice(2)[0].split("_")[0].split(".").slice(0, 2).join(".");
                if (old_ver !== new_ver) {
                    warn = `The installers below are for version ${new_ver} as ${old_ver} is no longer supported`;
                }
            }
            return false;  // do not continue
        });
    });
    if (warn) {
        let outer = document.createElement("div");
        let title = document.createElement("p");
        let inner = document.createElement("p");
        outer.setAttribute("class", "admonition warning");
        title.setAttribute("class", "admonition-title");
        title.innerText = "Warning";
        inner.innerText = warn;
        outer.append(title, inner);
        document.querySelectorAll('.install-selector-tabset')[0].before(outer);
    }
}

documentReady(warnVersion);
